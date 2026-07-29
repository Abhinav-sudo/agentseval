"""LLM-as-judge scoring for arbitrary (prompt, response) pairs.

This is the centre of the deliverable (PROJECT.md). The judge must score **arbitrary
(prompt, response) pairs from an external file supplied by a grader** — pairs produced by
someone else's model, in someone else's format, about subjects outside our corpus. So:

* the only required inputs are a prompt and a response. No trace, no tool calls, no
  reference answer, no dependency on `runs/` layout or on our agents having produced it;
* `load_pairs` is a real feature with real input handling (JSONL, JSON array, CSV; field
  aliases such as question/input and answer/output/completion), because a grader's file
  will not match our internal schema;
* a missing optional field degrades the rubric gracefully rather than raising.

The judge model is a third family, distinct from both agents, to avoid self-preference
bias, and runs at temperature 0 so re-scoring the same file reproduces. Every judgement is
logged as JSONL under a run manifest, including the raw judge completion — a score whose
reasoning was discarded cannot be audited.

Scoring is decoupled from running: `evals.runner` produces responses, this module scores
them, and a rubric change means re-scoring rather than re-running the agents.

Five things this module holds to, each of which is a claim a reader can check:

**Nothing from our internals reaches the prompt.** `build_judge_messages` is the only place a
message is assembled, and it passes exactly three strings: the prompt, the response, and an
optional `reference`. `EvalItem` is never imported here, so `expected_behavior` and `notes` —
which `schema.ANNOTATOR_ONLY_FIELDS` withholds from every model — cannot arrive by accident;
`load_pairs` also drops those two column names rather than carrying them into a record. The arm,
the `run_id`, and the model name stay in `JudgePair.metadata`, which is never rendered. A judge
that read our fields would behave differently on our items than on a grader's, which is the one
thing this module may not do.

**The judge is blinded, with `label.scrub_model_names`** rather than a second scrubber: that one
is built from `base.PRICING` plus a vendor list, so pricing a new model extends the blind
automatically. The redaction count is recorded, because a response that names its own model is
both a leak in the blind and a fact about the model.

**Scores are 1-`JUDGE_SCALE_MAX` per dimension plus `overall`, and nothing else.** No `label`,
no pass/fail: `schema.LabelSpace` keeps `rubric_1_5` and `binary_behavioral` apart on purpose,
and a judge emitting both would be inventing a per-call mapping between them. Any threshold is
pre-registered in README.md before a graded run.

**A parse failure is ours, not the candidate's.** It is recorded with `parse_ok=False`, `overall`
None, and the raw completion kept — never a zero, which would average in as a real judgement.
One repair attempt is made, and because `base.ResponseCache.key` covers the messages, that
attempt has to be a *different request*: the malformed completion goes back as an assistant turn
with an instruction naming the defect. Re-sending the original messages at temperature 0 would
replay the same cached completion and the retry would measure nothing. Repaired verdicts are
marked, so first-pass parse rate stays readable as the number describing the rubric's clarity.
None of this is `ChatAdapter`'s 429/5xx backoff, which stays where it is: that is the network,
this is the content, and a rate limit counted as a parse failure makes both unreadable.

**Evidence spans are verified.** Each is checked as a verbatim substring, after whitespace
normalisation, of the response the judge actually saw. Unverified spans are counted onto the
judgement rather than voiding it: a judge quoting text that is not there is a cheap deterministic
signal of a fabricating judge, and it is the judge-side analogue of
`deterministic.check_citation_grounding`.

Two things deliberately absent. There is no statistic here: `sample_verdicts` exposes the
sampling primitive and `validate_judge.check_stability` owns the number, as it owns block-order
sensitivity and self-preference. And there is no pairwise mode — every verdict is on one response
alone, which is why `axis="bias"` scores a single response and the within-pair delta that measures
bias is computed later, from two judgements, in `metrics.py`.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.core import strip_code_fence
from agent.manifest import JudgeRef, build_manifest
from agent.models.base import (
    DEFAULT_MAX_TOKENS,
    ChatMessage,
    ModelResponse,
    ResponseCache,
    add_cache_arguments,
    cache_enabled,
    load_env,
)
from agent.models.judge_model import JudgeAdapter, load_judge_model
from agent.prompts import (
    CANONICAL_BLOCK_ORDER,
    JUDGE_DEFAULT_RUBRIC,
    JUDGE_DIMENSIONS,
    JUDGE_EVIDENCE_KEY,
    JUDGE_SCALE_MAX,
    MAX_EVIDENCE_CHARS,
    MAX_EVIDENCE_SPANS,
    judge_rubric_names,
    judge_rubric_prompt,
    judge_rubric_sha256,
    judge_schema,
    normalise_whitespace,
    render_judge_pair,
    render_judge_repair_request,
)
from agent.trace import DEFAULT_RUNS_DIR, read_records, trace_path
from evals.label import scrub_model_names
from evals.schema import ANNOTATOR_ONLY_FIELDS, Axis

#: Temperature for every graded verdict. Fixed, not configurable: two judgements taken at
#: different temperatures are not two measurements of the same thing.
JUDGE_TEMPERATURE = 0.0

#: Temperature for stability sampling, fixed and recorded rather than tuned. See
#: `sample_verdicts` and `validate_judge.check_stability`.
STABILITY_TEMPERATURE = 0.7

#: Samples below this cannot show variance, so asking for one is a mistake worth naming.
MIN_STABILITY_SAMPLES = 2

#: How many repair attempts a malformed verdict gets. One: a second would mostly measure how
#: long the judge can be argued with, and the first-pass rate is the number that matters.
MAX_REPAIR_ATTEMPTS = 1

#: Column names accepted for each field, in preference order. Long enough to cover what a
#: grader's file plausibly calls these two things, and no longer: a guessed mapping that is
#: wrong is worse than an error naming the columns that were found.
PROMPT_ALIASES = ("prompt", "question", "input", "instruction", "query", "user_message")
RESPONSE_ALIASES = ("response", "answer", "output", "completion", "model_response", "generation")
REFERENCE_ALIASES = ("reference", "reference_answer", "gold", "gold_answer", "ideal")
PAIR_ID_ALIASES = ("pair_id", "id", "case_id", "item_id", "uid")
SOURCE_ALIASES = ("source", "dataset", "origin")

#: Column names accepted for a human label, for `validate_judge.load_labelled`. Here rather
#: than there so that one file owns every alias a grader's column might be resolved through: a
#: second table in a second module is how the two paths end up disagreeing about what `rating`
#: means. Nothing in this module reads them — `load_pairs` leaves an unrecognised label column
#: in `JudgePair.metadata`, since a judge must never see the answer it is being scored against.
#:
#: `gold` is deliberately in both this tuple and `REFERENCE_ALIASES`, and the two readings are
#: incompatible: in a grader's scoring file it is the reference answer, in a judge-validation
#: file it is the human label. Neither module may guess. `load_pairs` resolves it as a
#: reference, `validate_judge` resolves the label first and hides the consumed column, and both
#: say which they did in their errors.
LABEL_ALIASES = ("human_score", "human_label", "gold", "label", "rating", "score")
ANNOTATOR_ALIASES = ("annotator", "labeller", "labeler", "rater", "judge_by", "annotated_by")

#: Suffix of the judgements file, written beside the run's manifest. Its own file rather than
#: the candidate's trace: a judge failure must not be able to land in the candidate's rates.
JUDGE_SUFFIX = ".judge.jsonl"

EXIT_OK = 0
EXIT_FAILED = 1


@dataclass
class JudgePair:
    """One (prompt, response) pair to be scored.

    Everything except `prompt` and `response` is optional, so a grader's two-column file
    is valid input.

    Attributes:
        reference: External ground truth, if the file carries any. The rubric frames it as one
            acceptable answer rather than the only one, so a correct response worded differently
            is not penalised. This is the only slot a "right answer" may enter through; our own
            `expected_behavior` is an annotator instruction and never becomes one.
        metadata: Everything else the input file carried. Recorded with the judgement and never
            rendered into a message, which is where the arm and the run id live.
    """

    prompt: str
    response: str
    pair_id: str | None = None
    reference: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeAttempt:
    """One request to the judge, recorded whether or not it parsed.

    Both attempts of a repaired judgement are kept, because a repaired verdict is not the same
    measurement as a first-pass one and the difference has to be visible to the metric that
    reports either.

    Attributes:
        cache_key: `base.ResponseCache.key` for the request. Recorded so that "the repair was a
            different request" is checkable rather than asserted: the second attempt appends two
            messages, so its key differs and it cannot be served the first attempt's completion.
        cached: Whether this attempt was a replay. A cache hit is a replay, not a measurement
            (PROJECT.md), which is why `sample_verdicts` refuses to report on one.
        defect: Why the completion did not parse, or None when it did.
    """

    completion: str
    cache_key: str
    cached: bool
    defect: str | None = None
    latency_ms: float | None = None
    usd_cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "completion": self.completion,
            "cache_key": self.cache_key,
            "cached": self.cached,
            "defect": self.defect,
            "latency_ms": self.latency_ms,
            "usd_cost": self.usd_cost,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JudgeAttempt:
        """Rebuild an attempt from one written by `to_dict`."""
        return cls(
            completion=str(data.get("completion") or ""),
            cache_key=str(data.get("cache_key") or ""),
            cached=bool(data.get("cached")),
            defect=data.get("defect"),
            latency_ms=data.get("latency_ms"),
            usd_cost=data.get("usd_cost"),
        )


@dataclass
class JudgeScore:
    """A judgement, kept alongside the raw completion that produced it.

    Attributes:
        scores: One score per `prompts.JUDGE_DIMENSIONS`, each 1-`JUDGE_SCALE_MAX`. Empty when
            the verdict did not parse.
        overall: The judge's holistic score, or None when nothing parsed. Nullable precisely so
            that a failed parse cannot be averaged in as a zero — an unparsed judgement and a
            genuinely bad response are different findings, and `parse_ok` is what tells them
            apart.
        rubric: Which rubric file was read. Together with `rubric_sha256` this is what makes a
            score traceable to the text that produced it.
        evidence: Spans the judge quoted that are really in the response it saw.
        evidence_unverified: Spans that are not. Counted rather than discarded.
        redactions: How many model or vendor names were scrubbed out of the pair.
        repaired: True when the verdict came from the repair attempt rather than the first pass.
        attempts: Every request made for this judgement, in order.
        block_order: Which order the pair's blocks were rendered in, defaulting to
            `prompts.CANONICAL_BLOCK_ORDER` — the order every graded judgement is produced under. A
            judgement under any other order came from
            `validate_judge.check_block_order_sensitivity` and is not comparable to a graded one,
            so it is recorded per judgement rather than per run: a sensitivity check varies it
            within one run by design, which is exactly why it cannot live in the manifest.
    """

    pair_id: str
    scores: dict[str, float]
    overall: float | None
    rationale: str
    raw_completion: str
    judge_model: str
    parse_ok: bool = True
    error: str | None = None
    axis: str | None = None
    rubric: str = JUDGE_DEFAULT_RUBRIC
    rubric_sha256: str = ""
    evidence: list[str] = field(default_factory=list)
    evidence_unverified: list[str] = field(default_factory=list)
    redactions: int = 0
    repaired: bool = False
    temperature: float = JUDGE_TEMPERATURE
    attempts: list[JudgeAttempt] = field(default_factory=list)
    run_id: str | None = None
    usd_cost: float | None = None
    block_order: tuple[str, ...] = CANONICAL_BLOCK_ORDER

    @property
    def cached(self) -> bool:
        """True when every attempt was served from the cache, so nothing here was measured."""
        return bool(self.attempts) and all(attempt.cached for attempt in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        """Render one JSONL line: the judgement, its provenance, and its raw text."""
        return {
            "run_id": self.run_id,
            "pair_id": self.pair_id,
            "judge_model": self.judge_model,
            "axis": self.axis,
            "rubric": self.rubric,
            "rubric_sha256": self.rubric_sha256,
            "temperature": self.temperature,
            "parse_ok": self.parse_ok,
            "repaired": self.repaired,
            "scores": self.scores,
            "overall": self.overall,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "evidence_unverified": self.evidence_unverified,
            "redactions": self.redactions,
            "error": self.error,
            "usd_cost": self.usd_cost,
            "cached": self.cached,
            "block_order": list(self.block_order),
            "raw_completion": self.raw_completion,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JudgeScore:
        """Rebuild a judgement from one line of a `.judge.jsonl` file.

        The inverse of `to_dict`, and here rather than in `evals.metrics` so that the shape of a
        judgement has one definition and the reader of the file cannot drift from its writer.
        `overall` stays nullable through the round trip: an unparsed judgement must not come back
        as a zero, which is the whole reason the field is optional in the first place.

        Tolerant of a missing optional key, unlike `RunManifest.from_dict`, and for the opposite
        reason: a manifest's field set decides whether two runs are comparable, so an absent key
        there is a fact worth refusing over, whereas a judgement is a datum and a reader that
        rejected a file written by a slightly older judge would strand the scores rather than
        protect anything.

        `parse_ok` is the exception to that tolerance, and it defaults to False. A line that does
        not claim to have parsed is treated as one that did not, so the failure mode is a
        judgement dropped from a denominator — visible as `n_unjudged` — rather than an unparsed
        completion's zeros entering a rate as though a judge had really said them.
        """
        return cls(
            pair_id=str(data.get("pair_id") or ""),
            scores={str(name): float(value) for name, value in (data.get("scores") or {}).items()},
            overall=None if data.get("overall") is None else float(data["overall"]),
            rationale=str(data.get("rationale") or ""),
            raw_completion=str(data.get("raw_completion") or ""),
            judge_model=str(data.get("judge_model") or ""),
            parse_ok=bool(data.get("parse_ok")),
            error=data.get("error"),
            axis=data.get("axis"),
            rubric=str(data.get("rubric") or JUDGE_DEFAULT_RUBRIC),
            rubric_sha256=str(data.get("rubric_sha256") or ""),
            evidence=[str(span) for span in data.get("evidence") or []],
            evidence_unverified=[str(span) for span in data.get("evidence_unverified") or []],
            redactions=int(data.get("redactions") or 0),
            repaired=bool(data.get("repaired")),
            temperature=float(data.get("temperature", JUDGE_TEMPERATURE)),
            attempts=[JudgeAttempt.from_dict(item) for item in data.get("attempts") or []],
            run_id=data.get("run_id"),
            usd_cost=data.get("usd_cost"),
            block_order=tuple(data.get("block_order") or CANONICAL_BLOCK_ORDER),
        )


def read_scores(path: Path) -> list[JudgeScore]:
    """Read a `.judge.jsonl` file back into judgements.

    The counterpart to `write_scores`. `evals.metrics` joins these onto a candidate's trace on
    `pair_id`, which is the `item_id` when the pairs came from one of our runs.

    Raises:
        ValueError: a line is not a JSON object.
    """
    scores: list[JudgeScore] = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno} is not a JSON object")
            scores.append(JudgeScore.from_dict(record))
    return scores


@dataclass(frozen=True)
class JudgeRequest:
    """What `build_judge_messages` assembled, and what the judge will therefore see.

    Returned as a whole rather than as bare messages because three of these are needed after the
    call: `response` is the text an evidence span must be verbatim in — the *scrubbed* text, since
    that is what the judge read — and `redactions` and `block_order` belong on the judgement.
    """

    messages: list[ChatMessage]
    redactions: int
    response: str
    rubric: str
    block_order: tuple[str, ...] = CANONICAL_BLOCK_ORDER


# --------------------------------------------------------------------------------------
# Input handling: a grader's file, in whatever shape it arrives
# --------------------------------------------------------------------------------------


def first_alias(record: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    """Return the first alias present in `record` with a non-empty value.

    Public because `validate_judge` resolves two more columns (the human label and the
    annotator) out of the same records, against the alias tuples above. One resolver and one
    set of tables, so "what counts as a present column" cannot mean two things in one package.
    """
    for alias in aliases:
        value = record.get(alias)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return alias
    return None


def pair_from_mapping(record: Mapping[str, Any], index: int) -> JudgePair:
    """Build one pair from one record of a grader's file.

    Unrecognised columns are kept in `metadata`, which is never rendered into a message, so a
    grader's extra columns survive into the output without becoming judge input. The two
    annotator-only field names are dropped instead: they are instructions written for a human,
    and `schema.ANNOTATOR_ONLY_FIELDS` withholds them from every model.

    Raises:
        ValueError: no prompt or no response column. The message names the fields that were
            found and the aliases that were looked for, since a grader needs to know what to fix.
    """
    prompt_key = first_alias(record, PROMPT_ALIASES)
    response_key = first_alias(record, RESPONSE_ALIASES)
    if prompt_key is None or response_key is None:
        wanted = "prompt" if prompt_key is None else "response"
        aliases = PROMPT_ALIASES if prompt_key is None else RESPONSE_ALIASES
        raise ValueError(
            f"record {index} has no {wanted}: found fields {sorted(record)}, "
            f"expected one of {list(aliases)} with a non-empty value"
        )

    reference_key = first_alias(record, REFERENCE_ALIASES)
    pair_id_key = first_alias(record, PAIR_ID_ALIASES)
    source_key = first_alias(record, SOURCE_ALIASES)
    consumed = {prompt_key, response_key, reference_key, pair_id_key, source_key}

    metadata = {
        key: value
        for key, value in record.items()
        if key not in consumed and key not in ANNOTATOR_ONLY_FIELDS
    }
    return JudgePair(
        prompt=str(record[prompt_key]),
        response=str(record[response_key]),
        pair_id=str(record[pair_id_key]) if pair_id_key else f"pair-{index:04d}",
        reference=str(record[reference_key]) if reference_key else None,
        source=str(record[source_key]) if source_key else None,
        metadata=metadata,
    )


def _records_from_jsonl(text: str, path: Path) -> list[Mapping[str, Any]]:
    """Parse JSONL, naming the line that failed.

    A malformed line raises rather than being skipped: a grader who supplied 200 pairs and got
    198 scores back has no way to notice, and a silently short run is the failure mode that
    corrupts a comparison without announcing itself.
    """
    records: list[Mapping[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{lineno} is {type(record).__name__}, not an object")
        records.append(record)
    return records


def _is_json_document(text: str) -> bool:
    """True when the whole text is one JSON value, rather than JSONL."""
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _records_from_json(text: str, path: Path) -> list[Mapping[str, Any]]:
    """Parse a JSON array of pairs, an object wrapping one, or a single pair.

    Three shapes because all three turn up in files people hand over, and the difference between
    them is not a difference in what was meant.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if isinstance(payload, Mapping):
        lists = [value for value in payload.values() if isinstance(value, list)]
        if len(lists) == 1:
            payload = lists[0]
        elif not lists and first_alias(payload, PROMPT_ALIASES):
            payload = [payload]
        else:
            raise ValueError(
                f"{path} is a JSON object with {len(lists)} list-valued field(s) and is not "
                f"itself a pair; supply an array of pairs, JSONL, or an object with exactly one "
                f"list of pairs. Fields found: {sorted(payload)}"
            )
    if not isinstance(payload, list):
        raise ValueError(f"{path} holds {type(payload).__name__}, not an array of pairs")

    records: list[Mapping[str, Any]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"{path} item {index} is {type(record).__name__}, not an object")
        records.append(record)
    return records


def _records_from_csv(text: str, path: Path) -> list[Mapping[str, Any]]:
    """Parse CSV or TSV, sniffing the delimiter and normalising the header row.

    Headers are trimmed *and* case-folded, matching `validate_judge._normalise_keys`. Every
    alias tuple above is lower-case and `first_alias` looks columns up exactly, so a
    spreadsheet's `Question,Answer` resolved through neither and the first command a grader runs
    failed on the most ordinary file there is. `Question` versus `question` is not a difference
    in what was meant, and the two entry points must not disagree about that: the same file fed
    to `agentseval-judge` and to `agentseval-validate-judge` has to parse the same way.

    Only the header is touched. Values keep their exact text, since a response's leading
    whitespace is part of the response.
    """
    sample = text[:4096]
    try:
        dialect: Any = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError(f"{path} has no header row, so its columns cannot be identified")

    reader.fieldnames = [(name or "").strip().casefold() for name in reader.fieldnames]
    records: list[Mapping[str, Any]] = []
    for row in reader:
        # A short row leaves None values, and a long one collects the surplus under None.
        records.append({key: value for key, value in row.items() if key and value is not None})
    return records


def read_pair_records(path: Path) -> list[Mapping[str, Any]]:
    """Read a grader's file into raw records, without interpreting any column.

    The format half of `load_pairs`, split out because `validate_judge` needs the records
    before the columns are consumed: it resolves the human label first, and a `gold` column
    already turned into `JudgePair.reference` cannot be resolved as a label afterwards. Sharing
    this function is what keeps one set of format rules — the suffix dispatch, the delimiter
    sniff, and the errors naming the line that failed — behind both entry points.

    Raises:
        ValueError: the file is empty, holds no records, or has an unrecognised suffix.
        FileNotFoundError: there is no such file.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"{path} is empty: no pairs to score")

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records = _records_from_jsonl(text, path)
    elif suffix in {".csv", ".tsv"}:
        records = _records_from_csv(text, path)
    elif suffix == ".json":
        # Several objects one per line is not valid JSON as a whole, which is what distinguishes
        # JSONL-under-a-.json-name from the three JSON shapes. One object on one line parses
        # either way and means the same thing.
        records = (
            _records_from_json(text, path)
            if _is_json_document(text)
            else _records_from_jsonl(text, path)
        )
    else:
        raise ValueError(
            f"{path} has an unrecognised suffix {suffix!r}; supply .jsonl, .json, .csv, or .tsv"
        )

    if not records:
        raise ValueError(f"{path} holds no records: no pairs to score")
    return records


def load_pairs(path: Path) -> list[JudgePair]:
    """Load (prompt, response) pairs from an external, grader-supplied file.

    Accepts JSONL, a JSON array, or CSV, and maps common field aliases (prompt/question/
    input, response/answer/output/completion) onto `JudgePair`. Assigns positional
    `pair_id`s when the file has none, so results can be joined back to the input.

    The format is chosen by suffix, with one exception: a `.json` file whose first
    non-whitespace character starts a JSON object is read as JSONL, because a grader whose
    JSONL happens to be named `.json` should get scores rather than a lecture.

    Args:
        path: Any file of (prompt, response) pairs. Nothing about our agents, trace
            format, or knowledge base may be assumed.

    Raises:
        ValueError: the file cannot be interpreted as (prompt, response) pairs. The error
            names the fields that were found, since a grader needs to know what to fix.
        FileNotFoundError: there is no such file.
    """
    records = read_pair_records(path)
    return [pair_from_mapping(record, index) for index, record in enumerate(records)]


def pairs_from_trace(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> list[JudgePair]:
    """Extract the scored (prompt, response) pairs from one of our own run traces.

    The pair is the last user message of an item and the finished `turn` record that answered
    it, matching `schema.SCORED_TURN_INDEX`: on a multi-turn escalation item the earlier answers
    are context, and scoring one would grade the agent partway through the escalation. Since
    scoring is single-response, the judge sees that final exchange and not the turns before it —
    the same view a grader's two-column file gives, which is the point.

    The arm and the run id go into `metadata`, never into a message.

    Raises:
        FileNotFoundError: there is no trace for `run_id`.
        ValueError: the trace has no scored turns.
    """
    path = trace_path(run_id, runs_dir)
    if not path.exists():
        raise FileNotFoundError(f"no trace for run {run_id!r} at {path}")

    prompts: dict[str, str] = {}
    pairs: dict[str, JudgePair] = {}
    for record in read_records(path):
        item_id = record.get("item_id")
        if not item_id:
            continue
        role = record.get("role")
        if role == "user":
            prompts[item_id] = record.get("content") or ""
        elif role == "turn":
            pairs[item_id] = JudgePair(
                prompt=prompts.get(item_id, ""),
                response=record.get("content") or "",
                pair_id=item_id,
                source=str(path),
                metadata={"run_id": run_id, "turn_idx": record.get("turn_idx")},
            )

    missing_prompt = sorted(pair_id for pair_id, pair in pairs.items() if not pair.prompt)
    if missing_prompt:
        raise ValueError(
            f"{path}: no user message before the scored turn of {missing_prompt}; the judge "
            "cannot score a response to a prompt that is not in the trace"
        )
    if not pairs:
        raise ValueError(f"{path} has no completed turns to score")
    return list(pairs.values())


# --------------------------------------------------------------------------------------
# Assembling the judge's messages: the one seam our internals could leak through
# --------------------------------------------------------------------------------------


def _rubric_name(axis: Axis | str | None) -> str:
    """Resolve an axis to a rubric name, defaulting when there is no axis.

    Optional by design: a grader's file has no axis, and a missing optional field must degrade
    the rubric gracefully. A *named* axis with no rubric file raises instead — see
    `prompts.judge_rubric_prompt`.
    """
    if axis is None:
        return JUDGE_DEFAULT_RUBRIC
    return axis.value if isinstance(axis, Axis) else str(axis)


def build_judge_messages(
    pair: JudgePair,
    *,
    axis: Axis | None = None,
    block_order: Sequence[str] = CANONICAL_BLOCK_ORDER,
) -> JudgeRequest:
    """Assemble the messages for one pair, blinded.

    The single place a judge message is built, which is what makes "no model name and no
    annotator field reaches the judge" a property of the code rather than a convention. Exactly
    three strings are rendered — prompt, response, and reference — each passed through
    `label.scrub_model_names` first; `pair.metadata`, `pair.source`, and the pair id are not
    rendered at all.

    Args:
        block_order: Passed through to `prompts.render_judge_pair`. The default renders the
            canonical order, byte-identically to what a graded run produces. A non-default order is
            for `validate_judge.check_block_order_sensitivity` only, and the order is recorded on
            the resulting `JudgeScore` so a reordered judgement cannot be mistaken for a graded one.
    """
    rubric_name = _rubric_name(axis)
    rubric = judge_rubric_prompt(rubric_name)

    prompt, prompt_redactions = scrub_model_names(pair.prompt)
    response, response_redactions = scrub_model_names(pair.response)
    reference, reference_redactions = (
        scrub_model_names(pair.reference) if pair.reference else (None, 0)
    )

    messages: list[ChatMessage] = [
        {"role": "system", "content": rubric},
        {
            "role": "user",
            "content": render_judge_pair(prompt, response, reference, block_order=block_order),
        },
    ]
    return JudgeRequest(
        messages=messages,
        redactions=prompt_redactions + response_redactions + reference_redactions,
        response=response,
        rubric=rubric_name,
        block_order=tuple(block_order),
    )


# --------------------------------------------------------------------------------------
# Reading a verdict: strict, with one repair
# --------------------------------------------------------------------------------------


def _score_defect(key: str, value: Any) -> str | None:
    """Return why `value` is not a score, or None when it is one.

    An integral float is read as the integer it is — `4.0` is the documented shape arriving with
    a decimal point, not a different answer. A genuinely fractional score is refused: the human
    label space is 1-`JUDGE_SCALE_MAX` in whole steps, and a 4.5 has no counterpart to be
    compared against.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return f"{key} is {value!r}, not a number"
    if float(value) != int(value):
        return f"{key} is {value}, not a whole number on the 1-{JUDGE_SCALE_MAX} scale"
    if not 1 <= int(value) <= JUDGE_SCALE_MAX:
        return f"{key} is {int(value)}, outside 1-{JUDGE_SCALE_MAX}"
    return None


def _evidence_defect(spans: Any) -> str | None:
    """Return why `spans` is not a valid evidence list, or None when it is."""
    if not isinstance(spans, list) or any(not isinstance(span, str) for span in spans):
        return f"{JUDGE_EVIDENCE_KEY} is not a list of strings"
    if len(spans) > MAX_EVIDENCE_SPANS:
        return (
            f"{JUDGE_EVIDENCE_KEY} has {len(spans)} spans, more than the "
            f"{MAX_EVIDENCE_SPANS} allowed"
        )
    for span in spans:
        if len(span) > MAX_EVIDENCE_CHARS:
            return (
                f"an {JUDGE_EVIDENCE_KEY} span is {len(span)} characters, over the "
                f"{MAX_EVIDENCE_CHARS} limit"
            )
    return None


def parse_verdict(completion: str) -> tuple[dict[str, Any] | None, str | None]:
    """Read one judge completion strictly, returning `(verdict, defect)`.

    Exactly one of the two is None. A single markdown fence around the whole completion is
    accepted, because the rubric's own example could be fenced by a model that formats
    everything; anything else outside the object is a defect. This is deliberately stricter than
    `core.extract_json_object`, which searches for an object inside prose: there, leniency keeps a
    sloppy preamble from being scored as a total failure by the *candidate*, whereas here the
    judge is our instrument and its output format is a thing we control by fixing the rubric.

    The defect is phrased for `prompts.render_judge_repair_request`, so it has to name what was
    wrong rather than merely that something was.
    """
    text = strip_code_fence(completion).strip()
    if not text:
        return None, "the reply was empty"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"it was not one JSON object ({exc.msg} at line {exc.lineno})"
    if not isinstance(payload, dict):
        return None, f"it was a JSON {type(payload).__name__}, not an object"

    expected = set(judge_schema())
    missing = sorted(expected - set(payload))
    unexpected = sorted(set(payload) - expected)
    if missing or unexpected:
        return None, f"missing keys {missing} and unexpected keys {unexpected}"

    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        return None, "rationale was empty or not a string"

    for key in (*JUDGE_DIMENSIONS, "overall"):
        defect = _score_defect(key, payload[key])
        if defect is not None:
            return None, defect

    defect = _evidence_defect(payload[JUDGE_EVIDENCE_KEY])
    if defect is not None:
        return None, defect
    return payload, None


def verify_evidence(spans: Sequence[str], response: str) -> tuple[list[str], list[str]]:
    """Split `spans` into those really present in `response` and those not.

    Whitespace is normalised on both sides before comparing: a judge that re-wrapped a line it
    copied correctly has fabricated nothing, and counting that as a fabrication would be noise in
    the number that exists to catch real ones. Everything else — a changed word, an invented
    figure, a plausible sentence that was never written — fails.
    """
    haystack = normalise_whitespace(response)
    verified: list[str] = []
    unverified: list[str] = []
    for span in spans:
        needle = normalise_whitespace(span)
        target = verified if needle and needle in haystack else unverified
        target.append(span)
    return verified, unverified


def _ask(
    judge: JudgeAdapter,
    messages: list[ChatMessage],
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[ModelResponse, str]:
    """Call the judge, returning the response and the cache key of the request.

    The key is recomputed here rather than read off the adapter so that a caller can see two
    attempts were two different requests. It is the same function the adapter keys its cache
    with, so the two cannot disagree.
    """
    key = ResponseCache.key(
        judge.model_id,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=None,
    )
    return judge.generate(messages, temperature=temperature, max_tokens=max_tokens), key


def score_pair(
    pair: JudgePair,
    judge: JudgeAdapter,
    *,
    axis: Axis | None = None,
    temperature: float = JUDGE_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    run_id: str | None = None,
    block_order: Sequence[str] = CANONICAL_BLOCK_ORDER,
) -> JudgeScore:
    """Score one pair against the rubric in `agent.prompts.judge_rubric_prompt`.

    A judge response that does not parse is returned with `parse_ok=False` and the raw
    completion preserved, never as a silent zero — an unparsed judgement and a genuinely
    bad response are different findings.

    One repair attempt is made first, as a genuinely different request: the malformed completion
    goes back as an assistant turn together with an instruction naming the defect, which changes
    the messages and therefore the cache key. Re-sending the original messages would return the
    same cached completion and measure nothing.

    Args:
        axis: Selects the rubric text. None uses the default rubric, because a grader's file has
            no axis. It never changes the output schema: all four dimensions are scored on every
            axis, so scores stay comparable across axes.
        temperature: 0 for a graded verdict. `sample_verdicts` is the only caller that raises
            it, and its results never feed a graded score.
        max_tokens: Ceiling for the judge's reply. The evidence bounds in the rubric exist so a
            judge cannot spend this restating the response instead of judging it.
        block_order: Which order the pair's blocks are rendered in. The default is canonical and is
            what every graded judgement uses. `validate_judge.check_block_order_sensitivity` is the
            only caller that changes it, and the order used is recorded on the returned
            `JudgeScore` so a reordered verdict cannot enter a graded set unnoticed.

    Raises:
        ModelError: the provider failed and `ChatAdapter`'s retries did not clear it. Deliberately
            not caught: a transport failure recorded as a parse failure would make both numbers
            unreadable.
        ValueError: `axis` names a rubric with no file. See `prompts.judge_rubric_prompt`, or
            `block_order` is not a permutation of the canonical order.
    """
    request = build_judge_messages(pair, axis=axis, block_order=block_order)
    messages = list(request.messages)
    attempts: list[JudgeAttempt] = []
    verdict: dict[str, Any] | None = None
    defect: str | None = None

    for attempt_index in range(MAX_REPAIR_ATTEMPTS + 1):
        response, cache_key = _ask(judge, messages, temperature=temperature, max_tokens=max_tokens)
        verdict, defect = parse_verdict(response.text)
        attempts.append(
            JudgeAttempt(
                completion=response.text,
                cache_key=cache_key,
                cached=response.cached,
                defect=defect,
                latency_ms=response.latency_ms,
                usd_cost=response.usd_cost,
            )
        )
        if verdict is not None:
            break
        if attempt_index < MAX_REPAIR_ATTEMPTS:
            messages = [
                *messages,
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": render_judge_repair_request(defect or "")},
            ]

    costs = [attempt.usd_cost for attempt in attempts if attempt.usd_cost is not None]
    common = {
        "pair_id": pair.pair_id or "",
        "judge_model": judge.model_id,
        "axis": _rubric_name(axis) if axis is not None else None,
        "rubric": request.rubric,
        "rubric_sha256": judge_rubric_sha256(),
        "redactions": request.redactions,
        "temperature": temperature,
        "attempts": attempts,
        "run_id": run_id,
        "usd_cost": sum(costs) if costs else None,
        "raw_completion": attempts[-1].completion,
        "block_order": request.block_order,
    }

    if verdict is None:
        return JudgeScore(
            scores={},
            overall=None,
            rationale="",
            parse_ok=False,
            error=defect,
            repaired=False,
            **common,
        )

    verified, unverified = verify_evidence(verdict[JUDGE_EVIDENCE_KEY], request.response)
    return JudgeScore(
        scores={key: float(verdict[key]) for key in JUDGE_DIMENSIONS},
        overall=float(verdict["overall"]),
        rationale=str(verdict["rationale"]),
        parse_ok=True,
        evidence=verified,
        evidence_unverified=unverified,
        repaired=len(attempts) > 1,
        **common,
    )


def sample_verdicts(
    pair: JudgePair,
    *,
    n: int,
    temperature: float = STABILITY_TEMPERATURE,
    judge: JudgeAdapter | None = None,
    axis: Axis | None = None,
) -> list[JudgeScore]:
    """Take `n` independent verdicts on one pair, for `validate_judge.check_stability`.

    The primitive only. No statistic is computed here: majority share, mean pairwise agreement,
    and score variance are three different numbers, and `validate_judge` owns the choice as it
    owns block-order sensitivity and self-preference.

    **The cache has to be off, or this reports perfect agreement by construction.**
    `base.ResponseCache.key` covers the messages, the temperature, and `max_tokens`, so `n`
    identical requests share one key and the last `n - 1` would be replays of the first. A cache
    hit is a replay, not a measurement (PROJECT.md). So a cache-enabled adapter is refused up
    front, and a sample that still comes back cached raises rather than being counted.

    These verdicts never feed a graded score. That is the reason the paths are separate, and the
    reason 0.7 is fixed and recorded rather than tuned: a stability figure at 0.7 is an upper
    bound on the disagreement to expect at 0, which is worth having because providers are not
    bit-deterministic at 0 either.

    Raises:
        ValueError: `n` is below `MIN_STABILITY_SAMPLES`, the adapter has its cache on, or a
            sample came back cached.
    """
    if n < MIN_STABILITY_SAMPLES:
        raise ValueError(
            f"n={n} cannot show variance; stability needs at least {MIN_STABILITY_SAMPLES} samples"
        )

    judge = judge if judge is not None else load_judge_model(no_cache=True)
    if getattr(judge, "use_cache", False):
        raise ValueError(
            "stability sampling needs an adapter built with no_cache=True: identical messages "
            "share one cache key, so a cached run would report perfect agreement it never "
            "measured"
        )

    samples: list[JudgeScore] = []
    for index in range(n):
        sample = score_pair(pair, judge, axis=axis, temperature=temperature)
        if any(attempt.cached for attempt in sample.attempts):
            raise ValueError(
                f"sample {index} of {n} was served from the cache, so it is a replay of an "
                "earlier sample rather than a second measurement; no stability figure can be "
                "reported from these"
            )
        samples.append(sample)
    return samples


# --------------------------------------------------------------------------------------
# Judge runs: a run of its own, with its own manifest
# --------------------------------------------------------------------------------------


def judge_scores_path(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Return the judgements path for a judge run, sibling to its manifest."""
    return Path(runs_dir) / f"{run_id}{JUDGE_SUFFIX}"


def write_scores(scores: Sequence[JudgeScore], path: Path) -> Path:
    """Write judgements as JSONL, one object per line."""
    path = Path(path)
    if path.parent != Path():
        path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(score.to_dict(), ensure_ascii=False, default=str) for score in scores]
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    return path


def _score_pairs(
    pairs: Sequence[JudgePair],
    *,
    source: Path,
    judge: JudgeAdapter,
    axis: Axis | None,
    runs_dir: Path,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[JudgeScore]:
    """Score every pair as one judge run, appending each judgement as it is made.

    The manifest is written before the first call, and each judgement is flushed as it arrives,
    for the reason `TraceLogger` does the same: a run killed halfway must leave behind everything
    that happened up to the failure, and those are usually the records worth reading.
    """
    manifest = build_manifest(
        run_kind="judge",
        judge=JudgeRef.for_file(
            source,
            n_pairs=len(pairs),
            model_name=judge.model_id,
            provider=judge.provider,
            rubric_sha256=judge_rubric_sha256(),
            rubric_names=judge_rubric_names(),
            temperature=JUDGE_TEMPERATURE,
            max_tokens=max_tokens,
        ),
    )
    manifest.write(runs_dir)

    scores: list[JudgeScore] = []
    path = judge_scores_path(manifest.run_id, runs_dir)
    with path.open("a", encoding="utf-8") as handle:
        for pair in pairs:
            score = score_pair(
                pair,
                judge,
                axis=axis,
                max_tokens=max_tokens,
                run_id=manifest.run_id,
            )
            handle.write(json.dumps(score.to_dict(), ensure_ascii=False, default=str) + "\n")
            handle.flush()
            scores.append(score)
    return scores


def score_file(
    path: Path,
    *,
    judge: JudgeAdapter | None = None,
    runs_dir: Path = Path("runs"),
    axis: Axis | None = None,
) -> list[JudgeScore]:
    """Score every pair in an external file, logging each judgement as JSONL.

    Entry point for the grader-supplied case. Returns scores and writes a run directory
    with the manifest, so a published number can be traced to the judge model and rubric
    version behind it.

    The manifest is `run_kind="judge"`, whose conditions are the judge model and the rubric
    digest rather than an agent's configuration — `assert_comparable` therefore refuses to
    compare one of these against an eval run, which is the behaviour wanted.
    """
    pairs = load_pairs(path)
    judge = judge if judge is not None else load_judge_model()
    return _score_pairs(pairs, source=Path(path), judge=judge, axis=axis, runs_dir=runs_dir)


def score_run(
    run_id: str,
    *,
    judge: JudgeAdapter | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    axis: Axis | None = None,
) -> list[JudgeScore]:
    """Score the responses in one of our own runs.

    Convenience wrapper: extracts (prompt, response) pairs from the trace and calls the
    same scoring path as `score_file`. Our runs get no privileged treatment.

    The judgements go to a judge run of their own and never back into the candidate's trace, so
    a judge parse failure cannot be counted among the candidate's format violations. The trace
    is the scored file, digested as `pairs_sha256`: a re-run trace is different input, and the
    hash is what says so.
    """
    pairs = pairs_from_trace(run_id, runs_dir)
    judge = judge if judge is not None else load_judge_model()
    return _score_pairs(
        pairs,
        source=trace_path(run_id, runs_dir),
        judge=judge,
        axis=axis,
        runs_dir=runs_dir,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def render_summary(scores: Sequence[JudgeScore]) -> str:
    """Summarise a judge run: the judge-side numbers, kept separate from the candidate's.

    First-pass parse rate, repair rate, and unverified-span count are reported here because they
    describe the instrument rather than the responses. None of them belongs in a candidate's
    `format_violation_rate` (README.md).
    """
    total = len(scores)
    parsed = [score for score in scores if score.parse_ok]
    repaired = [score for score in parsed if score.repaired]
    first_pass = len(parsed) - len(repaired)
    unverified = sum(len(score.evidence_unverified) for score in scores)
    redactions = sum(score.redactions for score in scores)
    costs = [score.usd_cost for score in scores if score.usd_cost is not None]

    lines = [
        f"{total} pair(s) scored",
        f"first-pass parse: {first_pass}/{total}",
        f"repaired: {len(repaired)}",
        f"unparsed: {total - len(parsed)}",
        f"unverified evidence spans: {unverified}",
        f"names redacted from pairs: {redactions}",
    ]
    if parsed:
        mean_overall = sum(score.overall or 0.0 for score in parsed) / len(parsed)
        lines.append(f"mean overall (parsed only): {mean_overall:.2f}/{JUDGE_SCALE_MAX}")
    if costs:
        lines.append(f"judge cost: ${sum(costs):.4f}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `agentseval-judge --input pairs.jsonl [--out scores.jsonl]`.

    Returns 0 when every pair was scored and 1 when any judgement failed to parse — a run that
    could not read its own instrument's output should not look clean in CI.
    """
    load_env()
    parser = argparse.ArgumentParser(
        prog="agentseval-judge",
        description=(
            "Score (prompt, response) pairs with the judge rubric. Accepts an external file "
            "in JSONL, JSON, or CSV; nothing about our agents or corpus is assumed."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="file of (prompt, response) pairs to score")
    source.add_argument("--run", metavar="RUN_ID", help="score the responses in one of our runs")
    parser.add_argument(
        "--axis",
        choices=[axis.value for axis in Axis],
        help="read this axis's rubric instead of the default one",
    )
    parser.add_argument("--out", type=Path, help="also write the judgements here, as JSONL")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"where the judge run's manifest and judgements go (default: {DEFAULT_RUNS_DIR})",
    )
    add_cache_arguments(parser)
    args = parser.parse_args(argv)

    axis = Axis(args.axis) if args.axis else None
    judge = load_judge_model(no_cache=not cache_enabled(args.no_cache))

    if args.input is not None:
        scores = score_file(args.input, judge=judge, runs_dir=args.runs_dir, axis=axis)
    else:
        scores = score_run(args.run, judge=judge, runs_dir=args.runs_dir, axis=axis)

    run_id = scores[0].run_id if scores else None
    if args.out is not None:
        write_scores(scores, args.out)

    failed = [score for score in scores if not score.parse_ok]
    stream = sys.stderr if failed else sys.stdout
    print(f"judge run {run_id}", file=stream)
    print(f"judgements: {judge_scores_path(run_id or '', args.runs_dir)}", file=stream)
    if args.out is not None:
        print(f"copy: {args.out}", file=stream)
    print(render_summary(scores), file=stream)

    return EXIT_FAILED if failed else EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
