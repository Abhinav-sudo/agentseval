"""Execute eval sets against an agent and write traces.

Every turn is logged as JSONL against a run manifest (PROJECT.md) — a score that cannot be
traced back to the exact models, prompt version, tool inventory, corpus revision, and
configuration that produced it is not a result. Neither the manifest nor the writer belongs
here: `agent.manifest.build_manifest` builds every manifest in the project and
`agent.trace.TraceLogger` writes every trace, both of which the agent uses on its own. This
module supplies the one thing that is genuinely an eval-run property — the `DatasetRef` — and
hands the logger to the agent. It adds no manifest field of its own; a field the runner
recorded and the chat surface did not would be a condition nothing else could check.

Both agents run through the same runner over the same eval sets with the same tools and the
same `Budgets`. The only difference between arms is the model — a claim
`agent.manifest.assert_comparable` checks against the two manifests before their results are
compared, budgets included.

**The item shape lives in `evals.schema`, not here.** This module has no dataset type of its
own: two shapes in one package is a package where `deterministic.py` and `report.py` can read
one file and disagree about what is in it, with nothing raising to say so. `load_dataset`
returns `list[EvalItem]`, and `agentseval-validate-dataset` is what a file passes before a
graded run uses it.

Traces land in `runs/` (gitignored): `{run_id}.jsonl` plus `{run_id}.manifest.json`.

**This command runs; it does not score.** `evals.judge` is a run of its own — `run_kind="judge"`,
its own manifest carrying `judge_model`, `judge_rubric_sha256`, and `pairs_sha256`, its own
`runs/{judge_run_id}.judge.jsonl` — and its judgements never re-enter the candidate's trace, so
a judge parse failure cannot land in the candidate's `format_violation_rate`. A per-item judge
call inside the run would also make `pairs_sha256` the digest of a file still being appended to.
`--judge` is therefore an orchestrator and not a fusion: the eval finishes, then
`judge.score_run` opens a second run, and both ids are reported. The per-item join over the two
belongs with `metrics.summarise_run`, which already owns the trace + judge + checks join.

Four properties this module holds to, each of which is the reason a piece of it looks the way
it does:

* **One dataset per run.** A manifest holds one `DatasetRef` with one `dataset_sha256`, and
  `assert_comparable` compares those digests, so there is no manifest that could describe a run
  over several files. `--dataset` is repeatable and produces one run per path, never one
  manifest for several.
* **Nothing starts on a corpus or a dataset that cannot be trusted.** A dataset that fails the
  linter and a stale retrieval index are refused before the first model call, in process rather
  than by asking the operator to have run two other commands first. A dirty working tree only
  warns: the manifest records `git_dirty`, and the run is still worth having.
* **Concurrency is a condition, not a convenience.** Contention and provider rate limiting
  inflate `mean_latency_ms`, which is a reported metric, and they inflate it hardest on
  whichever provider throttles first — a harness difference wearing the costume of a model
  difference. Nothing in `RunManifest` records the worker count yet, so the default is 1 and a
  higher one warns that the run is not a graded latency measurement.
* **A resume is guarded on the whole manifest, not on `assert_comparable`.** That guard exempts
  `model_name` and `provider`, which is right for two arms of an A/B and catastrophic here: it
  would let the OSS arm append to the frontier arm's trace under one manifest asserting one
  model. See `resume_manifest`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from agent.core import ROLE_TURN, STOPPED_INFRASTRUCTURE_FAILED, AgentResult, Budgets
from agent.manifest import (
    IDENTITY_FIELDS,
    AgentConfig,
    DatasetRef,
    RunManifest,
    build_manifest,
    compare_manifests,
)
from agent.models.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    ModelAdapter,
    ModelError,
    add_cache_arguments,
    cache_enabled,
    load_agent_model,
    load_env,
)
from agent.tools.lookup_kb import (
    DEFAULT_KB_DIR,
    DEFAULT_MIN_SCORE,
    GROUNDING_MIN_SCORE,
    KnowledgeBase,
    corpus_files,
)
from agent.trace import DEFAULT_RUNS_DIR, TraceLogger, git_dirty, read_records, trace_path
from evals.judge import judge_scores_path, score_run
from evals.schema import SCORED_TURN_INDEX, EvalItem
from evals.validate_dataset import validate_dataset

logger = logging.getLogger(__name__)

#: Which agent an eval run puts under test. The two arms of the A/B, and nothing else: the
#: judge is not an arm, and it is reached through `evals.judge` rather than from here.
MODEL_CHOICES: tuple[str, ...] = ("frontier", "oss")

#: Workers by default. One, deliberately. See the module docstring: a worker count is an
#: unrecorded condition, and two arms run at different counts are not comparable on latency.
DEFAULT_CONCURRENCY = 1

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


class PreflightError(RuntimeError):
    """A precondition refused the run, before any model was called.

    Raised rather than warned because each of its causes invalidates the run's output rather
    than merely complicating it: a graded run over a dataset with a duplicate `id` produces
    scores joined to the wrong item, and one against a stale index retrieves from a corpus
    that is not the one the manifest digests.
    """


class ResumeRefused(PreflightError):
    """The run being resumed was executed under conditions that no longer hold.

    Its own type because the remedy differs from the other refusals: a stale index is fixed
    by rebuilding it, whereas this is fixed by starting a new run. A manifest is written once
    and never mutated, so a changed condition means a new `run_id` — exactly what
    `agent.session` does when a condition changes mid-session.
    """


# --------------------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------------------


def load_dataset(path: Path) -> list[EvalItem]:
    """Load an eval set from `evals/datasets/`: JSONL, one `EvalItem` per line.

    One format rather than several, because the manifest digests the file's bytes: a loader
    accepting two encodings of the same items invites a dataset being re-saved in the other one
    between two arms, which `assert_comparable` would then refuse for reasons no one could
    place. Unknown fields are an error — see `EvalItem`'s `extra="forbid"`.

    Loading is not linting. `validate_dataset` owns the byte-level and cross-item checks — a
    duplicate id, an unpaired bias item, CRLF endings — and `preflight` is what runs it before
    a graded run. This function raises only on a line that is not an item at all.

    Raises:
        ValueError: a line is not a valid `EvalItem`, or the file holds none. The line number
            is named, because a message without one is unactionable on a long file.
        FileNotFoundError: there is no such file.
    """
    path = Path(path)
    items: list[EvalItem] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(EvalItem.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{path}:{number} is not a valid eval item: {exc}") from exc
    if not items:
        raise ValueError(f"{path} holds no items: there is nothing to run")
    return items


def dataset_ref(path: Path) -> DatasetRef:
    """Describe `path` for the manifest: its bytes' digest and its item count.

    The digest is of the file rather than of the parsed cases, so a reformatted dataset counts
    as a different one. Deciding two files are equivalent is the judgement the hash exists to
    avoid.

    `n_items` counts the *file*, not what a run selected from it. `--limit` therefore does not
    move it, which is precisely why a limited run is warned about and why the manifest needs a
    field of its own to record one.
    """
    path = Path(path)
    return DatasetRef.for_file(path, n_items=len(load_dataset(path)))


# --------------------------------------------------------------------------------------
# Preflight: what must hold before the first model call
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Preflight:
    """What checking the run's preconditions found.

    Two lists rather than one severity-tagged list, because the two have different
    consequences and the caller acts on them differently: a refusal ends the run before it
    spends anything, and a warning is recorded next to a run that still happened.
    """

    refusals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refusals


def preflight(
    dataset_path: Path,
    kb_dir: Path = DEFAULT_KB_DIR,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    guardrails: bool = False,
    min_score: float = DEFAULT_MIN_SCORE,
) -> Preflight:
    """Check what has to hold before a run starts, returning refusals and warnings.

    The two refusals are the two failures that are cheap to detect now and expensive to
    discover afterwards. README tells an operator to lint the dataset and run
    `agentseval-index --check` before a graded eval; a runner that does both itself cannot
    forget, and neither check shells out — `validate_dataset` and `KnowledgeBase` are called in
    process, so there is one implementation of each rather than a second one hiding behind a
    subprocess.

    The staleness check is skipped when the corpus holds no documents. There is no index to be
    stale, `build_manifest` records `kb_sha256=None` for exactly this case, and refusing would
    make an empty corpus indistinguishable from an unbuilt one.

    A dirty working tree warns rather than refuses. The manifest records `git_dirty`, so the
    condition is not lost; what is lost is the ability of `git_sha` to identify the code that
    ran, and that is worth saying out loud without ending the run over it.

    Guardrails paired with an uncalibrated floor warns for a related reason: the run is valid and
    its manifest describes it correctly, but one of the three stages is inert, and finding that
    out from a suspiciously flat `grounding_abstained` count after four runs is expensive.
    """
    refusals: list[str] = []
    warnings: list[str] = []

    report = validate_dataset(Path(dataset_path))
    if report.errors:
        detail = "\n".join(diagnostic.format(report.path) for diagnostic in report.errors)
        refusals.append(
            f"{report.path} does not pass the linter ({len(report.errors)} error(s)); a graded "
            f"run over it is not worth discovering afterwards:\n{detail}"
        )
    elif report.warnings:
        warnings.append(
            f"{report.path} lints with {len(report.warnings)} warning(s); run "
            f"agentseval-validate-dataset --strict to see them before grading this run"
        )

    kb_dir = Path(kb_dir)
    if not corpus_files(kb_dir):
        warnings.append(
            f"{kb_dir} holds no corpus documents, so retrieval will return nothing and "
            "kb_sha256 will be null on this run's manifest"
        )
    else:
        reason = KnowledgeBase(kb_dir, auto_build=False).staleness_reason()
        if reason is not None:
            refusals.append(
                f"the retrieval index for {kb_dir} is stale ({reason}); the run would retrieve "
                f"from a corpus other than the one its manifest digests. Rebuild it with "
                f"agentseval-index --kb-dir {kb_dir}"
            )

    if git_dirty():
        warnings.append(
            "the working tree has uncommitted or untracked changes, so git_sha does not "
            "identify the code that ran; the manifest records git_dirty=true"
        )

    if concurrency > 1:
        warnings.append(
            f"concurrency={concurrency}: contention and provider rate limiting inflate "
            "mean_latency_ms, and no manifest field records the worker count, so this run must "
            "not be reported as a graded latency measurement. Both arms must use the same value"
        )

    if limit is not None:
        warnings.append(
            f"--limit {limit}: DatasetRef.n_items counts the file, so this run's manifest is "
            "indistinguishable from an unlimited one and assert_comparable would pass on a "
            "comparison of unequal item counts. This is a smoke run, not a graded one"
        )

    if guardrails and min_score <= DEFAULT_MIN_SCORE:
        warnings.append(
            f"--guardrails on with --min-score {min_score}: at or below the no-floor default, "
            f"every retrieved hit clears the floor, so the grounding stage can only fire on a "
            f"turn that retrieved nothing at all and the ablation measures two stages rather "
            f"than three. The calibrated value is {GROUNDING_MIN_SCORE} "
            f"(agentseval-calibrate-retrieval), and it must be identical in both arms"
        )

    return Preflight(refusals, warnings)


def manifest_diff(new: RunManifest, other: RunManifest) -> str:
    """Render `compare_manifests(new, other)` as an informational field diff.

    Never raises, and deliberately not a guard. `assert_comparable` belongs at comparison time,
    in `metrics.compare_runs` and the report, where it raises by design; catching it here to
    print a warning would remove the guard while leaving README's claim that a mismatched
    comparison "fails loudly instead of producing a plausible number" looking true. It would
    also be the wrong comparison — the run an operator reaches for is usually the other arm,
    which differs in `model_name` by construction.
    """
    try:
        names = compare_manifests(new, other)
    except ValueError as exc:
        return f"cannot diff {new.run_id} against {other.run_id}: {exc}"

    head = (
        f"informational diff against {other.run_id}: {len(names)} field(s) differ. This is a "
        "diff and not a comparability check; assert_comparable runs at comparison time"
    )
    lines = [f"  {name}: {getattr(other, name)!r} -> {getattr(new, name)!r}" for name in names]
    return "\n".join([head, *lines])


# --------------------------------------------------------------------------------------
# Writing one trace from several workers
# --------------------------------------------------------------------------------------


class LockedTraceLogger(TraceLogger):
    """A `TraceLogger` whose writes are serialised, for a run with more than one worker.

    One run is one trace, so the workers share this rather than getting a file each. But the
    base class holds one handle and writes-then-flushes per record, which under concurrent
    writers interleaves partial lines — and a trace with a half-written record is a trace
    `read_records` refuses to read at all, by design.

    A lock rather than a writer thread fed by a queue: the write is short, the contention is
    bounded by the worker count, and a queue would put an unflushed backlog between an event
    and the file that is supposed to survive a crash.
    """

    def __init__(self, run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> None:
        # Before `super().__init__`, which opens the handle: nothing may log until the lock
        # that protects the handle exists.
        self._lock = threading.Lock()
        super().__init__(run_id, runs_dir)

    def log(
        self,
        item_id: str | None,
        turn_idx: int,
        role: str,
        content: str,
        **fields: Any,
    ) -> dict[str, Any]:
        with self._lock:
            return super().log(item_id, turn_idx, role, content, **fields)


# --------------------------------------------------------------------------------------
# Running one item
# --------------------------------------------------------------------------------------


def run_item(agent: Any, item: EvalItem, logger_: TraceLogger) -> AgentResult:
    """Run one item, logging every turn, and return the result.

    An agent-level failure is logged and recorded as a failed item; it must not abort the
    run, or a single provider hiccup would discard all the completed work.

    Every turn in `item.turns` is sent, in order, through one `agent.memory.Conversation`, and
    the response scored is the one to the final turn (`schema.SCORED_TURN_INDEX`). Earlier
    turns are context: on a multi-turn escalation item, grading an intermediate answer would
    grade the agent partway through the escalation the item exists to provoke.

    `item.expected_behavior` and `item.notes` are not passed to the model. `item.model_visible()`
    is what a prompt builder should be handed.

    The conversation is the agent's own, reset between items rather than replaced. A freshly
    constructed `Conversation` would carry no summariser and no `on_summarised` hook, so
    compaction would silently drop folded turns instead of summarising them and the
    summariser's own model calls would never reach the trace — a memory policy different from
    the chat surface's, which is the harness variation PROJECT.md forbids.

    Budgets are per turn, so a three-turn item is allowed three turns' worth. That is correct,
    and it is why a per-item aggregate must sum over the `role="turn"` records rather than the
    `assistant` ones: the latter counts every model call, including the re-prompts.

    Returns:
        The result of the scored turn, or of the turn that ended the item when a failure did.
        An item that ended `infrastructure_failed` is returned as such and recorded as such;
        excluding it is `evals.metrics`'s job alone, applied identically to both arms.
    """
    agent.conversation.reset()
    result: AgentResult | None = None

    for turn_idx, text in enumerate(item.model_visible()["turns"]):
        try:
            result = agent.run_turn(text, item_id=item.id)
        except Exception as exc:
            # Deliberately broad. `run_turn` re-raises a `ModelError` after logging it, and
            # propagates anything a tool raised that was neither of the two known kinds. Both
            # end this item and neither should end the run, and narrowing this would let a
            # provider's new exception type discard every completed item behind it.
            detail = f"{type(exc).__name__}: {exc}"
            logger_.log(
                item.id,
                turn_idx,
                ROLE_TURN,
                "",
                error=detail,
                infrastructure_failed=True,
            )
            return AgentResult(
                final_text="",
                steps=[],
                stopped_reason=STOPPED_INFRASTRUCTURE_FAILED,
                run_id=logger_.run_id,
                tokens={"prompt": 0, "completion": 0, "total": 0},
                infrastructure_failed=True,
                infrastructure_error=detail,
            )

        if result.infrastructure_failed:
            # A tool broke and its retries did not clear it. The remaining turns would measure
            # the outage rather than the model.
            return result

    if result is None:  # pragma: no cover - EvalItem.turns has min_length=1
        raise ValueError(f"item {item.id!r} has no turns")
    return result


# --------------------------------------------------------------------------------------
# Resuming
# --------------------------------------------------------------------------------------


def completed_items(
    run_id: str,
    items: Sequence[EvalItem],
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> set[str]:
    """Return the ids of `items` already complete in `run_id`'s trace.

    The trace is the primary record, so completion is read from it rather than from a derived
    results file — a file that could be regenerated, deleted, or written from a different run.

    An item counts as complete only when the `role="turn"` record for its *final* turn is
    present. A partially run multi-turn item is therefore re-run from the start: half a
    conversation is not resumable context, and continuing from turn two would send the model a
    history it never produced.
    """
    path = trace_path(run_id, runs_dir)
    if not path.exists():
        return set()

    finished = {
        (record.get("item_id"), record.get("turn_idx"))
        for record in read_records(path)
        if record.get("role") == ROLE_TURN
    }
    return {item.id for item in items if (item.id, len(item.turns) + SCORED_TURN_INDEX) in finished}


def resume_manifest(
    run_id: str,
    cfg: AgentConfig,
    dataset: DatasetRef,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> RunManifest:
    """Return `run_id`'s stored manifest, or refuse if the current configuration differs.

    **`assert_comparable` is the wrong guard here.** It exempts `model_name` and `provider`,
    which is correct for two arms of an A/B and wrong for a resume: it would happily let the
    frontier arm's run be continued with the OSS model, writing both into one trace under one
    manifest asserting one of them. No downstream check could detect that, because there would
    only be the one manifest.

    So the check is the whole field set minus run identity, which refuses a resume after the
    corpus, the prompt, the budgets, the code version, or the model changed. The stored
    manifest is returned unmodified and never rewritten — a manifest is written once, and
    anything else means minting a new `run_id`, exactly as `agent.session` does when a
    condition changes mid-session.

    Raises:
        ResumeRefused: a condition differs, or there is no manifest for `run_id`.
    """
    try:
        stored = RunManifest.load(run_id, runs_dir)
    except (OSError, ValueError) as exc:
        raise ResumeRefused(f"cannot resume {run_id!r}: {exc}") from exc

    rebuilt = build_manifest(cfg, run_kind="eval", dataset=dataset)
    drift = [name for name in compare_manifests(stored, rebuilt) if name not in IDENTITY_FIELDS]
    if drift:
        lines = [
            f"  {name}: {getattr(stored, name)!r} != {getattr(rebuilt, name)!r}" for name in drift
        ]
        raise ResumeRefused(
            f"cannot resume {run_id!r}: {len(drift)} condition(s) differ from the manifest it "
            "was started under, so the trace would hold two sets of conditions under one "
            "manifest. Start a new run instead:\n" + "\n".join(lines)
        )
    return stored


# --------------------------------------------------------------------------------------
# Running a set
# --------------------------------------------------------------------------------------


def _execute(
    items: Sequence[EvalItem],
    cfg: AgentConfig,
    trace: TraceLogger,
    concurrency: int,
) -> dict[str, AgentResult]:
    """Run `items`, returning results keyed by `item_id`.

    Keyed rather than ordered, because with more than one worker the completion order is not
    the file order and nothing downstream may depend on either: the trace and any later join
    are joined on `(run_id, item_id)`.

    One `Agent` per worker, each built from the *same* `AgentConfig`, because `Agent` keeps
    per-turn state on the instance — the current item, the turn counter, the summariser's
    pending usage — and sharing one across threads would interleave all three. They share the
    single trace logger, which is what serialises the writes.
    """
    if concurrency == 1:
        agent = cfg.build_agent(trace)
        return {item.id: run_item(agent, item, trace) for item in items}

    local = threading.local()

    def worker(item: EvalItem) -> tuple[str, AgentResult]:
        agent = getattr(local, "agent", None)
        if agent is None:
            agent = cfg.build_agent(trace)
            local.agent = agent
        return item.id, run_item(agent, item, trace)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return dict(pool.map(worker, items))


def run_eval(
    model: ModelAdapter,
    dataset_path: Path,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    budgets: Budgets | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    kb_dir: Path = DEFAULT_KB_DIR,
    guardrails: bool = False,
    min_score: float = DEFAULT_MIN_SCORE,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    resume: str | None = None,
    compare_to: str | None = None,
) -> str:
    """Run a full eval set against one model and return the `run_id`.

    Assembles an `AgentConfig`, calls `build_manifest(cfg, run_kind="eval",
    dataset=dataset_ref(dataset_path))`, and writes the manifest before the first item, so an
    interrupted run still has a record of the conditions it was executed under. The same
    `AgentConfig` builds the agent, so the manifest cannot describe a run that did not happen.
    Scoring is deliberately a separate step (`evals.judge`) so responses can be re-scored
    under a revised rubric without re-running the agents.

    One `Budgets` for both arms, defaulting to `Budgets()`. Per-arm budgets are what
    `assert_comparable` exists to refuse, so this signature takes one object rather than
    three integers that could be varied one at a time.

    `kb_dir` exists for `evals/datasets/injection.jsonl`, which needs a corpus containing a
    poisoned document and therefore runs against a composed fixture rather than against `kb/`
    (PROJECT.md § "The injected fixture corpus"). It is passed straight through to
    `AgentConfig.kb_dir`, so `build_manifest` digests whatever corpus was actually read into
    `kb_sha256` and `assert_comparable` refuses a fixture run against a main one on its own.
    That refusal is the point: the two runs read different text. Nothing here should special-
    case the fixture or exempt it, and a caller that passes a `kb_dir` whose index is stale is
    refused before the first model call rather than measured against the wrong corpus.

    `guardrails` and `min_score` are conditions on the same footing as `kb_dir`, and both go
    straight to `AgentConfig` with nothing here branching on the model. That is the whole
    requirement for the ablation to mean anything: an arm screened by a different guardrail is
    not an arm of the same experiment, and `RunManifest.guardrails_sha256` is what makes the
    sameness checkable rather than assumed. `min_score` is held *fixed* across the ablation and
    only `guardrails` is varied — moving the floor and its enforcement together would leave the
    delta attributable to neither.

    Args:
        guardrails: Whether `agent.guardrails` screens the run.
        max_tokens: Output ceiling per model call, and a condition like the rest: it is what
            makes a reply truncate, `RunManifest.max_tokens` records it, and `assert_comparable`
            refuses two arms given different budgets. It has to be settable because the right
            value is a property of the models — one that meters thinking against this same
            ceiling needs far more of it than its visible replies suggest.
        min_score: The retrieval floor, bound into `lookup_kb` through
            `tools.bound_registry` and digested into `retrieval_config_sha256`. Defaults to the
            library's no-floor value; a graded ablation passes `lookup_kb.GROUNDING_MIN_SCORE`,
            the pre-registered figure from `agentseval-calibrate-retrieval`.
        concurrency: Workers. Above 1 the run is not a graded latency measurement — see the
            module docstring — and both arms must use the same value.
        limit: The first N items in file order. No shuffling; if that ever changes, the
            manifest has a `seeds` field for exactly this reason. A limited run is a smoke
            run, and warns as one.
        resume: A `run_id` to continue. Items already complete in its trace are skipped, and
            the run is refused if any condition has changed. See `resume_manifest`.
        compare_to: A `run_id` to print an informational manifest diff against. Never a guard.

    Raises:
        PreflightError: a precondition refused the run. Nothing was called and nothing written.
        ResumeRefused: `resume` names a run executed under different conditions.
        ValueError: `concurrency` or `limit` is not a usable number, or the dataset cannot be
            read as eval items.
    """
    dataset_path = Path(dataset_path)
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")

    checks = preflight(
        dataset_path,
        kb_dir,
        concurrency=concurrency,
        limit=limit,
        guardrails=guardrails,
        min_score=min_score,
    )
    for warning in checks.warnings:
        logger.warning("%s", warning)
    if not checks.ok:
        raise PreflightError("\n".join(checks.refusals))

    items = load_dataset(dataset_path)
    dataset = dataset_ref(dataset_path)
    cfg = AgentConfig(
        model=model,
        budgets=budgets if budgets is not None else Budgets(),
        temperature=temperature,
        max_tokens=max_tokens,
        kb_dir=Path(kb_dir),
        min_score=min_score,
        guardrails=guardrails,
    )

    if resume is not None:
        manifest = resume_manifest(resume, cfg, dataset, runs_dir)
    else:
        manifest = build_manifest(cfg, run_kind="eval", dataset=dataset)
        manifest.write(runs_dir)

    if compare_to is not None:
        try:
            other = RunManifest.load(compare_to, runs_dir)
        except (OSError, ValueError) as exc:
            logger.warning("cannot read manifest for %r: %s", compare_to, exc)
        else:
            logger.info("%s", manifest_diff(manifest, other))

    selected = items if limit is None else items[:limit]
    done = completed_items(manifest.run_id, selected, runs_dir) if resume is not None else set()
    pending = [item for item in selected if item.id not in done]

    with LockedTraceLogger(manifest.run_id, runs_dir) as trace:
        results = _execute(pending, cfg, trace, concurrency)

    failed = sum(1 for result in results.values() if result.infrastructure_failed)
    logger.info(
        "run %s: %d item(s) run, %d skipped as already complete, %d infrastructure failure(s)",
        manifest.run_id,
        len(results),
        len(done),
        failed,
    )
    return manifest.run_id


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentseval-run",
        description=(
            "Run an eval set against one agent and write its trace. Runs only: scoring is "
            "agentseval-judge, so responses can be re-scored under a revised rubric without "
            "re-running the agents."
        ),
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        required=True,
        help="which arm to run",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        metavar="PATH",
        help=(
            "one eval set. Repeatable, and each path is its own run with its own manifest: a "
            "manifest holds one dataset digest, so there is none that could describe several"
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"where traces and manifests go (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=DEFAULT_KB_DIR,
        help=(
            f"corpus directory (default: {DEFAULT_KB_DIR}). Pass the composed fixture corpus "
            "for the injection set; a run's corpus is a condition, so it is a flag rather than "
            "something inferred from the dataset's filename"
        ),
    )
    parser.add_argument(
        "--guardrails",
        choices=("on", "off"),
        default="off",
        help=(
            "screen the run with agent.guardrails. A condition, not a feature: it is recorded "
            "in the manifest as guardrails plus guardrails_sha256, so an on/off pair is an "
            "ablation that assert_ablation_comparable accepts and assert_comparable refuses. "
            "Identical for both arms, always"
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        metavar="FLOAT",
        help=(
            f"retrieval confidence floor, bound into lookup_kb and digested into "
            f"retrieval_config_sha256 (default: {DEFAULT_MIN_SCORE}, no floor). The "
            f"pre-registered value is {GROUNDING_MIN_SCORE}; hold it fixed across an ablation "
            f"and vary only --guardrails"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help=(
            f"output ceiling per model call (default: {DEFAULT_MAX_TOKENS}). A condition, "
            "recorded in the manifest and not exempt from assert_comparable, so both arms must "
            "use the same value. A model that meters thinking against this ceiling spends most "
            "of it before writing anything, so size it from billed output and not from the "
            "visible reply length"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        metavar="N",
        help=(
            f"workers (default: {DEFAULT_CONCURRENCY}). Above 1 the run is not a graded "
            "latency measurement, and both arms must use the same value"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="run the first N items in file order. A smoke run, not a graded one",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help=(
            "continue a run, skipping items already complete in its trace. Refused if any "
            "condition has changed since it started"
        ),
    )
    parser.add_argument(
        "--compare-to",
        metavar="RUN_ID",
        help="print an informational manifest diff against this run. Never raises",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "after the run finishes, score it as a separate judge run and print both ids. An "
            "orchestrator, not a fusion: the judgements go to their own run with their own "
            "manifest and never back into this trace"
        ),
    )
    add_cache_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `agentseval-run --model {frontier,oss} --dataset ... [--kb-dir DIR] [--no-cache]`.

    `--no-cache` comes from `agent.models.base.add_cache_arguments`, so every entry point
    spells the escape hatch identically.

    `--kb-dir` defaults to `kb/` and is only passed for the injection set, whose corpus is
    built by `agentseval-compose-fixture` and indexed with `agentseval-index --kb-dir`. It is
    a flag rather than something inferred from the dataset's name, because a run's corpus is a
    condition and inferring a condition from a filename puts it outside the manifest's reach.

    Every `--dataset` is its own run. The `run_id`s are printed to stdout, one per line and
    last, so a shell can capture them for `agentseval-judge --run`; warnings, diffs, and judge
    run ids go to stderr, where they cannot be mistaken for one.

    Returns 0 when every leg succeeded, 1 when any run or any judge pass failed, and 2 on a
    usage error.
    """
    load_env()
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Ours at INFO, everything else at WARNING, matching `agentseval-index`: the point of the
    # split there was that a dependency's request log must not bury our own output.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)

    if args.concurrency < 1:
        parser.error(f"--concurrency must be at least 1, got {args.concurrency}")
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be at least 1, got {args.limit}")
    if args.max_tokens < 1:
        parser.error(f"--max-tokens must be at least 1, got {args.max_tokens}")
    if args.resume is not None and len(args.dataset) > 1:
        parser.error(
            "--resume names one run, and a run has one dataset; pass a single --dataset or "
            "resume each run separately"
        )

    which = cast(Literal["frontier", "oss"], args.model)
    try:
        model = load_agent_model(which, no_cache=not cache_enabled(args.no_cache))
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED

    run_ids: list[str] = []
    failed = False

    for dataset_path in args.dataset:
        try:
            run_id = run_eval(
                model,
                dataset_path,
                runs_dir=args.runs_dir,
                kb_dir=args.kb_dir,
                guardrails=args.guardrails == "on",
                min_score=args.min_score,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency,
                limit=args.limit,
                resume=args.resume,
                compare_to=args.compare_to,
            )
        except (PreflightError, ValueError, OSError) as exc:
            print(f"error: {dataset_path}: {exc}", file=sys.stderr)
            failed = True
            continue

        run_ids.append(run_id)
        print(f"run {run_id}  {dataset_path}", file=sys.stderr)

        if args.judge and not _judge(run_id, args.runs_dir):
            failed = True

    for run_id in run_ids:
        print(run_id)

    return EXIT_FAILED if failed else EXIT_OK


def _judge(run_id: str, runs_dir: Path) -> bool:
    """Score `run_id` as a judge run of its own, returning whether it succeeded.

    Called after the eval finished rather than per item. A judgement made while the trace was
    still being appended to would make `pairs_sha256` the digest of a file that no longer
    exists in that form, which is the field deciding whether two judge runs read the same
    pairs.
    """
    try:
        scores = score_run(run_id, runs_dir=runs_dir)
    except (ModelError, OSError, ValueError) as exc:
        print(f"error: judging run {run_id}: {exc}", file=sys.stderr)
        return False

    if not scores:
        # An eval run always yields at least one scorable response unless every item failed
        # infrastructurally, so an empty result is a problem rather than an empty success.
        print(f"error: run {run_id} produced no pairs to score", file=sys.stderr)
        return False

    judge_run_id = scores[0].run_id
    unparsed = [score for score in scores if not score.parse_ok]
    print(f"judge run {judge_run_id}  scoring {run_id}", file=sys.stderr)
    print(f"judgements: {judge_scores_path(judge_run_id, runs_dir)}", file=sys.stderr)
    if unparsed:
        print(
            f"error: {len(unparsed)} of {len(scores)} judgement(s) did not parse",
            file=sys.stderr,
        )
        return False
    return True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
