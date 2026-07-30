"""Keystroke labeling for agent responses. No model calls, no network.

Human labels are the only artifact in this project that cannot be regenerated: a trace can be
re-run and a judge score can be recomputed, but an annotator's afternoon cannot. Everything
here follows from that.

**Append-only, flushed per keystroke.** Labels go to
`evals/datasets/labels/{dataset_stem}.{run_id}.{annotator}.{label_space}.jsonl`, one
`LabelRecord` per line,
flushed and fsynced as it is written exactly as `agent.trace.TraceLogger` does — so Ctrl-C, a
closed laptop, or a killed terminal keeps every label already given. Correcting a label appends
a superseding record instead of editing one; readers take the last record per
`(run_id, item_id)`. Editing in place would lose the fact that a label changed, which is the
first thing an audit asks about.

**Nothing here writes to the dataset.** `agent.manifest.DatasetRef` digests a dataset's bytes
and `assert_comparable` refuses two runs whose digests differ, so writing labels back into the
items would silently invalidate the comparison the labels exist to inform. The sidecar records
the `dataset_sha256` it was labelled against and warns loudly when it no longer matches, along
with the `run_id` and `response_sha256`, which is what makes a label verifiable against the
exact text it was made from rather than against a trace that may since have been regenerated.

**Blind, with its limits admitted.** Three things work against an annotator labelling the
response rather than the comparison:

* *Pair adjacency.* Shown both variants of a counterfactual pair together, an annotator labels
  the difference between them — which is precisely the judgement the within-pair delta is
  supposed to reach independently, from two labels formed apart. Variants are therefore kept at
  least `MIN_SEPARATION` apart, and pair membership is never displayed.
* *Arm ordering.* Labelling one arm's hundred responses and then the other's lands fatigue and
  drift asymmetrically on the arms. Multiple runs are merged into one shuffled pool and each
  label is routed back to its own run's sidecar. The same item from two arms is also held apart,
  for the pair-adjacency reason.
* *Self-identification.* Frontier models sometimes introduce themselves mid-answer, so model
  and vendor names are scrubbed before display and each scrub is counted and reported.

What none of that achieves is a real blind. Style tells cannot be scrubbed, and `run_id` sits
in the sidecar's filename where the run manifest maps it straight to a model. For a solo
annotator labelling their own project this is honour-system, and saying so is worth more than a
guarantee the filesystem does not provide.

**Two label spaces, no conversion.** `binary_behavioral` is `pass`/`fail` against the item's
`expected_behavior`; `rubric_1_5` matches the judge's own scale and feeds
`validate_judge.LabelledPair.human_score`. Choosing a threshold to collapse one into the other
after seeing a graded run means picking the statistic that flattered the result, so the space is
chosen before labelling and recorded on every line. See `schema.LabelSpace`.

The space is also in the sidecar's filename, so labelling one run in both spaces produces two
files rather than one mixed file that no reader would accept. That is a required workflow, not a
hypothetical: the judge-vs-rules baseline leg compares each instrument against humans in its own
space, so it needs native binary labels on the same items the ordinal report reads 1-5 labels for
(PROJECT.md). **Randomise the order between the two passes** — a different `--seed` — and leave time
between them. An annotator who labels the same items in the same order twice is partly recalling
the first pass rather than judging the second, and the two label sets are then not independent.
They are not independent in any case with one annotator; PROJECT.md says so rather than claiming
otherwise.

The annotator sees `expected_behavior` and `notes`; a model never does
(`schema.ANNOTATOR_ONLY_FIELDS`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.models.base import PRICING
from agent.prompts import JUDGE_SCALE_MAX
from agent.trace import DEFAULT_RUNS_DIR, sha256_of_paths, sha256_text, trace_path, utc_now_iso
from evals.schema import (
    ANNOTATOR_ONLY_FIELDS,
    Axis,
    EvalItem,
    HumanLabel,
    LabelRecord,
    LabelSpace,
)

DEFAULT_DATASETS_DIR = Path("evals/datasets")

#: Minimum number of other items between two responses that would invite comparison: the two
#: variants of a counterfactual pair, or the same item answered by two arms. Three is enough
#: that the earlier response is no longer on screen and no longer the thing being remembered.
MIN_SEPARATION = 3

#: Vendor and family words that identify a model but are not model ids. The ids themselves come
#: from `base.PRICING`, so adding a model to the price table extends the scrub automatically
#: rather than leaving a name that leaks until someone remembers this list.
VENDOR_NAMES = (
    "anthropic",
    "openai",
    "groq",
    "together",
    "claude",
    "chatgpt",
    "gpt",
    "qwen",
    "alibaba",
    "gemini",
    "google",
    "llama",
    "mistral",
)

#: What a scrubbed name is replaced with. Fixed-width and obviously a redaction, so an annotator
#: knows a name was removed rather than wondering at a strange sentence.
REDACTION = "[model]"


def _blinding_pattern() -> re.Pattern[str]:
    """Build the scrub pattern from the price table plus the vendor list.

    Longest first so `claude-sonnet-4` is redacted whole rather than leaving `-sonnet-4` behind.
    """
    words = {*VENDOR_NAMES, *PRICING}
    parts = sorted((re.escape(word) for word in words), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b[\w.\-]*", re.IGNORECASE)


BLINDING_PATTERN = _blinding_pattern()


def scrub_model_names(text: str) -> tuple[str, int]:
    """Redact model and vendor names, returning the text and how many were removed.

    The count is returned rather than swallowed because a response that self-identifies is worth
    knowing about: it is both a leak in the blind and a fact about the model.
    """
    return BLINDING_PATTERN.subn(REDACTION, text)


# --------------------------------------------------------------------------------------
# Reading the inputs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One response awaiting a label, with the item it answers."""

    item: EvalItem
    run_id: str
    response: str
    response_sha256: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.run_id, self.item.id)


def load_items(path: Path) -> list[EvalItem]:
    """Read a dataset. Reads only; see the module docstring.

    Deliberately not `runner.load_dataset`, which is still a stub — but the same shape, and
    `agentseval-validate-dataset` is what guarantees a file parses before it gets here.
    """
    items: list[EvalItem] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(EvalItem.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 - re-raised with the line, which is the point
            raise ValueError(
                f"{path}:{number}: not a valid EvalItem. Run agentseval-validate-dataset "
                f"for a full report.\n{exc}"
            ) from exc
    return items


def load_final_responses(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> dict[str, str]:
    """Map `item_id` to the assistant's answer to the item's final turn.

    The last assistant record per item, matching `schema.SCORED_TURN_INDEX`: on a multi-turn
    escalation item the earlier answers are context, and labelling one of them would label the
    agent partway through the escalation.
    """
    path = trace_path(run_id, runs_dir)
    if not path.exists():
        raise FileNotFoundError(f"no trace for run {run_id!r} at {path}")

    responses: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        with contextlib.suppress(json.JSONDecodeError):
            record = json.loads(line)
            if record.get("role") == "assistant" and record.get("item_id"):
                responses[record["item_id"]] = record.get("content") or ""
    return responses


def read_labels(path: Path) -> dict[tuple[str, str], LabelRecord]:
    """Read a sidecar, keeping the last record per `(run_id, item_id)`.

    Last rather than first because the file is append-only: a later record supersedes an earlier
    one, which is how `--redo` and undo work without editing anything.
    """
    if not path.exists():
        return {}
    latest: dict[tuple[str, str], LabelRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        with contextlib.suppress(Exception):
            record = LabelRecord.model_validate_json(line)
            latest[record.key] = record
    return latest


def labels_path(
    dataset: Path,
    run_id: str,
    annotator: str,
    labels_dir: Path | None = None,
    *,
    label_space: LabelSpace,
) -> Path:
    """Where one annotator's labels for one run of one dataset, in one space, live.

    **The label space is part of the filename, and `label_space` has no default.** One annotator
    labelling one run in both spaces is a real and required workflow — the judge-vs-rules baseline
    leg needs native `binary_behavioral` labels on the same items the ordinal report reads
    `rubric_1_5` labels for (PROJECT.md). Without the space in the name both passes would append to
    one file, and `validate_judge._require_single_space` would then refuse the result: an
    annotator's afternoon lost to a naming convention. Two files, same items, is the arrangement
    that works, and only the filename can carry that.

    No default, because the two callers want different spaces and a wrong guess writes labels
    where the reader will not look for them. Declaring the space is the same discipline
    `schema.LabelSpace` enforces on every record.
    """
    directory = labels_dir if labels_dir is not None else dataset.parent / "labels"
    return directory / f"{dataset.stem}.{run_id}.{annotator}.{label_space.value}.jsonl"


# --------------------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------------------


def _comparison_group(candidate: Candidate) -> str | None:
    """The key two candidates must not be presented near each other under.

    Two things invite labelling a comparison instead of a response: the two variants of a
    counterfactual pair, and the same item answered by two arms.
    """
    if candidate.item.counterfactual_id is not None:
        return f"cf:{candidate.item.counterfactual_id}"
    return f"item:{candidate.item.id}"


def order_candidates(candidates: Sequence[Candidate], seed: int) -> list[Candidate]:
    """Shuffle reproducibly, dealing one candidate per comparison group per round.

    Seeded so a session can be replayed and a reviewer can reconstruct what an annotator saw in
    what order.

    Round-robin rather than "shuffle, then fix up what landed too close": a greedy repair pass
    spends its freedom early and then has nothing but same-group candidates left, so it reliably
    puts the last pair's two variants next to each other — the one place the constraint matters
    most, since those are the last labels of a tiring session. Dealing like cards instead gives
    every group's members a gap equal to the number of groups still in play, with no endgame.

    The gap it achieves is therefore bounded by how many groups exist, and a dataset of a single
    pair cannot be spread at all. That is a real limit rather than an error: labelling such a set
    in a less ideal order beats refusing to label it. `min_group_gap` measures what was actually
    achieved and `main` says so when it falls short, which is better than a guarantee in a
    docstring that the data cannot always support.
    """
    rng = random.Random(seed)

    groups: dict[str | None, list[Candidate]] = {}
    for item in candidates:
        groups.setdefault(_comparison_group(item), []).append(item)
    for members in groups.values():
        rng.shuffle(members)

    keys = list(groups)
    rng.shuffle(keys)

    ordered: list[Candidate] = []
    while len(ordered) < len(candidates):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
    return ordered


def min_group_gap(ordered: Sequence[Candidate]) -> int | None:
    """The smallest distance between two candidates that invite comparison, or None if none do.

    Reported rather than asserted, because whether the target is reachable depends on the
    dataset. Anything at or below `MIN_SEPARATION` means two comparable responses were close
    enough for the first to still be in mind when the second is labelled.
    """
    last_seen: dict[str | None, int] = {}
    gaps: list[int] = []
    for index, item in enumerate(ordered):
        group = _comparison_group(item)
        if group in last_seen:
            gaps.append(index - last_seen[group])
        last_seen[group] = index
    return min(gaps) if gaps else None


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


class LabelWriter:
    """Append-only sidecar writer, one file per `(dataset, run, annotator)`.

    Flushed and fsynced per record for the reason `agent.trace.TraceLogger` is: an interrupted
    session must keep every label already given, and a human label cannot be reproduced by
    re-running anything.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = self.path.open("a", encoding="utf-8")
        self.written = 0

    def append(self, record: LabelRecord) -> None:
        if self._handle is None or self._handle.closed:
            raise ValueError(f"label writer for {self.path} is closed")
        self._handle.write(record.model_dump_json() + "\n")
        self._handle.flush()
        # Some filesystems reject fsync; the flush already makes the record visible, so a
        # refusal is not worth losing a session over.
        with contextlib.suppress(OSError):
            os.fsync(self._handle.fileno())
        self.written += 1

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._handle = None


# --------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------

#: Keys and what they do. `p`/`f` in the binary space, `1`-`JUDGE_SCALE_MAX` in the rubric one.
KEY_HELP = (
    ("p", "pass"),
    ("f", "fail"),
    ("s", "skip (no record written)"),
    ("u", "undo: relabel the previous item"),
    ("n", "attach a note to this item"),
    ("q", "quit"),
)


def read_key(stream: TextIO | None = None) -> str:
    """Read one keypress without waiting for Enter, or one line when stdin is not a tty.

    The tty path is what makes labelling fast enough that an annotator stays calibrated. The
    fallback is what makes this testable at all: under pytest stdin is a pipe, and `termios`
    raises on it.
    """
    stream = stream or sys.stdin
    if not stream.isatty():
        line = stream.readline()
        if not line:
            return "q"
        return line.strip()[:1].lower() or "s"

    import termios
    import tty

    fd = stream.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = stream.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    # Ctrl-C in raw mode arrives as a byte rather than as KeyboardInterrupt.
    if char == "\x03":
        return "q"
    return char.lower()


def read_line(prompt: str, stream: TextIO | None = None) -> str:
    """Read a whole line, for notes. Works the same on a tty and a pipe."""
    stream = stream or sys.stdin
    print(prompt, end="", flush=True)
    return (stream.readline() or "").strip()


# --------------------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------------------


@dataclass
class SessionResult:
    """What a session did, for the closing report and for tests."""

    labelled: int = 0
    skipped: int = 0
    redone: int = 0
    scrubbed_responses: int = 0
    quit_early: bool = False
    counts: Counter[str] = field(default_factory=Counter)


def _render(
    console: Console,
    candidate: Candidate,
    position: int,
    total: int,
    label_space: LabelSpace,
    note: str | None,
) -> int:
    """Show one item and its response. Returns the number of names scrubbed."""
    item = candidate.item
    scrubbed, n_scrubbed = scrub_model_names(candidate.response)

    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold")
    header.add_column()
    header.add_row("item", item.id)
    header.add_row("axis", f"{item.axis.value} / {item.subcategory}")
    header.add_row("answerable", "yes" if item.answerable else "no (kb does not cover it)")
    if item.expected_tool:
        header.add_row("expected tool", item.expected_tool)
    if item.must_include:
        header.add_row("must include", ", ".join(item.must_include))
    console.print(f"[dim]{position}/{total}[/dim]")
    console.print(header)

    turns = Table.grid(padding=(0, 1))
    turns.add_column(style="bold cyan", width=8)
    turns.add_column(overflow="fold")
    for index, turn in enumerate(item.turns):
        marker = "user*" if index == len(item.turns) - 1 else "user"
        turns.add_row(marker, turn)
    console.print(
        Panel(turns, title="conversation ([bold cyan]user*[/] is the turn being scored)")
    )

    console.print(Panel(scrubbed or "[dim](empty response)[/dim]", title="response"))
    if n_scrubbed:
        console.print(
            f"[yellow]{n_scrubbed} model/vendor name(s) redacted from the response.[/yellow]"
        )

    # Driven by the constant rather than by a hand-written list, so the fields withheld from
    # models are exactly the fields shown to the annotator. One definition, both directions.
    annotator_only = Table.grid(padding=(0, 1))
    annotator_only.add_column(style="bold green")
    annotator_only.add_column(overflow="fold")
    for name in sorted(ANNOTATOR_ONLY_FIELDS):
        value = getattr(item, name, None)
        if value:
            annotator_only.add_row(name.replace("_", " "), str(value))
    console.print(
        Panel(
            annotator_only,
            title="what a passing response does (annotator only, never shown to a model)",
            border_style="green",
        )
    )
    if note:
        console.print(f"[magenta]note attached:[/magenta] {note}")

    if label_space is LabelSpace.BINARY_BEHAVIORAL:
        keys = "  ".join(f"[bold]{key}[/bold] {what}" for key, what in KEY_HELP)
    else:
        rubric = f"[bold]1-{JUDGE_SCALE_MAX}[/bold] score"
        keys = "  ".join(
            [rubric]
            + [f"[bold]{key}[/bold] {what}" for key, what in KEY_HELP if key not in "pf"]
        )
    console.print(keys)
    return n_scrubbed


def run_session(
    candidates: Sequence[Candidate],
    *,
    dataset_sha256: str,
    annotator: str,
    label_space: LabelSpace,
    writers: dict[str, LabelWriter],
    console: Console | None = None,
    stream: TextIO | None = None,
) -> SessionResult:
    """Present each candidate and append a record per label. Returns what happened.

    `u` steps back one position and re-presents the previous item; the new label is appended as
    a superseding record rather than replacing the old line, because the file is append-only and
    the fact that a label changed is itself worth keeping.
    """
    console = console or Console()
    result = SessionResult()
    pending_note: str | None = None
    index = 0
    # Counted per response, not per render: attaching a note re-renders the same one, and a
    # leak count that grew every time an annotator typed would not mean anything.
    scrubbed_keys: set[tuple[str, str]] = set()

    while index < len(candidates):
        candidate = candidates[index]
        started = time.monotonic()
        if _render(console, candidate, index + 1, len(candidates), label_space, pending_note):
            scrubbed_keys.add(candidate.key)

        key = read_key(stream)

        if key == "q":
            result.quit_early = True
            result.scrubbed_responses = len(scrubbed_keys)
            return result
        if key == "s":
            result.skipped += 1
            pending_note = None
            index += 1
            continue
        if key == "n":
            pending_note = read_line("note: ", stream) or None
            continue
        if key == "u":
            if index == 0:
                console.print("[yellow]nothing to undo: this is the first item.[/yellow]")
                continue
            index -= 1
            result.redone += 1
            pending_note = None
            continue

        label: HumanLabel | None = None
        score: int | None = None
        if label_space is LabelSpace.BINARY_BEHAVIORAL:
            if key == "p":
                label = HumanLabel.PASS
            elif key == "f":
                label = HumanLabel.FAIL
            else:
                console.print(f"[red]unrecognised key {key!r}.[/red]")
                continue
        else:
            if not (key.isdigit() and 1 <= int(key) <= JUDGE_SCALE_MAX):
                console.print(f"[red]expected 1-{JUDGE_SCALE_MAX}, got {key!r}.[/red]")
                continue
            score = int(key)

        writers[candidate.run_id].append(
            LabelRecord(
                item_id=candidate.item.id,
                run_id=candidate.run_id,
                dataset_sha256=dataset_sha256,
                response_sha256=candidate.response_sha256,
                label_space=label_space,
                label=label,
                score=score,
                annotator=annotator,
                labelled_at=utc_now_iso(),
                seconds_spent=round(time.monotonic() - started, 3),
                notes=pending_note,
            )
        )
        result.labelled += 1
        result.counts[label.value if label else str(score)] += 1
        pending_note = None
        index += 1

    result.scrubbed_responses = len(scrubbed_keys)
    return result


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def build_candidates(
    items: Sequence[EvalItem],
    run_ids: Sequence[str],
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    console: Console,
) -> list[Candidate]:
    """Pair each item with each run's answer to its final turn."""
    by_id = {item.id: item for item in items}
    candidates: list[Candidate] = []
    for run_id in run_ids:
        responses = load_final_responses(run_id, runs_dir)
        missing = sorted(set(by_id) - set(responses))
        if missing:
            console.print(
                f"[yellow]run {run_id}: no response for {len(missing)} item(s) "
                f"({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}); skipping "
                "them.[/yellow]"
            )
        for item_id, response in responses.items():
            if item_id in by_id:
                candidates.append(
                    Candidate(
                        item=by_id[item_id],
                        run_id=run_id,
                        response=response,
                        response_sha256=sha256_text(response),
                    )
                )
    return candidates


def _filter(
    candidates: Sequence[Candidate],
    *,
    axis: str | None,
    subcategory: str | None,
    unlabelled_only: bool,
    redo: Sequence[str],
    existing: dict[str, dict[tuple[str, str], LabelRecord]],
    limit: int | None,
) -> list[Candidate]:
    """Apply the CLI filters, then resume.

    `--redo` overrides the resume skip for the named ids: the point of redoing a label is to
    reach an item that already has one.
    """
    redo_ids = set(redo)
    kept: list[Candidate] = []
    for candidate in candidates:
        if axis and candidate.item.axis.value != axis:
            continue
        if subcategory and candidate.item.subcategory != subcategory:
            continue
        already = candidate.key in existing.get(candidate.run_id, {})
        if candidate.item.id in redo_ids:
            kept.append(candidate)
            continue
        if unlabelled_only and already:
            continue
        kept.append(candidate)
    return kept[:limit] if limit else kept


def main(argv: Sequence[str] | None = None) -> int:
    """Label responses: `agentseval-label --dataset D --run R [--run R2] --annotator NAME`.

    Never writes to the dataset — only to `labels/`. See the module docstring.
    """
    parser = argparse.ArgumentParser(
        prog="agentseval-label",
        description="Collect human labels on agent responses. No model calls, no network.",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="the .jsonl eval set")
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        metavar="RUN_ID",
        help="a run whose responses to label; repeatable, and multiple runs are merged into "
        "one shuffled pool so fatigue does not land on one arm",
    )
    parser.add_argument("--annotator", required=True, help="who is labelling")
    parser.add_argument(
        "--label-space",
        type=LabelSpace,
        choices=list(LabelSpace),
        default=LabelSpace.BINARY_BEHAVIORAL,
        help="binary_behavioral (pass/fail) or rubric_1_5 (the judge's own scale)",
    )
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed; a session replays")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--labels-dir", type=Path, default=None, help="default: <dataset>/labels")
    parser.add_argument("--axis", choices=[a.value for a in Axis], default=None)
    parser.add_argument("--subcategory", default=None)
    parser.add_argument(
        "--unlabelled-only",
        action="store_true",
        help="skip what this annotator already labelled for these runs",
    )
    parser.add_argument(
        "--redo",
        action="append",
        default=[],
        metavar="ITEM_ID",
        help="relabel this item, superseding by appending rather than editing; repeatable",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    console = Console()

    try:
        items = load_items(args.dataset)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    dataset_sha256 = sha256_of_paths([args.dataset], root=args.dataset.parent) or ""

    existing: dict[str, dict[tuple[str, str], LabelRecord]] = {}
    for run_id in args.runs:
        path = labels_path(
            args.dataset,
            run_id,
            args.annotator,
            args.labels_dir,
            label_space=args.label_space,
        )
        existing[run_id] = read_labels(path)
        stale = {
            record.dataset_sha256
            for record in existing[run_id].values()
            if record.dataset_sha256 != dataset_sha256
        }
        if stale:
            console.print(
                Panel(
                    f"{len(existing[run_id])} existing label(s) in {path.name} were made "
                    f"against a different dataset ({', '.join(d[:12] for d in sorted(stale))}) "
                    f"than the file you just passed ({dataset_sha256[:12]}).\n\n"
                    "The text that was labelled is not the text here, so those labels cannot "
                    "be assumed to still apply. Nothing has been modified — decide whether to "
                    "relabel before using them.",
                    title="dataset has changed since these labels were written",
                    border_style="red",
                )
            )

    try:
        candidates = build_candidates(items, args.runs, runs_dir=args.runs_dir, console=console)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    selected = _filter(
        candidates,
        axis=args.axis,
        subcategory=args.subcategory,
        unlabelled_only=args.unlabelled_only,
        redo=args.redo,
        existing=existing,
        limit=args.limit,
    )
    if not selected:
        console.print("nothing to label.")
        return 0

    ordered = order_candidates(selected, args.seed)
    gap = min_group_gap(ordered)
    if gap is not None and gap <= MIN_SEPARATION:
        console.print(
            f"[yellow]two comparable responses are only {gap} apart (target: more than "
            f"{MIN_SEPARATION}). This set has too few distinct items to spread them further, so "
            "some pairs will be labelled close together — read each response on its own "
            "terms.[/yellow]"
        )

    writers = {
        run_id: LabelWriter(
            labels_path(
                args.dataset,
                run_id,
                args.annotator,
                args.labels_dir,
                label_space=args.label_space,
            )
        )
        for run_id in args.runs
    }

    try:
        result = run_session(
            ordered,
            dataset_sha256=dataset_sha256,
            annotator=args.annotator,
            label_space=args.label_space,
            writers=writers,
            console=console,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted; every label already given is on disk.[/yellow]")
        return 0
    finally:
        for writer in writers.values():
            writer.close()

    console.print(_render_result(result, writers))
    return 0


def _render_result(result: SessionResult, writers: dict[str, LabelWriter]) -> str:
    lines = [
        f"labelled {result.labelled}, skipped {result.skipped}, revisited {result.redone}",
        "  " + (", ".join(f"{k}={v}" for k, v in sorted(result.counts.items())) or "(none)"),
    ]
    for run_id, writer in sorted(writers.items()):
        lines.append(f"  {run_id}: {writer.written} record(s) appended to {writer.path}")
    if result.scrubbed_responses:
        lines.append(
            f"  {result.scrubbed_responses} response(s) had a model or vendor name redacted "
            "before display"
        )
    if result.quit_early:
        lines.append("  quit early; the rest are unlabelled and resumable with --unlabelled-only")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
