"""The read layer: find runs on disk, pair them with their judge runs, summarise them, and read a
chat transcript back.

Pure functions over `runs/`, with one Streamlit import for the cache decorator and nothing else
from Streamlit — the widgets live in the pages. Everything here is a thin arrangement of
`evals.metrics`: discovery is a glob, pairing is `metrics.find_judge_run`, a summary is
`metrics.summarise_run`, and the per-item join is `metrics.load_item_results`. None of it
recomputes anything the platform already computes.

Two properties worth stating, because both are load-bearing rather than incidental:

**Nothing is written.** `summarise_run` joins the trace, the judgements, and the dataset in memory
on every call, and the reason is PROJECT.md's: a derived results file is a second copy of a run
and therefore a second thing to keep truthful. The cache here is `st.cache_data` with no
`persist=`, so it lives in memory for the session and leaves no artifact behind. `chat_threads` is
keyed on its trace's size as well as its mtime, because a chat session appends to the file while
these pages are open and two appends inside one mtime tick are not rare.

**A run's directory is its own, not the search root.** `runs/` has subdirectories — the pilot
runs live in `runs/pilot/` — and `metrics.find_judge_run` globs one directory without recursing,
because a judge run is written beside the trace it scored. So discovery recurses to find
manifests and then hands each run the directory its own manifest sits in. Passing the search root
instead would silently fail to find any judge run for a nested trace, and the summary would report
every judge-derived rate with `n=0` as though the run had never been scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from agent.manifest import RunManifest
from agent.trace import (
    DEFAULT_RUNS_DIR,
    ROLE_GUARDRAIL,
    ROLE_TURN,
    ROLE_USER,
    by_conversation,
    carried_over_from,
    latest_segments,
    manifest_path,
    read_records,
    trace_path,
)
from evals.judge import judge_scores_path
from evals.metrics import (
    ItemResult,
    RunSummary,
    find_judge_run,
    load_item_results,
    summarise_run,
)
from evals.report import DEFAULT_REPORTS_DIR

#: Where the pages look for runs unless told otherwise. The same default the CLIs use, so the
#: browser lists what `agentseval-run` wrote without being pointed at it.
DEFAULT_RUNS_ROOT: Path = DEFAULT_RUNS_DIR

#: Where the report page looks, which is where `agentseval-report` writes. Unlike `runs/`, this
#: directory is tracked by git: a report is the one artifact of a run meant to be committed, and it
#: is the only form in which a result reaches anyone without this repository's runs on their disk.
DEFAULT_REPORTS_ROOT: Path = DEFAULT_REPORTS_DIR

#: The one `run_kind` these views can summarise. A chat session has no dataset to score against
#: and a judge run has no agent under test; `metrics.load_run` refuses both, and it is right to.
EVAL_RUN_KIND = "eval"

#: The `run_kind` the chat history page reads. A chat run has no summary because it has no dataset
#: and no judgements; it has a transcript, which is a different reader entirely.
CHAT_RUN_KIND = "chat"

#: What a mtime reads as when the file is not there. Zero rather than skipping the component, so
#: the cache key still changes when the file appears — a trace that has since been judged is a
#: different summary, and a key that ignored the absent file would serve the unjudged one.
MISSING_MTIME_NS = 0

#: The same idea for a size: absent reads as zero, so the key moves when the file arrives.
MISSING_SIZE = 0


@dataclass(frozen=True)
class RunRef:
    """One run found on disk, before anything has been aggregated over it.

    Attributes:
        runs_dir: The directory this run's own manifest sits in, which is what every
            `evals.metrics` call about it must be given. See the module docstring.
        judge_run_id: The judge run that scored this trace, joined through the judge manifest's
            recorded `pairs_path` rather than guessed from a filename. None on a run nothing
            scored, and on every non-eval run.
        judge_error: The message from `find_judge_run` when it refused to choose — two judge runs
            scored this trace, under conditions that may differ. Carried rather than raised: one
            ambiguous run should not blank the page for every other one, and the ambiguity is
            itself worth showing.
    """

    run_id: str
    runs_dir: Path
    manifest: RunManifest
    judge_run_id: str | None = None
    judge_error: str | None = None

    @property
    def is_eval(self) -> bool:
        return self.manifest.run_kind == EVAL_RUN_KIND

    @property
    def trace_path(self) -> Path:
        return trace_path(self.run_id, self.runs_dir)

    @property
    def manifest_path(self) -> Path:
        return manifest_path(self.run_id, self.runs_dir)

    @property
    def judge_scores_path(self) -> Path | None:
        """Where this run's judgements are, or None when nothing scored it."""
        if self.judge_run_id is None:
            return None
        return judge_scores_path(self.judge_run_id, self.runs_dir)


def _mtime_ns(path: Path | None) -> int:
    """`path`'s mtime in nanoseconds, or `MISSING_MTIME_NS` when it is not readable."""
    if path is None:
        return MISSING_MTIME_NS
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return MISSING_MTIME_NS


def _size(path: Path | None) -> int:
    """`path`'s size in bytes, or `MISSING_SIZE` when it is not readable.

    In the cache key beside the mtime, for the one file here that grows while a page is open: a
    chat session appends to its trace as it is being read, and two appends inside a single
    filesystem mtime tick are not rare on a coarse clock.
    """
    if path is None:
        return MISSING_SIZE
    try:
        return path.stat().st_size
    except OSError:
        return MISSING_SIZE


def discover_runs(root: Path = DEFAULT_RUNS_ROOT) -> tuple[list[RunRef], list[str]]:
    """Every run under `root`, newest first, with the problems found on the way.

    Recurses, because `runs/` holds subdirectories and a run in one of them is still a run. Each
    `RunRef` carries the directory its own manifest sits in.

    Returns:
        `(runs, problems)`. A manifest that cannot be read becomes a problem string rather than an
        exception or a silent omission: it is a file someone wrote intending it to be a run, and a
        browser that showed neither the run nor the reason would be the worst of the three
        outcomes.
    """
    root = Path(root)
    runs: list[RunRef] = []
    problems: list[str] = []
    for path in sorted(root.rglob("*.manifest.json")):
        try:
            manifest = RunManifest.read(path)
        except (OSError, ValueError) as exc:
            problems.append(f"{path}: {exc}")
            continue

        judge_run_id: str | None = None
        judge_error: str | None = None
        if manifest.run_kind == EVAL_RUN_KIND:
            try:
                judge_run_id = find_judge_run(manifest.run_id, path.parent)
            except ValueError as exc:
                judge_error = str(exc)
        runs.append(
            RunRef(
                run_id=manifest.run_id,
                runs_dir=path.parent,
                manifest=manifest,
                judge_run_id=judge_run_id,
                judge_error=judge_error,
            )
        )

    # Newest first, run id breaking the tie: two runs started in the same millisecond would
    # otherwise order by filesystem walk, which is not an order anyone can predict or quote.
    runs.sort(key=lambda run: (run.manifest.started_at, run.run_id), reverse=True)
    return runs, problems


def eval_runs(runs: list[RunRef]) -> list[RunRef]:
    """The runs these views can summarise. See `EVAL_RUN_KIND`."""
    return [run for run in runs if run.is_eval]


def find_run(runs: list[RunRef], run_id: str | None) -> RunRef | None:
    """The run called `run_id`, or None. A lookup rather than a guess at a default."""
    if run_id is None:
        return None
    return next((run for run in runs if run.run_id == run_id), None)


# --------------------------------------------------------------------------------------
# Chat transcripts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatTurn:
    """One exchange as a reader wants it: the question, the answer, and what it cost.

    Attributes:
        answer: The text the user was shown, which on a screened turn is the guardrail's
            substituted sentence rather than the model's own output. The two readings are held
            apart everywhere else in this codebase because every scorer wants the model's; a
            transcript is a record of what happened in front of a person, and wants the other.
            `guardrail_action` is what says the two differed.
    """

    item_id: str
    turn_idx: int
    ts: str
    question: str
    answer: str
    latency_ms: float | None
    usd_cost: float | None
    stopped_reason: str | None
    guardrail_action: str | None


@dataclass(frozen=True)
class ChatThread:
    """One conversation within one run, in turn order.

    Attributes:
        continues_run: The run this conversation's history was carried from, read off the
            `role="memory"` carry-over record `agent.session` writes. None when the conversation
            began here. It is what lets a reader follow a chat that crossed a model switch, since
            crossing one is what put it in two files.
    """

    item_id: str
    turns: tuple[ChatTurn, ...]
    continues_run: str | None = None

    @property
    def opened_at(self) -> str:
        """When this segment's first turn was asked, or empty for a segment with no turns."""
        return self.turns[0].ts if self.turns else ""


def chat_runs(runs: list[RunRef]) -> list[RunRef]:
    """The chat sessions among `runs`. See `CHAT_RUN_KIND`."""
    return [run for run in runs if run.manifest.run_kind == CHAT_RUN_KIND]


@st.cache_data(show_spinner="Reading the transcript...")
def _threads(run_id: str, runs_dir: str, trace_mtime_ns: int, trace_size: int) -> list[ChatThread]:
    """`chat_threads`, memoised on the run and on the bytes of the trace it read.

    Every parameter is part of the cache key, which is the whole point of the last two: they are
    not read in the body and must not be renamed with a leading underscore, because Streamlit
    reads that prefix as "exclude from the key" and a live session's later turns would never
    appear.

    In memory only. `persist=` is deliberately not set: see the module docstring.
    """
    grouped = by_conversation(read_records(trace_path(run_id, Path(runs_dir))))

    threads: list[ChatThread] = []
    for item_id, records in grouped.items():
        asked: dict[int, dict[str, Any]] = {}
        delivered: dict[int, str] = {}
        finished: dict[int, dict[str, Any]] = {}
        for record in records:
            turn_idx = int(record.get("turn_idx") or 0)
            role = record.get("role")
            if role == ROLE_USER:
                asked[turn_idx] = record
            elif role == ROLE_GUARDRAIL:
                delivered[turn_idx] = str(record.get("content") or "")
            elif role == ROLE_TURN:
                finished[turn_idx] = record

        turns: list[ChatTurn] = []
        for turn_idx, question in sorted(asked.items()):
            done = finished.get(turn_idx, {})
            turns.append(
                ChatTurn(
                    item_id=item_id,
                    turn_idx=turn_idx,
                    ts=str(question.get("ts") or ""),
                    question=str(question.get("content") or ""),
                    answer=delivered.get(turn_idx, str(done.get("content") or "")),
                    latency_ms=done.get("latency_ms"),
                    usd_cost=done.get("usd_cost"),
                    stopped_reason=done.get("error"),
                    guardrail_action=done.get("guardrail_action"),
                )
            )
        threads.append(
            ChatThread(
                item_id=item_id, turns=tuple(turns), continues_run=carried_over_from(records)
            )
        )
    return threads


def chat_threads(run: RunRef) -> list[ChatThread]:
    """Every conversation in `run`'s trace, in the order they were first spoken in.

    A turn with no `turn` record — a session that crashed mid-turn — still appears, with an empty
    answer. Omitting it would make a trace that recorded a question look like one where nobody
    asked.
    """
    # Absolute, for the reason `summary_for` gives: the cache outlives the working directory a
    # relative path is relative to.
    return _threads(
        run.run_id,
        str(run.runs_dir.resolve()),
        _mtime_ns(run.trace_path),
        _size(run.trace_path),
    )


def chat_thread_tips(runs: list[RunRef]) -> list[tuple[RunRef, ChatThread]]:
    """The latest segment of each conversation across `runs`.

    A conversation that crossed a model switch appears in several runs, and the tip is the one that
    stands for the whole of it. `agent.trace.latest_segments` holds the rule, which `agent.session`
    applies to the same question about resuming.
    """
    found = [(run, thread) for run in chat_runs(runs) for thread in chat_threads(run)]
    tips = latest_segments(
        (run.run_id, thread.item_id, thread.continues_run) for run, thread in found
    )
    return [(run, thread) for run, thread in found if (run.run_id, thread.item_id) in tips]


@st.cache_data(show_spinner="Joining trace, judgements, and dataset...")
def _summarise(
    run_id: str,
    runs_dir: str,
    trace_mtime_ns: int,
    judge_mtime_ns: int,
) -> RunSummary:
    """`summarise_run`, memoised on the run and on the files it read.

    Every parameter is part of the cache key, which is the whole point of the last two: they are
    not read in the body and must not be renamed with a leading underscore, because Streamlit reads
    that prefix as "exclude from the key" and the entry would then never be invalidated.

    Keying on the run id alone would serve a stale summary after a re-run or a re-judge wrote new
    bytes under the same id — a page showing last hour's numbers under this hour's run id is the
    provenance failure these views exist to prevent. The judge file's mtime is in the key as well as
    the trace's, because re-scoring a trace under a revised rubric changes every judge-derived rate
    without touching the trace at all.

    In memory only. `persist=` is deliberately not set: see the module docstring.
    """
    return summarise_run(run_id, runs_dir=Path(runs_dir))


def summary_for(run: RunRef) -> RunSummary:
    """The `RunSummary` for `run`, from cache when the files behind it have not changed.

    Raises:
        ValueError: `run` is not an eval run, or its dataset has changed since it was executed.
        FileNotFoundError: the dataset it was executed over is no longer there.
    """
    # Absolute, because the cache outlives the working directory a relative path is relative to.
    # Two runs sharing an id under two directories are two runs, and `runs/` is what a relative
    # path resolves to for both of them.
    return _summarise(
        run.run_id,
        str(run.runs_dir.resolve()),
        _mtime_ns(run.trace_path),
        _mtime_ns(run.judge_scores_path),
    )


def discover_reports(root: Path = DEFAULT_REPORTS_ROOT) -> list[Path]:
    """The markdown reports under `root`, sorted by name. Empty when there are none, or none yet.

    Not recursive, and not a manifest walk. A report is a document rather than a run: it has no
    conditions of its own to read, and the run whose conditions it carries is named inside it. So
    there is nothing here for `discover_runs`' pairing logic to do, and a flat listing is the whole
    of the discovery.
    """
    try:
        return sorted(path for path in root.glob("*.md") if path.is_file())
    except OSError:
        return []


@st.cache_data(show_spinner="Reading the report...")
def _report_text(path: str, mtime_ns: int, size: int) -> str:
    """One report's text, memoised on the file it read.

    `mtime_ns` and `size` are cache key and nothing else, on the same terms as `_summarise`: not
    read in the body, and not to be renamed with a leading underscore, or a regenerated report would
    keep serving the previous one's numbers under the same filename.
    """
    return Path(path).read_text(encoding="utf-8")


def report_text(path: Path) -> str:
    """The markdown at `path`, from cache when the file has not changed since it was read.

    Raises:
        OSError: the file is not readable — deleted between the listing and the read, most likely.
    """
    return _report_text(str(path.resolve()), _mtime_ns(path), _size(path))


@st.cache_data(show_spinner="Reading the judgements...")
def _judgements(
    run_id: str,
    runs_dir: str,
    judge_run_id: str | None,
    trace_mtime_ns: int,
    judge_mtime_ns: int,
) -> list[ItemResult]:
    """`load_item_results`, memoised on the run and on the files it read.

    Every parameter is part of the cache key, on the same terms as `_summarise`: the two mtimes are
    not read in the body and must not be renamed with a leading underscore, because Streamlit reads
    that prefix as "exclude from the key" and a re-judged trace would keep serving the old
    judgements under the same run id.

    In memory only. `persist=` is deliberately not set: see the module docstring.
    """
    return load_item_results(run_id, judge_run_id=judge_run_id, runs_dir=Path(runs_dir))


def judgements_for(run: RunRef) -> list[ItemResult]:
    """Every item of `run` joined to the judgement made of it, one per dataset item.

    An item nothing scored is here too, with `judge=None`; so is one whose judgement did not parse,
    with `parse_ok=False`. Both are absences of a score rather than low scores, which is why
    `ItemResult.dimension` answers None for them and why nothing here fills that in.

    `run.judge_run_id` is passed rather than left to `load_run` to resolve, so a caller reading
    these judgements and a caller reading the summary cannot disagree about which judge run scored
    the trace.

    Raises:
        ValueError: `run` is not an eval run, or its dataset has changed since it was executed.
        FileNotFoundError: the dataset it was executed over is no longer there.
    """
    # Absolute, for the reason `summary_for` gives: the cache outlives the working directory a
    # relative path is relative to.
    return _judgements(
        run.run_id,
        str(run.runs_dir.resolve()),
        run.judge_run_id,
        _mtime_ns(run.trace_path),
        _mtime_ns(run.judge_scores_path),
    )
