"""Append-only JSONL tracing, plus the paths, digests, and probes a run record needs.

Implements the "everything logged" requirement in PROJECT.md. A run produces two sibling
files in `runs/`:

* `runs/{run_id}.jsonl` — one JSON object per turn, appended and flushed as it happens;
* `runs/{run_id}.manifest.json` — the conditions the run was executed under.

This module writes the first. The second, and the `assert_comparable` guard that gives it
teeth, live in `agent.manifest`: assembling a manifest means knowing the rendered prompt, the
tool inventory, the corpus, and the price table, whereas writing a trace needs none of them.
The split is what keeps this module dependency-free — standard library only — so that the
hashing and git helpers below can be imported by `agent.tools.lookup_kb` and `agent.memory`
without dragging the rest of the harness along.

Records carry `run_id` rather than an inline copy of the manifest, so the two join on that
key without duplicating a dozen fields on every line.

This lives under `agent/` rather than `evals/` because `agent.core` and `app.py` log every
turn too; putting it under `evals/` would make the agent depend on the evaluation platform.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, TextIO

DEFAULT_RUNS_DIR = Path("runs")

# Used when the package is not installed (e.g. running from a source checkout).
# Keep in sync with `version` in pyproject.toml.
FALLBACK_CODE_VERSION = "0.1.0"

_GIT_TIMEOUT_S = 5


# --------------------------------------------------------------------------------------
# Paths and small helpers
# --------------------------------------------------------------------------------------


def trace_path(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Return the JSONL trace path for `run_id`."""
    return Path(runs_dir) / f"{run_id}.jsonl"


def manifest_path(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Return the manifest path for `run_id`, sibling to the trace."""
    return Path(runs_dir) / f"{run_id}.manifest.json"


def utc_now_iso() -> str:
    """Return the current UTC time as ISO-8601 with an explicit offset."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of `text`, encoded UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_paths(paths: Iterable[Path], *, root: Path | None = None) -> str | None:
    """Return one digest covering the names and contents of `paths`.

    Files are hashed in sorted name order, and each name is hashed alongside its content,
    so renaming a file changes the digest even when the bytes are unchanged. Names are
    taken relative to `root` when given, which keeps the digest stable across checkouts
    sitting at different absolute paths.

    Returns None for an empty input, distinguishing "no corpus" from "digest of nothing".
    """
    entries: list[tuple[str, Path]] = []
    for path in paths:
        path = Path(path)
        name = str(path.relative_to(root)) if root is not None else str(path)
        entries.append((Path(name).as_posix(), path))

    if not entries:
        return None

    digest = hashlib.sha256()
    for name, path in sorted(entries):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_git(args: list[str], repo: Path | None = None) -> str | None:
    """Run a git command, returning stripped stdout, or None if it could not succeed."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo) if repo is not None else None,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_sha(repo: Path | None = None) -> str | None:
    """Return the current commit SHA, or None if there isn't one.

    None covers all the ways this legitimately fails: git absent, not a repository, or a
    repository with no commits yet. A run is still valid without a SHA; it just cannot be
    tied to a commit, which is why the value is recorded rather than assumed.
    """
    return _run_git(["rev-parse", "HEAD"], repo)


def git_dirty(repo: Path | None = None) -> bool:
    """Return True if the working tree has uncommitted or untracked changes.

    A dirty tree means `git_sha` does not identify the code that ran, so this is recorded
    next to it. Untracked files count: an untracked module can still be imported.
    """
    status = _run_git(["status", "--porcelain"], repo)
    return bool(status)


def code_version(fallback: str = FALLBACK_CODE_VERSION) -> str:
    """Return the installed package version.

    Complements `git_sha`: the version says which release, the SHA says which commit.
    """
    try:
        return metadata.version("agentseval")
    except metadata.PackageNotFoundError:
        return fallback


# --------------------------------------------------------------------------------------
# Trace logging
# --------------------------------------------------------------------------------------

# The `role` values a record can carry. Here rather than in `agent.core`, which writes most of
# them, because they describe the record format this module defines: a reader assembling a
# transcript matches on them and should not have to import the agent loop to do it.
#
# `assistant` is one raw completion; `turn` is the finished turn with its aggregates. They are
# distinct so that summing tokens or latency over a trace can pick one and not double-count the
# same call twice.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
ROLE_TURN = "turn"
ROLE_SUMMARISER = "summariser"
ROLE_MEMORY = "memory"

#: A guardrail stage's own record. Its own role for the reason `summariser` has one: neither
#: `metrics.MODEL_CALL_ROLES` nor `run_cost` reads it, so screening latency and any screening
#: cost are attributed to the guardrail rather than folded into the model's. An arm whose
#: reported latency included our regex pass would be reporting us.
ROLE_GUARDRAIL = "guardrail"

#: `event` on the `role="memory"` record `agent.session` writes when a conversation crosses a run
#: boundary. Named here rather than there so that a reader can recognise one without importing the
#: chat lifecycle.
CARRY_OVER_EVENT = "carried_over"


def carried_over_from(records: Iterable[dict[str, Any]]) -> str | None:
    """The run one conversation's history was carried from, or None if it began here.

    `records` must already be narrowed to a single `item_id`. Tolerant by construction: `memory`
    is also the role compaction is recorded under, so a record whose content is not this payload
    is skipped rather than raised over.
    """
    for record in records:
        if record.get("role") != ROLE_MEMORY:
            continue
        try:
            payload = json.loads(str(record.get("content") or ""))
        except ValueError:
            continue
        if not isinstance(payload, dict) or payload.get("event") != CARRY_OVER_EVENT:
            continue
        source = payload.get("previous_run_id")
        if isinstance(source, str):
            return source
    return None


def by_conversation(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group `records` by `item_id`, in the order the ids first appear.

    Insertion order, so a reader lists conversations as the file tells them rather than
    alphabetically by a random hex id. A record with no `item_id` is skipped: it belongs to no
    conversation, and a group called "None" is not one.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        item_id = record.get("item_id")
        if isinstance(item_id, str) and item_id:
            grouped.setdefault(item_id, []).append(record)
    return grouped


def latest_segments(segments: Iterable[tuple[str, str, str | None]]) -> set[tuple[str, str]]:
    """Which of these `(run_id, item_id, continues_run)` triples are the tip of a conversation.

    A segment is not the tip when another names it as the run its history came from. The
    distinction matters wherever a conversation is reopened: a walk backward from a middle segment
    rebuilds the conversation as of a point it has already passed, and continuing its numbering
    would reuse turn indices the later segments have already written records under.

    Here, rather than in either caller, because there are two of them — `agent.session` offering a
    conversation to resume and `ui.data` offering one to read — and a rule about which history is
    safe to reopen should not be stated twice.
    """
    found = list(segments)
    continued = {(source, item_id) for _, item_id, source in found if source is not None}
    return {(run_id, item_id) for run_id, item_id, _ in found if (run_id, item_id) not in continued}


#: Every key present on every trace record. Fixed so that readers can rely on the shape
#: and a missing value is an explicit null rather than an absent key.
#:
#: The last four are typed outcomes rather than prose. `error` still carries the detail a
#: human reads, but a rate must not be computed by matching English against it: the first
#: reworded message would silently change every number derived from it. These four are what
#: `evals.metrics` counts.
#:
#: `cached` is the fifth thing counted rather than read, and it is here because a cache hit is
#: a replay and not a measurement (PROJECT.md): the `latency_ms` on a replayed call is the
#: original call's, so an average over both is a disk read wearing a model's costume. Most
#: calls in a re-run are hits, which is what makes the distinction load-bearing rather than
#: pedantic. `None` means the writer predates this field — unknown, not uncached — and
#: `evals.metrics` warns rather than assuming.
#:
#: `reasoning_tokens` and `usage` are appended after the guardrail pair, for the same reason
#: they were: this tuple grows at the end only. `reasoning_tokens` is billed output the
#: provider left out of `completion_tokens` (`models.base.derive_reasoning_tokens`), and None
#: on an older trace means the run predates the count — *not* that the model did no thinking,
#: which is the reading that made the first Gemini runs look four times cheaper than they were.
#: `usage` is the provider's verbatim usage object, carried because the hall-023 payload had to
#: be reconstructed from a replay once and should not have to be again: a trace that records
#: only the fields we thought to parse cannot answer a question we had not thought of. It is the
#: usage object rather than the whole response body, which would multiply trace size by the
#: length of every completion for no audit we need.
#:
#: `guardrail_action` and `guardrail_stage` are appended, never inserted.
#: They exist so that no guardrail rate is ever computed by matching the delivered text, which
#: matters more here than anywhere else in this tuple because that text is ours: a rate obtained
#: by recognising our own canned sentence would move on a copy-edit and would be measuring the
#: filter against its own vocabulary. Absent on an older trace they read as `None` — unknown, on
#: the `cached` idiom, and specifically *not* the same fact as `"none"`, which is a run that had
#: guardrails available and had nothing fire.
RECORD_FIELDS: tuple[str, ...] = (
    "ts",
    "run_id",
    "item_id",
    "turn_idx",
    "role",
    "content",
    "tool_calls",
    "retrieved_chunk_ids",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "usd_cost",
    "cached",
    "error",
    "format_violation",
    "budget_induced",
    "tool_error_reason",
    "infrastructure_failed",
    "guardrail_action",
    "guardrail_stage",
    "reasoning_tokens",
    "usage",
)


class TraceLogger:
    """Append-only JSONL writer for one run.

    Opened in append mode and flushed after every record, because a crashed or cancelled
    run must leave behind everything that happened up to the failure — those are usually
    the records worth reading.
    """

    def __init__(self, run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> None:
        self.run_id = run_id
        self.runs_dir = Path(runs_dir)
        self.path = trace_path(run_id, self.runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = self.path.open("a", encoding="utf-8")

    def log(
        self,
        item_id: str | None,
        turn_idx: int,
        role: str,
        content: str,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        retrieved_chunk_ids: list[str] | None = None,
        latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        usd_cost: float | None = None,
        cached: bool | None = None,
        error: str | None = None,
        format_violation: str | None = None,
        budget_induced: bool | None = None,
        tool_error_reason: str | None = None,
        infrastructure_failed: bool | None = None,
        guardrail_action: str | None = None,
        guardrail_stage: str | None = None,
        reasoning_tokens: int | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one record and return it.

        Args:
            item_id: Eval case or conversation id this turn belongs to.
            turn_idx: Zero-based turn index within `item_id`.
            role: What produced the content. Free-form, so a caller can name a producer this
                module does not know about; `agent.core` writes `user`, `assistant` (one raw
                completion), `tool`, `summariser`, `memory`, `guardrail` (a screening decision
                and the text it delivered), and `turn` (the finished turn with its aggregates,
                kept distinct from `assistant` so that summing over a trace does not count the
                same call twice).
            content: The raw text, unmodified — a model's malformed output is data.
            cached: True when the provider response was replayed from `base.ResponseCache`.
                None on a record that is not a model call, and on one written before this
                field existed — which is unknown rather than False, since a graded latency
                figure cannot be recovered from a trace that never said which calls were real.
            error: Failure detail, including protocol parse failures, which PROJECT.md
                requires be recorded rather than repaired.
            format_violation: An `agent.core.FormatViolation` value, when this record is a
                parse failure. None on every other record, so counting is a filter rather
                than a regex over `error`.
            budget_induced: True when the failure was our `max_tokens` truncating the
                response rather than the model misformatting it. None where it does not
                apply, which keeps "not a violation" distinct from "a violation that was not
                budget-induced".
            tool_error_reason: An `agent.tools.ToolErrorReason` value, when the model's tool
                call was rejected.
            infrastructure_failed: True when a tool failed for reasons outside the model's
                control and retries did not clear it, ending the item.
            guardrail_action: An `agent.guardrails.GuardrailAction` value, on a `guardrail`
                record and on the `turn` record it belongs to. None on every other record and
                on every record of a run without guardrails, so that "no guardrail was
                configured" stays distinct from "a guardrail ran and passed".
            guardrail_stage: An `agent.guardrails.GuardrailStage` value, alongside the action.
            reasoning_tokens: Billed output tokens the provider omitted from
                `completion_tokens`. None on a record that is not a model call, and on a
                provider that reports no total to derive it from.
            usage: The provider's usage object, verbatim, so a token question can be settled
                from the trace instead of by re-running the call.

        Raises:
            ValueError: the logger has been closed.
        """
        if self._handle is None or self._handle.closed:
            raise ValueError(f"TraceLogger for run {self.run_id!r} is closed")

        record: dict[str, Any] = {
            "ts": utc_now_iso(),
            "run_id": self.run_id,
            "item_id": item_id,
            "turn_idx": turn_idx,
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usd_cost": usd_cost,
            "cached": cached,
            "error": error,
            "format_violation": format_violation,
            "budget_induced": budget_induced,
            "tool_error_reason": tool_error_reason,
            "infrastructure_failed": infrastructure_failed,
            "guardrail_action": guardrail_action,
            "guardrail_stage": guardrail_stage,
            "reasoning_tokens": reasoning_tokens,
            "usage": usage,
        }

        # default=str keeps an unexpected object from raising and losing the whole record;
        # a stringified value is still evidence, a dropped line is not.
        self._handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        self._handle.flush()
        # Some filesystems and stream types reject fsync; the flush above already makes the
        # record visible to other readers, so a refusal here is not worth failing a run.
        with contextlib.suppress(OSError):
            os.fsync(self._handle.fileno())
        return record

    def close(self) -> None:
        """Close the trace file. Idempotent."""
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._handle = None

    def __enter__(self) -> TraceLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL trace.

    Blank lines are skipped. A malformed line raises rather than being dropped: a trace
    that silently loses records cannot be used to audit a result.

    Raises:
        ValueError: a line is not a JSON object.
    """
    records: list[dict[str, Any]] = []
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
            records.append(record)
    return records
