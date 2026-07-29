"""Tests for `agent.trace`: the JSONL writer and the probes a record is assembled from.

Weighted toward the two properties a trace is relied on for — that a record is on disk before
the run that produced it finishes, and that nothing is ever silently dropped. The manifest and
the comparability guard moved to `agent.manifest`, and so did their tests.

No network and no model calls, so the suite runs without API keys.
"""

from __future__ import annotations

import json
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

from agent import trace
from agent.trace import (
    CARRY_OVER_EVENT,
    ROLE_MEMORY,
    TraceLogger,
    carried_over_from,
    code_version,
    git_dirty,
    git_sha,
    read_records,
    sha256_of_paths,
    sha256_text,
    trace_path,
)

# --------------------------------------------------------------------------------------
# TraceLogger
# --------------------------------------------------------------------------------------


def test_log_writes_every_schema_key(tmp_path):
    with TraceLogger("run-1", tmp_path) as logger:
        logger.log(
            "case-1",
            0,
            "assistant",
            '{"tool": "lookup_kb", "arguments": {"query": "x"}}',
            tool_calls=[{"name": "lookup_kb", "arguments": {"query": "x"}}],
            retrieved_chunk_ids=["kb/a.md#2"],
            latency_ms=812.5,
            prompt_tokens=100,
            completion_tokens=20,
            usd_cost=0.0006,
        )

    (record,) = read_records(trace_path("run-1", tmp_path))
    assert record["run_id"] == "run-1"
    assert record["item_id"] == "case-1"
    assert record["turn_idx"] == 0
    assert record["role"] == "assistant"
    assert record["retrieved_chunk_ids"] == ["kb/a.md#2"]
    assert record["latency_ms"] == 812.5
    assert record["prompt_tokens"] == 100
    assert record["completion_tokens"] == 20
    assert record["usd_cost"] == 0.0006
    assert record["error"] is None
    assert record["ts"].endswith("+00:00")


def test_omitted_values_are_explicit_nulls(tmp_path):
    """Absent keys would force every reader to guess; nulls are unambiguous."""
    with TraceLogger("run-1", tmp_path) as logger:
        logger.log("case-1", 0, "user", "hello")

    (record,) = read_records(trace_path("run-1", tmp_path))
    for key in ("tool_calls", "retrieved_chunk_ids", "latency_ms", "usd_cost", "error"):
        assert record[key] is None


def test_records_are_one_json_object_per_line(tmp_path):
    with TraceLogger("run-1", tmp_path) as logger:
        logger.log("case-1", 0, "user", "line one\nline two")
        logger.log("case-1", 1, "assistant", "answer")

    lines = trace_path("run-1", tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_reopening_appends_rather_than_truncating(tmp_path):
    """A resumed or second-phase run must not destroy the first phase's records."""
    with TraceLogger("run-1", tmp_path) as logger:
        logger.log("case-1", 0, "user", "first")
    with TraceLogger("run-1", tmp_path) as logger:
        logger.log("case-2", 0, "user", "second")

    contents = [r["content"] for r in read_records(trace_path("run-1", tmp_path))]
    assert contents == ["first", "second"]


def test_records_are_readable_before_close(tmp_path):
    """Proves the per-record flush: a crashed run keeps what happened before the crash."""
    logger = TraceLogger("run-1", tmp_path)
    try:
        logger.log("case-1", 0, "user", "mid-run")
        assert len(read_records(logger.path)) == 1
    finally:
        logger.close()


def test_log_after_close_raises(tmp_path):
    logger = TraceLogger("run-1", tmp_path)
    logger.close()
    with pytest.raises(ValueError, match="closed"):
        logger.log("case-1", 0, "user", "too late")


def test_close_is_idempotent(tmp_path):
    logger = TraceLogger("run-1", tmp_path)
    logger.close()
    logger.close()


def test_logger_creates_missing_runs_dir(tmp_path):
    logger = TraceLogger("run-1", tmp_path / "nested" / "runs")
    try:
        logger.log("case-1", 0, "user", "hello")
    finally:
        logger.close()
    assert logger.path.exists()


def test_unserialisable_value_degrades_to_string(tmp_path):
    """A stringified value is still evidence; a dropped record is not."""

    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    with TraceLogger("run-1", tmp_path) as logger:
        logger.log("case-1", 0, "tool", "result", tool_calls=[{"arguments": Weird()}])

    (record,) = read_records(trace_path("run-1", tmp_path))
    assert record["tool_calls"] == [{"arguments": "<weird>"}]


def test_read_records_skips_blank_lines_and_rejects_malformed(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"run_id": "a"}\n\n', encoding="utf-8")
    assert read_records(path) == [{"run_id": "a"}]

    path.write_text('{"run_id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2 is not valid JSON"):
        read_records(path)


# --------------------------------------------------------------------------------------
# The carry-over record
# --------------------------------------------------------------------------------------


def carry_over(previous_run_id: str) -> dict[str, object]:
    """A carry-over record as `agent.session` writes one."""
    return {
        "role": ROLE_MEMORY,
        "content": json.dumps(
            {"event": CARRY_OVER_EVENT, "previous_run_id": previous_run_id, "messages_carried": 2}
        ),
    }


def test_carried_over_from_reads_the_run_a_conversation_came_from():
    records = [carry_over("run-a"), {"role": "user", "content": "and in the heat?"}]
    assert carried_over_from(records) == "run-a"


def test_carried_over_from_is_none_for_a_conversation_that_began_here():
    assert carried_over_from([{"role": "user", "content": "hello"}]) is None


@pytest.mark.parametrize("content", ['"a string"', "[1, 2]", "null", "not json at all", ""])
def test_carried_over_from_survives_a_memory_record_that_is_not_a_carry_over(content):
    """`memory` is also the role compaction is recorded under. A reader that raised over a fold
    record would refuse to render a transcript because memory management had happened in it."""
    assert carried_over_from([{"role": ROLE_MEMORY, "content": content}]) is None


def test_carried_over_from_ignores_a_payload_with_no_previous_run():
    records = [{"role": ROLE_MEMORY, "content": json.dumps({"event": CARRY_OVER_EVENT})}]
    assert carried_over_from(records) is None


def test_carried_over_from_reads_the_first_of_several():
    """One conversation is carried into a run once. Two records means two runs were folded in,
    and the earlier one is the source of the history the later one already contained."""
    assert carried_over_from([carry_over("run-a"), carry_over("run-b")]) == "run-a"


# --------------------------------------------------------------------------------------
# Environment probes and hashing
# --------------------------------------------------------------------------------------


def test_git_helpers_do_not_raise_outside_a_repo(tmp_path):
    """Absent git, or a repo with no commit, must degrade rather than abort a run."""
    assert git_sha(tmp_path) is None
    assert git_dirty(tmp_path) is False


def test_code_version_falls_back_when_package_is_not_installed(monkeypatch):
    def not_installed(_name: str) -> str:
        raise metadata.PackageNotFoundError("agentseval")

    monkeypatch.setattr(trace.metadata, "version", not_installed)
    assert code_version(fallback="0.0.0-test") == "0.0.0-test"


def test_fallback_version_matches_pyproject():
    """The fallback is a hand-copied literal, so pin it to the real version."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    fallback = trace.FALLBACK_CODE_VERSION
    assert fallback == declared


def test_sha256_text_is_stable_and_sensitive():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")


def test_sha256_of_paths_covers_names_and_content(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    paths = sorted(tmp_path.glob("*.md"))

    baseline = sha256_of_paths(paths, root=tmp_path)
    assert baseline == sha256_of_paths(list(reversed(paths)), root=tmp_path)

    (tmp_path / "b.md").write_text("beta edited", encoding="utf-8")
    assert sha256_of_paths(paths, root=tmp_path) != baseline

    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "b.md").rename(tmp_path / "c.md")
    renamed = sha256_of_paths(sorted(tmp_path.glob("*.md")), root=tmp_path)
    assert renamed != baseline


def test_sha256_of_paths_is_none_for_empty_input():
    """None distinguishes "no corpus" from "the digest of nothing"."""
    assert sha256_of_paths([]) is None
