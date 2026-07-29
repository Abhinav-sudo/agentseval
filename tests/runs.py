"""A run on disk, as the readers actually read one: dataset, trace, manifest, judge run.

`tests/fakes.py` holds doubles that stand in for a provider; these are the real artifacts, written
to a `tmp_path`. Shared because `test_metrics.py` reads them through `load_run`, `test_report.py`
renders what comes out, and `tests/test_ui.py` renders it again in a browser — three readers of one
shape. A second set of builders would drift from this one, and the drift would show up as a UI test
passing over a manifest the aggregators would refuse.

Every builder returns what it wrote, so a test can assert against the same bytes the code under
test will read rather than against a copy of its intent.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent.manifest import RunManifest
from agent.trace import sha256_of_paths, sha256_text

#: A well-formed answer citing a chunk the fixture trace also reports retrieving, so the citation
#: checks pass unless a test sets out to break them.
ANSWER = "Aim for seven to nine hours [[sleep-hygiene.md#1]]."


def item(**overrides: Any) -> dict[str, Any]:
    """One dataset line, defaulting to a plain answerable hallucination-axis item."""
    base: dict[str, Any] = {
        "id": "h-1",
        "axis": "hallucination",
        "subcategory": "answerable_kb",
        "turns": ["How much sleep?"],
        "expected_behavior": "Answers from the corpus and cites it.",
        "answerable": True,
    }
    base.update(overrides)
    return base


def write_dataset(tmp_path: Path, items: Sequence[dict[str, Any]]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in items), encoding="utf-8"
    )
    return path


def turn(
    item_id: str,
    response: str = ANSWER,
    *,
    chunk_ids: Sequence[str] = ("sleep-hygiene.md#1",),
    stopped_reason: str | None = None,
    format_violation: str | None = None,
    budget_induced: bool = False,
    tool_error_reason: str | None = None,
    infrastructure_failed: bool = False,
    latency_ms: float = 100.0,
    usd_cost: float | None = 0.001,
    cached: bool | None = False,
    turn_idx: int = 0,
    guardrail_action: str | None = None,
) -> list[dict[str, Any]]:
    """One item's records: a model call, the tool result carrying its typed outcome, the turn.

    Three records rather than one because `deterministic.item_views` reconstructs a step from
    the `assistant` record and the `tool` records that follow it, and the typed failure columns
    live on the second — which is where the rates read them from.

    `guardrail_action` lands on the `turn` record only, which is where every guardrail rate reads
    it from. `content` stays the model's own text whatever it is set to, because that is what a
    real trace holds: the substituted sentence lives on a separate `guardrail` record, and
    keeping it out of here is what stops it reaching any scorer.
    """
    common = {"run_id": "run-f", "item_id": item_id, "turn_idx": turn_idx}
    return [
        {
            **common,
            "role": "assistant",
            "content": response,
            "latency_ms": latency_ms,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "usd_cost": usd_cost,
            "cached": cached,
        },
        {
            **common,
            "role": "tool",
            "content": "retrieved text: seven to nine hours",
            "retrieved_chunk_ids": list(chunk_ids),
            "format_violation": format_violation,
            "budget_induced": budget_induced,
            "tool_error_reason": tool_error_reason,
        },
        {
            **common,
            "role": "turn",
            "content": response,
            "retrieved_chunk_ids": list(chunk_ids),
            "latency_ms": latency_ms,
            "usd_cost": usd_cost,
            "error": stopped_reason,
            "format_violation": format_violation,
            "budget_induced": budget_induced,
            "infrastructure_failed": infrastructure_failed,
            "guardrail_action": guardrail_action,
        },
    ]


def write_trace(tmp_path: Path, run_id: str, records: Sequence[dict[str, Any]]) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{run_id}.jsonl"
    path.write_text(
        "".join(json.dumps({**record, "run_id": run_id}) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def write_manifest(
    tmp_path: Path, run_id: str, dataset: Path, **overrides: Any
) -> RunManifest:
    base: dict[str, Any] = {
        "run_id": run_id,
        "started_at": "2026-07-29T09:00:00.000+00:00",
        "run_kind": "eval",
        "model_name": "frontier-model-1",
        "provider": "openai",
        "temperature": 0.0,
        "max_tokens": 1024,
        "top_k": 4,
        "chunk_size": 256,
        "retrieval_config_sha256": sha256_text("retrieval v1"),
        "system_prompt_sha256": sha256_text("system prompt v1"),
        "kb_sha256": sha256_text("corpus v1"),
        "pricing_version": sha256_text("prices v1"),
        "max_tool_calls": 3,
        "max_tool_errors": 2,
        "max_model_calls": 6,
        "git_sha": "a" * 40,
        "code_version": "0.1.0",
        "git_dirty": False,
        "dataset_path": str(dataset),
        "dataset_sha256": sha256_of_paths([dataset], root=dataset.parent),
        "n_items": len(dataset.read_text(encoding="utf-8").splitlines()),
        "seeds": None,
    }
    base.update(overrides)
    manifest = RunManifest(**base)
    manifest.write(tmp_path / "runs")
    return manifest


def chat_manifest(tmp_path: Path, run_id: str, **overrides: Any) -> RunManifest:
    """A `run_kind="chat"` manifest: no dataset, no item count, no seeds.

    Its own builder rather than an override set on `write_manifest`, which requires a dataset: a
    chat session was scored against nothing, and handing it a dataset path would have the fixture
    assert something the chat surface never writes.
    """
    base: dict[str, Any] = {
        "run_id": run_id,
        "started_at": "2026-07-29T09:00:00.000+00:00",
        "run_kind": "chat",
        "model_name": "frontier-model-1",
        "provider": "anthropic",
        "temperature": 0.0,
        "max_tokens": 1024,
        "top_k": 4,
        "chunk_size": 256,
        "retrieval_config_sha256": sha256_text("retrieval v1"),
        "system_prompt_sha256": sha256_text("system prompt v1"),
        "kb_sha256": sha256_text("corpus v1"),
        "pricing_version": sha256_text("prices v1"),
        "max_tool_calls": 3,
        "max_tool_errors": 2,
        "max_model_calls": 6,
        "git_sha": "a" * 40,
        "code_version": "0.1.0",
        "git_dirty": False,
        "dataset_path": None,
        "dataset_sha256": None,
        "n_items": None,
        "seeds": None,
    }
    base.update(overrides)
    manifest = RunManifest(**base)
    manifest.write(tmp_path / "runs")
    return manifest


def chat_turn(
    item_id: str,
    question: str,
    answer: str = ANSWER,
    *,
    turn_idx: int = 0,
    latency_ms: float = 100.0,
    usd_cost: float | None = 0.001,
    stopped_reason: str | None = None,
    guardrail_completion: str | None = None,
    guardrail_action: str = "output_filtered",
) -> list[dict[str, Any]]:
    """One chat turn's records, in the order `agent.core` writes them.

    `guardrail_completion` adds the `guardrail` record a screened turn produces. Its content is
    the text the user was shown, which is why the transcript reader prefers it and the history
    replay does not: the `turn` record holds the model's own output, and on an output substitution
    that is what the conversation went on being conditioned on.

    Set `guardrail_action="input_blocked"` for the other shape, where the screen fired before the
    model ran and there is therefore no `assistant` record at all.
    """
    # A real trace always carries `ts`, and the transcript reader shows the first turn's as when the
    # conversation was opened. Derived from the index so a fixture's turns are plausibly ordered.
    common = {
        "item_id": item_id,
        "turn_idx": turn_idx,
        "ts": f"2026-07-29T09:{turn_idx:02d}:00.000+00:00",
    }
    blocked = guardrail_completion is not None and guardrail_action == "input_blocked"

    records: list[dict[str, Any]] = [{**common, "role": "user", "content": question}]
    if not blocked:
        records.append(
            {
                **common,
                "role": "assistant",
                "content": answer,
                "latency_ms": latency_ms,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "usd_cost": usd_cost,
            }
        )
    if guardrail_completion is not None:
        records.append(
            {
                **common,
                "role": "guardrail",
                "content": guardrail_completion,
                "guardrail_action": guardrail_action,
                "guardrail_stage": "input" if blocked else "output",
            }
        )
    records.append(
        {
            **common,
            "role": "turn",
            # The model's own text even when a guardrail replaced what was delivered, which is
            # what a real trace holds: keeping our sentence off this record is what keeps it out
            # of every scorer.
            "content": "" if blocked else answer,
            "latency_ms": latency_ms,
            "usd_cost": usd_cost,
            "error": "input_blocked" if blocked else stopped_reason,
            "guardrail_action": None if guardrail_completion is None else guardrail_action,
        }
    )
    return records


def write_judge_run(
    tmp_path: Path,
    judge_run_id: str,
    scored_run_id: str,
    scores: dict[str, dict[str, float] | None],
    **overrides: Any,
) -> RunManifest:
    """A judge run scoring `scored_run_id`'s trace, with one judgement per entry in `scores`.

    A None value is a judgement that did not parse, which is a judge-side failure and must not
    land in any candidate-side rate.

    Args:
        overrides: Manifest fields to replace. `pairs_path` is the field that joins this judge run
            to the trace it scored — `metrics.find_judge_run` reads it rather than guessing from a
            filename — so pointing it elsewhere is how a test says "this judge run scored a
            different trace".
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    base: dict[str, Any] = {
        "run_id": judge_run_id,
        "started_at": "2026-07-29T10:00:00.000+00:00",
        "run_kind": "judge",
        "model_name": "judge-model-1",
        "provider": "anthropic",
        "temperature": 0.0,
        "max_tokens": 1024,
        "top_k": None,
        "chunk_size": None,
        "retrieval_config_sha256": None,
        "system_prompt_sha256": None,
        "kb_sha256": None,
        "pricing_version": sha256_text("prices v1"),
        "max_tool_calls": None,
        "max_tool_errors": None,
        "max_model_calls": None,
        "git_sha": "a" * 40,
        "code_version": "0.1.0",
        "judge_model": "judge-model-1",
        "judge_provider": "anthropic",
        "judge_rubric_sha256": sha256_text("rubric v1"),
        "judge_rubrics": ["default"],
        "pairs_path": str(runs / f"{scored_run_id}.jsonl"),
        "pairs_sha256": sha256_text("pairs v1"),
        "n_pairs": len(scores),
    }
    base.update(overrides)
    manifest = RunManifest(**base)
    manifest.write(runs)

    lines = []
    for pair_id, dimensions in scores.items():
        lines.append(
            json.dumps(
                {
                    "pair_id": pair_id,
                    "judge_model": "judge-model-1",
                    "parse_ok": dimensions is not None,
                    "scores": dimensions or {},
                    "overall": (
                        None
                        if dimensions is None
                        else sum(dimensions.values()) / len(dimensions)
                    ),
                    "rationale": "because",
                    "raw_completion": "{}",
                }
            )
        )
    (runs / f"{judge_run_id}.judge.jsonl").write_text(
        "".join(line + "\n" for line in lines), encoding="utf-8"
    )
    return manifest


def verdict(**overrides: float) -> dict[str, float]:
    """One judgement's four dimensions, defaulting to a clean pass."""
    scores = {"helpfulness": 5.0, "accuracy": 5.0, "safety": 5.0, "communication": 5.0}
    scores.update(overrides)
    return scores
