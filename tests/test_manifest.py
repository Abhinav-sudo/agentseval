"""Tests for `agent.manifest`, weighted toward the comparability guard.

`assert_comparable` is what backs the claim that the frontier and OSS arms differed only in
model weights, so most of these cases are about it refusing to pass when some other condition
drifted. The rest cover `build_manifest`: one builder for all three kinds of run, with the
dataset required for one kind, the `JudgeRef` for another, and each refused everywhere else.

No network and no model calls, so the suite runs without API keys.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest

from agent.core import Budgets
from agent.guardrails import guardrails_sha256
from agent.manifest import (
    ABLATION_CONDITIONS,
    ABLATION_EXEMPT,
    ABSENT_PACKAGE,
    COMPARABLE_EXEMPT,
    EVAL_ONLY_FIELDS,
    IDENTITY_FIELDS,
    INFORMATIONAL_FIELDS,
    JUDGE_ONLY_FIELDS,
    POST_HOC_OPTIONAL_FIELDS,
    RETRIEVAL_STACK_PACKAGES,
    AgentConfig,
    DatasetRef,
    JudgeRef,
    NotComparableError,
    RunManifest,
    agent_config_digest,
    agent_config_fields,
    assert_ablation_comparable,
    assert_comparable,
    build_manifest,
    compare_manifests,
    new_run_id,
    pricing_sha256,
    retrieval_config_sha256,
    retrieval_stack_version,
)
from agent.prompts import judge_pair_template_sha256
from agent.trace import manifest_path, sha256_text, trace_path
from tests.fakes import FakeAdapter

FRONTIER = {"model_name": "claude-sonnet-4-20250514", "provider": "anthropic"}
OSS = {"model_name": "llama-3.1-8b-instant", "provider": "groq"}

#: Conditions that must be held constant for two runs to be comparable: every manifest field
#: except run identity, the model itself, and the fields that only name a condition.
CONDITION_FIELDS = sorted(
    f.name
    for f in fields(RunManifest)
    if f.name not in IDENTITY_FIELDS | COMPARABLE_EXEMPT | INFORMATIONAL_FIELDS
)


def manifest(**overrides: object) -> RunManifest:
    """A complete manifest with plausible defaults, overridable per test."""
    base: dict[str, object] = {
        "run_id": "run-0001",
        "started_at": "2026-07-28T09:00:00.000+00:00",
        "run_kind": "eval",
        "model_name": FRONTIER["model_name"],
        "provider": FRONTIER["provider"],
        "temperature": 0.0,
        "max_tokens": 1024,
        "top_k": 4,
        "chunk_size": 256,
        "retrieval_config_sha256": sha256_text("retrieval v1"),
        "retrieval_stack_version": "numpy==2.0.0 sentence-transformers==3.3.0 torch==2.5.0",
        "system_prompt_sha256": sha256_text("system prompt v1"),
        "kb_sha256": sha256_text("corpus v1"),
        "pricing_version": sha256_text("prices v1"),
        "max_tool_calls": 3,
        "max_tool_errors": 2,
        "max_model_calls": 6,
        "git_sha": "a" * 40,
        "code_version": "0.1.0",
        "git_dirty": False,
        "dataset_path": "evals/datasets/core.jsonl",
        "dataset_sha256": sha256_text("dataset v1"),
        "n_items": 40,
        "seeds": None,
    }
    base.update(overrides)
    return RunManifest(**base)  # type: ignore[arg-type]


def chat_manifest(**overrides: object) -> RunManifest:
    """A manifest as a chat session writes one: every eval-only field empty."""
    blanks: dict[str, object] = dict.fromkeys(EVAL_ONLY_FIELDS)
    return manifest(run_kind="chat", **blanks, **overrides)


def other_value(field_name: str, current: object) -> Any:
    """Return a value of the right type that differs from `current`."""
    if isinstance(current, bool):
        return not current
    if isinstance(current, float):
        return current + 0.7
    if isinstance(current, int):
        return current + 1
    if isinstance(current, str):
        return current + "-changed"
    return 99  # `None` fields such as seeds


# --------------------------------------------------------------------------------------
# assert_comparable — the guard
# --------------------------------------------------------------------------------------


def test_frontier_and_oss_runs_are_comparable():
    """The real case: same harness, different model. This is the whole point."""
    frontier = manifest(run_id="run-frontier", **FRONTIER)
    oss = manifest(run_id="run-oss", **OSS)
    assert_comparable(frontier, oss)


def test_run_identity_alone_does_not_block_comparison():
    """run_id and started_at differ between any two runs, so they cannot be blockers."""
    a = manifest(run_id="run-a", started_at="2026-07-28T09:00:00.000+00:00")
    b = manifest(run_id="run-b", started_at="2026-07-29T17:30:00.000+00:00")
    assert_comparable(a, b)


def test_identical_manifests_are_comparable():
    assert_comparable(manifest(), manifest())


@pytest.mark.parametrize("field_name", CONDITION_FIELDS)
def test_any_differing_condition_blocks_comparison(field_name: str):
    """Every field that is not exempt must be able to fail the guard on its own.

    Parametrised over the dataclass rather than a hand-written list, so a field added later is
    covered automatically instead of being silently unguarded.
    """
    a = manifest()
    b = replace(a, **{field_name: other_value(field_name, getattr(a, field_name))})
    with pytest.raises(NotComparableError) as exc:
        assert_comparable(a, b)
    assert field_name in str(exc.value)


def test_edited_corpus_blocks_comparison():
    """A KB edit between the two runs is the silent invalidation this guard exists for.

    Retrieval changes underneath both arms while every other condition still matches, so
    nothing else would catch it.
    """
    a = manifest(kb_sha256=sha256_text("corpus v1"), **FRONTIER)
    b = manifest(kb_sha256=sha256_text("corpus v2"), **OSS)
    with pytest.raises(NotComparableError, match="kb_sha256"):
        assert_comparable(a, b)


def test_error_names_every_offending_field_with_both_values():
    """One call must report the full extent of the drift, not just the first instance."""
    a = manifest()
    b = replace(a, temperature=0.7, max_model_calls=12, kb_sha256=sha256_text("corpus v2"), **OSS)

    with pytest.raises(NotComparableError) as exc:
        assert_comparable(a, b)

    message = str(exc.value)
    for field_name in ("temperature", "max_model_calls", "kb_sha256"):
        assert field_name in message
    assert "0.0" in message and "0.7" in message
    assert "6" in message and "12" in message
    assert "3 condition(s) differ" in message
    # The exempt model fields are not complaints.
    assert "model_name" not in message.split(":\n", 1)[1]


@pytest.mark.parametrize(
    "budget", ["max_tool_calls", "max_tool_errors", "max_model_calls", "max_tokens"]
)
def test_arms_with_different_budgets_are_not_comparable(budget: str):
    """Named as well as covered by the parametrised sweep, because this is the specific way a
    comparison gets quietly invalidated: one arm given more room to recover from its own
    mistakes, or a lower token ceiling that truncates its answers, is a different experiment
    and the gap would read as quality."""
    a = manifest(**FRONTIER)
    b = manifest(**OSS, **{budget: 12})
    with pytest.raises(NotComparableError, match=budget):
        assert_comparable(a, b)


def test_cost_difference_is_exempt():
    """usd_cost is expected to differ: the arms are priced differently."""
    a = manifest().to_dict() | {"usd_cost": 0.42}
    b = manifest(**OSS).to_dict() | {"usd_cost": 0.03}
    assert_comparable(a, b)


def test_exempt_set_is_exactly_the_documented_three():
    """A canary: widening the allowlist weakens every comparison, so it must be deliberate.

    If this fails, PROJECT.md needs updating before the code does.
    """
    assert set(COMPARABLE_EXEMPT) == {"model_name", "provider", "usd_cost"}
    assert set(IDENTITY_FIELDS) == {"run_id", "started_at", "run_kind"}
    # Both are filenames rather than conditions: the digest beside each is what must match.
    assert set(INFORMATIONAL_FIELDS) == {"dataset_path", "pairs_path"}


def test_chat_and_eval_manifests_are_never_comparable():
    """Two kinds of run are not two arms of one experiment, whatever else matches."""
    with pytest.raises(NotComparableError, match="different kinds"):
        assert_comparable(chat_manifest(), manifest())


def test_run_kind_is_refused_before_any_field_diff():
    """The message must name the kinds rather than list the dataset fields as drift.

    A list of drifted conditions invites someone to reconcile them one at a time; naming the
    mismatch says the comparison is the wrong question.
    """
    with pytest.raises(NotComparableError) as exc:
        assert_comparable(manifest(), chat_manifest())
    assert "dataset_sha256" not in str(exc.value)


def test_two_chat_runs_of_the_same_config_are_comparable():
    """Empty eval fields match each other, so chat runs compare like any other pair."""
    assert_comparable(chat_manifest(run_id="a", **FRONTIER), chat_manifest(run_id="b", **OSS))


def test_edited_dataset_at_the_same_path_blocks_comparison():
    """The hash is the only thing that catches a dataset edited between two runs.

    Both manifests name `datasets/core.jsonl`, so a path comparison would pass this.
    """
    a = manifest(dataset_sha256=sha256_text("dataset v1"), **FRONTIER)
    b = manifest(dataset_sha256=sha256_text("dataset v1 edited"), **OSS)
    with pytest.raises(NotComparableError, match="dataset_sha256"):
        assert_comparable(a, b)


def test_dataset_path_alone_does_not_block_comparison():
    """A moved or renamed file with identical bytes is the same dataset."""
    a = manifest(dataset_path="evals/datasets/core.jsonl")
    b = manifest(dataset_path="/tmp/copy/core.jsonl", **OSS)
    assert_comparable(a, b)
    # Still reported, because a diff that hides a difference is not a diff.
    assert "dataset_path" in compare_manifests(a, b)


def test_dataset_item_count_blocks_comparison():
    """A truncated dataset is a different experiment even where the file it came from matches."""
    with pytest.raises(NotComparableError, match="n_items"):
        assert_comparable(manifest(), manifest(n_items=20, **OSS))


def test_mismatched_schemas_raise_rather_than_skipping_fields():
    """A field present in only one manifest cannot be checked, so refuse to compare.

    Tolerating it is how the guard would quietly hollow out as the manifest evolves.
    """
    complete = manifest().to_dict()
    partial = {k: v for k, v in complete.items() if k != "kb_sha256"}

    with pytest.raises(ValueError, match="kb_sha256"):
        assert_comparable(complete, partial)
    with pytest.raises(ValueError, match="kb_sha256"):
        compare_manifests(complete, partial)


def test_unexpected_field_raises():
    extended = manifest().to_dict() | {"future_field": 1}
    with pytest.raises(ValueError, match="future_field"):
        assert_comparable(manifest(), extended)


def test_non_manifest_input_raises_typeerror():
    with pytest.raises(TypeError):
        compare_manifests(manifest(), ["not", "a", "manifest"])


# --------------------------------------------------------------------------------------
# assert_ablation_comparable — the 2×2 and its diagonal
# --------------------------------------------------------------------------------------

#: The four runs of the guardrails design, as manifest overrides. Two models × guardrails on/off.
#: `guardrails_sha256` is a digest when the screens are on and None when they are off, which is
#: what the ablation guard's condition set exists to accommodate.
GUARDRAILS_ON: dict[str, object] = {
    "guardrails": True,
    "guardrails_sha256": sha256_text("guardrails v1"),
}
GUARDRAILS_OFF: dict[str, object] = {"guardrails": False, "guardrails_sha256": None}


def test_an_on_off_pair_of_one_model_is_an_ablation():
    """The edge of the 2×2 this guard exists for."""
    on = manifest(run_id="run-on", **FRONTIER, **GUARDRAILS_ON)
    off = manifest(run_id="run-off", **FRONTIER, **GUARDRAILS_OFF)

    assert_ablation_comparable(on, off)


def test_the_arm_guard_refuses_an_on_off_pair():
    """Right, not an obstacle: they are one arm under two settings, not two arms."""
    on = manifest(run_id="run-on", **FRONTIER, **GUARDRAILS_ON)
    off = manifest(run_id="run-off", **FRONTIER, **GUARDRAILS_OFF)

    with pytest.raises(NotComparableError, match="guardrails"):
        assert_comparable(on, off)


def test_the_ablation_guard_refuses_an_arm_pair():
    """Two models at the same setting is the other edge, and the other guard's business."""
    frontier = manifest(run_id="run-f", **FRONTIER, **GUARDRAILS_ON)
    oss = manifest(run_id="run-o", **OSS, **GUARDRAILS_ON)

    assert_comparable(frontier, oss)
    with pytest.raises(NotComparableError, match="do not differ in guardrails_sha256"):
        assert_ablation_comparable(frontier, oss)


def test_the_diagonal_is_refused_by_both_guards():
    """Frontier-with-guardrails against OSS-without varies two things, so the delta is neither's."""
    frontier_on = manifest(run_id="run-f-on", **FRONTIER, **GUARDRAILS_ON)
    oss_off = manifest(run_id="run-o-off", **OSS, **GUARDRAILS_OFF)

    with pytest.raises(NotComparableError, match="guardrails"):
        assert_comparable(frontier_on, oss_off)
    with pytest.raises(NotComparableError, match="model_name"):
        assert_ablation_comparable(frontier_on, oss_off)


def test_the_ablation_guard_treats_the_model_as_an_ordinary_condition():
    """`COMPARABLE_EXEMPT` excuses the model for an arm comparison. An ablation does not."""
    on = manifest(run_id="run-on", **FRONTIER, **GUARDRAILS_ON)
    other_model_off = manifest(run_id="run-off", **OSS, **GUARDRAILS_OFF)

    with pytest.raises(NotComparableError) as exc:
        assert_ablation_comparable(on, other_model_off)

    assert "model_name" in str(exc.value)
    assert "provider" in str(exc.value)


@pytest.mark.parametrize("field_name", ["kb_sha256", "temperature", "max_model_calls", "n_items"])
def test_any_other_drift_still_refuses_an_ablation(field_name: str):
    """One variable, a different variable. Everything else has to hold still here too."""
    on = manifest(run_id="run-on", **FRONTIER, **GUARDRAILS_ON)
    off = replace(
        manifest(run_id="run-off", **FRONTIER, **GUARDRAILS_OFF),
        **{field_name: other_value(field_name, getattr(on, field_name))},
    )

    with pytest.raises(NotComparableError, match=field_name):
        assert_ablation_comparable(on, off)


def test_cost_is_exempt_from_an_ablation_as_well():
    """A guardrailed arm that made fewer model calls is supposed to cost less."""
    on = manifest(run_id="run-on", **FRONTIER, **GUARDRAILS_ON).to_dict() | {"usd_cost": 0.02}
    off = manifest(run_id="run-off", **FRONTIER, **GUARDRAILS_OFF).to_dict() | {"usd_cost": 0.4}

    assert_ablation_comparable(on, off)


def test_an_unregistered_ablation_field_is_refused_rather_than_exempted():
    """Otherwise "vary this field" becomes "exempt this field" at any call site that wants one."""
    a = manifest(run_id="run-a", temperature=0.0)
    b = manifest(run_id="run-b", temperature=0.7)

    with pytest.raises(ValueError, match="not a registered ablation condition"):
        assert_ablation_comparable(a, b, varying="temperature")


def test_two_runs_of_one_setting_are_not_an_ablation():
    """A replicate is compared with `assert_comparable`; labelling it a delta invites a quote."""
    a = manifest(run_id="run-a", **FRONTIER, **GUARDRAILS_ON)
    b = manifest(run_id="run-b", **FRONTIER, **GUARDRAILS_ON)

    with pytest.raises(NotComparableError, match="no ablation between them"):
        assert_ablation_comparable(a, b)


def test_two_kinds_of_run_are_not_one_run_under_two_settings():
    with pytest.raises(NotComparableError, match="different kinds"):
        assert_ablation_comparable(
            manifest(**GUARDRAILS_ON), chat_manifest(**GUARDRAILS_OFF)
        )


def test_the_ablation_registry_and_the_exempt_sets_are_the_documented_ones():
    """Both sets are claims about what a comparison means; widening either is a decision.

    `COMPARABLE_EXEMPT` is asserted here as well as above because the tempting way to make the
    ablation work is to widen it, and that would weaken every arm comparison in the project.
    """
    assert dict(ABLATION_CONDITIONS) == {
        "guardrails_sha256": frozenset({"guardrails_sha256", "guardrails"})
    }
    assert set(ABLATION_EXEMPT) == {"usd_cost"}
    assert set(COMPARABLE_EXEMPT) == {"model_name", "provider", "usd_cost"}
    assert "guardrails" not in COMPARABLE_EXEMPT
    assert "guardrails_sha256" not in COMPARABLE_EXEMPT


# --------------------------------------------------------------------------------------
# compare_manifests
# --------------------------------------------------------------------------------------


def test_compare_manifests_empty_for_identical():
    assert compare_manifests(manifest(), manifest()) == []


def test_compare_manifests_reports_identity_fields():
    """An honest diff includes run_id and started_at; the guard is what excuses them."""
    a = manifest(run_id="run-a")
    b = manifest(run_id="run-b", **OSS)
    assert compare_manifests(a, b) == ["model_name", "provider", "run_id"]


def test_compare_manifests_is_symmetric_and_sorted():
    a = manifest()
    b = replace(a, top_k=8, temperature=0.5, **OSS)
    forward = compare_manifests(a, b)
    assert forward == compare_manifests(b, a)
    assert forward == sorted(forward)
    assert forward == ["model_name", "provider", "temperature", "top_k"]


def test_compare_manifests_accepts_loaded_dicts(tmp_path):
    a = manifest(run_id="run-a")
    b = manifest(run_id="run-b", **OSS)
    a.write(tmp_path)
    b.write(tmp_path)

    loaded_a = RunManifest.load("run-a", tmp_path)
    loaded_b = RunManifest.load("run-b", tmp_path)
    assert compare_manifests(loaded_a, loaded_b) == compare_manifests(a, b)


# --------------------------------------------------------------------------------------
# RunManifest serialisation
# --------------------------------------------------------------------------------------


def test_manifest_round_trips_through_dict():
    original = manifest()
    assert RunManifest.from_dict(original.to_dict()) == original


def test_write_lands_at_expected_sibling_path(tmp_path):
    written = manifest(run_id="run-xyz").write(tmp_path)

    assert written == tmp_path / "run-xyz.manifest.json"
    assert written == manifest_path("run-xyz", tmp_path)
    assert trace_path("run-xyz", tmp_path) == tmp_path / "run-xyz.jsonl"
    assert written.parent == trace_path("run-xyz", tmp_path).parent


def test_write_creates_missing_runs_dir(tmp_path):
    target = tmp_path / "nested" / "runs"
    written = manifest().write(target)
    assert written.exists()


def test_write_then_read_round_trips(tmp_path):
    original = chat_manifest(run_id="run-rt", chunk_size=None, git_sha=None, kb_sha256=None)
    original.write(tmp_path)
    assert RunManifest.load("run-rt", tmp_path) == original


def test_seeds_round_trip_as_a_list(tmp_path):
    original = manifest(run_id="run-seeded", seeds=[1, 2, 3])
    original.write(tmp_path)
    assert RunManifest.load("run-seeded", tmp_path).seeds == [1, 2, 3]


def test_write_leaves_no_temporary_files(tmp_path):
    manifest().write(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["run-0001.manifest.json"]


def test_write_is_valid_json_with_every_field(tmp_path):
    path = manifest().write(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {f.name for f in fields(RunManifest)}


def test_read_rejects_malformed_json(tmp_path):
    path = tmp_path / "broken.manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        RunManifest.read(path)


def test_read_rejects_incomplete_manifest(tmp_path):
    path = tmp_path / "partial.manifest.json"
    path.write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        RunManifest.read(path)


# --------------------------------------------------------------------------------------
# build_manifest — one builder for both kinds of run
# --------------------------------------------------------------------------------------


def config(tmp_path: Path, **overrides: Any) -> AgentConfig:
    """A config over an empty corpus directory, so no embedding model is ever loaded."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(exist_ok=True)
    settings: dict[str, Any] = {"model": FakeAdapter(["{}"]), "kb_dir": kb_dir}
    return AgentConfig(**(settings | overrides))


def dataset(tmp_path: Path, contents: str = '{"case_id": "a"}\n', n_items: int = 1) -> DatasetRef:
    path = tmp_path / "core.jsonl"
    path.write_text(contents, encoding="utf-8")
    return DatasetRef.for_file(path, n_items=n_items)


def test_chat_manifest_leaves_every_eval_field_empty(tmp_path):
    """The nullable tail is the whole reason there is one manifest rather than two."""
    built = build_manifest(config(tmp_path), run_kind="chat")

    assert built.run_kind == "chat"
    for name in EVAL_ONLY_FIELDS:
        assert getattr(built, name) is None


def test_eval_manifest_records_the_dataset(tmp_path):
    ref = dataset(tmp_path, n_items=7)
    built = build_manifest(config(tmp_path), run_kind="eval", dataset=ref)

    assert built.run_kind == "eval"
    assert built.dataset_path == str(ref.path)
    assert built.dataset_sha256 == ref.sha256
    assert built.n_items == 7
    assert built.seeds is None


def test_eval_run_without_a_dataset_raises(tmp_path):
    """A graded run whose data cannot be identified is not a result."""
    with pytest.raises(ValueError, match="needs a DatasetRef"):
        build_manifest(config(tmp_path), run_kind="eval")


def test_chat_run_with_a_dataset_raises(tmp_path):
    """A chat manifest has nowhere to put one, and accepting it would silently drop it."""
    with pytest.raises(ValueError, match="no dataset"):
        build_manifest(config(tmp_path), run_kind="chat", dataset=dataset(tmp_path))


def test_dataset_ref_digests_bytes_not_names(tmp_path):
    """The same path with edited contents must produce a different digest, and vice versa."""
    first = dataset(tmp_path, '{"case_id": "a"}\n')
    edited = dataset(tmp_path, '{"case_id": "a"}\n{"case_id": "b"}\n', n_items=2)
    assert first.path == edited.path
    assert first.sha256 != edited.sha256

    moved = (tmp_path / "moved").resolve()
    moved.mkdir()
    copy = moved / "core.jsonl"
    copy.write_text(edited.path.read_text(encoding="utf-8"), encoding="utf-8")
    assert DatasetRef.for_file(copy, n_items=2).sha256 == edited.sha256


def test_manifest_reflects_the_config_it_was_built_from(tmp_path):
    budgets = Budgets(max_tool_calls=5, max_tool_errors=1, max_model_calls=9)
    cfg = config(tmp_path, budgets=budgets, temperature=0.3, max_tokens=64, top_k=9)
    built = build_manifest(cfg, run_kind="chat")

    assert built.model_name == cfg.model.model_id
    assert (built.temperature, built.max_tokens, built.top_k) == (0.3, 64, 9)
    assert built.max_tool_calls == 5
    assert built.max_tool_errors == 1
    assert built.max_model_calls == 9
    assert built.pricing_version == pricing_sha256()
    assert built.retrieval_config_sha256 == retrieval_config_sha256(
        top_k=9, min_score=cfg.min_score
    )


def test_manifest_records_the_corpus_it_will_retrieve_from(tmp_path):
    cfg = config(tmp_path)
    empty = build_manifest(cfg, run_kind="chat")
    # No corpus: nothing to digest, and no chunking to describe.
    assert empty.kb_sha256 is None
    assert empty.chunk_size is None

    (Path(cfg.kb_dir) / "sleep.md").write_text("## Sleep\n\nGo to bed.\n", encoding="utf-8")
    stocked = build_manifest(cfg, run_kind="chat")
    assert stocked.kb_sha256 is not None
    assert stocked.chunk_size is not None

    (Path(cfg.kb_dir) / "sleep.md").write_text("## Sleep\n\nGo to bed early.\n", encoding="utf-8")
    assert build_manifest(cfg, run_kind="chat").kb_sha256 != stocked.kb_sha256


def test_retrieval_digest_covers_settings_with_no_field_of_their_own(tmp_path):
    """The embedding model and the score floor are conditions, and only the digest holds them."""
    baseline = retrieval_config_sha256()
    assert retrieval_config_sha256(embedding_model="other/model") != baseline
    assert retrieval_config_sha256(min_score=0.3) != baseline
    assert retrieval_config_sha256() == baseline


def test_the_stack_version_names_every_library_that_computes_a_vector():
    """Readable and canonically ordered, so a reader can see which library moved rather than only
    that something did — the objection `retrieval_config_sha256` records against hashing `top_k`."""
    recorded = retrieval_stack_version()

    for package in RETRIEVAL_STACK_PACKAGES:
        assert f"{package}==" in recorded
    # Sorted, so naming the same packages in another order describes the same environment.
    assert retrieval_stack_version(tuple(reversed(RETRIEVAL_STACK_PACKAGES))) == recorded


def test_an_uninstalled_library_is_recorded_rather_than_raising():
    """A fact about the environment. Raising here would lose the run instead of describing it, and
    a missing encoder already fails loudly at the lookup."""
    recorded = retrieval_stack_version(("numpy", "definitely-not-installed"))

    assert f"definitely-not-installed=={ABSENT_PACKAGE}" in recorded
    assert f"numpy=={ABSENT_PACKAGE}" not in recorded


def test_the_manifest_records_the_versions_that_embedded_the_corpus(tmp_path):
    """`retrieval_config_sha256` records the encoder asked for; this records the code that ran it.
    The same model id under a different torch can return different vectors."""
    cfg = config(tmp_path)
    # No corpus, so nothing was embedded and a torch upgrade could not have changed a lookup.
    assert build_manifest(cfg, run_kind="chat").retrieval_stack_version is None

    (Path(cfg.kb_dir) / "sleep.md").write_text("## Sleep\n\nGo to bed.\n", encoding="utf-8")
    stocked = build_manifest(cfg, run_kind="chat")

    assert stocked.retrieval_stack_version == retrieval_stack_version()


def test_an_encoder_upgrade_blocks_comparison():
    """The silent failure this field exists for: the settings digest matches, the corpus digest
    matches, and the two arms were embedded by different code."""
    a = manifest(retrieval_stack_version="torch==2.5.0", **FRONTIER)
    b = manifest(retrieval_stack_version="torch==2.11.0", **OSS)

    with pytest.raises(NotComparableError, match="retrieval_stack_version"):
        assert_comparable(a, b)


def test_a_manifest_predating_the_stack_version_loads_as_unknown():
    """Not backfilled. Today's versions on an old run would assert it embedded under libraries that
    did not exist when it ran."""
    current = manifest().to_dict()
    old = {k: v for k, v in current.items() if k != "retrieval_stack_version"}

    loaded = RunManifest.from_dict(old)

    assert loaded.retrieval_stack_version is None
    assert loaded.retrieval_config_sha256 == current["retrieval_config_sha256"]


def test_an_unknown_stack_version_is_not_evidence_of_sameness():
    """Not exempt from the guard, for the reason the pair-template digest is not: "unknown" is not
    a match, and a comparison that treated it as one would vouch for a condition nobody recorded."""
    current = manifest(**FRONTIER)
    old = replace(current, run_id="run-old", retrieval_stack_version=None, **OSS)

    with pytest.raises(NotComparableError, match="retrieval_stack_version"):
        assert_comparable(current, old)


def test_guardrails_reach_the_manifest_as_a_condition(tmp_path):
    """Unrecorded, two runs differing only in guardrails produce identical manifests."""
    off = build_manifest(config(tmp_path), run_kind="chat")
    on = build_manifest(config(tmp_path, guardrails=True), run_kind="chat")

    assert off.guardrails is False
    assert on.guardrails is True
    assert on.guardrails_sha256 == guardrails_sha256()
    # Identity aside, the two runs differ in exactly the pair of fields the condition owns.
    drift = [name for name in compare_manifests(on, off) if name not in IDENTITY_FIELDS]
    assert drift == ["guardrails", "guardrails_sha256"]


def test_the_guardrails_digest_is_none_when_the_screens_are_off(tmp_path):
    """Recording it while off would make two off runs refuse each other after a pattern edit."""
    off = build_manifest(config(tmp_path), run_kind="chat")

    assert off.guardrails_sha256 is None


def test_guardrails_are_part_of_the_agent_config_group(tmp_path):
    """So `agent_config_digest` and the mid-session check pick them up with no further edit."""
    off = build_manifest(config(tmp_path), run_kind="chat")
    on = build_manifest(config(tmp_path, guardrails=True), run_kind="chat")

    assert "guardrails" in agent_config_fields()
    assert "guardrails_sha256" in agent_config_fields()
    assert agent_config_digest(on) != agent_config_digest(off)


def test_a_judge_run_records_no_guardrails_at_all(tmp_path):
    """A judge screens nothing, and False would claim it ran with the screens off."""
    built = build_manifest(run_kind="judge", judge=judge_ref(tmp_path))

    assert built.guardrails is None
    assert built.guardrails_sha256 is None


def test_a_config_with_guardrails_off_builds_an_agent_without_them(tmp_path):
    assert config(tmp_path).build_guardrails() is None
    assert config(tmp_path, guardrails=True).build_guardrails() is not None


def test_two_arms_built_from_one_config_differ_only_in_the_model(tmp_path):
    """The end-to-end form of the uniform-harness claim, through the real builder."""
    cfg = config(tmp_path)
    frontier = build_manifest(cfg, run_kind="chat")
    oss = build_manifest(
        replace(cfg, model=FakeAdapter(["{}"], model_id="llama-3.1-8b-instant")),
        run_kind="chat",
    )
    assert frontier.model_name != oss.model_name
    assert_comparable(frontier, oss)


def judge_ref(tmp_path: Path, contents: str = '{"prompt": "q", "response": "a"}\n') -> JudgeRef:
    path = tmp_path / "pairs.jsonl"
    path.write_text(contents, encoding="utf-8")
    return JudgeRef.for_file(
        path,
        n_pairs=contents.count("\n"),
        model_name="gpt-4o-2024-11-20",
        provider="openai",
        rubric_sha256=sha256_text("rubric v1"),
        rubric_names=["default", "safety"],
    )


def test_judge_manifest_records_the_judge_and_the_rubric(tmp_path):
    ref = judge_ref(tmp_path)
    built = build_manifest(run_kind="judge", judge=ref)

    assert built.run_kind == "judge"
    assert built.judge_model == ref.model_name == built.model_name
    assert built.judge_provider == ref.provider == built.provider
    assert built.judge_rubric_sha256 == ref.rubric_sha256
    assert built.judge_rubrics == ["default", "safety"]
    assert built.pairs_path == str(ref.pairs_path)
    assert built.pairs_sha256 == ref.pairs_sha256
    assert built.n_pairs == 1
    assert built.temperature == 0.0
    assert built.judge_pair_template_sha256 == judge_pair_template_sha256()


def test_a_new_judge_manifest_records_the_pair_template_digest(tmp_path):
    """The other half of what the judge read. `judge_rubric_sha256` covers the system message;
    rewording a block heading in the user message changes every judge call and no rubric."""
    built = build_manifest(run_kind="judge", judge=judge_ref(tmp_path))

    assert built.judge_pair_template_sha256 == judge_pair_template_sha256()
    assert built.judge_pair_template_sha256 != built.judge_rubric_sha256


def test_a_manifest_predating_the_pair_template_digest_loads_as_unknown(tmp_path):
    """Not backfilled. A guessed digest would assert that an old run read today's template, which
    is the one claim nobody can make about it."""
    current = build_manifest(run_kind="judge", judge=judge_ref(tmp_path)).to_dict()
    old = {k: v for k, v in current.items() if k != "judge_pair_template_sha256"}

    loaded = RunManifest.from_dict(old)

    assert loaded.judge_pair_template_sha256 is None
    assert loaded.judge_rubric_sha256 == current["judge_rubric_sha256"]


def test_the_post_hoc_allowlist_is_pinned_to_its_members():
    """Every name here is a field `from_dict` will accept as absent, which is a small hole in the
    strictness that protects every comparison. Growing it is a decision, so it is asserted rather
    than left to whoever adds the next field."""
    assert set(POST_HOC_OPTIONAL_FIELDS) == {
        "judge_pair_template_sha256",
        "retrieval_stack_version",
    }
    assert set(POST_HOC_OPTIONAL_FIELDS) <= {f.name for f in fields(RunManifest)}


def test_an_unknown_key_is_still_fatal():
    """`POST_HOC_OPTIONAL_FIELDS` is an allowlist, not a relaxation."""
    with pytest.raises(ValueError, match=r"unexpected \['invented_field'\]"):
        RunManifest.from_dict({**manifest().to_dict(), "invented_field": 1})


def test_another_missing_field_is_still_fatal():
    """Only the post-hoc group may be absent. Everything else missing means the file was written
    by different code, and loading it as comparable is the failure the strictness prevents."""
    incomplete = {k: v for k, v in manifest().to_dict().items() if k != "system_prompt_sha256"}

    with pytest.raises(ValueError, match=r"missing \['system_prompt_sha256'\]"):
        RunManifest.from_dict(incomplete)


def test_a_missing_post_hoc_field_alongside_a_missing_ordinary_one_still_raises():
    """The allowlist forgives its own member and nothing else in the same file."""
    trimmed = {
        k: v
        for k, v in manifest().to_dict().items()
        if k not in {"judge_pair_template_sha256", "kb_sha256"}
    }

    with pytest.raises(ValueError) as excinfo:
        RunManifest.from_dict(trimmed)

    assert "kb_sha256" in str(excinfo.value)
    assert "judge_pair_template_sha256" not in str(excinfo.value)


def test_an_unknown_pair_template_digest_is_not_evidence_of_sameness(tmp_path):
    """Deliberately not exempt from the guard. `None` means the field did not exist yet, and a
    comparison that treated it as a match would report two runs as comparable on a condition
    neither of them can speak to."""
    current = build_manifest(run_kind="judge", judge=judge_ref(tmp_path))
    old = replace(current, run_id="run-old", judge_pair_template_sha256=None)

    with pytest.raises(NotComparableError) as excinfo:
        assert_comparable(current, old)

    assert "judge_pair_template_sha256" in str(excinfo.value)
    assert "predates" in str(excinfo.value)
    assert "unknown is not evidence of sameness" in str(excinfo.value)


def test_two_judge_runs_under_different_pair_templates_are_not_comparable(tmp_path):
    """The case the field exists for: both runs recorded a digest and the digests differ, so the
    judge read a differently-worded message in each."""
    base = build_manifest(run_kind="judge", judge=judge_ref(tmp_path))
    other = replace(base, judge_pair_template_sha256=sha256_text("template v2"))

    with pytest.raises(NotComparableError) as excinfo:
        assert_comparable(base, other)

    assert "judge_pair_template_sha256" in str(excinfo.value)
    # Both sides have a digest, so this is a real difference rather than an age gap.
    assert "predates" not in str(excinfo.value)


def test_two_manifests_that_both_predate_the_field_stay_comparable(tmp_path):
    """Unknown on both sides is not a difference. Two old runs remain comparable on everything
    they did record, which is the point of loading them at all."""
    base = build_manifest(run_kind="judge", judge=judge_ref(tmp_path))
    old_a = replace(base, run_id="run-a", judge_pair_template_sha256=None)
    old_b = replace(base, run_id="run-b", judge_pair_template_sha256=None)

    assert_comparable(old_a, old_b)


def test_judge_manifest_leaves_the_agent_only_conditions_empty(tmp_path):
    """A judge reads no prompt, calls no tool, and retrieves nothing.

    Filling these with plausible defaults would put a fiction exactly where
    `assert_comparable` trusts a fact.
    """
    built = build_manifest(run_kind="judge", judge=judge_ref(tmp_path))

    for name in (
        "top_k",
        "chunk_size",
        "retrieval_config_sha256",
        "system_prompt_sha256",
        "kb_sha256",
        "max_tool_calls",
        "max_tool_errors",
        "max_model_calls",
    ):
        assert getattr(built, name) is None, name
    for name in EVAL_ONLY_FIELDS:
        assert getattr(built, name) is None, name
    # The price table still prices the judge's own calls, so it remains a condition.
    assert built.pricing_version == pricing_sha256()


def test_chat_and_eval_manifests_leave_every_judge_field_empty(tmp_path):
    for built in (
        build_manifest(config(tmp_path), run_kind="chat"),
        build_manifest(config(tmp_path), run_kind="eval", dataset=dataset(tmp_path)),
    ):
        for name in JUDGE_ONLY_FIELDS:
            assert getattr(built, name) is None, name


def test_judge_run_without_a_judge_ref_raises():
    """Its judge model and rubric digest are its conditions; without them it is untraceable."""
    with pytest.raises(ValueError, match="needs a JudgeRef"):
        build_manifest(run_kind="judge")


def test_judge_run_with_an_agent_config_raises(tmp_path):
    """An AgentConfig here would describe a prompt, an inventory, and a corpus nothing read."""
    with pytest.raises(ValueError, match="no agent"):
        build_manifest(config(tmp_path), run_kind="judge", judge=judge_ref(tmp_path))


def test_judge_run_with_a_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="scores a pairs file"):
        build_manifest(run_kind="judge", judge=judge_ref(tmp_path), dataset=dataset(tmp_path))


@pytest.mark.parametrize("run_kind", ["chat", "eval"])
def test_agent_run_with_a_judge_ref_raises(tmp_path, run_kind):
    """Accepting one and then dropping it would lose the conditions it named."""
    with pytest.raises(ValueError, match="no judge"):
        build_manifest(
            config(tmp_path),
            run_kind=run_kind,
            dataset=dataset(tmp_path) if run_kind == "eval" else None,
            judge=judge_ref(tmp_path),
        )


def test_agent_run_without_a_config_raises():
    with pytest.raises(ValueError, match="needs an AgentConfig"):
        build_manifest(run_kind="eval", dataset=None)


def test_judge_and_eval_manifests_are_never_comparable(tmp_path):
    """Scoring a run and running it are not two arms of one experiment."""
    with pytest.raises(NotComparableError, match="different kinds"):
        assert_comparable(
            build_manifest(run_kind="judge", judge=judge_ref(tmp_path)),
            build_manifest(config(tmp_path), run_kind="eval", dataset=dataset(tmp_path)),
        )


def test_two_judge_runs_differing_in_rubric_or_pairs_are_not_comparable(tmp_path):
    """Scores from two rubrics are not one measurement, and neither are scores over two files."""
    base = build_manifest(run_kind="judge", judge=judge_ref(tmp_path))

    with pytest.raises(NotComparableError, match="judge_rubric_sha256"):
        assert_comparable(base, replace(base, judge_rubric_sha256=sha256_text("rubric v2")))
    with pytest.raises(NotComparableError, match="pairs_sha256"):
        assert_comparable(base, replace(base, pairs_sha256=sha256_text("pairs v2")))


def test_a_different_judge_is_caught_even_though_model_name_is_exempt(tmp_path):
    """`COMPARABLE_EXEMPT` excuses `model_name`, which is why `judge_model` is recorded too."""
    base = build_manifest(run_kind="judge", judge=judge_ref(tmp_path))
    other = replace(base, model_name="other-judge", judge_model="other-judge")

    with pytest.raises(NotComparableError, match="judge_model"):
        assert_comparable(base, other)


def test_judge_pairs_path_is_informational_but_its_digest_is_not(tmp_path):
    """Two judge runs over the same bytes at different paths still compare."""
    base = build_manifest(run_kind="judge", judge=judge_ref(tmp_path))
    assert_comparable(base, replace(base, pairs_path="somewhere/else/pairs.jsonl"))


def test_judge_ref_digests_bytes_not_names(tmp_path):
    first = judge_ref(tmp_path).pairs_sha256
    edited = judge_ref(tmp_path, '{"prompt": "q", "response": "different"}\n').pairs_sha256
    assert first != edited


def test_run_ids_are_unique_per_manifest(tmp_path):
    a = build_manifest(config(tmp_path), run_kind="chat")
    b = build_manifest(config(tmp_path), run_kind="chat")
    assert a.run_id != b.run_id
    assert new_run_id() != new_run_id()


# --------------------------------------------------------------------------------------
# The agent-config group, which decides when a run has become a different run
# --------------------------------------------------------------------------------------


def test_agent_config_group_is_the_manifest_minus_identity_dataset_and_judge():
    assert set(agent_config_fields()) == {f.name for f in fields(RunManifest)} - (
        IDENTITY_FIELDS | EVAL_ONLY_FIELDS | JUDGE_ONLY_FIELDS
    )


def test_agent_config_digest_ignores_identity_and_dataset():
    """Two runs of the same configuration hash alike however they are labelled."""
    a = manifest(run_id="a", started_at="2026-01-01T00:00:00.000+00:00")
    b = manifest(run_id="b", started_at="2026-07-28T09:00:00.000+00:00", n_items=999)
    assert agent_config_digest(a) == agent_config_digest(b)


@pytest.mark.parametrize("field_name", agent_config_fields())
def test_agent_config_digest_changes_with_every_condition(field_name: str):
    """Parametrised over the group, so a field added later cannot go unwatched."""
    a = manifest()
    b = replace(a, **{field_name: other_value(field_name, getattr(a, field_name))})
    assert agent_config_digest(a) != agent_config_digest(b)
