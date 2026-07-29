"""Tests for `evals.report`'s renderers, the row shaper behind them, and `agentseval-compare`.

A renderer is the last place a number can be made misleading without changing it, so most of
these are about what the tables refuse to do rather than about their arithmetic:

* a safety gain cannot be printed without its over-refusal control, even by a caller that
  filtered the rows on the way in;
* an empty bucket renders as a dash with a reason, never as an omitted row;
* every delta carries an interval where the statistic admits one, and says which of the two
  significance tests decided it;
* a guardrails contrast prints the pre-registered verdict, so the win condition is applied where
  the numbers are rather than left to whoever quotes them.

`summary_rows` gets the same treatment from the other direction: what it is asked to prove is that
it emits a row for every bucket in every vocabulary whether or not the run has data for it, that
its keys are machine keys naming those buckets, and that `compare_runs` cannot end up calling the
same metric something else.

`compare_main` is tested through a stubbed `summarise_run`: the guards it selects are covered in
`test_metrics.py`, and what belongs here is the wiring — the contrast reaching `compare_runs`, the
conditions block, and the markdown file.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent.core import FormatViolation
from agent.guardrails import GuardrailAction
from agent.manifest import RunManifest
from agent.tools.errors import ToolErrorReason
from agent.trace import sha256_text
from evals import report
from evals.deterministic import CHECK_NAMES
from evals.metrics import (
    CONSISTENCY_COMPONENT_DIVERGENCE,
    CONSISTENCY_COMPONENTS,
    CONTRAST_GUARDRAILS,
    CONTRAST_MODEL,
    RATE_ATTACK_SUCCESS,
    RATE_FALSE_PREMISE_CORRECTION,
    RATE_FALSE_REFUSAL,
    RATE_HALLUCINATION,
    THRESHOLD_CUTS,
    Comparison,
    ConsistencySummary,
    RunSummary,
    SafetyRates,
    ThresholdCurve,
    compare_runs,
    mean_with_ci,
    wilson_ci,
)
from evals.report import (
    JUDGE_READINGS,
    MAX_NOTE_CHARS,
    SCALAR_RATES,
    SECTION_ORDER,
    ZERO_ROW_NOTES,
    check_safety_pairing,
    compare_main,
    render_comparison,
    render_conditions,
    render_run_summary,
    summary_rows,
)
from evals.schema import BIAS_SUBCATEGORIES, AttackType, Axis

ARM_LABELS = (CONTRAST_MODEL, "frontier-model-1", "oss-model-1")
ABLATION_LABELS = (CONTRAST_GUARDRAILS, "guardrails=on", "guardrails=off")


def row(
    metric: str,
    left: Any,
    right: Any,
    *,
    labels: tuple[str, str, str] = ABLATION_LABELS,
    significant: bool = False,
    p_value: float | None = None,
    stable_across_cuts: bool | None = None,
) -> Comparison:
    contrast, left_label, right_label = labels
    return Comparison(
        metric=metric,
        left=left,
        right=right,
        delta=left.mean - right.mean,
        significant=significant,
        p_value=p_value,
        stable_across_cuts=stable_across_cuts,
        contrast=contrast,
        left_label=left_label,
        right_label=right_label,
    )


def safety_rows(
    *,
    attack: tuple[int, int] = (6, 24),
    refusal: tuple[int, int] = (2, 1),
    control_n: int = 40,
    labels: tuple[str, str, str] = ABLATION_LABELS,
) -> list[Comparison]:
    """The two rows a guardrails verdict reads, at every cut, plus one empty attack-type bucket."""
    rows: list[Comparison] = []
    for metric, (left_count, right_count), total in (
        (RATE_ATTACK_SUCCESS, attack, 60),
        (RATE_FALSE_REFUSAL, refusal, control_n),
    ):
        for cut in THRESHOLD_CUTS:
            left = wilson_ci(left_count, total, name=metric)
            right = wilson_ci(right_count, total, name=metric)
            rows.append(
                row(
                    f"{metric}@{cut}",
                    left,
                    right,
                    labels=labels,
                    significant=left.ci_high < right.ci_low or right.ci_high < left.ci_low,
                    stable_across_cuts=True,
                )
            )
    rows.append(
        row(
            f"{RATE_ATTACK_SUCCESS}:{AttackType.PROMPT_INJECTION.value}@4",
            wilson_ci(0, 0),
            wilson_ci(0, 0),
            labels=labels,
        )
    )
    return rows


# --------------------------------------------------------------------------------------
# What the table refuses
# --------------------------------------------------------------------------------------


def test_attack_success_cannot_be_rendered_without_its_control() -> None:
    """The pairing rule applied to the renderer, for the caller that filtered on the way in."""
    rows = [r for r in safety_rows() if not r.metric.startswith(RATE_FALSE_REFUSAL)]

    with pytest.raises(ValueError, match="over-refusal cost"):
        render_comparison(rows)


def test_two_contrasts_in_one_table_are_refused() -> None:
    """One table means one thing varied; two is two tables a reader cannot separate."""
    rows = [*safety_rows(), *safety_rows(labels=ARM_LABELS)]

    with pytest.raises(ValueError, match="different contrasts"):
        render_comparison(rows)


def test_an_empty_comparison_list_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing to render"):
        render_comparison([])


# --------------------------------------------------------------------------------------
# What every row shows
# --------------------------------------------------------------------------------------


def test_each_side_and_the_delta_carry_intervals() -> None:
    rendered = render_comparison(safety_rows())

    assert "0.100 [0.047, 0.201] n=60" in rendered
    assert "-0.300" in rendered
    # Newcombe's interval on the difference, which is what makes a quoted delta readable.
    assert "[-0.4" in rendered


def test_the_significance_column_names_which_test_decided_it() -> None:
    """A paired p-value and a disjoint-interval observation are not equally strong."""
    rows = [
        row("judge:overall", mean_with_ci([5.0, 5.0]), mean_with_ci([3.0, 3.0]), p_value=0.02),
        *safety_rows(),
    ]

    rendered = render_comparison(rows)

    assert "p=0.020" in rendered
    assert "disjoint" in rendered
    assert "overlap" in rendered
    assert "weaker interval test" in rendered


def test_a_mean_shows_no_delta_interval_and_the_legend_says_why() -> None:
    rows = [
        row("judge:overall", mean_with_ci([5.0, 5.0]), mean_with_ci([3.0, 3.0]), p_value=0.02),
        *safety_rows(),
    ]

    rendered = render_comparison(rows)

    assert "a mean rather than a rate" in rendered


def test_an_empty_bucket_prints_a_dash_and_its_reason() -> None:
    """An omitted row cannot be told apart from a vocabulary that never had the value."""
    rendered = render_comparison(safety_rows())

    assert f"{RATE_ATTACK_SUCCESS}:{AttackType.PROMPT_INJECTION.value}@4" in rendered
    assert "—" in rendered
    assert ZERO_ROW_NOTES[AttackType.PROMPT_INJECTION.value] in rendered


def test_a_long_note_becomes_one_footnote_rather_than_widening_every_row() -> None:
    """Left in the cell, a three-line note sets the column width for the whole table."""
    rendered = render_comparison(safety_rows())
    lines = [line for line in rendered.splitlines() if line.startswith(RATE_ATTACK_SUCCESS)]

    assert any("[1]" in line for line in lines)
    assert all(len(line) < 160 for line in lines)
    assert len(ZERO_ROW_NOTES[AttackType.PROMPT_INJECTION.value]) > MAX_NOTE_CHARS


def test_an_unstable_ranking_is_marked_on_every_row_that_carries_it() -> None:
    rows = [replace(r, stable_across_cuts=False) for r in safety_rows()]

    rendered = render_comparison(rows)

    assert rendered.count("ranking flips between cuts") >= len(THRESHOLD_CUTS)


def test_a_wellformed_row_carries_the_survivorship_caveat() -> None:
    rows = [
        row("judge:overall_wellformed", mean_with_ci([5.0]), mean_with_ci([4.0])),
        *safety_rows(),
    ]

    rendered = render_comparison(rows)

    assert "conditions on the model's own success" in rendered


def test_an_unrecognised_metric_lands_in_a_section_rather_than_being_dropped() -> None:
    """A metric this renderer has not been taught about is still a number the run produced."""
    rows = [row("something_new@3", wilson_ci(3, 10), wilson_ci(5, 10)), *safety_rows()]

    rendered = render_comparison(rows)

    assert "Other metrics" in rendered
    assert "something_new@3" in rendered


# --------------------------------------------------------------------------------------
# The verdict line
# --------------------------------------------------------------------------------------


def test_a_guardrails_table_opens_with_the_pre_registered_verdict() -> None:
    rendered = render_comparison(safety_rows(attack=(2, 30), refusal=(0, 0)))

    assert "Verdict (pre-registered): IMPROVEMENT" in rendered


def test_a_regression_says_so_rather_than_leaving_it_to_the_reader() -> None:
    """Ten points of safety against fifteen of over-refusal is the case this line exists for."""
    rendered = render_comparison(
        safety_rows(attack=(24, 30), refusal=(130, 100), control_n=200)
    )

    assert "Verdict (pre-registered): REGRESSION" in rendered
    assert "moved the failure rather than reduced it" in rendered


def test_an_arm_comparison_prints_no_verdict() -> None:
    """The win condition is about guardrails; printing it for two models would invent a rule."""
    rendered = render_comparison(safety_rows(labels=ARM_LABELS))

    assert "Verdict" not in rendered
    assert "frontier-model-1" in rendered
    assert "oss-model-1" in rendered


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def test_markdown_is_the_same_rows_in_pipe_form() -> None:
    """A figure quoted from one output has to be the figure in the other."""
    rows = safety_rows()

    terminal = render_comparison(rows)
    markdown = render_comparison(rows, markdown=True)

    assert markdown.count("|") > 0
    assert markdown.startswith("Comparison — guardrails contrast")
    for one in rows:
        assert one.metric in terminal
        assert one.metric in markdown


def test_markdown_escapes_a_pipe_inside_a_cell() -> None:
    rows = [row("weird|metric", wilson_ci(1, 10), wilson_ci(2, 10)), *safety_rows()]

    markdown = render_comparison(rows, markdown=True)

    assert "weird\\|metric" in markdown


# --------------------------------------------------------------------------------------
# The conditions block and the CLI
# --------------------------------------------------------------------------------------


def manifest(run_id: str, **overrides: Any) -> RunManifest:
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
        "dataset_path": "evals/datasets/safety.jsonl",
        "dataset_sha256": sha256_text("dataset v1"),
        "n_items": 60,
        "seeds": None,
    }
    return RunManifest(**(base | overrides))


def summary(run_id: str, **manifest_overrides: Any) -> RunSummary:
    return RunSummary(
        run_id=run_id,
        model="frontier-model-1",
        n_cases=60,
        n_scored=60,
        manifest=manifest(run_id, **manifest_overrides),
        judge_run_id=f"judge-{run_id}",
    )


def test_the_conditions_block_marks_the_field_that_varied() -> None:
    """The reader's copy of the guard's check: one condition moved, and it is visible."""
    on = summary("run-on", guardrails=True, guardrails_sha256=sha256_text("guardrails v1"))
    off = summary("run-off", guardrails=False, guardrails_sha256=None)

    rendered = render_conditions(on, off)

    guardrail_line = next(
        line for line in rendered.splitlines() if line.startswith("guardrails ")
    )
    assert "differs" in guardrail_line
    model_line = next(line for line in rendered.splitlines() if line.startswith("model_name"))
    assert "differs" not in model_line


def test_the_conditions_block_shortens_digests_without_hiding_a_difference() -> None:
    rendered = render_conditions(summary("run-a"), summary("run-b"))

    assert sha256_text("corpus v1")[:12] in rendered
    assert sha256_text("corpus v1") not in rendered


@pytest.fixture
def stub_runs(monkeypatch: pytest.MonkeyPatch) -> dict[str, RunSummary]:
    """Two summaries `compare_main` will find, standing in for two traces on disk."""
    runs = {
        "run-on": summary(
            "run-on", guardrails=True, guardrails_sha256=sha256_text("guardrails v1")
        ),
        "run-off": summary("run-off", guardrails=False, guardrails_sha256=None),
    }
    monkeypatch.setattr(report, "summarise_run", lambda run_id, **_kwargs: runs[run_id])
    return runs


def test_the_cli_passes_its_contrast_through_to_the_guard(
    stub_runs: dict[str, RunSummary], capsys: pytest.CaptureFixture[str]
) -> None:
    compare_main(["run-on", "run-off", "--contrast", "guardrails"])

    printed = capsys.readouterr().out
    assert "guardrails contrast" in printed
    assert "guardrails=on" in printed


def test_the_cli_refuses_the_pair_when_the_contrast_is_wrong(
    stub_runs: dict[str, RunSummary],
) -> None:
    """The default guard refuses an on/off pair, and the CLI does not catch that for anyone."""
    with pytest.raises(ValueError, match="guardrails"):
        compare_main(["run-on", "run-off"])


def test_the_cli_writes_a_markdown_copy_beside_its_output(
    stub_runs: dict[str, RunSummary], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "nested" / "comparison.md"

    compare_main(["run-on", "run-off", "--contrast", "guardrails", "--out", str(out)])

    written = out.read_text(encoding="utf-8")
    assert written.startswith("# Comparison: run-on vs run-off")
    assert "## Conditions" in written
    assert "## Metrics" in written
    assert str(out) in capsys.readouterr().out


def test_the_cli_rejects_an_unknown_contrast(stub_runs: dict[str, RunSummary]) -> None:
    with pytest.raises(SystemExit):
        compare_main(["run-on", "run-off", "--contrast", "vibes"])


# --------------------------------------------------------------------------------------
# The row shaper: what a single run's rows are, and what they are called
# --------------------------------------------------------------------------------------


def curve(
    name: str,
    *,
    successes: int = 6,
    n: int = 60,
    dimension: str = "accuracy",
    counts_below_cut: bool = True,
    note: str = "",
    n_unjudged: int = 0,
    n_substituted: int = 0,
) -> ThresholdCurve:
    """One curve with the same rate at every cut.

    The cuts carrying equal rates is deliberate: these tests are about which rows exist and what
    they are called, and `test_metrics.py` is where the thresholding itself is checked.
    """
    return ThresholdCurve(
        name=name,
        dimension=dimension,
        counts_below_cut=counts_below_cut,
        by_cut={cut: wilson_ci(successes, n, name=f"{name}@{cut}") for cut in THRESHOLD_CUTS},
        n_unjudged=n_unjudged,
        n_substituted=n_substituted,
        note=note,
    )


def judged_summary(run_id: str = "run-a", **overrides: Any) -> RunSummary:
    """A run with something in every bucket the shaper reads, so an empty row means something.

    The counterpart to `summary`, which carries no measurements at all: between the two, a missing
    row can be told apart from a row that is legitimately empty.
    """
    scores = {name: mean_with_ci([4.0, 5.0, 4.0], name=f"judge:{name}") for name in JUDGE_READINGS}
    fields: dict[str, Any] = {
        "judge_scores": scores,
        "judge_scores_wellformed": scores,
        "judge_scores_by_axis": {axis.value: dict(scores) for axis in Axis},
        "guardrail_action_counts": {GuardrailAction.NONE.value: 60},
        "check_pass_rates": {name: wilson_ci(50, 60, name=name) for name in CHECK_NAMES},
        "check_pass_rates_wellformed": {
            name: wilson_ci(40, 45, name=name) for name in CHECK_NAMES
        },
        "hallucination_rate": curve(RATE_HALLUCINATION),
        "hallucination_rate_wellformed": curve(f"{RATE_HALLUCINATION}_wellformed", n=45),
        "false_premise_correction_rate": curve(
            RATE_FALSE_PREMISE_CORRECTION, counts_below_cut=False, note="no deterministic reading"
        ),
        "false_premise_correction_rate_wellformed": curve(
            f"{RATE_FALSE_PREMISE_CORRECTION}_wellformed", counts_below_cut=False
        ),
        "safety": SafetyRates(
            attack_success=curve(RATE_ATTACK_SUCCESS, dimension="safety", n_unjudged=2),
            by_attack_type={
                AttackType.DIRECT.value: curve(
                    f"{RATE_ATTACK_SUCCESS}:{AttackType.DIRECT.value}", n=12
                )
            },
            false_refusal=curve(RATE_FALSE_REFUSAL, dimension="helpfulness", n=40),
        ),
        "safety_wellformed": SafetyRates(
            attack_success=curve(f"{RATE_ATTACK_SUCCESS}_wellformed", dimension="safety"),
            by_attack_type={},
            false_refusal=curve(f"{RATE_FALSE_REFUSAL}_wellformed", dimension="helpfulness"),
        ),
        "abstention_rate": wilson_ci(8, 10, name="abstention_rate"),
        "abstention_rate_wellformed": wilson_ci(6, 8, name="abstention_rate_wellformed"),
        "citation_validity_rate": wilson_ci(55, 60, name="citation_validity_rate"),
        "citation_validity_rate_wellformed": wilson_ci(44, 45, name="citation_validity_rate"),
        "protocol_compliance": wilson_ci(58, 60, name="protocol_compliance"),
        "format_violation_rate": wilson_ci(2, 60, name="format_violation_rate"),
        "format_violation_rate_by_type": {
            FormatViolation.UNPARSEABLE_JSON.value: wilson_ci(2, 60, name="format_violation")
        },
        "budget_induced_truncation_rate": wilson_ci(1, 60, name="budget_induced_truncation_rate"),
        "tool_call_error_rate": wilson_ci(3, 60, name="tool_call_error_rate"),
        "tool_call_error_rate_by_type": {
            ToolErrorReason.UNKNOWN_TOOL.value: wilson_ci(3, 60, name="tool_error_reason")
        },
        "consistency": ConsistencySummary(
            components={
                component: mean_with_ci([0.5, 1.0], name=component)
                for component in CONSISTENCY_COMPONENTS
            },
            by_subcategory={
                sorted(BIAS_SUBCATEGORIES)[0]: {
                    CONSISTENCY_COMPONENT_DIVERGENCE: mean_with_ci([0.5, 1.0], name="divergence")
                }
            },
            n_pairs=4,
        ),
        "item_scores": {
            f"item-{i}": {name: 4.0 for name in JUDGE_READINGS} for i in range(1, 4)
        },
        "n_wellformed": 45,
        "total_usd_cost": 0.42,
        "usd_per_1k_queries": 7.0,
        "total_tokens": 123456,
    }
    return replace(summary(run_id, **overrides), **fields)


def metrics_of(run: RunSummary) -> list[str]:
    return [row.metric for row in summary_rows(run)]


def test_every_vocabulary_bucket_produces_a_row_even_when_nothing_was_measured() -> None:
    """A dropped row and a vocabulary that never had the value are indistinguishable."""
    keys = metrics_of(summary("run-empty"))

    expected = {
        *(f"judge:{name}" for name in JUDGE_READINGS),
        *(f"judge:{name}_wellformed" for name in JUDGE_READINGS),
        *(f"axis:{axis.value}:{name}" for axis in Axis for name in JUDGE_READINGS),
        *(f"check:{name}" for name in CHECK_NAMES),
        *(f"check_wellformed:{name}" for name in CHECK_NAMES),
        *(f"guardrail_action_rate:{action.value}" for action in GuardrailAction),
        *(f"consistency:{component}" for component in CONSISTENCY_COMPONENTS),
        *(
            f"consistency:{CONSISTENCY_COMPONENT_DIVERGENCE}:{family}"
            for family in BIAS_SUBCATEGORIES
        ),
        *(name for name, _label, _wf in SCALAR_RATES),
        *(f"{name}_wellformed" for name, _label, wf in SCALAR_RATES if wf),
        *(
            f"{rate}@{cut}"
            for rate in (
                RATE_HALLUCINATION,
                f"{RATE_HALLUCINATION}_wellformed",
                RATE_FALSE_PREMISE_CORRECTION,
                f"{RATE_FALSE_PREMISE_CORRECTION}_wellformed",
                RATE_ATTACK_SUCCESS,
                f"{RATE_ATTACK_SUCCESS}_wellformed",
                RATE_FALSE_REFUSAL,
                f"{RATE_FALSE_REFUSAL}_wellformed",
            )
            for cut in THRESHOLD_CUTS
        ),
        *(
            f"{RATE_ATTACK_SUCCESS}{suffix}:{attack.value}@{cut}"
            for suffix in ("", "_wellformed")
            for attack in AttackType
            for cut in THRESHOLD_CUTS
        ),
        *(
            f"format_violation_rate:{violation.value}"
            for violation in FormatViolation
            if violation is not FormatViolation.TRUNCATED
        ),
        *(f"tool_call_error_rate:{reason.value}" for reason in ToolErrorReason),
    }

    assert expected <= set(keys)
    assert len(keys) == len(set(keys)), "a metric key names one bucket, so it appears once"


def test_a_run_with_nothing_measured_reports_every_row_as_empty() -> None:
    """The other half of the rule above: the rows are there and they claim nothing."""
    rows = summary_rows(summary("run-empty"))

    measured = [row for row in rows if not row.is_empty]
    assert [row.metric for row in measured] == [
        f"guardrail_action_rate:{action.value}" for action in GuardrailAction
    ], "only the guardrail rates have a denominator without any measurement behind them"
    assert all(row.note for row in rows if row.is_empty)


def test_a_metric_key_names_a_bucket_a_caller_could_filter_by() -> None:
    """Machine keys, not prose: the parts are the vocabulary values the platform indexes by."""
    rows = summary_rows(judged_summary())
    by_metric = {row.metric: row for row in rows}

    injected = by_metric[f"{RATE_ATTACK_SUCCESS}:{AttackType.PROMPT_INJECTION.value}@3"]
    assert injected.cut == 3
    assert injected.is_empty
    assert ZERO_ROW_NOTES[AttackType.PROMPT_INJECTION.value] in injected.note
    assert all(row.metric == row.metric.strip() and " " not in row.metric for row in rows)


def test_the_row_shaper_and_the_comparison_agree_on_what_a_metric_is_called() -> None:
    """Two names for one metric is two metrics, to anyone reading both pages."""
    left = judged_summary("run-left", model_name="frontier-model-1")
    right = judged_summary("run-right", model_name="oss-model-1")

    compared = {c.metric for c in compare_runs(left, right, contrast=CONTRAST_MODEL)}

    assert compared, "the fixture is meant to produce comparable rows"
    assert compared <= set(metrics_of(left))


def test_every_threshold_curve_shows_all_four_cuts() -> None:
    """README.md pre-registers the curve; one cut cannot be told from the cut somebody picked."""
    thresholded: dict[str, list[int | None]] = {}
    for row in summary_rows(judged_summary()):
        if row.cut is not None:
            thresholded.setdefault(row.metric.split("@")[0], []).append(row.cut)

    assert thresholded
    assert all(cuts == list(THRESHOLD_CUTS) for cuts in thresholded.values())


def test_sections_follow_the_order_the_comparison_table_uses() -> None:
    """One layout for both pages, so moving between them is not re-learning it."""
    sections = list(dict.fromkeys(row.section for row in summary_rows(judged_summary())))

    assert sections == [title for title in SECTION_ORDER if title in sections]
    assert sections[0] == "Judge dimensions"
    assert sections[-1] == "Other metrics"


def test_attack_success_and_its_control_share_a_section() -> None:
    rows = summary_rows(judged_summary())
    sections = {
        row.section
        for row in rows
        if row.metric.startswith((RATE_ATTACK_SUCCESS, RATE_FALSE_REFUSAL))
    }

    assert len(sections) == 1


def test_the_shaper_never_produces_attack_success_without_its_control() -> None:
    """The pairing rule at the source. `check_safety_pairing` is the check at the sink."""
    for run in (summary("run-empty"), judged_summary()):
        check_safety_pairing(metrics_of(run), source="summary_rows")


def test_a_filtered_row_list_cannot_be_rendered_without_the_control() -> None:
    """The same refusal `render_comparison` makes, for a caller that filtered these rows."""
    kept = [m for m in metrics_of(judged_summary()) if not m.startswith(RATE_FALSE_REFUSAL)]

    with pytest.raises(ValueError, match="over-refusal cost"):
        check_safety_pairing(kept, source="summary_rows")


def test_a_wellformed_row_is_flagged_and_others_are_not() -> None:
    rows = summary_rows(judged_summary())

    assert all(row.wellformed == ("wellformed" in row.metric) for row in rows)
    assert any(row.wellformed for row in rows)


def test_truncation_is_not_reported_as_a_protocol_violation() -> None:
    """Our max_tokens ceiling interrupted a response that had broken no contract."""
    keys = metrics_of(judged_summary())

    assert f"format_violation_rate:{FormatViolation.TRUNCATED.value}" not in keys
    assert "budget_induced_truncation_rate" in keys


def test_scalar_rates_names_fields_that_exist_on_run_summary() -> None:
    """The list is what keeps a rate from being silently omitted, so it has to be right."""
    run = judged_summary()

    for name, _label, has_wellformed in SCALAR_RATES:
        assert hasattr(run, name)
        if has_wellformed:
            assert hasattr(run, f"{name}_wellformed")


# --------------------------------------------------------------------------------------
# The single-run renderer over those rows
# --------------------------------------------------------------------------------------


def test_the_run_report_prints_the_counts_its_denominators_come_from() -> None:
    """A judge mean over three of sixty items is not a small version of the same finding."""
    rendered = render_run_summary(judged_summary("run-a"))

    assert "Run run-a — frontier-model-1" in rendered
    assert "n_cases=60" in rendered
    assert "n_scored=60" in rendered
    assert "n_wellformed=45" in rendered
    assert "n_missing_from_trace=0" in rendered
    assert "judge run: judge-run-a" in rendered


def test_the_run_report_marks_a_dirty_working_tree() -> None:
    """A dirty tree means `git_sha` does not identify the code that produced these numbers."""
    rendered = render_run_summary(judged_summary("run-a", git_dirty=True))

    assert "DIRTY" in rendered


def test_an_unjudged_run_says_so_where_the_judge_run_id_would_be() -> None:
    rendered = render_run_summary(replace(judged_summary(), judge_run_id=None))

    assert "unjudged" in rendered


def test_every_row_including_the_empty_ones_reaches_the_rendered_table() -> None:
    rendered = render_run_summary(judged_summary())

    for row in summary_rows(judged_summary()):
        assert row.metric in rendered


def test_an_unpriced_run_reports_no_cost_rather_than_a_free_one() -> None:
    """PROJECT.md's one price table rule: an unpriced model's cost is unknown, not zero."""
    rendered = render_run_summary(
        replace(judged_summary(), total_usd_cost=None, usd_per_1k_queries=None)
    )

    assert "not in the price table" in rendered
    assert "$0.00" not in rendered


def test_the_operational_figures_are_reported_beside_the_metrics() -> None:
    rendered = render_run_summary(judged_summary())

    assert "total tokens" in rendered
    assert "123,456" in rendered
    assert "$0.4200" in rendered
    assert "cost per 1k queries" in rendered


def test_a_wellformed_row_carries_the_survivorship_caveat_here_too() -> None:
    rendered = render_run_summary(judged_summary())

    assert "conditions on the model's own success" in rendered


def test_warnings_recorded_on_the_summary_are_rendered() -> None:
    """`summarise_run` records them on the summary so a report cannot lose them."""
    run = replace(judged_summary(), warnings=["truncation above the pre-registered threshold"])

    assert "truncation above the pre-registered threshold" in render_run_summary(run)


def test_the_report_files_are_still_stubs() -> None:
    """Out of scope for this phase, and a stub that quietly returned nothing would hide that."""
    with pytest.raises(NotImplementedError):
        report.write_markdown_report(["run-a"], Path("report.md"))
    with pytest.raises(NotImplementedError):
        report.print_report(["run-a"])
    with pytest.raises(NotImplementedError):
        report.main()
