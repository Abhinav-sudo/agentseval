"""Covers `evals.validate_judge`, the module every judge-derived number is downstream of.

Weighted toward four things rather than toward line coverage.

**The statistics are checked against hand-computed values.** A bootstrap interval is hard to
verify by eye, so the point estimates are pinned to arithmetic a reader can redo, and the
weighted and unweighted kappas are computed on the same data to show that the choice of
weighting is load-bearing rather than decorative.

**Degeneracies return a reading, not a number.** A constant rater makes kappa and both
correlations undefined, and the tests assert `None` plus a recorded reason — `0.0` there would
read as "the judge agrees with nobody" and lead to the opposite decision.

**Provenance failures are refusals.** A label made against different text, a mixed label space,
two annotators on one response: each is asserted to raise rather than to warn, because a
validation number that looks like every other validation number while describing something else
is the failure this module exists to prevent.

**The artifact is byte-stable.** Two runs over identical data must produce identical JSON, or the
artifact cannot be diffed and "the judge changed" becomes indistinguishable from "the serialiser
did".

`FakeAdapter`, `tmp_path`, no network.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from rich.console import Console

import evals.validate_judge as validate_judge_module
from agent.manifest import RunManifest
from agent.models.judge_model import JudgeAdapter
from agent.prompts import (
    CANONICAL_BLOCK_ORDER,
    JUDGE_DIMENSIONS,
    JUDGE_PAIR_HEADINGS,
    JUDGE_SCALE_MAX,
    JUDGE_SCORE_BANDS,
)
from agent.trace import sha256_of_paths, sha256_text
from evals.deterministic import (
    CHECK_CITATION_GROUNDING,
    CHECK_KB_GROUNDED,
    CHECK_NAMES,
    CHECK_NO_REFUSAL,
    CaseChecks,
    CheckResult,
    rules_version,
)
from evals.judge import JudgeScore
from evals.label import labels_path
from evals.schema import Axis, HumanLabel, LabelRecord, LabelSpace
from evals.validate_judge import (
    AGREEMENT_GATE_KAPPA,
    AGREEMENT_GATE_MIN_N,
    ALL_AXES,
    BANDS_CITATION,
    BASELINE_COVERAGE,
    BLOCK_REORDERINGS,
    EXIT_FAILED,
    EXIT_OK,
    MAX_EXCERPT_CHARS,
    MIN_KAPPA_N,
    AgreementReport,
    BaselineInputs,
    LabelledDataError,
    LabelledPair,
    ValidationReport,
    _identity_weights,
    agreement_by_axis,
    agreement_from_scores,
    baseline_comparison,
    check_block_order_sensitivity,
    check_self_preference,
    check_stability,
    cohens_kappa,
    confusion_matrix,
    evaluate_gate,
    judge_band,
    judge_validation_path,
    load_baseline_inputs,
    load_binary_labels_from_run,
    load_labelled,
    load_labelled_from_run,
    main,
    measure_agreement,
    parse_column_map,
    print_report,
    rank_disagreements,
    score_labelled,
    select_stability_items,
    validate_judge,
)
from tests.fakes import EnvLoaded, FakeAdapter, refuse_env_load

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

HALL: dict[str, object] = {
    "id": "h-1",
    "axis": "hallucination",
    "subcategory": "answerable_kb",
    "turns": ["How much water during exercise?"],
    "expected_behavior": "Cites the hydration doc.",
    "answerable": True,
}


def verdict(overall: int, *, rationale: str = "A considered judgement.") -> str:
    """One well-formed judge completion scoring every dimension `overall`."""
    payload: dict[str, object] = {
        "rationale": rationale,
        "evidence": ["an answer"],
        **{dimension: overall for dimension in JUDGE_DIMENSIONS},
        "overall": overall,
    }
    return json.dumps(payload)


def adapter(
    *completions: str, use_cache: bool = False, cached: bool = False
) -> JudgeAdapter:
    """A `FakeAdapter` in the position the real judge adapter occupies.

    Cast rather than subclassed: `FakeAdapter` implements the slice of the adapter surface the
    judge path uses, and inheriting from `JudgeAdapter` would drag in the constructor's
    `OPENAI_API_KEY` requirement and defeat the purpose of a fake.
    """
    fake = FakeAdapter(
        completions=[*completions], use_cache=use_cache, cached=cached
    )
    return cast(JudgeAdapter, fake)


def scoring_adapter(judge_scores: Sequence[int | None]) -> JudgeAdapter:
    """An adapter scripted to produce one judgement per entry, in order.

    `None` scripts a judgement that never parses. It costs two completions rather than one,
    because `judge.score_pair` makes exactly one repair attempt — so the script stays aligned
    with the pairs.
    """
    completions: list[str] = []
    for score in judge_scores:
        if score is None:
            completions.extend(["not a verdict at all", "still not a verdict"])
        else:
            completions.append(verdict(score))
    return adapter(*completions)


def fake(*completions: str) -> FakeAdapter:
    """The `FakeAdapter` itself, for a test that inspects the calls it recorded."""
    return FakeAdapter(completions=[*completions])


def alternating_adapter(
    *, default: int, reordered: int | None, pairs_scored: int = 2
) -> FakeAdapter:
    """An adapter that answers one score under the canonical order and another under a reordering.

    `check_block_order_sensitivity` scores each pair twice in that order, so the script alternates.
    `None` for `reordered` scripts a judgement that never parses, which costs two completions
    because `judge.score_pair` makes exactly one repair attempt.
    """
    completions: list[str] = []
    for _ in range(pairs_scored):
        completions.append(verdict(default))
        if reordered is None:
            completions.extend(["not a verdict at all", "still not a verdict"])
        else:
            completions.append(verdict(reordered))
    return FakeAdapter(completions=completions)


def block_orders_of(recorded: FakeAdapter) -> list[list[str]]:
    """The block order each recorded judge call was made under, read off the message text.

    Read from what the judge was actually sent rather than from a recorded parameter: a
    `block_order` faithfully recorded on `JudgeScore` while the message went out in the canonical
    order would be the one failure this measurement cannot survive.
    """
    orders: list[list[str]] = []
    for index in range(recorded.count):
        text = recorded.prompt(index)
        positions = {
            key: text.index(heading)
            for key, heading in JUDGE_PAIR_HEADINGS.items()
            if heading in text
        }
        orders.append(sorted(positions, key=lambda key: positions[key]))
    return orders


def pairs(humans: Sequence[int], *, axis: str | None = None) -> list[LabelledPair]:
    """Labelled pairs with distinct ids and responses, in file order."""
    from evals.schema import Axis

    return [
        LabelledPair(
            pair_id=f"v-{index:03d}",
            prompt=f"question {index}",
            response=f"response {index}",
            human_score=float(human),
            annotator="alice",
            axis=Axis(axis) if axis else None,
        )
        for index, human in enumerate(humans)
    ]


def report_for(
    humans: Sequence[int], judged: Sequence[int | None], **kwargs: object
) -> AgreementReport:
    """Score `humans` against `judged` through the real scoring path."""
    labelled = pairs(humans)
    scores = score_labelled(labelled, scoring_adapter(judged))
    return agreement_from_scores(labelled, scores, **kwargs)  # type: ignore[arg-type]


def write_labelled(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def silent() -> Console:
    """A console that renders nowhere, so a test's output is its assertions."""
    return Console(file=io.StringIO(), width=100)


def rendered(report: ValidationReport) -> str:
    """The console output with whitespace collapsed.

    Collapsed because a `rich` table wraps a cell to fit the terminal, so asserting on the exact
    spacing would be asserting on the width these tests happen to pass rather than on what the
    operator is told.
    """
    stream = io.StringIO()
    print_report(report, Console(file=stream, width=120))
    return " ".join(stream.getvalue().split())


#: One judgement off by one at the top of the scale. Every hand-computed figure below is derived
#: from this, so the arithmetic in the comments can be checked in one place.
HUMANS = [1, 2, 3, 4, 5]
JUDGED = [1, 2, 3, 4, 4]
SCORED = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 4.0)]


# --------------------------------------------------------------------------------------
# The ordinal statistics, against hand-computed values
# --------------------------------------------------------------------------------------


def test_exact_within_one_and_mae_are_the_arithmetic() -> None:
    report = report_for(HUMANS, JUDGED)

    assert report.n == 5
    assert report.exact_agreement == pytest.approx(4 / 5)
    assert report.within_one == pytest.approx(1.0)
    assert report.mean_absolute_error == pytest.approx(1 / 5)
    assert report.mean_judge_score == pytest.approx(2.8)
    assert report.mean_human_score == pytest.approx(3.0)


def test_pearson_matches_the_hand_computation() -> None:
    # cov = 8.0, ss_human = 10, ss_judge = 6.8 -> 8 / sqrt(68)
    report = report_for(HUMANS, JUDGED)

    assert report.pearson_r == pytest.approx(8.0 / 68**0.5)


def test_spearman_uses_mid_ranks_for_ties() -> None:
    """The judge's two 4s share rank 4.5; breaking the tie by position would make rho depend
    on the order the file was written in."""
    # human ranks 1..5 against judge ranks [1, 2, 3, 4.5, 4.5]: 9.5 / sqrt(10 * 9.5)
    report = report_for(HUMANS, JUDGED)

    assert report.spearman_rho == pytest.approx(9.5 / 95**0.5)


def test_quadratic_kappa_matches_the_hand_computation() -> None:
    # One disagreement at (5, 4): weighted observed = 1/16. Weighted expected = 85 / 80.
    assert cohens_kappa(SCORED) == pytest.approx(1.0 - (1 / 16) / (85 / 80))


def test_weighted_and_unweighted_kappa_differ_on_the_same_data() -> None:
    """The weighting is load-bearing, not decoration.

    Unweighted kappa scores this single off-by-one disagreement as harshly as it would score a
    1-vs-5, which is why README.md pre-registers the quadratic form. `_identity_weights` exists
    only so this comparison can be made on real data; nothing reports it.
    """
    weighted = cohens_kappa(SCORED)
    unweighted = cohens_kappa(SCORED, weights=_identity_weights)

    assert weighted == pytest.approx(0.9411764705882353)
    assert unweighted == pytest.approx(0.75)
    assert weighted is not None and unweighted is not None
    assert weighted > unweighted


def test_a_far_disagreement_costs_more_than_a_near_one() -> None:
    near = cohens_kappa([*SCORED[:4], (5.0, 4.0)])
    far = cohens_kappa([*SCORED[:4], (5.0, 1.0)])

    assert near is not None and far is not None
    assert far < near


def test_the_confusion_matrix_is_five_by_five_with_every_cell() -> None:
    report = report_for(HUMANS, JUDGED)

    assert len(report.confusion) == JUDGE_SCALE_MAX
    assert all(len(row) == JUDGE_SCALE_MAX for row in report.confusion)
    # Rows are the human label, columns the judge's overall.
    assert report.confusion[0][0] == 1
    assert report.confusion[4][3] == 1
    assert sum(sum(row) for row in report.confusion) == 5


def test_an_empty_confusion_matrix_still_has_every_cell() -> None:
    """The set of cells is known before the data is, so a shape that varied would hide the fact
    that nobody labelled a 1."""
    matrix = confusion_matrix([])

    assert matrix == [[0] * JUDGE_SCALE_MAX for _ in range(JUDGE_SCALE_MAX)]


def test_direction_counts_separate_lenient_from_harsh() -> None:
    report = report_for([1, 2, 3], [2, 2, 2])

    assert report.n_judge_lenient == 1
    assert report.n_judge_harsh == 1
    assert report.n_tied == 1


def test_every_headline_figure_carries_an_interval() -> None:
    report = report_for(HUMANS, JUDGED)

    for name in (
        "exact_agreement",
        "within_one",
        "mean_absolute_error",
        "mean_judge_score",
        "mean_human_score",
        "pearson_r",
        "spearman_rho",
    ):
        assert name in report.cis, name
        assert report.cis[name].ci_low <= report.cis[name].mean <= report.cis[name].ci_high


# --------------------------------------------------------------------------------------
# Degeneracies: None with a reason, never 0.0
# --------------------------------------------------------------------------------------


def test_a_constant_judge_makes_kappa_and_both_correlations_undefined() -> None:
    """`0.0` here would read as "the judge agrees with nobody", which is the opposite of "this
    data cannot say"."""
    report = report_for([1, 2, 3, 4, 5], [3, 3, 3, 3, 3])

    assert report.cohens_kappa is None
    assert report.pearson_r is None
    assert report.spearman_rho is None
    assert report.kappa_unavailable_reason is not None
    assert "the judge scored every pair 3" in report.kappa_unavailable_reason


def test_a_constant_human_rater_is_reported_as_such() -> None:
    report = report_for([4, 4, 4, 4, 4], [1, 2, 3, 4, 5])

    assert report.cohens_kappa is None
    assert report.correlation_unavailable_reason is not None
    assert "every human label was 4" in report.correlation_unavailable_reason


def test_kappa_is_suppressed_below_the_minimum_sample() -> None:
    """A kappa over three items describes the sample, not the judge."""
    report = report_for([1, 3, 5], [1, 3, 5])

    assert report.n == 3
    assert report.cohens_kappa is None
    assert report.kappa_unavailable_reason is not None
    assert f"MIN_KAPPA_N={MIN_KAPPA_N}" in report.kappa_unavailable_reason


def test_kappa_is_reported_once_there_are_enough_pairs() -> None:
    humans = [1, 2, 3, 4, 5] * 2
    report = report_for(humans, humans)

    assert report.n == MIN_KAPPA_N
    assert report.cohens_kappa == pytest.approx(1.0)
    assert report.kappa_unavailable_reason is None


def test_perfect_agreement_on_a_two_point_spread_is_still_defined() -> None:
    pairs_seen = [(1.0, 1.0), (5.0, 5.0)]

    assert cohens_kappa(pairs_seen) == pytest.approx(1.0)


def test_an_empty_report_says_so_rather_than_reporting_zeros() -> None:
    report = agreement_from_scores([], {})

    assert report.n == 0
    assert report.cohens_kappa is None
    assert report.kappa_unavailable_reason == "no pairs were scored"
    assert report.confusion == confusion_matrix([])


# --------------------------------------------------------------------------------------
# Unparsed judgements are ours, not the annotator's
# --------------------------------------------------------------------------------------


def test_unparsed_judgements_leave_every_denominator() -> None:
    """Counting an unparsed verdict as a zero would make a rubric that confuses the judge look
    like a judge that disagrees with people."""
    report = report_for([1, 2, 3], [1, None, 3])

    assert report.n == 2
    assert report.n_unparsed == 1
    assert report.n_labelled == 3
    assert report.exact_agreement == pytest.approx(1.0)
    assert report.mean_absolute_error == pytest.approx(0.0)


def test_an_all_unparsed_run_reports_no_agreement_rather_than_perfect() -> None:
    report = report_for([1, 2], [None, None])

    assert report.n == 0
    assert report.n_unparsed == 2
    assert report.cohens_kappa is None


def test_unparsed_pairs_are_not_listed_as_disagreements() -> None:
    report = report_for([1, 5], [1, None])

    assert report.disagreements == []


# --------------------------------------------------------------------------------------
# Path (a): a self-contained labelled file
# --------------------------------------------------------------------------------------


def test_a_jsonl_file_loads(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [
            {"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 4},
            {"pair_id": "v-2", "prompt": "q2", "response": "a2", "human_score": 2},
        ],
    )

    labelled = load_labelled(path)

    assert [pair.pair_id for pair in labelled] == ["v-1", "v-2"]
    assert [pair.human_score for pair in labelled] == [4.0, 2.0]
    assert labelled[0].label_space is LabelSpace.RUBRIC_1_5


def test_csv_headers_are_trimmed_and_casefolded(tmp_path: Path) -> None:
    path = tmp_path / "labelled.csv"
    path.write_text(
        "Pair ID,Prompt , RESPONSE,Human_Score\nv-1,q,a,3\n",
        encoding="utf-8",
    )

    labelled = load_labelled(path)

    assert labelled[0].human_score == 3.0
    assert labelled[0].response == "a"


def test_aliases_resolve_a_graders_own_column_names(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "theirs.jsonl",
        [{"id": "t-1", "question": "q", "completion": "a", "rating": 5, "rater": "bob"}],
    )

    labelled = load_labelled(path)

    assert labelled[0].pair_id == "t-1"
    assert labelled[0].human_score == 5.0
    assert labelled[0].annotator == "bob"


def test_gold_is_the_human_label_in_a_validation_file(tmp_path: Path) -> None:
    """`gold` means the reference answer to `judge.load_pairs` and the label here.

    The collision is real and neither module guesses: this path resolves the label first and
    hides the column from the pair, so the judge never sees the answer it is scored against.
    """
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "gold": 2}],
    )

    labelled = load_labelled(path)

    assert labelled[0].human_score == 2.0
    assert "2" not in json.dumps(labelled[0].to_judge_pair().metadata)


def test_the_label_column_is_hidden_from_the_judge_pair(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 5, "notes": "n"}],
    )

    metadata = load_labelled(path)[0].to_judge_pair().metadata

    assert "human_score" not in metadata
    assert "notes" not in metadata


def test_a_missing_label_column_lists_found_needed_and_tried(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl", [{"pair_id": "v-1", "prompt": "q", "response": "a"}]
    )

    with pytest.raises(LabelledDataError) as excinfo:
        load_labelled(path)

    message = str(excinfo.value)
    assert "columns found" in message
    assert "columns needed" in message
    assert "aliases tried" in message
    assert "human_score" in message


def test_a_reference_only_file_is_told_about_the_gold_collision(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "reference": "the answer"}],
    )

    with pytest.raises(LabelledDataError, match="reference-answer column is present"):
        load_labelled(path)


def test_a_missing_response_column_names_the_response(tmp_path: Path) -> None:
    path = write_labelled(tmp_path / "labelled.jsonl", [{"prompt": "q", "human_score": 3}])

    with pytest.raises(LabelledDataError) as excinfo:
        load_labelled(path)

    assert "no response" in str(excinfo.value)


def test_a_column_map_renames_a_graders_column(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "theirs.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "verdict_1_5": 4}],
    )

    labelled = load_labelled(path, column_map=parse_column_map(["human_score=verdict_1_5"]))

    assert labelled[0].human_score == 4.0


def test_a_column_map_with_an_unknown_internal_name_is_rejected() -> None:
    with pytest.raises(LabelledDataError) as excinfo:
        parse_column_map(["rating=human_score"])

    message = str(excinfo.value)
    assert "not an internal column name" in message
    assert "human_score" in message
    # The most likely mistake is the direction, so the error offers the reverse.
    assert "Did you mean human_score=rating?" in message


def test_a_column_map_without_an_equals_sign_is_rejected() -> None:
    with pytest.raises(LabelledDataError, match="internal=external"):
        parse_column_map(["human_score"])


def test_a_binary_verdict_in_a_rubric_file_is_refused_by_name(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": "pass"}],
    )

    with pytest.raises(LabelledDataError) as excinfo:
        load_labelled(path)

    assert "binary verdict" in str(excinfo.value)
    assert "no pass/fail cut is pre-registered" in str(excinfo.value)


def test_a_score_off_the_scale_is_refused(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 7}],
    )

    with pytest.raises(LabelledDataError, match=f"1-{JUDGE_SCALE_MAX}"):
        load_labelled(path)


def test_a_half_step_score_is_refused(tmp_path: Path) -> None:
    """The judge's scale has no 3.5 for a 3.5 to be compared against."""
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 3.5}],
    )

    with pytest.raises(LabelledDataError, match="whole number"):
        load_labelled(path)


def test_the_declared_label_space_must_be_ordinal(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 3}],
    )

    with pytest.raises(LabelledDataError, match="agreement is ordinal"):
        load_labelled(path, label_space=LabelSpace.BINARY_BEHAVIORAL)


def test_a_row_contradicting_the_declared_space_is_refused(tmp_path: Path) -> None:
    """The space is declared, never inferred: a file of 1-5 values declaring binary is not a
    rubric file with a typo, it is two claims about what was labelled and one is wrong."""
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [
            {"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 3},
            {
                "pair_id": "v-2",
                "prompt": "q",
                "response": "a",
                "human_score": 4,
                "label_space": "binary_behavioral",
            },
        ],
    )

    with pytest.raises(LabelledDataError, match="One report covers one label space"):
        load_labelled(path)


def test_an_unknown_axis_is_refused(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 3, "axis": "vibes"}],
    )

    with pytest.raises(LabelledDataError, match="axis 'vibes' is not one of"):
        load_labelled(path)


def test_a_duplicate_pair_id_is_refused(tmp_path: Path) -> None:
    """Judgements are keyed by pair_id, so a duplicate would drop a pair silently."""
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [
            {"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 3},
            {"pair_id": "v-1", "prompt": "q2", "response": "a2", "human_score": 4},
        ],
    )

    with pytest.raises(LabelledDataError, match="appear more than once"):
        load_labelled(path)


def test_pairs_without_ids_get_positional_ones(tmp_path: Path) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [
            {"prompt": "q", "response": "a", "human_score": 3},
            {"prompt": "q2", "response": "a2", "human_score": 4},
        ],
    )

    labelled = load_labelled(path)

    assert len({pair.pair_id for pair in labelled}) == 2


# --------------------------------------------------------------------------------------
# Path (b): dataset + run + label sidecars
# --------------------------------------------------------------------------------------


def write_dataset(tmp_path: Path, *items: dict[str, object]) -> Path:
    path = tmp_path / "ds" / "core.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    return path


def write_trace(tmp_path: Path, run_id: str, responses: dict[str, str]) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{run_id}.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "run_id": run_id,
                    "item_id": item_id,
                    "turn_idx": index,
                    "role": "assistant",
                    "content": content,
                }
            )
            + "\n"
            for index, (item_id, content) in enumerate(responses.items())
        ),
        encoding="utf-8",
    )
    return path


def label_record(
    item_id: str,
    run_id: str,
    *,
    dataset_sha256: str,
    response: str,
    score: int | None = 4,
    annotator: str = "alice",
    label_space: LabelSpace = LabelSpace.RUBRIC_1_5,
    response_sha256: str | None = None,
    notes: str | None = None,
) -> dict[str, object]:
    record = LabelRecord(
        item_id=item_id,
        run_id=run_id,
        dataset_sha256=dataset_sha256,
        response_sha256=response_sha256 or sha256_text(response),
        label_space=label_space,
        score=score if label_space is LabelSpace.RUBRIC_1_5 else None,
        label=None if label_space is LabelSpace.RUBRIC_1_5 else HumanLabel.PASS,
        annotator=annotator,
        labelled_at="2026-07-29T00:00:00Z",
        seconds_spent=3.0,
        notes=notes,
    )
    return record.model_dump(mode="json")


def write_sidecar(
    tmp_path: Path,
    dataset: Path,
    run_id: str,
    annotator: str,
    records: list[dict[str, object]],
    *,
    label_space: LabelSpace = LabelSpace.RUBRIC_1_5,
) -> Path:
    """Write a sidecar through `label.labels_path`, so the test cannot hold a naming convention
    the code has moved away from.

    `label_space` names the *file*, which is not necessarily the space of the records inside it:
    the mixed-space and wrong-space refusals exist precisely because the record's declared space is
    the authority and a filename is not evidence.
    """
    directory = tmp_path / "ds" / "labels"
    directory.mkdir(parents=True, exist_ok=True)
    path = labels_path(dataset, run_id, annotator, directory, label_space=label_space)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def sidecar_fixture(
    tmp_path: Path,
    *,
    score: int = 4,
    response: str = "the agent's answer",
    labelled_response: str | None = None,
    dataset_sha256: str | None = None,
    label_space: LabelSpace = LabelSpace.RUBRIC_1_5,
) -> tuple[Path, str]:
    """A dataset, a trace, and one sidecar that agree with each other unless told otherwise."""
    dataset = write_dataset(tmp_path, HALL)
    write_trace(tmp_path, "run-1", {"h-1": response})
    digest = dataset_sha256 or (sha256_of_paths([dataset], root=dataset.parent) or "")
    write_sidecar(
        tmp_path,
        dataset,
        "run-1",
        "alice",
        [
            label_record(
                "h-1",
                "run-1",
                dataset_sha256=digest,
                response=labelled_response or response,
                score=score,
                label_space=label_space,
            )
        ],
    )
    return dataset, "run-1"


def test_the_sidecar_path_joins_dataset_trace_and_labels(tmp_path: Path) -> None:
    dataset, run_id = sidecar_fixture(tmp_path, score=5)

    labelled = load_labelled_from_run(dataset, run_id, runs_dir=tmp_path / "runs")

    assert len(labelled) == 1
    assert labelled[0].pair_id == "h-1"
    assert labelled[0].human_score == 5.0
    assert labelled[0].response == "the agent's answer"
    assert labelled[0].prompt == HALL["turns"][0]  # type: ignore[index]
    assert labelled[0].axis is not None and labelled[0].axis.value == "hallucination"
    assert labelled[0].response_sha256 == sha256_text("the agent's answer")


def test_the_last_record_for_an_item_wins(tmp_path: Path) -> None:
    """Sidecars are append-only: a correction is a newer record, not an edit."""
    dataset = write_dataset(tmp_path, HALL)
    write_trace(tmp_path, "run-1", {"h-1": "answer"})
    digest = sha256_of_paths([dataset], root=dataset.parent) or ""
    write_sidecar(
        tmp_path,
        dataset,
        "run-1",
        "alice",
        [
            label_record("h-1", "run-1", dataset_sha256=digest, response="answer", score=2),
            label_record("h-1", "run-1", dataset_sha256=digest, response="answer", score=5),
        ],
    )

    labelled = load_labelled_from_run(dataset, "run-1", runs_dir=tmp_path / "runs")

    assert [pair.human_score for pair in labelled] == [5.0]


def test_a_label_made_against_different_text_is_refused(tmp_path: Path) -> None:
    """A human label cannot be re-derived the way a judge score can, so this is a refusal."""
    dataset, run_id = sidecar_fixture(
        tmp_path, response="what the trace holds", labelled_response="what was labelled"
    )

    with pytest.raises(LabelledDataError) as excinfo:
        load_labelled_from_run(dataset, run_id, runs_dir=tmp_path / "runs")

    assert "made against a different response" in str(excinfo.value)


def test_a_label_made_against_a_different_dataset_is_refused(tmp_path: Path) -> None:
    dataset, run_id = sidecar_fixture(tmp_path, dataset_sha256="d" * 64)

    with pytest.raises(LabelledDataError, match="different dataset"):
        load_labelled_from_run(dataset, run_id, runs_dir=tmp_path / "runs")


def test_a_binary_sidecar_is_refused(tmp_path: Path) -> None:
    dataset, run_id = sidecar_fixture(tmp_path, label_space=LabelSpace.BINARY_BEHAVIORAL)

    with pytest.raises(LabelledDataError, match="agreement here is ordinal"):
        load_labelled_from_run(dataset, run_id, runs_dir=tmp_path / "runs")


# --------------------------------------------------------------------------------------
# The binary sidecar: the baseline leg's human side
# --------------------------------------------------------------------------------------


def binary_sidecar_fixture(
    tmp_path: Path,
    *,
    labels: dict[str, HumanLabel] | None = None,
    record_space: LabelSpace = LabelSpace.BINARY_BEHAVIORAL,
) -> tuple[Path, str]:
    """A dataset, a trace, and a natively-binary sidecar under the binary filename.

    `record_space` changes what is *inside* the file, never the filename: the wrong-space refusal
    is about the space each record declares, which is the authority a filename is not.
    """
    wanted = labels or {"h-1": HumanLabel.PASS}
    objects = [HALL if item_id == "h-1" else {**HALL, "id": item_id} for item_id in wanted]
    dataset = write_dataset(tmp_path, *objects)
    responses = {item_id: f"answer to {item_id}" for item_id in wanted}
    write_trace(tmp_path, "run-1", responses)
    digest = sha256_of_paths([dataset], root=dataset.parent) or ""

    records = []
    for item_id, label in wanted.items():
        record = LabelRecord(
            item_id=item_id,
            run_id="run-1",
            dataset_sha256=digest,
            response_sha256=sha256_text(responses[item_id]),
            label_space=record_space,
            score=None if record_space is LabelSpace.BINARY_BEHAVIORAL else 4,
            label=label if record_space is LabelSpace.BINARY_BEHAVIORAL else None,
            annotator="alice",
            labelled_at="2026-07-29T00:00:00Z",
            seconds_spent=3.0,
            notes=None,
        )
        records.append(record.model_dump(mode="json"))

    write_sidecar(
        tmp_path,
        dataset,
        "run-1",
        "alice",
        records,
        label_space=LabelSpace.BINARY_BEHAVIORAL,
    )
    return dataset, "run-1"


def test_binary_labels_load_keyed_by_item_id(tmp_path: Path) -> None:
    dataset, run_id = binary_sidecar_fixture(
        tmp_path, labels={"h-1": HumanLabel.PASS, "h-2": HumanLabel.FAIL}
    )

    labels = load_binary_labels_from_run(dataset, run_id, runs_dir=tmp_path / "runs")

    assert labels == {"h-1": HumanLabel.PASS, "h-2": HumanLabel.FAIL}


def test_the_binary_loader_refuses_rubric_labels(tmp_path: Path) -> None:
    """The mirror image of `_require_single_space`. Collapsing 1-5 human labels to pass/fail would
    need a threshold over the human side, which the pre-registered rules refuse outright."""
    dataset, run_id = binary_sidecar_fixture(tmp_path, record_space=LabelSpace.RUBRIC_1_5)

    with pytest.raises(LabelledDataError) as excinfo:
        load_binary_labels_from_run(dataset, run_id, runs_dir=tmp_path / "runs")

    assert "threshold over the human side" in str(excinfo.value)
    assert "--label-space binary_behavioral" in str(excinfo.value)


def test_the_binary_loader_reads_its_own_file_not_the_ordinal_one(tmp_path: Path) -> None:
    """Two files, same items. The space is in the filename, so neither loader can be handed the
    other's pass by a glob."""
    dataset, run_id = binary_sidecar_fixture(tmp_path)
    digest = sha256_of_paths([dataset], root=dataset.parent) or ""
    write_sidecar(
        tmp_path,
        dataset,
        run_id,
        "alice",
        [
            label_record(
                "h-1", run_id, dataset_sha256=digest, response="answer to h-1", score=5
            )
        ],
    )

    binary = load_binary_labels_from_run(dataset, run_id, runs_dir=tmp_path / "runs")
    ordinal = load_labelled_from_run(dataset, run_id, runs_dir=tmp_path / "runs")

    assert binary == {"h-1": HumanLabel.PASS}
    assert [pair.human_score for pair in ordinal] == [5.0]


def test_the_binary_loader_makes_every_provenance_refusal(tmp_path: Path) -> None:
    """A cheaper loader would hold the binary half of the comparison to a lower standard than the
    ordinal half."""
    dataset, run_id = binary_sidecar_fixture(tmp_path)
    write_trace(tmp_path, run_id, {"h-1": "a regenerated answer"})

    with pytest.raises(LabelledDataError, match="made against a different response"):
        load_binary_labels_from_run(dataset, run_id, runs_dir=tmp_path / "runs")


def test_the_binary_loader_says_which_file_it_looked_for(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path, HALL)
    write_trace(tmp_path, "run-1", {"h-1": "answer"})

    with pytest.raises(LabelledDataError) as excinfo:
        load_binary_labels_from_run(dataset, "run-1", runs_dir=tmp_path / "runs")

    assert "core.run-1.<annotator>.binary_behavioral.jsonl" in str(excinfo.value)


def test_a_mixed_binary_sidecar_is_refused_as_mixed(tmp_path: Path) -> None:
    """`_single_space` is shared, so the mixed-set refusal is the same one both loaders make.

    The rubric record is for a *second* item: appending one for `h-1` would supersede its binary
    label rather than sit beside it, which is the append-only semantics working."""
    dataset, run_id = binary_sidecar_fixture(
        tmp_path, labels={"h-1": HumanLabel.PASS, "h-2": HumanLabel.FAIL}
    )
    digest = sha256_of_paths([dataset], root=dataset.parent) or ""
    path = labels_path(
        dataset,
        run_id,
        "alice",
        tmp_path / "ds" / "labels",
        label_space=LabelSpace.BINARY_BEHAVIORAL,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                label_record(
                    "h-2", run_id, dataset_sha256=digest, response="answer to h-2", score=4
                )
            )
            + "\n"
        )

    with pytest.raises(LabelledDataError, match="span 2 label spaces"):
        load_binary_labels_from_run(dataset, run_id, runs_dir=tmp_path / "runs")


# --------------------------------------------------------------------------------------
# The baseline leg: the judge against rules that cost nothing
# --------------------------------------------------------------------------------------


def checks_for(passed: dict[str, bool], *, name: str = CHECK_NO_REFUSAL) -> dict[str, CaseChecks]:
    """Rule outcomes by item id, one named check per item."""
    return {
        item_id: CaseChecks(
            item_id=item_id, results=[CheckResult(name=name, passed=outcome, detail="")]
        )
        for item_id, outcome in passed.items()
    }


def baseline_inputs(
    labels: dict[str, HumanLabel],
    checks: dict[str, CaseChecks],
    **overrides: object,
) -> BaselineInputs:
    return BaselineInputs(
        binary_labels=labels,
        checks=checks,
        arm_model="frontier-model-1",
        arm_run_id="run-1",
        **overrides,  # type: ignore[arg-type]
    )


def judged(scores: dict[str, int]) -> dict[str, JudgeScore]:
    """Parsed judgements by item id, every dimension scored the same as `overall`."""
    return {
        item_id: JudgeScore(
            pair_id=item_id,
            scores=dict.fromkeys(JUDGE_DIMENSIONS, float(score)),
            overall=float(score),
            rationale="a judgement",
            raw_completion=verdict(score),
            judge_model="fake-model-1",
            parse_ok=True,
        )
        for item_id, score in scores.items()
    }


def test_the_bands_are_read_from_the_citation_rather_than_a_literal() -> None:
    assert judge_band(1) == "fail"
    assert judge_band(2) == "fail"
    assert judge_band(3) == "adequate"
    assert judge_band(4) == "pass"
    assert judge_band(JUDGE_SCALE_MAX) == "pass"
    assert judge_band(None) is None
    assert judge_band(JUDGE_SCALE_MAX + 1) is None
    assert BANDS_CITATION == "agent/prompts.py:JUDGE_SCORE_BANDS"
    assert dict(JUDGE_SCORE_BANDS) == {"fail": (1, 2), "adequate": (3, 3), "pass": (4, 5)}


def test_the_registered_coverage_is_pinned_to_the_table_in_the_readme() -> None:
    """Rows are added by registering them in README.md first. Comparing a rule against a human
    label without a registered row is an unregistered comparison."""
    assert [
        (row.key, row.checks, row.judge_dimension) for row in BASELINE_COVERAGE
    ] == [
        ("all_rules_pass", (), "overall"),
        ("reached_an_answer", (CHECK_NO_REFUSAL,), "helpfulness"),
        ("citations_resolve", (CHECK_CITATION_GROUNDING,), "accuracy"),
        ("quantitative_claims_supported", (CHECK_KB_GROUNDED,), "accuracy"),
    ]
    assert all(
        row.judge_dimension in {"overall", *JUDGE_DIMENSIONS} for row in BASELINE_COVERAGE
    )
    assert all(check in CHECK_NAMES for row in BASELINE_COVERAGE for check in row.checks)


def test_every_registered_row_appears_in_the_readme_table() -> None:
    """The table is the registration; the code reads it rather than inventing rows. If the two drift
    apart, the artifact cites a rule nobody registered."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    for row in BASELINE_COVERAGE:
        assert f"| `{row.key}` |" in readme
        assert f"| `{row.judge_dimension}` |" in readme


def test_each_instrument_is_scored_against_the_human_labels_in_its_own_space() -> None:
    """The rules answer natively; the judge is binarised by the cited bands. Nothing subtracts an
    ordinal figure from a binary one and nothing cuts the human side."""
    labels = {"h-1": HumanLabel.PASS, "h-2": HumanLabel.FAIL}
    inputs = baseline_inputs(labels, checks_for({"h-1": True, "h-2": False}))

    report = baseline_comparison(inputs, judged({"h-1": 5, "h-2": 1}))
    row = report["rows"]["reached_an_answer"]

    assert row["n"] == 2
    assert row["rules_agreement"]["mean"] == pytest.approx(1.0)
    assert row["judge_agreement"]["mean"] == pytest.approx(1.0)
    assert row["target"] == "human binary_behavioral label"


def test_a_row_the_judge_wins_records_a_strictly_better_verdict() -> None:
    labels = {"h-1": HumanLabel.PASS, "h-2": HumanLabel.FAIL}
    inputs = baseline_inputs(labels, checks_for({"h-1": True, "h-2": True}))

    row = baseline_comparison(inputs, judged({"h-1": 5, "h-2": 1}))["rows"]["reached_an_answer"]

    assert row["judge_agreement"]["mean"] == pytest.approx(1.0)
    assert row["rules_agreement"]["mean"] == pytest.approx(0.5)
    assert row["difference"] == pytest.approx(0.5)
    assert row["winner"] == "judge"
    assert row["judge_strictly_better"] is True


def test_equal_agreement_is_a_win_for_the_rules() -> None:
    """Rules cost nothing and never drift, so on a question where the judge merely matches them it
    adds no information — a finding worth publishing rather than a null result to bury."""
    labels = {"h-1": HumanLabel.PASS, "h-2": HumanLabel.FAIL}
    inputs = baseline_inputs(labels, checks_for({"h-1": True, "h-2": False}))

    row = baseline_comparison(inputs, judged({"h-1": 5, "h-2": 1}))["rows"]["reached_an_answer"]

    assert row["difference"] == pytest.approx(0.0)
    assert row["winner"] == "rules"
    assert row["judge_strictly_better"] is False


def test_the_difference_is_tested_paired_over_identical_items() -> None:
    labels = {f"h-{index}": HumanLabel.PASS for index in range(6)}
    inputs = baseline_inputs(labels, checks_for(dict.fromkeys(labels, False)))

    row = baseline_comparison(inputs, judged(dict.fromkeys(labels, 5)))["rows"][
        "reached_an_answer"
    ]

    assert row["n"] == 6
    assert row["p_value"] is not None
    # Never 0.0: a permutation p-value is bounded below by 1 / draws.
    assert 0.0 < row["p_value"] <= 1.0


def test_an_item_the_judge_scored_three_leaves_the_leg_and_is_counted() -> None:
    """`adequate` is its own band, not a tie broken in some direction."""
    labels = {"h-1": HumanLabel.PASS, "h-2": HumanLabel.PASS, "h-3": HumanLabel.FAIL}
    inputs = baseline_inputs(labels, checks_for(dict.fromkeys(labels, True)))

    row = baseline_comparison(inputs, judged({"h-1": 5, "h-2": 3, "h-3": 3}))["rows"][
        "reached_an_answer"
    ]

    assert row["n"] == 1
    assert row["n_excluded_adequate"] == 2
    assert row["excluded_adequate_items"] == ["h-2", "h-3"]
    assert row["adequate_rate"] == pytest.approx(2 / 3)


def test_the_three_rate_is_recorded_beside_the_arm_that_produced_it() -> None:
    """One invocation sees one arm, so the artifact carries the arm and its rate; comparing the two
    arms' rates is a precondition for reading the leg."""
    labels = {"h-1": HumanLabel.PASS, "h-2": HumanLabel.PASS}
    inputs = baseline_inputs(labels, checks_for(dict.fromkeys(labels, True)))

    report = baseline_comparison(inputs, judged({"h-1": 3, "h-2": 5}))

    assert report["arm"]["model"] == "frontier-model-1"
    assert report["arm"]["run_id"] == "run-1"
    assert "precondition for reading the binary leg" in report["arm"]["note"]
    assert report["rows"]["reached_an_answer"]["adequate_rate"] == pytest.approx(0.5)


def test_an_excluded_item_never_reaches_the_rules_either() -> None:
    """The exclusion removes an item from the paired comparison entirely, so the rules are never
    scored on an item the judge was dropped from."""
    labels = {"h-1": HumanLabel.FAIL, "h-2": HumanLabel.PASS}
    inputs = baseline_inputs(labels, checks_for({"h-1": True, "h-2": True}))

    row = baseline_comparison(inputs, judged({"h-1": 3, "h-2": 5}))["rows"]["reached_an_answer"]

    assert [entry["item_id"] for entry in row["per_item"]] == ["h-2"]
    assert row["rules_agreement"]["n"] == 1


def test_a_row_with_nothing_left_reports_none_with_a_reason_never_zero() -> None:
    """`0.0` there would read as a judge that agreed with nobody, which is a different finding."""
    labels = {"h-1": HumanLabel.PASS, "h-2": HumanLabel.FAIL}
    inputs = baseline_inputs(labels, checks_for(dict.fromkeys(labels, True)))

    row = baseline_comparison(inputs, judged({"h-1": 3, "h-2": 3}))["rows"]["reached_an_answer"]

    assert row["n"] == 0
    assert row["judge_agreement"] is None
    assert row["rules_agreement"] is None
    assert row["difference"] is None
    assert row["p_value"] is None
    assert row["winner"] is None
    assert "would read as a judge that agreed with nobody" in row["unavailable_reason"]


def test_exclusions_are_applied_before_degeneracy_is_evaluated() -> None:
    """Evaluating degeneracy first would let an excluded item make a comparison look possible."""
    labels = {"h-1": HumanLabel.PASS, "h-2": HumanLabel.PASS}
    inputs = baseline_inputs(labels, checks_for(dict.fromkeys(labels, True)))

    row = baseline_comparison(inputs, judged({"h-1": 3, "h-2": 4}))["rows"]["reached_an_answer"]

    assert row["n"] == 1
    assert row["n_excluded_adequate"] == 1
    assert row["judge_agreement"]["n"] == 1


def test_an_unparsed_judgement_leaves_the_leg_and_is_counted_separately() -> None:
    labels = {"h-1": HumanLabel.PASS, "h-2": HumanLabel.PASS}
    scores = judged({"h-1": 5, "h-2": 5})
    scores["h-2"] = JudgeScore(
        pair_id="h-2",
        scores={},
        overall=None,
        rationale="",
        raw_completion="not a verdict",
        judge_model="fake-model-1",
        parse_ok=False,
    )
    inputs = baseline_inputs(labels, checks_for(dict.fromkeys(labels, True)))

    row = baseline_comparison(inputs, scores)["rows"]["reached_an_answer"]

    assert row["n"] == 1
    assert row["n_unparsed"] == 1


def test_a_check_the_dataset_never_asked_for_is_not_counted_as_a_failure() -> None:
    """A skipped check counted as a failure would charge an item for a rule nobody applied."""
    labels = {"h-1": HumanLabel.PASS}
    inputs = baseline_inputs(labels, checks_for({"h-1": True}, name=CHECK_NO_REFUSAL))

    rows = baseline_comparison(inputs, judged({"h-1": 5}))["rows"]

    assert rows["reached_an_answer"]["n"] == 1
    assert rows["citations_resolve"]["n"] == 0
    assert rows["citations_resolve"]["n_rule_did_not_run"] == 1


def test_the_headline_row_conjoins_every_check_that_ran() -> None:
    labels = {"h-1": HumanLabel.FAIL}
    checks = {
        "h-1": CaseChecks(
            item_id="h-1",
            results=[
                CheckResult(name=CHECK_NO_REFUSAL, passed=True, detail=""),
                CheckResult(name=CHECK_KB_GROUNDED, passed=False, detail=""),
            ],
        )
    }

    row = baseline_comparison(baseline_inputs(labels, checks), judged({"h-1": 1}))["rows"][
        "all_rules_pass"
    ]

    assert row["n"] == 1
    assert row["per_item"][0]["rule_answer"] is False
    assert row["rules_agreement"]["mean"] == pytest.approx(1.0)


def test_the_rules_version_and_the_bands_citation_travel_with_every_result() -> None:
    """A later regex tweak must not be able to move a number already published."""
    labels = {"h-1": HumanLabel.PASS}
    inputs = baseline_inputs(labels, checks_for({"h-1": True}))

    report = baseline_comparison(inputs, judged({"h-1": 5}))

    assert report["rules_version"] == rules_version()
    assert report["bands"]["citation"] == BANDS_CITATION
    assert report["bands"]["excluded_band"] == "adequate"
    assert report["pre_registered_at"] == "README.md#pre-registered-scoring-rules"
    assert "nothing here binarises a human" in report["human_side"]


def test_the_artifact_lifts_the_rules_version_out_of_the_baseline_section(
    tmp_path: Path,
) -> None:
    labelled = pairs([4])
    labelled[0].item_id = "h-1"
    source = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "h-1", "prompt": "q", "response": "a", "human_score": 4}],
    )

    payload = validate_judge(
        labelled,
        source=source,
        judge=adapter(verdict(5)),
        runs_dir=tmp_path / "runs",
        baseline_inputs=baseline_inputs(
            {"h-1": HumanLabel.PASS}, checks_for({"h-1": True})
        ),
    ).to_dict()

    assert payload["rules_version"] == rules_version()
    assert payload["baseline"]["status"] == "ok"


def test_items_excluded_before_the_leg_are_recorded_with_their_reason() -> None:
    labels = {"h-1": HumanLabel.PASS}
    inputs = baseline_inputs(
        labels,
        checks_for({"h-1": True}),
        excluded={"h-9": "the item ended infrastructure_failed"},
    )

    report = baseline_comparison(inputs, judged({"h-1": 5}))

    assert report["excluded_before_the_leg"] == {
        "h-9": "the item ended infrastructure_failed"
    }


def write_full_trace(
    tmp_path: Path,
    run_id: str,
    items: dict[str, str],
    *,
    infrastructure_failed: frozenset[str] = frozenset(),
) -> Path:
    """A trace with the records the rules read: an `assistant` completion and a `turn` answer.

    `write_trace` above logs only the assistant record the labeller reads.
    `deterministic.item_views` reconstructs steps and the final answer from the whole turn, so the
    baseline leg needs both.
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for item_id, answer in items.items():
        lines.append(
            json.dumps(
                {
                    "run_id": run_id,
                    "item_id": item_id,
                    "turn_idx": 0,
                    "role": "assistant",
                    "content": answer,
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "run_id": run_id,
                    "item_id": item_id,
                    "turn_idx": 0,
                    "role": "turn",
                    "content": answer,
                    "error": None,
                    "infrastructure_failed": item_id in infrastructure_failed,
                }
            )
        )
    path = runs / f"{run_id}.jsonl"
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def write_run_manifest(tmp_path: Path, run_id: str, **overrides: object) -> RunManifest:
    """An eval-run manifest for `run_id`, which is where the rules read their budgets."""
    base: dict[str, object] = {
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
        "dataset_path": "evals/datasets/core.jsonl",
        "dataset_sha256": sha256_text("dataset v1"),
        "n_items": 1,
        "seeds": None,
    }
    base.update(overrides)
    manifest = RunManifest(**base)  # type: ignore[arg-type]
    manifest.write(tmp_path / "runs")
    return manifest


def baseline_fixture(
    tmp_path: Path,
    *,
    answer: str = "Drink water regularly during exercise.",
    label: HumanLabel = HumanLabel.PASS,
    infrastructure_failed: bool = False,
) -> tuple[Path, str]:
    """A dataset, a full trace, a manifest, and a native binary sidecar that agree."""
    dataset = write_dataset(tmp_path, HALL)
    write_full_trace(
        tmp_path,
        "run-1",
        {"h-1": answer},
        infrastructure_failed=frozenset({"h-1"}) if infrastructure_failed else frozenset(),
    )
    write_run_manifest(tmp_path, "run-1")
    digest = sha256_of_paths([dataset], root=dataset.parent) or ""
    write_sidecar(
        tmp_path,
        dataset,
        "run-1",
        "alice",
        [
            label_record(
                "h-1",
                "run-1",
                dataset_sha256=digest,
                response=answer,
                label_space=LabelSpace.BINARY_BEHAVIORAL,
            )
            | {"label": label.value}
        ],
        label_space=LabelSpace.BINARY_BEHAVIORAL,
    )
    return dataset, "run-1"


def test_the_leg_runs_the_rules_over_the_run_trace(tmp_path: Path) -> None:
    dataset, run_id = baseline_fixture(tmp_path)

    inputs = load_baseline_inputs(dataset, run_id, runs_dir=tmp_path / "runs")

    assert inputs.binary_labels == {"h-1": HumanLabel.PASS}
    assert inputs.arm_model == "frontier-model-1"
    assert inputs.arm_run_id == run_id
    assert inputs.checks["h-1"].by_name(CHECK_NO_REFUSAL) is not None
    assert inputs.sidecars and inputs.sidecars[0].endswith("binary_behavioral.jsonl")


def test_the_budget_checks_read_the_run_manifest_rather_than_a_guess(tmp_path: Path) -> None:
    """A guessed ceiling would report a budget failure the run never had."""
    dataset, run_id = baseline_fixture(tmp_path)

    with_budgets = load_baseline_inputs(dataset, run_id, runs_dir=tmp_path / "runs")

    write_run_manifest(tmp_path, run_id, max_model_calls=None, max_tool_errors=None)
    without = load_baseline_inputs(dataset, run_id, runs_dir=tmp_path / "runs")

    names = {result.name for result in with_budgets.checks["h-1"].results}
    fewer = {result.name for result in without.checks["h-1"].results}

    assert "model_call_budget" in names
    assert "tool_call_errors" in names
    assert names - fewer == {"model_call_budget", "tool_call_errors"}


def test_an_infrastructure_failed_item_leaves_before_the_leg_with_its_reason(
    tmp_path: Path,
) -> None:
    """The pre-registered rules exclude it from scoring everywhere: it is our failure, not the
    model's answer."""
    dataset, run_id = baseline_fixture(tmp_path, infrastructure_failed=True)

    inputs = load_baseline_inputs(dataset, run_id, runs_dir=tmp_path / "runs")

    assert "h-1" not in inputs.checks
    assert "infrastructure_failed" in inputs.excluded["h-1"]


def add_rubric_sidecar(tmp_path: Path, dataset: Path, run_id: str, *, answer: str) -> None:
    """The ordinal pass's sidecar, beside the binary one. Two files, same item — which is the
    arrangement the two label spaces need and the reason the space is in the filename."""
    digest = sha256_of_paths([dataset], root=dataset.parent) or ""
    write_sidecar(
        tmp_path,
        dataset,
        run_id,
        "alice",
        [label_record("h-1", run_id, dataset_sha256=digest, response=answer, score=5)],
    )


def test_the_cli_runs_the_baseline_end_to_end_on_the_sidecar_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = "Drink water regularly during exercise."
    dataset, run_id = baseline_fixture(tmp_path, answer=answer)
    add_rubric_sidecar(tmp_path, dataset, run_id, answer=answer)

    cli(
        monkeypatch,
        adapter(verdict(5)),
        "--dataset",
        str(dataset),
        "--run",
        run_id,
        "--labels-dir",
        str(tmp_path / "ds" / "labels"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--baseline",
    )

    artifacts = sorted((tmp_path / "runs").glob("*.judge_validation.json"))
    baseline = json.loads(artifacts[0].read_text())["baseline"]

    assert baseline["status"] == "ok"
    assert baseline["rules_version"] == rules_version()
    assert baseline["arm"]["model"] == "frontier-model-1"
    assert baseline["rows"]["reached_an_answer"]["n"] == 1


def test_the_cli_baseline_names_an_unimplemented_rule_rather_than_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule that is not implemented is not a rule instrument — a reason this leg cannot run, not a
    traceback in the middle of one."""
    answer = "Drink water regularly during exercise."
    dataset, run_id = baseline_fixture(tmp_path, answer=answer)
    add_rubric_sidecar(tmp_path, dataset, run_id, answer=answer)

    def unimplemented(*args: object, **kwargs: object) -> CaseChecks:
        raise NotImplementedError("check_kb_grounded needs a corpus index")

    monkeypatch.setattr(validate_judge_module, "run_all", unimplemented)

    code = cli(
        monkeypatch,
        adapter(verdict(5)),
        "--dataset",
        str(dataset),
        "--run",
        run_id,
        "--labels-dir",
        str(tmp_path / "ds" / "labels"),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--baseline",
    )

    assert code == EXIT_FAILED  # the gate fails on n, which is a separate matter
    artifacts = sorted((tmp_path / "runs").glob("*.judge_validation.json"))
    baseline = json.loads(artifacts[0].read_text())["baseline"]

    assert baseline["status"] == "unavailable"
    assert any("not implemented" in reason for reason in baseline["reasons"])
    assert any("check_kb_grounded" in reason for reason in baseline["reasons"])


def test_only_items_all_three_sides_cover_enter_a_denominator() -> None:
    """An item labelled but never judged cannot enter one row's denominator and not another's."""
    labels = {"h-1": HumanLabel.PASS, "h-2": HumanLabel.PASS}
    inputs = baseline_inputs(labels, checks_for({"h-1": True}))

    report = baseline_comparison(inputs, judged({"h-1": 5, "h-2": 5}))

    assert report["n_items_joined"] == 1
    assert report["n_binary_labels"] == 2


def test_mixed_label_spaces_are_refused(tmp_path: Path) -> None:
    """Kappa is undefined across mismatched category sets, and dropping half the labels would
    show an n smaller than the labelling effort with nothing saying why."""
    dataset = write_dataset(tmp_path, HALL, {**HALL, "id": "h-2"})
    write_trace(tmp_path, "run-1", {"h-1": "answer one", "h-2": "answer two"})
    digest = sha256_of_paths([dataset], root=dataset.parent) or ""
    write_sidecar(
        tmp_path,
        dataset,
        "run-1",
        "alice",
        [
            label_record("h-1", "run-1", dataset_sha256=digest, response="answer one", score=4),
            label_record(
                "h-2",
                "run-1",
                dataset_sha256=digest,
                response="answer two",
                label_space=LabelSpace.BINARY_BEHAVIORAL,
            ),
        ],
    )

    with pytest.raises(LabelledDataError, match="span 2 label spaces"):
        load_labelled_from_run(dataset, "run-1", runs_dir=tmp_path / "runs")


def test_two_annotators_on_one_item_are_refused_until_one_is_chosen(tmp_path: Path) -> None:
    """Two labels on one response is inter-annotator agreement, a different statistic."""
    dataset = write_dataset(tmp_path, HALL)
    write_trace(tmp_path, "run-1", {"h-1": "answer"})
    digest = sha256_of_paths([dataset], root=dataset.parent) or ""
    for annotator, score in (("alice", 4), ("bob", 2)):
        write_sidecar(
            tmp_path,
            dataset,
            "run-1",
            annotator,
            [
                label_record(
                    "h-1",
                    "run-1",
                    dataset_sha256=digest,
                    response="answer",
                    score=score,
                    annotator=annotator,
                )
            ],
        )

    with pytest.raises(LabelledDataError) as excinfo:
        load_labelled_from_run(dataset, "run-1", runs_dir=tmp_path / "runs")

    assert "more than one annotator" in str(excinfo.value)
    assert "--annotator" in str(excinfo.value)

    chosen = load_labelled_from_run(
        dataset, "run-1", runs_dir=tmp_path / "runs", annotator="bob"
    )
    assert [pair.human_score for pair in chosen] == [2.0]


def test_a_label_for_an_item_the_run_never_answered_is_refused(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path, HALL, {**HALL, "id": "h-2"})
    write_trace(tmp_path, "run-1", {"h-1": "answer"})
    digest = sha256_of_paths([dataset], root=dataset.parent) or ""
    write_sidecar(
        tmp_path,
        dataset,
        "run-1",
        "alice",
        [
            label_record("h-1", "run-1", dataset_sha256=digest, response="answer"),
            label_record("h-2", "run-1", dataset_sha256=digest, response="missing"),
        ],
    )

    with pytest.raises(LabelledDataError, match="no response for"):
        load_labelled_from_run(dataset, "run-1", runs_dir=tmp_path / "runs")


def test_no_sidecar_at_all_says_where_it_looked(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path, HALL)
    write_trace(tmp_path, "run-1", {"h-1": "answer"})

    with pytest.raises(LabelledDataError) as excinfo:
        load_labelled_from_run(dataset, "run-1", runs_dir=tmp_path / "runs")

    assert "core.run-1.<annotator>.rubric_1_5.jsonl" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# Disagreements: deterministically ordered, scrubbed, truncated
# --------------------------------------------------------------------------------------


def test_disagreements_are_ordered_by_magnitude_then_pair_id() -> None:
    """Magnitude first because the four-point gap is the one worth reading; pair_id second so a
    tie cannot reshuffle between two runs over identical data."""
    labelled = pairs([5, 1, 1, 3])
    scores = score_labelled(labelled, scoring_adapter([1, 4, 5, 3]))

    entries = rank_disagreements(labelled, scores)

    # Deltas: v-000 -4, v-001 +3, v-002 +4, v-003 0. The two 4s tie and sort by id.
    assert [entry["pair_id"] for entry in entries] == ["v-000", "v-002", "v-001"]


def test_disagreement_order_is_stable_across_calls() -> None:
    labelled = pairs([1, 5, 1, 5])
    scores = score_labelled(labelled, scoring_adapter([3, 3, 3, 3]))

    first = rank_disagreements(labelled, scores)
    second = rank_disagreements(labelled, scores)

    assert [entry["pair_id"] for entry in first] == [entry["pair_id"] for entry in second]


def test_disagreements_are_truncated_to_the_limit() -> None:
    labelled = pairs([1, 1, 1, 1])
    scores = score_labelled(labelled, scoring_adapter([5, 5, 5, 5]))

    entries = rank_disagreements(labelled, scores, limit=2)

    assert len(entries) == 2


def test_a_disagreement_carries_its_signed_direction() -> None:
    labelled = pairs([2, 4])
    scores = score_labelled(labelled, scoring_adapter([4, 2]))

    entries = rank_disagreements(labelled, scores)

    assert {entry["pair_id"]: entry["direction"] for entry in entries} == {
        "v-000": "judge_lenient",
        "v-001": "judge_harsh",
    }


def test_excerpts_are_scrubbed_of_model_names() -> None:
    labelled = [
        LabelledPair(
            pair_id="v-1",
            prompt="Which model are you?",
            response="I am Claude, made by Anthropic.",
            human_score=1.0,
        )
    ]
    scores = score_labelled(labelled, scoring_adapter([5]))

    entry = rank_disagreements(labelled, scores)[0]

    assert "Claude" not in entry["response_excerpt"]
    assert "Anthropic" not in entry["response_excerpt"]


def test_excerpts_are_truncated_and_say_so() -> None:
    labelled = [
        LabelledPair(
            pair_id="v-1",
            prompt="q",
            response="x" * (MAX_EXCERPT_CHARS + 500),
            human_score=1.0,
        )
    ]
    scores = score_labelled(labelled, scoring_adapter([5]))

    excerpt = rank_disagreements(labelled, scores)[0]["response_excerpt"]

    assert "truncated" in excerpt
    assert len(excerpt) < MAX_EXCERPT_CHARS + 100


def test_a_disagreement_cites_the_run_rather_than_copying_the_completion() -> None:
    labelled = pairs([1])
    scores = score_labelled(labelled, scoring_adapter([5]))

    entry = rank_disagreements(labelled, scores, judge_run_id="jr-1")[0]

    assert entry["judge_run_id"] == "jr-1"
    assert entry["pair_id"] == "v-000"


# --------------------------------------------------------------------------------------
# Per-axis reporting
# --------------------------------------------------------------------------------------


def test_every_axis_gets_a_row_even_when_nobody_labelled_it() -> None:
    """A dropped row is indistinguishable from a vocabulary that never had the value."""
    from evals.schema import Axis

    labelled = pairs([1, 2, 3], axis="safety")
    scores = score_labelled(labelled, scoring_adapter([1, 2, 3]))

    by_axis = agreement_by_axis(labelled, scores)

    assert set(by_axis) == {axis.value for axis in Axis}
    assert by_axis["safety"].n == 3
    assert by_axis["bias"].n == 0
    assert by_axis["bias"].kappa_unavailable_reason == "no pairs were scored"


def test_the_no_axis_bucket_appears_only_when_it_holds_something() -> None:
    labelled = pairs([1, 2])
    scores = score_labelled(labelled, scoring_adapter([1, 2]))

    with_none = agreement_by_axis(labelled, scores)
    axed = agreement_by_axis(*_scored(pairs([1, 2], axis="safety"), [1, 2]))

    assert "(no axis)" in with_none
    assert "(no axis)" not in axed


def _scored(
    labelled: list[LabelledPair], judged: Sequence[int | None]
) -> tuple[list[LabelledPair], dict[str, JudgeScore]]:
    return labelled, score_labelled(labelled, scoring_adapter(judged))


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------


def passing_set() -> tuple[list[LabelledPair], list[int]]:
    """Enough pairs, and enough agreement, to clear the pre-registered gate."""
    humans = [1, 2, 3, 4, 5] * 5
    judged = [1, 2, 3, 4, 5] * 4 + [1, 2, 3, 4, 4]
    return pairs(humans), judged


def test_the_gate_passes_on_a_well_agreeing_set() -> None:
    labelled, judged = passing_set()
    report = agreement_from_scores(labelled, score_labelled(labelled, scoring_adapter(judged)))

    gate = evaluate_gate(report)

    assert gate["passed"] is True
    assert gate["observed_n"] >= AGREEMENT_GATE_MIN_N
    assert gate["observed_kappa"] >= AGREEMENT_GATE_KAPPA


def test_the_gate_fails_when_there_are_too_few_pairs() -> None:
    report = report_for([1, 2, 3, 4, 5] * 2, [1, 2, 3, 4, 5] * 2)

    gate = evaluate_gate(report)

    assert gate["passed"] is False
    assert any("below the pre-registered minimum" in failure for failure in gate["failures"])


def test_the_gate_fails_when_agreement_is_poor() -> None:
    humans = [1, 2, 3, 4, 5] * 5
    report = report_for(humans, [5, 4, 3, 2, 1] * 5)

    gate = evaluate_gate(report)

    assert gate["passed"] is False
    assert gate["observed_kappa"] is not None and gate["observed_kappa"] < AGREEMENT_GATE_KAPPA
    assert any("below the pre-registered" in failure for failure in gate["failures"])


def test_an_undefined_kappa_fails_the_gate() -> None:
    """"Not computable from this data" is not evidence that the judge agrees with anyone."""
    humans = [4] * 25
    report = report_for(humans, [1, 2, 3, 4, 5] * 5)

    gate = evaluate_gate(report)

    assert gate["passed"] is False
    assert gate["observed_kappa"] is None
    assert any("not defined for this data" in failure for failure in gate["failures"])


def test_the_gate_names_where_it_was_pre_registered() -> None:
    gate = evaluate_gate(report_for([1, 2], [1, 2]))

    assert gate["pre_registered_at"].startswith("README.md")
    assert gate["statistic"] == "quadratic_weighted_cohens_kappa"


# --------------------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------------------


def test_stability_reports_perfect_agreement_when_the_judge_repeats_itself() -> None:
    labelled = pairs([3, 3], axis="safety")
    judge = adapter(verdict(3))

    result = check_stability(labelled, judge, repeats=3)

    assert result["overall"]["mean_pairwise_exact_agreement"]["mean"] == pytest.approx(1.0)
    assert result["temperature"] == 0.7
    assert result["samples_per_pair"] == 3


def test_stability_falls_when_the_judge_wavers() -> None:
    labelled = pairs([3])
    # Three samples on one pair: two agree, one does not, so one of three unordered pairs matches.
    judge = adapter(verdict(3), verdict(3), verdict(5))

    result = check_stability(labelled, judge, repeats=3)

    assert result["overall"]["mean_pairwise_exact_agreement"]["mean"] == pytest.approx(1 / 3)


def test_stability_records_where_the_judge_wavered() -> None:
    labelled = pairs([3])
    judge = adapter(verdict(3), verdict(5))

    result = check_stability(labelled, judge, repeats=2)

    variances = result["overall"]["mean_dimension_variance"]
    assert set(variances) == set(JUDGE_DIMENSIONS)
    assert all(value == pytest.approx(1.0) for value in variances.values())


def test_stability_is_reported_per_axis() -> None:
    from evals.schema import Axis

    labelled = pairs([3], axis="safety")
    judge = adapter(verdict(3))

    result = check_stability(labelled, judge, repeats=2)

    assert "safety" in result["by_axis"]
    assert set(result["by_axis"]) == {axis.value for axis in Axis}


def test_stability_carries_an_interval() -> None:
    labelled = pairs([3, 3, 3, 3])
    judge = adapter(verdict(3))

    result = check_stability(labelled, judge, repeats=2)
    headline = result["overall"]["mean_pairwise_exact_agreement"]

    assert headline["n"] == 4
    assert headline["ci_low"] <= headline["mean"] <= headline["ci_high"]


def test_one_sample_cannot_show_variance() -> None:
    with pytest.raises(ValueError, match="cannot show variance"):
        check_stability(pairs([3]), adapter(verdict(3)), repeats=1)


def test_a_cache_enabled_adapter_is_refused_on_the_stability_leg() -> None:
    """A cache hit is a replay: n identical requests share one key, so this would report perfect
    agreement it never measured."""
    judge = adapter(verdict(3), use_cache=True)

    with pytest.raises(ValueError, match="no_cache=True"):
        check_stability(pairs([3]), judge, repeats=2)


def test_a_cached_sample_is_refused_even_when_the_adapter_claims_otherwise() -> None:
    judge = adapter(verdict(3), cached=True)

    with pytest.raises(ValueError, match="served from the cache"):
        check_stability(pairs([3]), judge, repeats=2)


def test_the_stability_subsample_is_first_in_file_order_by_default() -> None:
    labelled = pairs([1, 2, 3, 4, 5])

    chosen = select_stability_items(labelled, 2)

    assert [pair.pair_id for pair in chosen] == ["v-000", "v-001"]


def test_a_seeded_stability_subsample_is_reproducible() -> None:
    labelled = pairs([1, 2, 3, 4, 5])

    first = select_stability_items(labelled, 3, seed=7)
    second = select_stability_items(labelled, 3, seed=7)

    assert [pair.pair_id for pair in first] == [pair.pair_id for pair in second]
    assert len(first) == 3


def test_a_limit_at_or_above_the_set_takes_everything() -> None:
    labelled = pairs([1, 2, 3])

    assert len(select_stability_items(labelled, 0)) == 3
    assert len(select_stability_items(labelled, 99)) == 3


# --------------------------------------------------------------------------------------
# Block-order sensitivity
# --------------------------------------------------------------------------------------


def test_position_bias_is_gone_under_its_old_name() -> None:
    """The A/B flip rate it named is not a statistic a single-response judge can produce."""
    assert not hasattr(validate_judge_module, "check_position_bias")


def test_self_preference_names_the_dependency_it_is_missing() -> None:
    """It stops raising bare; what is missing is stated in the same style as the others."""
    with pytest.raises(NotImplementedError, match="human labels on both arms"):
        check_self_preference(adapter(verdict(3)), [])


def test_a_reordering_reports_drift_with_an_interval_and_its_own_n() -> None:
    drift = check_block_order_sensitivity(pairs([3, 3, 3]), adapter(verdict(4)))

    row = drift["reorderings"]["response_before_prompt"]

    assert row["n"] == 3
    assert row["block_order"] == ["response", "prompt", "reference"]
    assert row["overall_drift"]["mean"] == pytest.approx(0.0)
    assert row["overall_drift"]["ci_low"] is not None
    assert row["overall_drift"]["ci_high"] is not None
    assert drift["canonical_block_order"] == list(CANONICAL_BLOCK_ORDER)


def test_the_drift_is_signed_so_a_lenient_reordering_reads_as_lenient() -> None:
    """An absolute value would report a systematically more generous judge and a noisy one
    identically."""
    judge = alternating_adapter(default=3, reordered=5)

    drift = check_block_order_sensitivity(pairs([3, 3]), cast(JudgeAdapter, judge))

    assert drift["reorderings"]["response_before_prompt"]["overall_drift"]["mean"] == pytest.approx(
        2.0
    )


def test_a_harsher_reordering_reads_as_negative_drift() -> None:
    judge = alternating_adapter(default=4, reordered=2)

    drift = check_block_order_sensitivity(pairs([3, 3]), cast(JudgeAdapter, judge))

    assert drift["reorderings"]["response_before_prompt"]["overall_drift"]["mean"] == pytest.approx(
        -2.0
    )


def test_drift_is_reported_per_dimension_as_well() -> None:
    judge = alternating_adapter(default=3, reordered=5, pairs_scored=1)

    drift = check_block_order_sensitivity(pairs([3]), cast(JudgeAdapter, judge))

    per_dimension = drift["reorderings"]["response_before_prompt"]["dimension_drift"]

    assert set(per_dimension) == set(JUDGE_DIMENSIONS)
    assert all(value == pytest.approx(2.0) for value in per_dimension.values())


def test_drift_is_reported_per_axis_with_the_axis_own_n() -> None:
    labelled = [
        LabelledPair(pair_id="v-1", prompt="q", response="a", human_score=3, axis=Axis.SAFETY),
        LabelledPair(pair_id="v-2", prompt="q", response="a", human_score=3, axis=Axis.SAFETY),
        LabelledPair(
            pair_id="v-3", prompt="q", response="a", human_score=3, axis=Axis.HALLUCINATION
        ),
    ]

    drift = check_block_order_sensitivity(labelled, adapter(verdict(4)))

    by_axis = drift["reorderings"]["response_before_prompt"]["by_axis"]

    assert by_axis["safety"]["n"] == 2
    assert by_axis["hallucination"]["n"] == 1


def test_every_reordering_reports_its_own_n_rather_than_a_pooled_one() -> None:
    """Two reorderings measured over different numbers of pairs are two findings."""
    drift = check_block_order_sensitivity(pairs([3, 3]), adapter(verdict(4)))

    assert set(drift["reorderings"]) == set(BLOCK_REORDERINGS)
    assert all("n" in row for row in drift["reorderings"].values())
    assert "n" not in drift


def test_the_reordering_that_matters_most_is_always_measured() -> None:
    """A judge more generous when it reads the answer first cannot rank two agents."""
    assert "response_before_prompt" in BLOCK_REORDERINGS

    drift = check_block_order_sensitivity(pairs([3]), adapter(verdict(3)))

    assert drift["reorderings"]["response_before_prompt"]["n"] == 1
    assert drift["reorderings"]["response_before_prompt"].get("reason") is None


def test_the_reference_reordering_reports_n_zero_with_its_reason() -> None:
    """`LabelledPair` carries no reference, so the block is never rendered on this path. Reported
    as a row with `n=0` and the reason rather than omitted or as a reassuring 0.0."""
    drift = check_block_order_sensitivity(pairs([3, 3]), adapter(verdict(3)))

    row = drift["reorderings"]["reference_before_response"]

    assert row["n"] == 0
    assert row["overall_drift"] is None
    assert "LabelledPair carries no reference" in row["reason"]
    assert "not a finding that the judge ignores reference position" in row["reason"]


def test_an_inert_reordering_costs_no_judge_calls() -> None:
    """Measuring a guaranteed zero is not worth a judge call, and reporting it as drift would say
    the judge was asked and did not move."""
    judge = fake(verdict(3))

    check_block_order_sensitivity(pairs([3, 3]), cast(JudgeAdapter, judge))

    # Two pairs, scored twice, for the one reordering that changes the rendering.
    assert judge.count == 4


def test_an_unparsed_judgement_leaves_the_drift_denominator() -> None:
    judge = alternating_adapter(default=3, reordered=None)

    drift = check_block_order_sensitivity(pairs([3, 3]), cast(JudgeAdapter, judge))

    row = drift["reorderings"]["response_before_prompt"]

    assert row["n"] == 0
    assert row["n_unparsed"] == 2


def test_the_reordered_call_really_moves_the_blocks_in_the_message() -> None:
    """A `block_order` recorded on the judgement while the message went out canonically would be
    the one failure this measurement cannot survive."""
    judge = fake(verdict(3))

    check_block_order_sensitivity(pairs([3]), cast(JudgeAdapter, judge))

    assert block_orders_of(judge) == [["prompt", "response"], ["response", "prompt"]]


def test_every_reordered_judgement_records_the_order_it_was_made_under() -> None:
    """Reordered verdicts must never be mistaken for graded ones."""
    scores = score_labelled(pairs([3]), adapter(verdict(3)))

    assert tuple(next(iter(scores.values())).block_order) == CANONICAL_BLOCK_ORDER


# --------------------------------------------------------------------------------------
# The artifact and the manifest
# --------------------------------------------------------------------------------------


def run_validation(
    tmp_path: Path, humans: Sequence[int], judged: Sequence[int | None]
) -> ValidationReport:
    labelled = pairs(humans)
    source = write_labelled(
        tmp_path / "labelled.jsonl",
        [
            {
                "pair_id": pair.pair_id,
                "prompt": pair.prompt,
                "response": pair.response,
                "human_score": pair.human_score,
            }
            for pair in labelled
        ],
    )
    return validate_judge(
        labelled,
        source=source,
        judge=scoring_adapter(judged),
        runs_dir=tmp_path / "runs",
    )


def test_a_validation_run_writes_a_judge_manifest_and_its_judgements(tmp_path: Path) -> None:
    """A validation number without its conditions is not a result."""
    labelled, judged = passing_set()
    report = run_validation(tmp_path, [int(p.human_score) for p in labelled], judged)

    runs = tmp_path / "runs"
    manifest = json.loads((runs / f"{report.run_id}.manifest.json").read_text())

    assert manifest["run_kind"] == "judge"
    assert manifest["n_pairs"] == len(labelled)
    assert manifest["judge_rubric_sha256"]
    assert manifest["judge_model"] == report.judge_model
    judgements = (runs / f"{report.run_id}.judge.jsonl").read_text().strip().splitlines()
    assert len(judgements) == len(labelled)


def test_the_artifact_lands_under_runs_beside_the_manifest(tmp_path: Path) -> None:
    labelled, judged = passing_set()
    report = run_validation(tmp_path, [int(p.human_score) for p in labelled], judged)

    path = judge_validation_path(report.run_id, tmp_path / "runs")

    assert path.exists()
    assert path.parent == tmp_path / "runs"


def test_two_runs_over_identical_data_produce_identical_bytes(tmp_path: Path) -> None:
    """Otherwise the artifact cannot be diffed and "the judge changed" becomes
    indistinguishable from "the serialiser did".

    A run's identity is the one thing that legitimately differs — a new run mints a new id, and
    every path and citation derived from it moves with it — so the comparison is made after
    substituting the two ids for a placeholder. Everything else, including the bootstrap bounds,
    must match byte for byte.
    """
    labelled, judged = passing_set()
    humans = [int(pair.human_score) for pair in labelled]

    first = run_validation(tmp_path / "a", humans, judged)
    second = run_validation(tmp_path / "b", humans, judged)

    def normalised(report: ValidationReport, root: Path) -> str:
        text = json.dumps(report.to_dict(), indent=2, sort_keys=True)
        return text.replace(report.run_id, "RUN_ID").replace(str(root), "ROOT")

    assert normalised(first, tmp_path / "a") == normalised(second, tmp_path / "b")


def test_the_written_artifact_is_the_dict_it_reports(tmp_path: Path) -> None:
    labelled, judged = passing_set()
    report = run_validation(tmp_path, [int(pair.human_score) for pair in labelled], judged)

    written = judge_validation_path(report.run_id, tmp_path / "runs").read_text()

    assert json.loads(written) == report.to_dict()
    assert written.endswith("\n")


def test_the_artifact_records_the_absent_threshold_rather_than_omitting_it(
    tmp_path: Path,
) -> None:
    """A reader checking whether a cut was applied should find the field and see that none was."""
    labelled, judged = passing_set()
    payload = run_validation(tmp_path, [int(p.human_score) for p in labelled], judged).to_dict()

    assert payload["threshold"] is None
    assert payload["rules_version"] is None
    assert "no pass/fail cut is pre-registered" in payload["threshold_note"]
    # Null because no rules ran, and the baseline section rather than a second note says why: two
    # places explaining one absence is two places to fall out of step.
    assert payload["baseline"]["status"] == "not requested"


def test_the_artifact_reports_no_binary_statistics(tmp_path: Path) -> None:
    labelled, judged = passing_set()
    payload = run_validation(tmp_path, [int(p.human_score) for p in labelled], judged)
    text = json.dumps(payload.to_dict())

    for absent in ('"accuracy"', '"precision"', '"recall"', '"f1"'):
        assert absent not in text.lower(), absent


def test_the_artifact_labels_the_confusion_matrix_direction(tmp_path: Path) -> None:
    labelled, judged = passing_set()
    payload = run_validation(tmp_path, [int(p.human_score) for p in labelled], judged).to_dict()

    confusion = payload["agreement"]["confusion"]

    assert confusion["rows"] == "human_score"
    assert confusion["columns"] == "judge_overall"
    assert confusion["scale"] == list(range(1, JUDGE_SCALE_MAX + 1))


def test_the_artifact_carries_a_version(tmp_path: Path) -> None:
    labelled, judged = passing_set()
    payload = run_validation(tmp_path, [int(p.human_score) for p in labelled], judged).to_dict()

    assert payload["report_version"] >= 1
    assert payload["run_kind"] == "judge"


def test_an_unrequested_leg_says_so_rather_than_reading_as_a_null_result(tmp_path: Path) -> None:
    """A reader who finds nothing cannot tell a leg that was skipped from one that ran and found
    nothing, and those lead to opposite conclusions."""
    labelled, judged = passing_set()
    payload = run_validation(tmp_path, [int(p.human_score) for p in labelled], judged).to_dict()

    for section in (payload["block_order_sensitivity"], payload["baseline"]):
        assert section["status"] == "not requested"
        assert section["reasons"]

    assert payload["block_order_sensitivity"]["reorderings"] == {}


def test_the_block_order_leg_lands_in_the_artifact_when_it_is_asked_for(tmp_path: Path) -> None:
    labelled = pairs([3, 3])
    source = write_labelled(
        tmp_path / "labelled.jsonl",
        [
            {
                "pair_id": pair.pair_id,
                "prompt": pair.prompt,
                "response": pair.response,
                "human_score": pair.human_score,
            }
            for pair in labelled
        ],
    )

    payload = validate_judge(
        labelled,
        source=source,
        judge=adapter(verdict(3)),
        runs_dir=tmp_path / "runs",
        block_order=True,
    ).to_dict()

    drift = payload["block_order_sensitivity"]

    assert drift["reorderings"]["response_before_prompt"]["n"] == 2
    assert drift["reorderings"]["reference_before_response"]["n"] == 0
    assert any("measured nothing" in warning for warning in payload["warnings"])


# --------------------------------------------------------------------------------------
# Console output
# --------------------------------------------------------------------------------------


def test_the_console_prints_an_unlabelled_axis_as_an_empty_row(tmp_path: Path) -> None:
    labelled, judged = passing_set()
    report = run_validation(tmp_path, [int(p.human_score) for p in labelled], judged)

    text = rendered(report)

    assert "bias" in text
    assert "none labelled" in text


def test_the_per_axis_table_renders_whole_in_eighty_columns(tmp_path: Path) -> None:
    """`rich` reclaims width from the widest column, so a long cell truncates the axis names —
    and an axis printed as `hallucin...` is not a row anyone can read."""
    from evals.schema import Axis
    from evals.validate_judge import _per_axis_table

    labelled, judged = passing_set()
    report = run_validation(tmp_path, [int(pair.human_score) for pair in labelled], judged)

    stream = io.StringIO()
    Console(file=stream, width=80).print(_per_axis_table(report.by_axis))
    text = stream.getvalue()

    for axis in Axis:
        assert axis.value in text, axis.value
    assert "…" not in text


def test_the_console_explains_the_direction_column(tmp_path: Path) -> None:
    labelled, judged = passing_set()
    report = run_validation(tmp_path, [int(pair.human_score) for pair in labelled], judged)

    text = rendered(report)

    assert "L/H/T: pairs the judge scored above the human label" in text


def test_the_console_prints_the_five_by_five_table(tmp_path: Path) -> None:
    labelled, judged = passing_set()
    report = run_validation(tmp_path, [int(p.human_score) for p in labelled], judged)

    text = rendered(report)

    assert "rows: human label" in text
    assert "columns: judge overall" in text


def test_the_console_says_n_a_rather_than_zero_for_an_undefined_statistic(
    tmp_path: Path,
) -> None:
    report = run_validation(tmp_path, [3] * 25, [3] * 25)

    text = rendered(report)

    assert "n/a" in text
    assert "GATE FAILED" in text


def test_the_console_flags_the_lenient_direction_as_the_dangerous_one(tmp_path: Path) -> None:
    labelled, judged = passing_set()
    report = run_validation(tmp_path, [int(p.human_score) for p in labelled], judged)

    text = rendered(report)

    assert "lenient direction is the dangerous one" in text


def test_the_console_reports_unparsed_judgements_separately(tmp_path: Path) -> None:
    report = run_validation(tmp_path, [1, 2, 3], [1, None, 3])

    text = rendered(report)

    assert "did not parse" in text
    assert "our failure" in text


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def cli(
    monkeypatch: pytest.MonkeyPatch, judge: JudgeAdapter, *args: str
) -> int:
    monkeypatch.setattr(validate_judge_module, "load_judge_model", lambda **_: judge)
    return main(list(args))


def test_the_cli_loads_dotenv_before_reading_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate must not fail for want of an export when the key is in `.env`."""
    monkeypatch.setattr(validate_judge_module, "load_env", refuse_env_load)

    with pytest.raises(EnvLoaded):
        main(["--labelled", "unread.jsonl"])


def test_the_cli_exits_zero_when_the_gate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labelled, judged = passing_set()
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [
            {
                "pair_id": pair.pair_id,
                "prompt": pair.prompt,
                "response": pair.response,
                "human_score": pair.human_score,
            }
            for pair in labelled
        ],
    )

    code = cli(
        monkeypatch,
        scoring_adapter(judged),
        "--labelled",
        str(path),
        "--runs-dir",
        str(tmp_path / "runs"),
    )

    assert code == EXIT_OK


def test_the_cli_exits_non_zero_when_agreement_is_too_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exit code is the gate: an unvalidated judge must not look clean in CI."""
    humans = [1, 2, 3, 4, 5] * 5
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [
            {"pair_id": f"v-{index:03d}", "prompt": "q", "response": f"a{index}", "human_score": s}
            for index, s in enumerate(humans)
        ],
    )

    code = cli(
        monkeypatch,
        scoring_adapter([5, 4, 3, 2, 1] * 5),
        "--labelled",
        str(path),
        "--runs-dir",
        str(tmp_path / "runs"),
    )

    assert code == EXIT_FAILED


def test_the_cli_exits_non_zero_when_the_labels_cannot_be_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_labelled(tmp_path / "labelled.jsonl", [{"prompt": "q", "response": "a"}])

    code = cli(monkeypatch, scoring_adapter([3]), "--labelled", str(path))

    assert code == EXIT_FAILED
    assert "columns needed" in capsys.readouterr().err


def test_the_cli_refuses_a_dataset_without_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path, HALL)

    code = cli(monkeypatch, scoring_adapter([3]), "--dataset", str(dataset))

    assert code == EXIT_FAILED
    assert "--dataset needs --run" in capsys.readouterr().err


def test_the_cli_walks_the_sidecar_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, run_id = sidecar_fixture(tmp_path, score=4)

    code = cli(
        monkeypatch,
        scoring_adapter([4]),
        "--dataset",
        str(dataset),
        "--run",
        run_id,
        "--runs-dir",
        str(tmp_path / "runs"),
        "--labels-dir",
        str(tmp_path / "ds" / "labels"),
    )

    # One pair is far below the gate's minimum n, so this exits non-zero by design.
    assert code == EXIT_FAILED
    artifacts = list((tmp_path / "runs").glob("*.judge_validation.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text())
    assert payload["provenance"]["response_sha256_verified"] is True
    assert payload["provenance"]["labelled_run_id"] == run_id


def test_the_cli_refuses_a_stability_run_below_the_minimum_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 3}],
    )

    code = cli(
        monkeypatch,
        scoring_adapter([3]),
        "--labelled",
        str(path),
        "--stability-samples",
        "1",
    )

    assert code == EXIT_FAILED
    assert "cannot show variance" in capsys.readouterr().err


def test_the_cli_reports_a_cached_stability_adapter_as_an_operator_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Not a traceback: a cache hit is a replay, and that is a configuration fact rather than a
    bug in this code."""
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 3}],
    )
    judge = adapter(verdict(3), use_cache=True)

    code = cli(
        monkeypatch,
        judge,
        "--labelled",
        str(path),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--stability-samples",
        "2",
    )

    assert code == EXIT_FAILED
    assert "stability sampling could not be measured" in capsys.readouterr().err


def test_the_cli_records_the_stability_leg_in_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subsample and its seed live in the artifact, not the manifest: a stability figure over
    an unrecorded subsample cannot be re-checked against the run that produced it."""
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [
            {"pair_id": f"v-{index}", "prompt": "q", "response": f"a{index}", "human_score": 3}
            for index in range(4)
        ],
    )

    cli(
        monkeypatch,
        adapter(verdict(3)),
        "--labelled",
        str(path),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--stability-samples",
        "2",
        "--stability-items",
        "2",
        "--stability-seed",
        "11",
    )

    artifacts = list((tmp_path / "runs").glob("*.judge_validation.json"))
    stability = json.loads(artifacts[0].read_text())["stability"]

    assert stability["seed"] == 11
    assert stability["n_items_sampled"] == 2
    assert stability["samples_per_pair"] == 2
    assert stability["temperature"] == 0.7
    assert "cache hit is a replay" in stability["cache"]
    assert stability["overall"]["mean_pairwise_exact_agreement"]["mean"] == pytest.approx(1.0)


def test_a_subsampled_stability_leg_warns_that_it_is_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [
            {"pair_id": f"v-{index}", "prompt": "q", "response": f"a{index}", "human_score": 3}
            for index in range(4)
        ],
    )

    cli(
        monkeypatch,
        adapter(verdict(3)),
        "--labelled",
        str(path),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--stability-samples",
        "2",
        "--stability-items",
        "2",
    )

    artifacts = list((tmp_path / "runs").glob("*.judge_validation.json"))
    warnings = json.loads(artifacts[0].read_text())["warnings"]

    assert any("stability was measured over 2 of 4 pairs" in warning for warning in warnings)


def test_the_cli_maps_a_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_labelled(
        tmp_path / "theirs.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "their_rating": 3}],
    )

    code = cli(
        monkeypatch,
        scoring_adapter([3]),
        "--labelled",
        str(path),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--column-map",
        "human_score=their_rating",
    )

    # Resolution succeeded; the gate then fails on n, which is a separate matter.
    assert code == EXIT_FAILED
    artifacts = list((tmp_path / "runs").glob("*.judge_validation.json"))
    assert json.loads(artifacts[0].read_text())["agreement"]["n"] == 1


def test_the_cli_refuses_the_baseline_on_the_grader_path_with_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `--labelled` file has two columns and a score; the rules read a trace, so on this path
    there is no rule instrument to compare the judge against."""
    path = write_labelled(
        tmp_path / "labelled.jsonl",
        [{"pair_id": "v-1", "prompt": "q", "response": "a", "human_score": 3}],
    )

    cli(
        monkeypatch,
        scoring_adapter([3]),
        "--labelled",
        str(path),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--baseline",
    )

    output = " ".join(capsys.readouterr().out.split())

    assert "baseline: unavailable" in output
    assert "--baseline needs --dataset and --run" in output

    artifacts = list((tmp_path / "runs").glob("*.judge_validation.json"))
    baseline = json.loads(artifacts[0].read_text())["baseline"]

    assert baseline["status"] == "unavailable"
    assert any("no rule instrument" in reason for reason in baseline["reasons"])


def test_the_two_input_paths_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        main(["--labelled", "a.jsonl", "--dataset", "b.jsonl"])


def test_one_input_path_is_required() -> None:
    with pytest.raises(SystemExit):
        main([])


# --------------------------------------------------------------------------------------
# measure_agreement, the documented entry point
# --------------------------------------------------------------------------------------


def test_measure_agreement_scores_and_aggregates_in_one_call() -> None:
    labelled = pairs(HUMANS)

    report = measure_agreement(labelled, scoring_adapter(JUDGED))

    assert report.n == 5
    assert report.exact_agreement == pytest.approx(0.8)
    assert report.axis == ALL_AXES
