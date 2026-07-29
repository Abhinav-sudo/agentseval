"""Covers `evals.schema`, weighted toward the invariants that would otherwise fail silently.

The tests worth having here are the ones asserting that a *wrong* item is rejected rather than
accepted with a field quietly unset, because that is the failure mode a dataset has: an item
that loads fine and measures something other than what its author meant.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.prompts import JUDGE_DIMENSIONS, JUDGE_SCALE_MAX
from evals.schema import (
    ALL_SUBCATEGORIES,
    ANNOTATOR_ONLY_FIELDS,
    BENIGN_CONTROL_SUBCATEGORY,
    COUNTERFACTUAL_FIELDS,
    ERROR_ATTACK_TYPE_FORBIDDEN,
    ERROR_ATTACK_TYPE_ON_CONTROL,
    ERROR_ATTACK_TYPE_REQUIRED,
    ERROR_BIAS_UNPAIRED,
    ERROR_COUNTERFACTUAL_INCOMPLETE,
    ERROR_LABEL_SPACE_MISMATCH,
    ERROR_SUBCATEGORY_UNKNOWN,
    ERROR_TURN_EMPTY,
    MODEL_VISIBLE_FIELDS,
    SCORED_TURN_INDEX,
    SUBCATEGORIES,
    AttackType,
    Axis,
    EvalItem,
    HumanLabel,
    LabelRecord,
    LabelSpace,
)


def item(**overrides: object) -> EvalItem:
    """A minimal valid hallucination item, overridable field by field."""
    base: dict[str, object] = {
        "id": "h-1",
        "axis": Axis.HALLUCINATION,
        "subcategory": "answerable_kb",
        "turns": ["How much water during exercise?"],
        "expected_behavior": "Cites the hydration doc.",
        "answerable": True,
    }
    base.update(overrides)
    return EvalItem(**base)  # type: ignore[arg-type]


def error_types(exc: pytest.ExceptionInfo[ValidationError]) -> list[str]:
    return [str(err["type"]) for err in exc.value.errors()]


# --------------------------------------------------------------------------------------
# extra="forbid"
# --------------------------------------------------------------------------------------


def test_unknown_field_is_rejected_not_ignored() -> None:
    """A typo'd field name must fail rather than vanish.

    This is the whole reason for `extra="forbid"`: `attack_typ` silently ignored would leave a
    safety item scored as though no attack type had been set, and nothing downstream could tell.
    """
    with pytest.raises(ValidationError) as exc:
        item(attack_typ="direct")
    assert error_types(exc) == ["extra_forbidden"]


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        EvalItem(id="x", axis=Axis.HALLUCINATION, subcategory="answerable_kb")  # type: ignore[call-arg]
    assert "missing" in error_types(exc)


def test_item_is_frozen_against_reassignment() -> None:
    with pytest.raises(ValidationError):
        item().id = "other"


# --------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------


def test_subcategory_must_be_in_the_axis_vocabulary() -> None:
    with pytest.raises(ValidationError) as exc:
        item(subcategory="made_up_bucket")
    assert error_types(exc) == [ERROR_SUBCATEGORY_UNKNOWN]


def test_subcategory_from_another_axis_is_rejected() -> None:
    """`gender` is a real subcategory, just not on this axis."""
    with pytest.raises(ValidationError) as exc:
        item(subcategory="gender")
    assert error_types(exc) == [ERROR_SUBCATEGORY_UNKNOWN]


def test_unknown_axis_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        item(axis="toxicity")
    assert "enum" in error_types(exc)


def test_every_axis_has_a_nonempty_vocabulary() -> None:
    assert set(SUBCATEGORIES) == set(Axis)
    assert all(SUBCATEGORIES[axis] for axis in Axis)


def test_subcategories_do_not_overlap_between_axes() -> None:
    """Overlap would make `by_subcategory` ambiguous without also carrying the axis."""
    total = sum(len(values) for values in SUBCATEGORIES.values())
    assert len(ALL_SUBCATEGORIES) == total


def test_hallucination_vocabulary_covers_the_documented_corpus_gaps() -> None:
    """PROJECT.md lists six areas the corpus is deliberately silent on.

    They are what make the axis measurable, so each needs a bucket: an unanswerable item with
    nowhere to be filed gets filed somewhere wrong.
    """
    expected = {
        "unanswerable_medication",
        "unanswerable_diagnosis",
        "unanswerable_pregnancy",
        "unanswerable_pediatric",
        "unanswerable_rehab",
        "unanswerable_mental_health",
    }
    assert expected <= SUBCATEGORIES[Axis.HALLUCINATION]


# --------------------------------------------------------------------------------------
# attack_type iff safety
# --------------------------------------------------------------------------------------


def test_safety_item_requires_an_attack_type() -> None:
    with pytest.raises(ValidationError) as exc:
        item(axis=Axis.SAFETY, subcategory="overtraining")
    assert error_types(exc) == [ERROR_ATTACK_TYPE_REQUIRED]


def test_attack_type_is_forbidden_off_the_safety_axis() -> None:
    with pytest.raises(ValidationError) as exc:
        item(attack_type=AttackType.ROLEPLAY)
    assert error_types(exc) == [ERROR_ATTACK_TYPE_FORBIDDEN]


def test_safety_item_with_an_attack_type_is_valid() -> None:
    built = item(axis=Axis.SAFETY, subcategory="overtraining", attack_type=AttackType.DIRECT)
    assert built.attack_type is AttackType.DIRECT


def test_a_benign_control_needs_no_attack_type() -> None:
    """The one safety subcategory exempt from the requirement: nothing elicited it."""
    built = item(axis=Axis.SAFETY, subcategory=BENIGN_CONTROL_SUBCATEGORY, answerable=True)
    assert built.attack_type is None


def test_a_benign_control_may_not_carry_an_attack_type() -> None:
    """Exempt from the requirement is not the same as free to set it."""
    with pytest.raises(ValidationError) as exc:
        item(
            axis=Axis.SAFETY,
            subcategory=BENIGN_CONTROL_SUBCATEGORY,
            attack_type=AttackType.DIRECT,
        )
    assert error_types(exc) == [ERROR_ATTACK_TYPE_ON_CONTROL]


def test_the_control_exemption_does_not_leak_to_other_safety_subcategories() -> None:
    """The narrow relaxation stays narrow: every other harm topic still requires one."""
    for subcategory in sorted(SUBCATEGORIES[Axis.SAFETY] - {BENIGN_CONTROL_SUBCATEGORY}):
        with pytest.raises(ValidationError) as exc:
            item(axis=Axis.SAFETY, subcategory=subcategory)
        assert error_types(exc) == [ERROR_ATTACK_TYPE_REQUIRED], subcategory


def test_benign_control_is_only_a_safety_subcategory() -> None:
    """A control on another axis would be an ordinary item with a misleading name."""
    assert BENIGN_CONTROL_SUBCATEGORY in SUBCATEGORIES[Axis.SAFETY]
    assert BENIGN_CONTROL_SUBCATEGORY not in SUBCATEGORIES[Axis.HALLUCINATION]
    assert BENIGN_CONTROL_SUBCATEGORY not in SUBCATEGORIES[Axis.BIAS]


def test_no_attack_type_names_a_benign_control() -> None:
    """`AttackType` stays a vocabulary of techniques; the control is a subcategory instead."""
    assert not any("benign" in member.value for member in AttackType)


# --------------------------------------------------------------------------------------
# Counterfactual pairing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("present", COUNTERFACTUAL_FIELDS)
def test_counterfactual_fields_are_all_or_nothing(present: str) -> None:
    with pytest.raises(ValidationError) as exc:
        item(**{present: "value"})
    assert error_types(exc) == [ERROR_COUNTERFACTUAL_INCOMPLETE]


def test_all_three_counterfactual_fields_together_is_valid() -> None:
    built = item(
        counterfactual_id="cf-1",
        counterfactual_variant="male",
        counterfactual_attribute="gender",
    )
    assert built.counterfactual_id == "cf-1"


def test_bias_item_must_be_paired() -> None:
    """A lone bias item yields no within-pair delta but would be counted as though it did."""
    with pytest.raises(ValidationError) as exc:
        item(axis=Axis.BIAS, subcategory="gender")
    assert error_types(exc) == [ERROR_BIAS_UNPAIRED]


def test_paired_bias_item_is_valid() -> None:
    built = item(
        axis=Axis.BIAS,
        subcategory="gender",
        counterfactual_id="cf-1",
        counterfactual_variant="male",
        counterfactual_attribute="gender",
    )
    assert built.axis is Axis.BIAS


def test_non_bias_items_may_be_paired() -> None:
    """Pairing is required on bias, not exclusive to it."""
    built = item(
        counterfactual_id="cf-1",
        counterfactual_variant="short",
        counterfactual_attribute="phrasing",
    )
    assert built.counterfactual_attribute == "phrasing"


# --------------------------------------------------------------------------------------
# Turns
# --------------------------------------------------------------------------------------


def test_turns_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        item(turns=[])


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_whitespace_only_turn_is_rejected(blank: str) -> None:
    with pytest.raises(ValidationError) as exc:
        item(turns=["real question", blank])
    assert error_types(exc) == [ERROR_TURN_EMPTY]


def test_multi_turn_flag_and_scored_turn() -> None:
    built = item(turns=["first", "second", "third"])
    assert built.is_multi_turn
    assert built.scored_turn == "third"
    assert built.turns[SCORED_TURN_INDEX] == built.scored_turn


def test_single_turn_item_is_not_multi_turn() -> None:
    assert not item().is_multi_turn


# --------------------------------------------------------------------------------------
# What a model may see
# --------------------------------------------------------------------------------------


def test_annotator_only_fields_are_not_model_visible() -> None:
    assert not (ANNOTATOR_ONLY_FIELDS & MODEL_VISIBLE_FIELDS)
    assert {"expected_behavior", "notes"} <= ANNOTATOR_ONLY_FIELDS


def test_model_visible_excludes_expected_behavior_and_notes() -> None:
    """Showing the annotator's instruction to the agent makes the eval an instruction-following
    test, which raises both arms' scores while measuring less."""
    built = item(expected_behavior="say the kb does not cover it", notes="tricky one")
    visible = built.model_visible()
    assert visible == {"turns": built.turns}
    assert "say the kb does not cover it" not in str(visible)
    assert "tricky one" not in str(visible)


def test_model_visible_and_annotator_only_are_real_fields() -> None:
    """A constant naming a field that no longer exists would silently stop protecting it."""
    fields = set(EvalItem.model_fields)
    assert fields >= MODEL_VISIBLE_FIELDS
    assert fields >= ANNOTATOR_ONLY_FIELDS


# --------------------------------------------------------------------------------------
# Axis is not a judge dimension
# --------------------------------------------------------------------------------------


def test_axes_and_judge_dimensions_are_distinct_taxonomies() -> None:
    """They share the word 'safety' and nothing else; conflating them would score bias as a
    rubric dimension, which no single response can support."""
    axes = {axis.value for axis in Axis}
    dimensions = set(JUDGE_DIMENSIONS)
    assert axes != dimensions
    assert axes & dimensions == {"safety"}
    assert "bias" not in dimensions


# --------------------------------------------------------------------------------------
# Label records
# --------------------------------------------------------------------------------------


def label(**overrides: object) -> LabelRecord:
    base: dict[str, object] = {
        "item_id": "h-1",
        "run_id": "run-1",
        "dataset_sha256": "a" * 64,
        "response_sha256": "b" * 64,
        "label_space": LabelSpace.BINARY_BEHAVIORAL,
        "label": HumanLabel.PASS,
        "annotator": "alice",
        "labelled_at": "2026-07-28T12:00:00.000+00:00",
        "seconds_spent": 4.5,
    }
    base.update(overrides)
    return LabelRecord(**base)  # type: ignore[arg-type]


def test_binary_label_record_is_valid() -> None:
    assert label().label is HumanLabel.PASS
    assert label().score is None


def test_binary_space_rejects_a_score() -> None:
    """The two label spaces must not be mixed on one line: kappa is undefined across mismatched
    category sets, so a reader must never have to guess which space a record is in."""
    with pytest.raises(ValidationError) as exc:
        label(score=4)
    assert error_types(exc) == [ERROR_LABEL_SPACE_MISMATCH]


def test_rubric_space_rejects_a_binary_label() -> None:
    with pytest.raises(ValidationError) as exc:
        label(label_space=LabelSpace.RUBRIC_1_5, score=4, label=HumanLabel.PASS)
    assert error_types(exc) == [ERROR_LABEL_SPACE_MISMATCH]


def test_rubric_space_requires_a_score() -> None:
    with pytest.raises(ValidationError) as exc:
        label(label_space=LabelSpace.RUBRIC_1_5, label=None)
    assert error_types(exc) == [ERROR_LABEL_SPACE_MISMATCH]


def test_binary_space_requires_a_label() -> None:
    with pytest.raises(ValidationError) as exc:
        label(label=None)
    assert error_types(exc) == [ERROR_LABEL_SPACE_MISMATCH]


@pytest.mark.parametrize("score", [0, JUDGE_SCALE_MAX + 1, -1])
def test_rubric_score_is_bounded_by_the_judges_scale(score: int) -> None:
    """The human's categories have to be the judge's categories, so the bound comes from
    `JUDGE_SCALE_MAX` rather than a retyped literal."""
    with pytest.raises(ValidationError):
        label(label_space=LabelSpace.RUBRIC_1_5, label=None, score=score)


def test_label_record_key_is_run_and_item() -> None:
    """The key a later record supersedes. Two arms' answers to one item are two labels."""
    assert label().key == ("run-1", "h-1")
    assert label(run_id="run-2").key != label().key


def test_label_record_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError) as exc:
        label(labeled_at="typo")
    assert error_types(exc) == ["extra_forbidden"]
