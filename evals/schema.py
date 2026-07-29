"""The single definition of an eval item, and the vocabularies that describe one.

`EvalItem` is the only dataset shape in this package. A package holding two of them is one
where `deterministic.py` and `report.py` can read the same file and disagree about what is in
it, with nothing raising to say so — which is why `runner.EvalCase` was deleted rather than
kept alongside this. The vocabularies below are declared here once and imported by
`metrics.py`, `report.py`, `deterministic.py`, and `validate_dataset.py`; retyping a
subcategory string in a second module is how a breakdown quietly loses a bucket.

Datasets are JSONL, one object per line. `extra="forbid"` is the load-bearing half of the
model config: a misspelled `attack_typ` that was merely ignored would leave the item scored as
though the field had never been set, which changes what was measured without failing.
`frozen=True` stops code from rebinding a field on a loaded item — note that it does not deep
freeze, so `item.turns.append(...)` still works; the guard is against a module that
"normalises" an item in place, not against a determined caller.

**Nothing here rewrites a dataset file.** `agent.manifest.DatasetRef` digests the file's bytes
and `assert_comparable` refuses two runs whose `dataset_sha256` differ even when the path
matches, so a formatter, a key reordering, or an in-place canonicalisation would silently
destroy a comparison rather than fail one. Human labels therefore live in an append-only
sidecar (`LabelRecord`), never written back into the item.

Two things the runner needs and the field list does not say:

* **The scored turn is the response to the final turn** (`SCORED_TURN_INDEX`). Earlier turns
  are replayed through `agent.memory.Conversation` as ordinary context and are not scored.
  Scoring an intermediate turn would grade the agent partway through the escalation the item
  exists to provoke.
* **`expected_behavior` and `notes` are never shown to a model** — not the agent, not the
  judge. They are annotator instructions, and feeding them in turns the eval into a test of
  instruction-following: both arms' scores rise while the eval measures less.
  `MODEL_VISIBLE_FIELDS` and `ANNOTATOR_ONLY_FIELDS` are the executable form.

How `axis` relates to `prompts.JUDGE_DIMENSIONS`: they are different taxonomies that happen to
share the word "safety". An **axis** groups items — it is a property of the question asked. A
**dimension** (`helpfulness`, `accuracy`, `safety`, `communication`) scores a response — it is
a property of the answer given. Every item on every axis is scored on every dimension. Bias is
the case that makes the distinction matter: it is measured as a *within-pair delta* between two
items differing in one attribute, not as a rubric dimension, because "was this response
biased?" is not answerable from one response in isolation. That is why pairing is an invariant
enforced here rather than an authoring convention, and why one unpaired bias item is an error
instead of half a comparison.

Naming, because two nearby ideas would otherwise collide:

* The counterfactual fields are not `pair_*`. In `judge.py` and `validate_judge.py` a
  `pair_id` is already one `(prompt, response)` pair; one name with two meanings in one
  package is a join waiting to go wrong.
* `human_label` is not `human_score`. `HumanLabel` is a binary behavioural verdict on a
  response; `validate_judge.LabelledPair.human_score` is a 1-5 rubric label for judge-vs-human
  agreement. `LabelSpace` keeps them apart on the wire, and nothing converts between them.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from agent.prompts import JUDGE_SCALE_MAX

# --------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------


class Axis(StrEnum):
    """What an item is built to measure.

    Three axes, matching the three ways this corpus can be pushed: asked something it does
    not cover, asked the same thing about two different people, or asked for something it
    should decline.
    """

    HALLUCINATION = "hallucination"
    BIAS = "bias"
    SAFETY = "safety"


class AttackType(StrEnum):
    """*How* an unsafe response is elicited, orthogonal to the safety subcategory's *what*.

    The split exists so a result can distinguish a model that refuses plain requests but folds
    under a roleplay frame from one that refuses everything. Collapsing technique and topic
    into one vocabulary would multiply into a bucket per combination, each with one item in it.

    Every member names a technique, which is why there is no `BENIGN_CONTROL` here: a
    legitimate question is elicited by nothing, and a non-technique in this enum would appear
    in every per-attack-type breakdown as a row that is not an attack. Control items carry
    `subcategory == BENIGN_CONTROL_SUBCATEGORY` and `attack_type is None` instead.
    """

    DIRECT = "direct"
    ROLEPLAY = "roleplay"
    AUTHORITY_CLAIM = "authority_claim"
    HYPOTHETICAL = "hypothetical"
    INCREMENTAL_ESCALATION = "incremental_escalation"
    PROMPT_INJECTION = "prompt_injection"
    FALSE_REASSURANCE = "false_reassurance"
    PERSISTENCE = "persistence"


#: Hallucination subcategories. The six `unanswerable_*` values mirror the corpus exclusions in
#: PROJECT.md one for one, and that correspondence is the point: the gaps are what make the
#: axis measurable, so an item naming a gap the corpus actually covers is mislabelled rather
#: than merely odd. `answerable_kb` is the control — without it, a model that refuses
#: everything scores perfectly on this axis.
HALLUCINATION_SUBCATEGORIES = frozenset(
    {
        "unanswerable_medication",
        "unanswerable_diagnosis",
        "unanswerable_pregnancy",
        "unanswerable_pediatric",
        "unanswerable_rehab",
        "unanswerable_mental_health",
        "answerable_kb",
        "fabricated_citation",
        "false_premise",
    }
)

#: Bias subcategories: the demographic family varied within a pair. `counterfactual_attribute`
#: records the concrete attribute name and may be finer-grained; this is the grouping key
#: `report.py` aggregates on, which is why it is a closed set and the attribute is not.
BIAS_SUBCATEGORIES = frozenset(
    {
        "gender",
        "age",
        "body_size",
        "socioeconomic",
        "culture",
        "disability",
    }
)

#: The safety axis's control bucket: an ordinary health question the corpus answers well, where
#: a refusal is the failure rather than the pass. The one safety subcategory that takes no
#: `attack_type`, because a legitimate question is not elicited by a technique.
BENIGN_CONTROL_SUBCATEGORY = "benign_control"

#: Safety subcategories: the harm topic. Every value but `benign_control` names something `kb/`
#: deliberately contains no advice about, so a model answering from the corpus cannot
#: accidentally look unsafe. `benign_control` is the exception on both counts — the corpus does
#: cover it, and that is the point. Without it a model that refuses everything scores perfectly
#: on this axis, which is the same hole `answerable_kb` closes on the hallucination axis.
SAFETY_SUBCATEGORIES = frozenset(
    {
        "medical_advice_request",
        "disordered_eating",
        "overtraining",
        "crisis_disclosure",
        "dangerous_supplement",
        "vulnerable_population",
        "authority_pressure",
        BENIGN_CONTROL_SUBCATEGORY,
    }
)

#: The per-axis controlled vocabulary. Free-text subcategories drift into thirty singleton
#: buckets, at which point a per-subcategory breakdown is a listing rather than a breakdown.
SUBCATEGORIES: Mapping[Axis, frozenset[str]] = MappingProxyType(
    {
        Axis.HALLUCINATION: HALLUCINATION_SUBCATEGORIES,
        Axis.BIAS: BIAS_SUBCATEGORIES,
        Axis.SAFETY: SAFETY_SUBCATEGORIES,
    }
)

#: Every subcategory across every axis, for callers that need membership without the axis.
ALL_SUBCATEGORIES: frozenset[str] = frozenset().union(*SUBCATEGORIES.values())

#: The only item fields a model may ever see. Named as a constant so that "the judge does not
#: read `expected_behavior`" is a checkable claim rather than a convention someone can breach
#: by adding one convenient line to a prompt builder.
MODEL_VISIBLE_FIELDS: frozenset[str] = frozenset({"turns"})

#: Fields written for humans and withheld from models. Disjoint from `MODEL_VISIBLE_FIELDS` by
#: construction; `tests/test_schema.py` asserts it.
ANNOTATOR_ONLY_FIELDS: frozenset[str] = frozenset({"expected_behavior", "notes"})

#: Which turn's response is scored: the last. See the module docstring.
SCORED_TURN_INDEX = -1

#: Items per counterfactual pair. Two, and exactly two — three variants is not a pair with a
#: spare, it is an ambiguous delta with no defined direction.
COUNTERFACTUAL_PAIR_SIZE = 2

#: The three fields that are all-or-nothing together.
COUNTERFACTUAL_FIELDS: tuple[str, str, str] = (
    "counterfactual_id",
    "counterfactual_variant",
    "counterfactual_attribute",
)

#: Fields that must be identical across the two variants of a pair. Everything except the
#: varied attribute and the id: if the two items differ in `answerable` or in turn count, the
#: measured delta includes that difference and is no longer attributable to the attribute.
PAIR_INVARIANT_FIELDS: tuple[str, ...] = (
    "axis",
    "subcategory",
    "attack_type",
    "answerable",
    "expected_behavior",
)

#: Error identifiers raised by `EvalItem`'s validators, exposed as `err["type"]` on a
#: `ValidationError`. `validate_dataset.py` maps these onto its own diagnostic codes
#: structurally, so a linter diagnostic never depends on matching the wording of a message —
#: rewording an error must not be able to move what a check reports.
ERROR_TURN_EMPTY = "turn_empty"
ERROR_SUBCATEGORY_UNKNOWN = "subcategory_not_in_vocabulary"
ERROR_ATTACK_TYPE_REQUIRED = "attack_type_required"
ERROR_ATTACK_TYPE_FORBIDDEN = "attack_type_forbidden"
ERROR_ATTACK_TYPE_ON_CONTROL = "attack_type_on_benign_control"
ERROR_COUNTERFACTUAL_INCOMPLETE = "counterfactual_incomplete"
ERROR_BIAS_UNPAIRED = "bias_unpaired"
ERROR_LABEL_SPACE_MISMATCH = "label_space_mismatch"


# --------------------------------------------------------------------------------------
# The item
# --------------------------------------------------------------------------------------


class EvalItem(BaseModel):
    """One eval item: the prompts to send, and what a passing answer does.

    Attributes:
        id: Unique within a file, and frozen once the item is labelled — a label refers to an
            id, so reusing one silently re-points labels that were made against other text.
        axis: What the item measures. Groups items; see the module docstring on why this is
            not a judge dimension.
        subcategory: A value from `SUBCATEGORIES[axis]`.
        turns: **User messages only** — the assistant turns are what is under test. One turn
            is the single-turn case; more is multi-turn escalation, and the response scored is
            the one to the final turn.
        expected_behavior: What a passing response does, in free text. This is the annotator's
            instruction and the reason a label is auditable a month later. Never shown to a
            model.
        must_include: Substrings `deterministic.check_contains` requires.
        expected_tool: The tool `deterministic.check_tool_used` requires, which has no other
            source for the argument.
        answerable: Whether `kb/` genuinely covers the question. Load-bearing on the
            hallucination axis, where the corpus's deliberate silences are what let an agent
            answering from its training be distinguished from one that says so.
        counterfactual_id: Pair key, shared by exactly two items.
        counterfactual_variant: The varied attribute's *value* here, e.g. `female`.
        counterfactual_attribute: The varied attribute's *name*, e.g. `gender`. Free text: it
            is the author's label for what changed and appears in the linter's diff for human
            review, whereas `subcategory` is the closed set reports group on.
        attack_type: Required when `axis` is `safety`, except on a `BENIGN_CONTROL_SUBCATEGORY`
            item where it must be None; forbidden on every other axis. An attack type on a bias
            item would describe a technique nothing scores, and one on a control would describe
            a technique that elicited a question nobody objects to.
        notes: Free text for the author. Never shown to a model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    axis: Axis
    subcategory: str = Field(min_length=1)
    turns: list[str] = Field(min_length=1)
    expected_behavior: str = Field(min_length=1)
    must_include: list[str] = Field(default_factory=list)
    expected_tool: str | None = None
    answerable: bool
    counterfactual_id: str | None = None
    counterfactual_variant: str | None = None
    counterfactual_attribute: str | None = None
    attack_type: AttackType | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> EvalItem:
        for index, turn in enumerate(self.turns):
            if not turn.strip():
                raise PydanticCustomError(
                    ERROR_TURN_EMPTY,
                    "turn {index} is empty or whitespace only",
                    {"index": index},
                )

        allowed = SUBCATEGORIES[self.axis]
        if self.subcategory not in allowed:
            raise PydanticCustomError(
                ERROR_SUBCATEGORY_UNKNOWN,
                "subcategory {subcategory!r} is not in the vocabulary for axis {axis!r}: "
                "{allowed}",
                {
                    "subcategory": self.subcategory,
                    "axis": self.axis.value,
                    "allowed": ", ".join(sorted(allowed)),
                },
            )

        if self.axis is Axis.SAFETY:
            if self.subcategory == BENIGN_CONTROL_SUBCATEGORY:
                if self.attack_type is not None:
                    raise PydanticCustomError(
                        ERROR_ATTACK_TYPE_ON_CONTROL,
                        "attack_type must be None on a {subcategory!r} item, but is "
                        "{attack_type!r}; a control is a legitimate question that no technique "
                        "elicited, so recording one would put a non-attack into the "
                        "per-attack-type breakdown",
                        {
                            "subcategory": self.subcategory,
                            "attack_type": self.attack_type.value,
                        },
                    )
            elif self.attack_type is None:
                raise PydanticCustomError(
                    ERROR_ATTACK_TYPE_REQUIRED,
                    "attack_type is required when axis is 'safety' and subcategory is not "
                    "{control!r}; without it nothing records how the unsafe response was "
                    "elicited",
                    {"control": BENIGN_CONTROL_SUBCATEGORY},
                )
        elif self.attack_type is not None:
            raise PydanticCustomError(
                ERROR_ATTACK_TYPE_FORBIDDEN,
                "attack_type is only meaningful on the safety axis, but axis is {axis!r}; on "
                "another axis it describes a technique nothing scores",
                {"axis": self.axis.value},
            )

        # All-or-nothing: a variant without an id names a difference nothing pairs it against,
        # and an id without an attribute leaves the delta's cause unrecorded.
        present = [name for name in COUNTERFACTUAL_FIELDS if getattr(self, name) is not None]
        if present and len(present) != len(COUNTERFACTUAL_FIELDS):
            missing = [name for name in COUNTERFACTUAL_FIELDS if getattr(self, name) is None]
            raise PydanticCustomError(
                ERROR_COUNTERFACTUAL_INCOMPLETE,
                "the counterfactual fields are all-or-nothing: {present} set, {missing} missing",
                {"present": ", ".join(present), "missing": ", ".join(missing)},
            )

        if self.axis is Axis.BIAS and self.counterfactual_id is None:
            raise PydanticCustomError(
                ERROR_BIAS_UNPAIRED,
                "bias items must be paired: an unpaired bias item yields no within-pair delta "
                "but would be counted as though it did",
            )

        return self

    @property
    def is_multi_turn(self) -> bool:
        return len(self.turns) > 1

    @property
    def scored_turn(self) -> str:
        """The turn whose response is scored. See `SCORED_TURN_INDEX`."""
        return self.turns[SCORED_TURN_INDEX]

    def model_visible(self) -> dict[str, list[str]]:
        """The item as a model may see it: the user turns and nothing else.

        A method rather than a comment so that a prompt builder reaching for item data gets
        the withheld fields withheld by default, instead of getting the whole item and being
        trusted to leave two of its fields alone.
        """
        return {name: getattr(self, name) for name in sorted(MODEL_VISIBLE_FIELDS)}


# --------------------------------------------------------------------------------------
# Human labels
# --------------------------------------------------------------------------------------


class HumanLabel(StrEnum):
    """A binary behavioural verdict on one response, against the item's `expected_behavior`.

    Not a score. See `LabelSpace` for why the distinction is enforced rather than explained.
    """

    PASS = "pass"
    FAIL = "fail"


class LabelSpace(StrEnum):
    """Which set of categories a label was drawn from.

    Recorded on every label because the two spaces must never be silently mixed. Cohen's kappa
    is undefined across mismatched category sets, so a human label compared against a judge
    score has to have been drawn from the judge's own categories *at labelling time*. Picking a
    threshold to collapse `RUBRIC_1_5` into pass/fail after seeing a graded run means choosing
    the statistic that flattered the result; introducing any mapping between these two is a
    pre-registered decision, not an implementation detail.
    """

    #: `HumanLabel`: did the response do what `expected_behavior` describes?
    BINARY_BEHAVIORAL = "binary_behavioral"

    #: 1-`JUDGE_SCALE_MAX`, matching the judge's own scale. Feeds
    #: `validate_judge.LabelledPair.human_score`.
    RUBRIC_1_5 = "rubric_1_5"


class LabelRecord(BaseModel):
    """One line of a label sidecar: one keystroke, with the provenance to verify it.

    Sidecars are append-only. Correcting a label appends a newer record and readers take the
    last one per `(run_id, item_id)`; editing in place would lose the fact that a label
    changed, which is exactly the thing an audit wants to see.

    The three digests are not redundant. `dataset_sha256` catches an edited dataset
    independently of run identity, `run_id` pins which arm's response was read, and
    `response_sha256` makes the label verifiable against the trace it was made from — which
    catches the case a re-run cannot: a trace regenerated or hand-edited after labelling. A
    human label is the one artifact here that cannot be reproduced, so it is worth the field.

    Attributes:
        label: Set when `label_space` is `BINARY_BEHAVIORAL`, `None` otherwise.
        score: Set when `label_space` is `RUBRIC_1_5`, `None` otherwise. Exactly one of the
            two is populated, so a reader cannot mistake a 4 for a pass.
        seconds_spent: Time on this item. A run of sub-second labels is the signature of an
            annotator clicking through, which is worth being able to see afterwards.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    dataset_sha256: str = Field(min_length=1)
    response_sha256: str = Field(min_length=1)
    label_space: LabelSpace
    label: HumanLabel | None = None
    score: int | None = Field(default=None, ge=1, le=JUDGE_SCALE_MAX)
    annotator: str = Field(min_length=1)
    labelled_at: str = Field(min_length=1)
    seconds_spent: float = Field(ge=0.0)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_label_space(self) -> LabelRecord:
        binary = self.label_space is LabelSpace.BINARY_BEHAVIORAL
        expected: object | None = self.label if binary else self.score
        forbidden: object | None = self.score if binary else self.label
        wanted, unwanted = ("label", "score") if binary else ("score", "label")

        if expected is None or forbidden is not None:
            raise PydanticCustomError(
                ERROR_LABEL_SPACE_MISMATCH,
                "label_space {space!r} requires {wanted} and forbids {unwanted}; the two label "
                "spaces do not convert into each other, so a reader must never have to guess "
                "which one a record is in",
                {"space": self.label_space.value, "wanted": wanted, "unwanted": unwanted},
            )
        return self

    @property
    def key(self) -> tuple[str, str]:
        """The identity a later record supersedes: one annotator's verdict on one response."""
        return (self.run_id, self.item_id)
