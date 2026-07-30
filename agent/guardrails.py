"""Three screening stages around a turn: harmful input, unsafe output, ungrounded answer.

A guardrail is a *condition of a run*, not a feature. Two runs differing only in whether this
module was active are two arms of an ablation, and the only reason the comparison means anything
is that everything else about them is identical — same model, same prompt, same corpus, same
retrieval floor, same budgets. `manifest.assert_ablation_comparable` enforces that, and
`guardrails_sha256()` is what makes "same guardrails" a checkable claim rather than a habit.

**Independent of the scorer, on purpose, and this is the load-bearing decision in the module.**
`evals.deterministic` already owns lexical machinery for refusal and grounding, and this module
does not import it — `agent` may not import `evals` at all, and a test asserts that. But the
layering rule is downstream of a measurement problem, so consider what sharing would do.

`deterministic.check_no_refusal` is what computes the false-refusal rate: it decides whether a
response declined. Suppose this module emitted a canned refusal assembled from
`deterministic.RULE_PATTERNS["refusal"]`. The detector would then recognise our text by
construction, and the reported over-refusal cost of the guardrail would be a measurement of our
own vocabulary against itself. Every point of false-refusal rate would be an artifact of one
phrase appearing in two files. So: two pattern sets, two digests (`guardrails_sha256()` here,
`deterministic.rules_version()` there), written independently and allowed to disagree.

The independence is made **structural**, not merely lexical, because a lexical convention decays.
`deterministic.item_views` builds the scored `response` from the `role="turn"` record's
`content`, and `core.run_turn` writes the *model's own text* there whatever a guardrail did.
Our substituted sentence is therefore never fed to the refusal detector at all; it reaches
metrics only as a typed `guardrail_action`, which is what PROJECT.md's rule that no reported
number is computed by matching text demands, and it demands it most sharply here, where the text
would be ours.

**Rule-based throughout, and that is also a measurement decision.** No stage calls a model. A
screen using the candidate model would make the guardrail's quality a function of the arm, so
"guardrails help" and "this arm's classifier is weaker" could not be separated — the
harness-uniformity failure the project exists to avoid. A screen using the judge model would
have the judge scoring responses it had itself filtered. If a model-based stage is ever added it
must be one fixed model, recorded in the manifest, identical for both arms, and it gets
`GuardrailBudgets` rather than the candidate's `core.Budgets`.

**Deliberately narrow.** Each set is a handful of tight patterns rather than a broad net. A
filter that fires often catches more, and pays for it in `false_refusal_rate` on benign controls;
since the report prints the safety gain and the over-refusal cost in the same row, a broad set
would mostly be measuring its own collateral. Narrow is the better instrument here, and the
cost of every additional pattern is visible in the same table as its benefit.

**What the input screen does not block.** It matches requests for a *method* — how to injure
oneself, how to purge, how to exceed a labelled dose. A disclosure of distress is not matched,
and that is a decision rather than an oversight: the right response to "I have been feeling like
I do not want to be here" is a crisis resource, and blocking the turn withholds exactly that
while looking, in every aggregate, like a safety success.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from agent.trace import sha256_text

__all__ = [
    "GUARDRAIL_PATTERNS",
    "SAFE_COMPLETIONS",
    "GuardrailAction",
    "GuardrailBudgets",
    "GuardrailDecision",
    "GuardrailStage",
    "Guardrails",
    "guardrails_sha256",
]


class GuardrailAction(StrEnum):
    """What a guardrail did to a turn, as a closed vocabulary.

    This is the field every guardrail rate is computed from. Typed rather than derived from the
    delivered prose, because the prose is ours: a rate obtained by matching our own canned
    sentence would move whenever someone reworded it, and would report a copy-edit as a change
    in behaviour.

    `NONE` is a member rather than an absence so that a guardrails-on run and a guardrails-off
    run produce the same set of keys, and a per-action breakdown prints a zero row instead of
    dropping a row. A *missing* action — an older trace written before the field existed — is
    None and reads as unknown, which is a different fact from `NONE`.

    Members:
        NONE: Every stage passed, or guardrails were off. The turn was delivered as written.
        INPUT_BLOCKED: The input screen matched, and no candidate model call was made.
        OUTPUT_FILTERED: The model answered and the output screen matched its answer.
        GROUNDING_ABSTAINED: The model answered with nothing retrieved above the floor.
    """

    NONE = "none"
    INPUT_BLOCKED = "input_blocked"
    OUTPUT_FILTERED = "output_filtered"
    GROUNDING_ABSTAINED = "grounding_abstained"


class GuardrailStage(StrEnum):
    """Which screen fired, recorded next to the action.

    Nearly redundant with `GuardrailAction` today, and kept separate anyway: the action says
    what happened to the turn and the stage says which code decided it. They come apart as soon
    as one stage can produce two actions, and a trace that recorded only the action would then
    need re-deriving rather than reading.
    """

    INPUT_SCREEN = "input_screen"
    OUTPUT_SCREEN = "output_screen"
    GROUNDING = "grounding"


#: Every pattern any stage matches, in one frozen mapping so `guardrails_sha256()` covers all of
#: them. The discipline is `deterministic.RULE_PATTERNS`': stored as source strings, compiled on
#: demand, and never written inline — an inlined regex would be a rule the digest cannot see,
#: and the digest is the only thing making "both arms ran the same guardrail" checkable.
#:
#: Two sets, keyed by the stage that reads them:
#:
#: * `harmful_intent` — a user turn requesting a method of self-injury, purging, extreme
#:   restriction, or dose misuse. Requests, not topics: "is 400 mg of caffeine a lot" asks about
#:   a dose and "how much before it becomes toxic" asks for a threshold to cross, and a screen
#:   that could not tell those apart would block the corpus's own subject matter.
#: * `unsafe_content` — an assistant answer that would be unsafe to deliver whatever elicited
#:   it. Written against the *imperative* rather than against a list of drug names: a substance
#:   list is a maintenance burden that silently rots, whereas "you can stop taking your
#:   medication" is unsafe in a wellness assistant regardless of which medication follows.
#:
#: No pattern here appears in `RULE_PATTERNS`, and the two sets are not merely disjoint by
#: accident: one matches what a user asks for and the other what an assistant says, so they
#: describe different speakers. See the module docstring for why that separation is the reason
#: the false-refusal number means anything.
GUARDRAIL_PATTERNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "harmful_intent": (
            # A method of self-injury. Proximity-bounded rather than a bare keyword, so that a
            # disclosure ("I have been having thoughts about dying") is not caught by a rule
            # written for a request ("how do I ...").
            r"\bhow (?:do|can|could|should|would) I\b[^.?!]{0,60}"
            r"\b(?:kill myself|end my life|end it all|overdose|hurt myself|harm myself)\b",
            r"\b(?:best|easiest|fastest|quickest|painless|most effective) way to\b"
            r"[^.?!]{0,40}\b(?:kill myself|end my life|overdose|hurt myself|harm myself)\b",
            # Purging and appetite suppression as an instruction request.
            r"\bhow (?:do|can|to|could) (?:I |you )?"
            r"(?:purge|throw up|make (?:myself|yourself) (?:sick|vomit)|vomit)\b",
            r"\b(?:how (?:do|can) I |ways? to |trick to )"
            r"(?:completely |totally |permanently )?(?:kill|stop|suppress|shut off) "
            r"(?:my |your )?(?:appetite|hunger)\b",
            # Deliberately exceeding a labelled dose, or asking where the harm threshold is so
            # as to approach it. The distinguishing feature is the direction of the question:
            # not "what is the limit" but "how far past it can I go".
            r"\bhow (?:much|many|high)\b[^.?!]{0,60}\b(?:before (?:it|I|this) (?:becomes?|gets?|"
            r"turns?) (?:toxic|dangerous|lethal|fatal)|to overdose|overdose on|would kill me)\b",
            r"\b(?:double|triple|quadruple|mega-?dose|megadosing|stack(?:ing)?)\b"
            r"[^.?!]{0,40}\b(?:dose|dosage|my (?:meds|medication|prescription))\b",
            # A request to work around clinical supervision is a request for the one thing this
            # assistant is least able to give safely.
            r"\bwithout (?:my |a |the )?(?:doctor|physician|clinician|GP)\b"
            r"[^.?!]{0,30}\b(?:knowing|finding out|approval|permission)\b",
        ),
        "unsafe_content": (
            # Directing a change to prescribed treatment, or waving off clinical care.
            r"\byou (?:should|can|could|may|might)\b[^.?!]{0,20}"
            r"\b(?:stop|quit|discontinue|skip|halve|double)\b[^.?!]{0,40}"
            r"\b(?:medication|meds|prescription|dose|dosage|antidepressant|insulin|thyroid)\b",
            r"\b(?:no need|(?:do not|don't|doesn't) need|not necessary|unnecessary)\b"
            r"[^.?!]{0,20}\bto (?:see|consult|contact|call|talk to)\b[^.?!]{0,30}"
            r"\b(?:doctor|physician|clinician|GP|professional|emergency)\b",
            # Endorsing purging or extreme restriction as a method that works.
            r"\b(?:purging|vomiting|laxatives?|diuretics?|fasting for (?:days|a week))\b"
            r"[^.?!]{0,40}\b(?:is|are|can be)\b[^.?!]{0,20}"
            r"\b(?:fine|safe|okay|ok|effective|harmless)\b",
            r"\b(?:under|below|fewer than|less than)\s*[1-9]\d{2}\s*(?:kcal|calories)\b"
            r"[^.?!]{0,20}\b(?:a|per|each) day\b",
            # Telling the reader to go past a labelled maximum. The label is the safety claim
            # the manufacturer stands behind, and inviting someone past it is the failure mode
            # a corpus carrying no dosages cannot possibly support.
            r"\b(?:exceed|go (?:above|over|past)|more than|well over)\b[^.?!]{0,30}"
            r"\b(?:labell?ed|recommended|maximum|safe) (?:dose|dosage|limit|amount)\b",
        ),
    }
)

#: What the user is shown when a stage fires, one message per action.
#:
#: Here rather than in `agent.prompts` because `prompts.py` holds text a *model* reads and this
#: is text a *user* reads. Putting them together would also put this inside
#: `system_prompt_sha256`, which would make a guardrails-on run and a guardrails-off run differ
#: on the prompt digest and be refused by every comparison guard for the wrong reason.
#:
#: There is no `NONE` entry: nothing is substituted when nothing fired, and a mapping that
#: returned "" for the ordinary case would invite a caller to deliver it.
#:
#: Whether `deterministic.check_no_refusal` would match this wording is not something these
#: sentences are written to control, in either direction. The detector never sees them — the
#: scorers read the model's own output — and writing our prose to evade a detector would be the
#: same circularity as writing it to satisfy one.
SAFE_COMPLETIONS: Mapping[GuardrailAction, str] = MappingProxyType(
    {
        GuardrailAction.INPUT_BLOCKED: (
            "This assistant only covers general wellness information from its own reference "
            "material, and what you have asked for falls outside that. If this is about your "
            "own safety right now, please contact a local emergency number or a crisis line, "
            "where someone can actually help."
        ),
        GuardrailAction.OUTPUT_FILTERED: (
            "The answer prepared for this question described a change to medication, dosing, or "
            "eating that this assistant is not in a position to recommend, so it has been "
            "withheld. Please raise the question with a clinician who knows your history."
        ),
        GuardrailAction.GROUNDING_ABSTAINED: (
            "The reference material behind this assistant does not contain anything close "
            "enough to this question to answer it, so no answer is offered rather than one "
            "assembled from elsewhere."
        ),
    }
)


@dataclass(frozen=True)
class GuardrailBudgets:
    """What the screening stages may spend, kept apart from `core.Budgets`.

    `core.Budgets` bounds the *candidate's* reasoning. Screening that drew on it would leave an
    arm with less of its own budget than the other arm's, which is a different experiment run
    under the same name — and the shortfall would surface as a `tool_budget` stop that looked
    like the model's failure.

    Zero by default, and every stage here is rule-based, so the default is not a limit anyone
    is working around: it is a statement that no stage calls a model. It is a field rather than
    a constant so a future model-based screen has somewhere to declare its ceiling, and so
    `guardrails_sha256()` covers that ceiling when it appears.
    """

    max_model_calls: int = 0

    def __post_init__(self) -> None:
        if self.max_model_calls < 0:
            raise ValueError(
                f"max_model_calls cannot be negative, got {self.max_model_calls}"
            )


@dataclass(frozen=True)
class GuardrailDecision:
    """One stage's finding: what it did, which stage did it, and what is delivered instead.

    Attributes:
        action: The typed outcome. Every reported guardrail number is computed from this.
        stage: Which screen produced it.
        matched: The span that triggered the match, for a human reading a trace. Diagnostic
            only — nothing aggregates it, because a rate over regex spans would be a rate over
            how the patterns happen to be written.
        completion: The text delivered in place of the model's. Never assigned to
            `AgentResult.final_text`: see that field's docstring, and the module docstring here.
    """

    action: GuardrailAction
    stage: GuardrailStage
    matched: str
    completion: str


def _compiled(name: str) -> tuple[re.Pattern[str], ...]:
    """Compile one pattern set from `GUARDRAIL_PATTERNS`, case-insensitively.

    Compiled on demand rather than cached at import, matching `deterministic._compiled`: a test
    that swaps the mapping gets the swap rather than a stale compilation, and a dozen short
    patterns are not worth the staleness.
    """
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in GUARDRAIL_PATTERNS[name])


def _first_match(text: str, name: str) -> str | None:
    """Return the first span in `text` matching pattern set `name`, or None."""
    for pattern in _compiled(name):
        found = pattern.search(text)
        if found is not None:
            return " ".join(found.group(0).split())
    return None


def guardrails_sha256(
    *,
    budgets: GuardrailBudgets | None = None,
    model_id: str | None = None,
) -> str:
    """Digest of everything that decides what these guardrails do.

    Recorded in the manifest whenever guardrails are on, and it is the field the ablation guard
    varies. Without it, two runs differing only in guardrails would produce identical manifests
    and `assert_comparable` would certify a comparison of different conditions as a comparison
    of one — silently, which is the failure the manifest exists to prevent.

    Covers the patterns, the delivered completions, the screening budget, and the model id of a
    model-based stage (None while every stage is rule-based). The completions are in scope
    because they are what the user received: a run that delivered different text delivered a
    different intervention, even if the same patterns chose it.

    Deliberately **excludes `min_score`**. The floor is already covered by
    `manifest.retrieval_config_sha256`, and digesting it in two places would leave a reader
    unable to say which condition moved when both digests changed together.

    Args:
        budgets: The screening budget in force. Defaults to `GuardrailBudgets()`.
        model_id: The fixed model a model-based stage would use, or None while none does.

    The container types are normalised the way `rules_version()` normalises them, so a refactor
    from tuple to list is not reported as a change in behaviour.
    """
    budgets = budgets if budgets is not None else GuardrailBudgets()
    payload = {
        "patterns": {name: list(group) for name, group in GUARDRAIL_PATTERNS.items()},
        "completions": {
            action.value: text for action, text in sorted(SAFE_COMPLETIONS.items())
        },
        "budgets": {"max_model_calls": budgets.max_model_calls},
        "model_id": model_id,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


class Guardrails:
    """The three screens, as one object a turn consults.

    Stateless between turns on purpose: a screen that remembered would make an item's outcome
    depend on which items preceded it, and the eval runner's per-item isolation would stop
    meaning anything.
    """

    def __init__(
        self,
        *,
        budgets: GuardrailBudgets | None = None,
        model_id: str | None = None,
    ) -> None:
        """Configure the screens.

        Args:
            budgets: What screening may spend. Defaults to `GuardrailBudgets()` — no model
                calls, because no stage makes one.
            model_id: The fixed model a model-based stage would use. Must be the same for both
                arms if it is ever set; a per-arm screening model would make the guardrail's
                quality a property of the arm.
        """
        self.budgets = budgets if budgets is not None else GuardrailBudgets()
        self.model_id = model_id

    def digest(self) -> str:
        """This instance's `guardrails_sha256()`, for the manifest.

        Through the instance rather than the module function directly, so a non-default budget
        or a screening model reaches the digest. A manifest recording the module default while
        the run used something else would be the same lie `bound_registry` was added to stop.
        """
        return guardrails_sha256(budgets=self.budgets, model_id=self.model_id)

    def screen_input(self, text: str) -> GuardrailDecision | None:
        """Screen a user turn before any candidate model call.

        Called before the model loop, so a block costs the candidate nothing: no model call, no
        tool call, no budget. That ordering is the point of the stage — screening after the fact
        would pay for the completion it then discards, and an arm whose budget went on discarded
        completions is not comparable with one whose did not.

        Returns None when nothing matched, which is the ordinary case.
        """
        matched = _first_match(text, "harmful_intent")
        if matched is None:
            return None
        return GuardrailDecision(
            action=GuardrailAction.INPUT_BLOCKED,
            stage=GuardrailStage.INPUT_SCREEN,
            matched=matched,
            completion=SAFE_COMPLETIONS[GuardrailAction.INPUT_BLOCKED],
        )

    def screen_output(self, text: str) -> GuardrailDecision | None:
        """Screen the model's answer before it is delivered.

        The model's text is not modified and not discarded — `core.run_turn` writes it to the
        trace verbatim and leaves `AgentResult.final_text` holding it. This decides only what
        the *user* receives, which is why the two readings the ablation compares can both be
        recovered from a single guardrails-on trace.

        An empty answer is not screened: there is nothing in it to be unsafe, and a stage that
        fired on emptiness would convert every budget-exhausted turn into a filtered one.
        """
        if not text.strip():
            return None
        matched = _first_match(text, "unsafe_content")
        if matched is None:
            return None
        return GuardrailDecision(
            action=GuardrailAction.OUTPUT_FILTERED,
            stage=GuardrailStage.OUTPUT_SCREEN,
            matched=matched,
            completion=SAFE_COMPLETIONS[GuardrailAction.OUTPUT_FILTERED],
        )

    def enforce_grounding(
        self, retrieved_chunk_ids: Sequence[str], answer_text: str
    ) -> GuardrailDecision | None:
        """Abstain when an answer would be delivered with nothing retrieved to support it.

        **The floor is not re-derived here.** `min_score` already exists on `search` and
        `lookup_kb`, sits on `AgentConfig`, and is covered by `retrieval_config_sha256`; with it
        bound into the tool by `tools.bound_registry`, a hit below the floor is dropped before
        it ever reaches the loop. So "nothing cleared the floor" and "the turn retrieved no
        chunk" are the same observation, and this stage reads the second one. A second threshold
        here would be a second floor, and the ablation would then vary two conditions — the
        floor and its enforcement — leaving the delta attributable to neither.

        Two cases reach this, and they are deliberately treated alike: a turn whose retrieval
        came back empty, and a turn that never retrieved at all. Both deliver an answer with no
        corpus text behind it, which is the thing being enforced against. It also means an
        answer built from `search_web` abstains, since web results carry no chunk ids — correct
        under this rule, and worth knowing before reading the false-refusal column.

        An empty answer is left alone: it makes no claim to be ungrounded, and there is nothing
        to substitute for.
        """
        if not answer_text.strip():
            return None
        if retrieved_chunk_ids:
            return None
        return GuardrailDecision(
            action=GuardrailAction.GROUNDING_ABSTAINED,
            stage=GuardrailStage.GROUNDING,
            matched="",
            completion=SAFE_COMPLETIONS[GuardrailAction.GROUNDING_ABSTAINED],
        )
