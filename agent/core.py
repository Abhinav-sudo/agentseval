"""The agent loop.

One turn is: render prompt -> call model -> try to parse a tool call out of the raw text
-> run the tool -> feed the result back -> repeat until the model answers or the step
budget is exhausted.

Two properties this module must preserve (PROJECT.md):

* **Uniform harness.** The loop is model-agnostic. Tool calls are parsed out of message
  content as JSON, never taken from a provider's native tool-calling response. Nothing
  here may branch on `model.name`; a frontier-only fast path would confound model quality
  with harness quality and void the comparison.
* **Everything logged.** Every step — prompt, raw completion, parsed call, tool result,
  parse failures, retries — is appended to the run's JSONL trace via `agent.trace`, under
  a run manifest describing the conditions. Malformed JSON from a model is data about that
  model and is recorded, not silently repaired.

The parser is tolerant of framing and strict about content. It will find a JSON object wrapped
in a fence or buried in prose, because that is a recoverable deviation, and it records a
`FORMAT_VIOLATION` whenever no protocol object could be read at all. The per-model violation
rate is a headline result rather than a nuisance to smooth over, so the tolerance is identical
for both arms and the failures are counted rather than quietly repaired. `FormatViolation`
types them, because "this arm fails to format 20% of the time" and "it omits the citations key
20% of the time" are different findings with different fixes.

One failure is carved out of that rate: a response the provider stopped at `max_tokens`.
Half a JSON object is what our own token ceiling produced, so it is recorded as
`TRUNCATED`/`budget_induced` and kept out of the contract-violation count. Charging it to the
model would report a harness setting as a model weakness.

Three budgets bound a turn, and they are independent because they measure different things:

* `max_tool_calls` counts *successful* dispatches. Once spent, the model is told to answer.
* `max_tool_errors` counts model-caused failures — an invented tool, arguments that do not
  fit the schema. Sharing one counter with successes would let a model that cannot call a
  tool correctly consume the same budget as one doing useful work, and hide the difference.
* `max_model_calls` is the absolute ceiling on LLM calls, which also bounds a model that
  keeps replying with unparseable text.

Infrastructure failures are charged to none of them: they are retried, and if they persist
the item ends as `infrastructure_failed` and is excluded from axis scoring (see README.md).
An outage on our side is not evidence about a model.

Runs are at temperature 0.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent.guardrails import GuardrailAction, GuardrailDecision, Guardrails
from agent.memory import Conversation
from agent.models.base import (
    DEFAULT_MAX_TOKENS,
    ChatMessage,
    ModelAdapter,
    ModelError,
    ModelResponse,
    RetryPolicy,
)
from agent.prompts import (
    ARGS_KEY,
    CITATIONS_KEY,
    FINAL_ANSWER_REQUIRED,
    FINAL_KEY,
    TOOL_KEY,
    build_system_prompt,
    render_bad_argument_type_error,
    render_missing_argument_error,
    render_tool_error,
    render_tool_result,
    render_unknown_argument_error,
    render_unknown_tool_error,
)
from agent.tools import Tool, ToolErrorReason, ToolInfraError, ToolInputError, registry
from agent.trace import (
    ROLE_ASSISTANT,
    ROLE_GUARDRAIL,
    ROLE_MEMORY,
    ROLE_SUMMARISER,
    ROLE_TOOL,
    ROLE_TURN,
    ROLE_USER,
    TraceLogger,
)

DEFAULT_TEMPERATURE = 0.0

#: Successful tool calls per turn before the model is told to answer. Three is enough for
#: retrieve, refine, fall back to the web, and small enough that a model stuck in a loop
#: costs a bounded amount.
DEFAULT_MAX_TOOL_CALLS = 3

#: Model-caused tool errors tolerated per turn. Two, because the first is a mistake and the
#: second is a mistake repeated after being told exactly what was wrong; a third would spend
#: the turn's remaining calls to learn nothing new.
DEFAULT_MAX_TOOL_ERRORS = 2

#: Hard ceiling on model calls per turn. Above the sum of the other two, so a turn that
#: spends every tool call and every error still has calls left to produce an answer.
DEFAULT_MAX_MODEL_CALLS = 6

#: Attempts *after* the first for a tool that failed for infrastructure reasons. Lower than
#: the provider default in `RetryPolicy`: a local tool that has failed three times is broken
#: rather than busy, and the item is excluded from scoring either way.
DEFAULT_TOOL_RETRIES = 2

#: Consecutive unparseable completions tolerated. The first failure buys one re-prompt
#: carrying the parse error; a model that cannot answer that has demonstrated the thing being
#: measured, and further attempts would only spend budget to confirm it.
MAX_CONSECUTIVE_VIOLATIONS = 2

#: Metric name for a completion that could not be parsed under the protocol. Defined here
#: because both `evals.deterministic` and `evals.metrics` report it and the string is the
#: join between them.
FORMAT_VIOLATION = "format_violation"

# Why a turn ended. Recorded on `AgentResult` and in the trace. One per budget, because
# "ran out of tool calls" and "kept calling tools wrong" need different follow-up.
STOPPED_ANSWERED = "answered"
STOPPED_TOOL_BUDGET = "tool_budget"
STOPPED_TOOL_ERROR_BUDGET = "tool_error_budget"
STOPPED_MODEL_CALL_BUDGET = "model_call_budget"
STOPPED_PROTOCOL_ERROR = "protocol_error"
STOPPED_INFRASTRUCTURE_FAILED = "infrastructure_failed"

#: The input screen matched and the turn never reached the model. Its own reason rather than
#: `answered` or a budget: nothing was answered and no budget ran out, and folding it into
#: either would put a guardrail decision in a column that reports model behaviour.
#:
#: `output_filtered` and `grounding_abstained` deliberately have no entry here. In both the
#: model did answer, and only the delivery was replaced — recording a stop would report a turn
#: that completed as one that did not, and would drop it out of `n_wellformed` for something the
#: model did correctly.
STOPPED_INPUT_BLOCKED = "input_blocked"

#: Longest excerpt of unparseable output quoted back to the model. Enough to identify what
#: went wrong, short enough that a rambling completion cannot crowd out the context window.
ERROR_EXCERPT_CHARS = 200

#: JSON type names mapped to what Python calls them, for argument validation. `bool` is
#: excluded from the numeric types on purpose: `True` is an `int` in Python, and accepting it
#: as `top_k` would send a nonsense argument to a tool that then has to defend itself.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}

_FENCE = re.compile(r"^\s*```[^\n`]*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)


class FormatViolation(StrEnum):
    """How a completion failed the protocol, typed so the failures can be told apart.

    A single `format_violation_rate` says a model is unreliable; these say *how*, and the
    difference decides what to do about it. A model that omits `citations` needs one line of
    the prompt reworded; a model that cannot emit JSON at all needs a different protocol.
    Reporting the two as one number would hide a cheap fix inside an expensive-looking result.

    Members:
        UNPARSEABLE_JSON: No protocol object could be read from the text at all — a syntax
            error, or valid JSON that was not an object.
        UNKNOWN_TOP_LEVEL_KEY: An object arrived, but with neither `tool` nor `final` in it.
        WRONG_VALUE_TYPE: A protocol key was present holding something unusable: a non-string
            `final` or `tool`, a non-object `args`, or `tool` and `final` together, which
            leaves no single readable instruction.
        MISSING_CITATIONS_KEY: An answer with no `citations` key. Kept separate from
            `CITATIONS_WRONG_TYPE` because "forgot the field" and "sent the wrong shape" are
            different mistakes, and separate from the rest because it is the one violation
            that citation grounding depends on.
        CITATIONS_WRONG_TYPE: `citations` present but not an array of strings.
        TRUNCATED: The provider stopped at `max_tokens` mid-object. Recorded as
            `budget_induced` and excluded from the contract-violation rate: our ceiling cut
            the response off, so this is a fact about the harness.
    """

    UNPARSEABLE_JSON = "unparseable_json"
    UNKNOWN_TOP_LEVEL_KEY = "unknown_top_level_key"
    WRONG_VALUE_TYPE = "wrong_value_type"
    MISSING_CITATIONS_KEY = "missing_citations_key"
    CITATIONS_WRONG_TYPE = "citations_wrong_type"
    TRUNCATED = "truncated"


class ProtocolError(ValueError):
    """The model's output was not a valid tool call under the JSON protocol.

    Carries the `FormatViolation` describing which way it failed. Required rather than
    defaulted, so a new raise site cannot land in the report as an untyped violation: the
    classification is made where the failure is detected and by the code that detected it.
    """

    def __init__(self, message: str, violation: FormatViolation) -> None:
        super().__init__(message)
        self.violation = violation


@dataclass(frozen=True)
class Budgets:
    """What one turn is allowed to spend, as one object.

    Grouped rather than passed as three arguments so both arms are configured from a single
    value that the manifest records whole. Three separate knobs are three chances to set one
    of them differently for one arm and compare the arms anyway; `assert_comparable` refuses
    runs whose budgets differ, and it can only do that because these reach the manifest.

    Attributes:
        max_tool_calls: Successful dispatches per turn. Errors do not consume this — a model
            that spends its calls productively and one that spends them on invalid arguments
            should not look alike.
        max_tool_errors: Model-caused tool failures per turn. Infrastructure failures are not
            charged here or anywhere.
        max_model_calls: Hard ceiling on LLM calls per turn, including re-prompts after a
            parse failure and the call that produces the final answer.
    """

    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_tool_errors: int = DEFAULT_MAX_TOOL_ERRORS
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS

    def __post_init__(self) -> None:
        """Reject budgets that cannot produce a turn.

        A `max_model_calls` of zero would yield an empty answer from every item and a run of
        perfectly comparable zeros, which is worse than an error because it looks like data.
        """
        if self.max_model_calls < 1:
            raise ValueError(f"max_model_calls must be at least 1, got {self.max_model_calls}")
        if self.max_tool_calls < 0 or self.max_tool_errors < 0:
            raise ValueError(
                f"tool budgets cannot be negative, got max_tool_calls={self.max_tool_calls}, "
                f"max_tool_errors={self.max_tool_errors}"
            )


@dataclass
class ToolCall:
    """A tool invocation parsed from model output."""

    name: str
    arguments: dict[str, Any]
    raw: str


@dataclass
class FinalAnswer:
    """An answer parsed from model output, with the citations it declared.

    `citations` is what the model *said* it cited. The inline `[[id]]` markers remain in
    `text` and are recoverable with `lookup_kb.parse_citations`; keeping both is what lets
    `evals.deterministic.check_citation_grounding` notice a declared citation that never
    appears inline, or an inline one that was never retrieved.
    """

    text: str
    citations: list[str]
    raw: str


@dataclass
class ToolOutcome:
    """What a dispatched tool call produced, successful or not.

    `reason` is the discriminator: None means the tool ran, and anything else means the model
    got the call wrong and the failure is charged to `max_tool_errors`. Infrastructure
    failures never appear here — they are raised, because they end the item.
    """

    rendered: str
    chunk_ids: list[str] = field(default_factory=list)
    error: str | None = None
    reason: ToolErrorReason | None = None


@dataclass
class Step:
    """One model call and its consequence, as written to the trace.

    Attributes:
        format_violation: How this completion failed the protocol, or None if it parsed.
        budget_induced: True when the failure was our `max_tokens` cutting the response off
            rather than the model misformatting it. Kept next to `format_violation` so a
            reader of one always sees the other.
        tool_error: Why the model's tool call was rejected, or None. Typed for the same
            reason as `format_violation`: the report breaks the rate down by these, and
            deriving them from `error` prose would break on the next rewording.
    """

    index: int
    messages_sent: list[dict[str, str]]
    completion: str
    tool_call: ToolCall | None = None
    tool_result: Any = None
    error: str | None = None
    latency_ms: float | None = None
    usage: dict[str, int | None] = field(default_factory=dict)
    cached: bool = False
    usd_cost: float | None = None
    format_violation: FormatViolation | None = None
    budget_induced: bool = False
    tool_error: ToolErrorReason | None = None


@dataclass
class AgentResult:
    """The outcome of one turn: final answer plus the full step-by-step trace.

    Attributes:
        final_text: The answer, or "" when the turn ended without one. Deliberately not
            filled in with an apology the model never wrote: an empty answer is what
            happened, and a synthesised one would flatter the model in every downstream
            score.
        citations: Knowledge-base ids the model declared in its final JSON.
        tool_calls: Calls actually dispatched, in order. Attempts refused because the tool
            budget was spent are visible in `steps` instead.
        retrieved_chunk_ids: Every chunk id the tools returned this turn, deduplicated in
            first-seen order. The ground truth `citations` is checked against.
        latency_ms: Wall clock for the whole turn, including tool time. Per-call model
            latency lives on each `Step`, alongside `cached`, so an aggregate can exclude
            cache replays (PROJECT.md).
        tokens: `prompt`, `completion`, and `total`, summed over every model call in the
            turn — the summariser's included, since it spent the same budget.
        format_violations: Completions that broke the protocol contract. Truncations are
            *not* counted here; they are in `budget_induced_truncations`. The per-model rate
            is a reported result, and mixing our token ceiling into it would make the number
            partly about us.
        format_violation: The first violation type of the turn, or None. A turn can produce
            two, so this is the headline and `format_violations_by_type` is the detail.
        format_violations_by_type: Counts keyed by `FormatViolation` value, including
            `truncated`, so one field answers both "how often" and "which way".
        budget_induced_truncations: Completions cut off at `max_tokens`. Reported separately
            and loudly: above a couple of percent it means `max_tokens` is too low and the
            run is measuring the harness.
        tool_errors: Model-caused tool failures — an invented tool, invalid arguments.
        tool_errors_by_type: Those counts keyed by `ToolErrorReason` value.
        infrastructure_failed: True when a tool failed for reasons outside the model's
            control and retries did not clear it. Such items are excluded from axis scoring
            (README.md), which is only defensible because the flag is recorded per item and
            applied identically to both arms.
        infrastructure_error: The failure that ended the turn, or None.
        usd_cost: Summed over calls that reported a cost, or None when none did.
        guardrail_action: What `agent.guardrails` did to this turn, as a typed
            `GuardrailAction` value. `none` when guardrails were off or every stage passed.
            This is the field every reported guardrail rate is computed from — never the
            delivered prose, which is ours and would make a rate move on a copy-edit.
        guardrail_stage: Which screen fired, or None. Diagnostic company for the action.
        guardrail_completion: The text delivered in place of the model's answer, or None. Held
            apart from `final_text` rather than replacing it: see `final_text` above, and
            `agent.guardrails`. A judge handed this would score our sentence as the
            candidate's.
    """

    final_text: str
    steps: list[Step]
    stopped_reason: str
    run_id: str
    citations: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)
    format_violations: int = 0
    format_violation: FormatViolation | None = None
    format_violations_by_type: dict[str, int] = field(default_factory=dict)
    budget_induced_truncations: int = 0
    tool_errors: int = 0
    tool_errors_by_type: dict[str, int] = field(default_factory=dict)
    infrastructure_failed: bool = False
    infrastructure_error: str | None = None
    usd_cost: float | None = None
    # Appended, like every field before them: a reader pulling `citations` out of position
    # four would find a guardrail action there instead.
    guardrail_action: str = GuardrailAction.NONE.value
    guardrail_stage: str | None = None
    guardrail_completion: str | None = None

    @property
    def well_formed(self) -> bool:
        """True when the turn produced an answer with no protocol contract violation.

        The condition behind every `_wellformed` metric in `evals.metrics`. A truncation does
        not disqualify a turn here: it is not the model's failure, and treating it as one
        would move the survivorship bias this property exists to make visible.

        An `output_filtered` or `grounding_abstained` turn is still well-formed, because the
        model produced a conforming answer and a guardrail replaced only what was delivered.
        An `input_blocked` turn is not, and cannot be: there is no answer and no protocol
        exchange to have conformed to. That asymmetry is why the guardrail actions are read
        from `guardrail_action` and not inferred from this property.
        """
        return self.stopped_reason == STOPPED_ANSWERED and self.format_violations == 0

    @property
    def delivered_text(self) -> str:
        """What the user actually received: the substituted completion, or the model's answer.

        Derived rather than stored, so the two readings the ablation compares cannot drift
        apart. `final_text` is always the model's own output and `guardrail_completion` is
        always ours; "as delivered" is a view over both, and a third stored field would be a
        third thing to keep consistent with the other two.

        The distinction is the whole point of the guardrails ablation: as-delivered is what the
        guardrailed system did, which is the thing under evaluation, and `final_text` is what
        the model would have said, which is what the guardrails-off arm measures directly.
        """
        substituted = self.guardrail_completion
        return substituted if substituted is not None else self.final_text


# --------------------------------------------------------------------------------------
# The tolerant parser
# --------------------------------------------------------------------------------------


def strip_code_fence(completion: str) -> str:
    """Remove one markdown fence wrapping the whole completion.

    The protocol accepts a ```json fence, so unwrapping it is not leniency; it is reading the
    documented shape. Anything other than a single fence around the entire text is left
    alone for `extract_json_object` to search through.
    """
    match = _FENCE.match(completion)
    return match.group("body") if match else completion


def _balanced_end(text: str, start: int) -> int | None:
    """Return the index just past the `}` closing the object at `start`, or None.

    Tracks string state and backslash escapes. Counting braces without doing so breaks on a
    `{` inside a string value, which is exactly what an answer quoting JSON — or discussing
    a protocol — contains.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _excerpt(text: str) -> str:
    """Shorten model output for an error message, collapsing whitespace."""
    flat = " ".join(text.split())
    if len(flat) <= ERROR_EXCERPT_CHARS:
        return flat
    return flat[:ERROR_EXCERPT_CHARS] + "..."


def extract_json_object(completion: str) -> dict[str, Any]:
    """Return the protocol object in `completion`.

    Reads the compliant shape first — the whole text, fence removed, as one JSON object —
    and only then searches for the first balanced `{...}` that parses. The search exists
    because a model that prefixes "Sure, I'll look that up" has produced a recoverable
    deviation, and scoring that as a total failure would overstate the gap between a model
    with sloppy framing and one that cannot emit JSON at all.

    Raises:
        ProtocolError: no JSON object could be read, or the JSON was not an object.
    """
    stripped = strip_code_fence(completion)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        payload = _search_for_object(stripped, exc)

    if not isinstance(payload, dict):
        raise ProtocolError(
            f"expected one JSON object, got {type(payload).__name__}: {_excerpt(completion)}",
            FormatViolation.UNPARSEABLE_JSON,
        )
    return payload


def _search_for_object(text: str, first_error: json.JSONDecodeError) -> Any:
    """Return the first balanced `{...}` in `text` that parses as JSON.

    Candidates are tried left to right, because the first `{` may open a brace in prose
    rather than the object the model meant. A candidate that fails to parse is skipped
    *whole*: the objects nested inside it are part of that one broken attempt, not further
    attempts, and returning one of them would answer a question about `{"args": {}}` when the
    model's actual mistake was a stray comma three characters earlier.

    Raises:
        ProtocolError: no candidate parsed.
    """
    start = text.find("{")
    while start != -1:
        end = _balanced_end(text, start)
        if end is None:
            break
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            start = text.find("{", end)

    if "{" not in text:
        raise ProtocolError(
            f"expected exactly one JSON object and found none in: {_excerpt(text)}",
            FormatViolation.UNPARSEABLE_JSON,
        )
    raise ProtocolError(
        f"could not parse a JSON object ({first_error.msg}): {_excerpt(text)}",
        FormatViolation.UNPARSEABLE_JSON,
    )


def _require_citations(payload: Mapping[str, Any]) -> list[str]:
    """Return the `citations` array, which every answer must carry.

    Required because the prompt requires it, including the empty list for an answer that
    cited nothing. Accepting a missing array would leave "cited nothing" and "forgot the
    field" indistinguishable, and citation grounding rests on telling them apart.
    """
    if CITATIONS_KEY not in payload:
        raise ProtocolError(
            f'"{CITATIONS_KEY}" is required on every answer; use [] when nothing was cited',
            FormatViolation.MISSING_CITATIONS_KEY,
        )
    citations = payload[CITATIONS_KEY]
    if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
        raise ProtocolError(
            f'"{CITATIONS_KEY}" must be an array of chunk id strings',
            FormatViolation.CITATIONS_WRONG_TYPE,
        )
    return list(citations)


def parse_reply(completion: str) -> ToolCall | FinalAnswer:
    """Parse one completion into the tool call or the answer it carries.

    The loop's single entry point: one parse per completion, then a branch on the result.
    `parse_tool_call` and `parse_final` are the same logic behind narrower signatures.

    Raises:
        ProtocolError: the output is not a valid protocol object.
    """
    payload = extract_json_object(completion)
    has_tool = TOOL_KEY in payload
    has_final = FINAL_KEY in payload

    if has_tool and has_final:
        raise ProtocolError(
            f'object has both "{TOOL_KEY}" and "{FINAL_KEY}"; send one or the other, not both',
            FormatViolation.WRONG_VALUE_TYPE,
        )

    if has_final:
        text = payload[FINAL_KEY]
        if not isinstance(text, str):
            raise ProtocolError(
                f'"{FINAL_KEY}" must be a string, got {type(text).__name__}',
                FormatViolation.WRONG_VALUE_TYPE,
            )
        return FinalAnswer(text=text, citations=_require_citations(payload), raw=completion)

    if has_tool:
        name = payload[TOOL_KEY]
        if not isinstance(name, str) or not name.strip():
            raise ProtocolError(
                f'"{TOOL_KEY}" must be a non-empty tool name',
                FormatViolation.WRONG_VALUE_TYPE,
            )
        arguments = payload.get(ARGS_KEY, {})
        if not isinstance(arguments, dict):
            raise ProtocolError(
                f'"{ARGS_KEY}" must be a JSON object, got {type(arguments).__name__}',
                FormatViolation.WRONG_VALUE_TYPE,
            )
        return ToolCall(name=name.strip(), arguments=dict(arguments), raw=completion)

    raise ProtocolError(
        f'object has neither "{TOOL_KEY}" nor "{FINAL_KEY}": {_excerpt(completion)}',
        FormatViolation.UNKNOWN_TOP_LEVEL_KEY,
    )


def parse_tool_call(completion: str) -> ToolCall | None:
    """Extract a tool call from raw model output, or None if it is a final answer.

    Must behave identically for every model: one parser, no per-model leniency. Where the
    protocol permits a shape (e.g. a fenced code block), it permits it for everyone.

    Raises:
        ProtocolError: output looks like an attempted tool call but is not valid under the
            protocol. The caller logs this and tells the model, rather than guessing at
            the intent.
    """
    reply = parse_reply(completion)
    return reply if isinstance(reply, ToolCall) else None


def parse_final(completion: str) -> tuple[str, list[str]] | None:
    """Extract `(answer, citations)` from raw model output, or None if it is a tool call.

    Raises:
        ProtocolError: as `parse_tool_call`.
    """
    reply = parse_reply(completion)
    if isinstance(reply, ToolCall):
        return None
    return reply.text, reply.citations


def classify_violation(
    error: ProtocolError, response: ModelResponse
) -> tuple[FormatViolation, bool]:
    """Return `(violation, budget_induced)` for a failed parse.

    Truncation is checked first and overrides whatever the parser concluded, because the
    parser is looking at half a response and its verdict describes the half. A response the
    provider stopped at `max_tokens` did not break the contract; our ceiling interrupted it,
    which is why the pair is returned together and the truncation is not counted as a
    violation anywhere downstream.

    The check is `finish_reason`, not a heuristic on the text. Guessing truncation from an
    unbalanced brace would let a model earn the carve-out by emitting one.
    """
    if response.truncated:
        return FormatViolation.TRUNCATED, True
    return error.violation, False


# --------------------------------------------------------------------------------------
# Tool arguments
# --------------------------------------------------------------------------------------


def validate_arguments(
    tool_name: str, schema: Mapping[str, Any], arguments: Mapping[str, Any]
) -> None:
    """Check `arguments` against a tool's schema: required keys, no strays, right types.

    Not a JSON Schema implementation, and not trying to be: it covers required, unknown, and
    scalar types, which is every constraint the two tool schemas in this project express.
    Pulling in `jsonschema` to validate four arguments would add a dependency to enforce
    rules nobody has written.

    Reports one problem at a time, in a fixed order — the first missing required argument,
    then the first unrecognised one, then the first mistyped one — so the same bad call
    produces the same sentence every time and for both arms. The wording itself comes from
    `agent.prompts`, because the model reads it.

    Raises:
        ToolInputError: the arguments do not fit the schema. A model-caused failure, charged
            to `max_tool_errors` — not a `FormatViolation`, since the JSON was valid and it
            was the request inside it that was wrong.
    """
    properties = schema.get("properties") or {}
    required = schema.get("required") or ()

    for name in required:
        if name not in arguments:
            raise ToolInputError(
                render_missing_argument_error(tool_name, name),
                ToolErrorReason.MISSING_ARG,
            )

    for name in arguments:
        if name not in properties:
            raise ToolInputError(
                render_unknown_argument_error(tool_name, name, properties),
                ToolErrorReason.SCHEMA_INVALID,
            )

    for name, value in arguments.items():
        spec = properties.get(name)
        expected = spec.get("type") if isinstance(spec, Mapping) else None
        allowed = _JSON_TYPES.get(expected) if isinstance(expected, str) else None
        if allowed is None:
            continue
        # `True` is an `int` in Python, so a boolean passes an integer check unless it is
        # rejected first; `top_k=True` is not a top_k.
        if not isinstance(value, allowed) or (isinstance(value, bool) and bool not in allowed):
            raise ToolInputError(
                render_bad_argument_type_error(tool_name, name, str(expected), value),
                ToolErrorReason.BAD_ARG_TYPE,
            )


def chunk_ids_in(result: Any) -> list[str]:
    """Return the chunk ids a tool result carries, in order, without duplicates.

    Ids are read through `to_dict()` where a result defines one, which is the same view
    `prompts.render_tool_result` shows the model. That is the invariant worth holding: this
    list is what the model could legitimately have cited, so reading it from a different view
    of the same object than the model saw would make citation grounding score the mismatch.

    A result with no chunk ids — a web search, an empty knowledge-base lookup — yields an
    empty list rather than failing.
    """
    if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
        return []
    found: list[str] = []
    for item in result:
        to_dict = getattr(item, "to_dict", None)
        payload = to_dict() if callable(to_dict) else item
        chunk_id = payload.get("chunk_id") if isinstance(payload, Mapping) else None
        if isinstance(chunk_id, str) and chunk_id and chunk_id not in found:
            found.append(chunk_id)
    return found


def tool_specs_for(tools: Mapping[str, Tool]) -> list[dict[str, Any]]:
    """Render prompt specs from the tools the loop will actually dispatch to.

    Taken from the registry rather than from a separate list, so the prompt cannot document
    a tool the loop does not have or omit one it does. For the default registry this is
    exactly `agent.tools.tool_specs()`, which is what the manifest's digest covers.

    Public because `agent.manifest` digests the prompt this renders. Both callers must derive
    the specs the same way, or a manifest would record the digest of a prompt the agent never
    saw.
    """
    return [
        {"name": tool.name, "description": tool.description, "schema": tool.schema}
        for tool in tools.values()
    ]


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


class Agent:
    """A tool-using agent, identical for every model behind `ModelAdapter`."""

    def __init__(
        self,
        model: ModelAdapter,
        tools: dict[str, Any] | None = None,
        *,
        budgets: Budgets | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        run_id: str | None = None,
        logger: TraceLogger | None = None,
        tool_retry_policy: RetryPolicy | None = None,
        guardrails: Guardrails | None = None,
    ) -> None:
        """Wire up model, tool registry, and trace logging.

        `temperature` defaults to 0 and graded runs must leave it there. Latency is in
        milliseconds throughout, matching `ModelResponse.latency_ms` and the trace record,
        so no unit conversion happens anywhere in the loop.

        The conversation's summariser is this same model, so compaction costs the arm that
        caused it and no third model quietly edits one arm's context.

        `tool_retry_policy` governs infrastructure retries only, and exists mainly so tests
        can inject `sleep` instead of waiting out a backoff.

        `guardrails` is None for an unguarded run, which is the off arm of the ablation and the
        default everywhere else. When present it must be the *same* configuration for both arms
        — `manifest.RunManifest.guardrails_sha256` is what makes that checkable — and nothing in
        this loop branches on which model is answering.
        """
        self.model = model
        self.tools: dict[str, Tool] = dict(tools) if tools is not None else registry()
        self.budgets = budgets if budgets is not None else Budgets()
        self.guardrails = guardrails
        self.tool_retry_policy = tool_retry_policy or RetryPolicy(max_retries=DEFAULT_TOOL_RETRIES)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger
        self.run_id = run_id or (logger.run_id if logger is not None else uuid.uuid4().hex[:12])

        self.system_prompt = build_system_prompt(tool_specs_for(self.tools))
        self.conversation = Conversation(
            system_prompt=self.system_prompt,
            summariser=model,
            on_summarised=self._log_summariser_call,
        )

        self._item_id: str | None = None
        self._next_turn_idx = 0
        # Which turn's records `_log` is writing. Held on the instance so the summariser
        # callback, which memory invokes mid-assembly, lands on the right turn.
        self._turn_idx = 0
        # Model calls the summariser made while assembling the current turn's messages.
        self._summariser_usage: list[ModelResponse] = []

    # -- one turn ----------------------------------------------------------------------

    @property
    def next_turn_idx(self) -> int:
        """The index the next turn will be logged under.

        Read by `agent.session` when a changed condition forces a new agent: the conversation
        continues onto it, so its numbering has to continue too.
        """
        return self._next_turn_idx

    def resume_numbering(self, item_id: str | None, next_turn_idx: int) -> None:
        """Continue `item_id`'s numbering from `next_turn_idx` on a freshly built agent.

        Only the counter — `Conversation.adopt` moves the history. Called before the first
        `run_turn`, so that a carried conversation's turns keep ascending across a run boundary
        instead of restarting at zero and leaving two traces each holding a record keyed
        `(item_id, 0)` with nothing but a timestamp to order them by.

        A later turn under a different `item_id` still resets to zero, as always: the counter is
        per conversation, not per agent.
        """
        self._item_id = item_id
        self._next_turn_idx = next_turn_idx

    def run_turn(
        self,
        user_message: str,
        conversation: Conversation | None = None,
        *,
        item_id: str | None = None,
    ) -> AgentResult:
        """Run the loop for one user message until an answer or a budget ends it.

        `conversation` defaults to the agent's own, which is what makes a multi-turn chat
        remember anything; the eval runner passes a fresh one per case so cases cannot leak
        into each other. `item_id` labels the trace records, and restarts the turn counter
        when it changes.

        Raises:
            ModelError: the provider failed and retries were exhausted. Logged into the
                trace first, then re-raised, so the caller decides whether one bad case
                should end the run.
            Exception: anything a tool raises that is neither `ToolInputError` nor
                `ToolInfraError` is logged and then propagated. An unclassified failure is a
                gap in this harness, and booking it as one of the two known kinds would file
                it under a heading that is probably wrong and stop anyone noticing.
        """
        if item_id != self._item_id:
            self._item_id = item_id
            self._next_turn_idx = 0
        self._turn_idx = self._next_turn_idx
        self._next_turn_idx += 1

        conversation = conversation if conversation is not None else self.conversation
        started = time.perf_counter()

        self._log(ROLE_USER, user_message)
        conversation.add_user(user_message)

        if self.guardrails is not None:
            blocked = self.guardrails.screen_input(user_message)
            if blocked is not None:
                return self._input_blocked(conversation, blocked, started)

        budgets = self.budgets
        steps: list[Step] = []
        attempted: list[dict[str, Any]] = []
        retrieved: list[str] = []
        violations: Counter[str] = Counter()
        tool_errors: Counter[str] = Counter()
        consecutive_violations = 0
        successes = 0
        errors = 0
        exhausted: str | None = None
        infrastructure_error: str | None = None
        answer: FinalAnswer | None = None
        stopped_reason = STOPPED_MODEL_CALL_BUDGET

        for index in range(budgets.max_model_calls):
            messages, response = self._call_model(conversation)
            step = Step(
                index=index,
                messages_sent=messages,
                completion=response.text,
                latency_ms=response.latency_ms,
                usage={
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "reasoning_tokens": response.reasoning_tokens,
                },
                cached=response.cached,
                usd_cost=response.usd_cost,
            )
            steps.append(step)
            # Kept in the conversation whatever it turns out to be. Dropping unparseable
            # output would hide from the model what it just did, and leave memory telling a
            # different story from the trace.
            conversation.add_assistant(response.text)

            try:
                reply = parse_reply(response.text)
            except ProtocolError as exc:
                violation, budget_induced = classify_violation(exc, response)
                step.error = str(exc)
                step.format_violation = violation
                step.budget_induced = budget_induced
                violations[violation.value] += 1
                # A truncation still counts toward the consecutive-failure stop even though
                # it is not a contract violation: a model truncating every reply is making
                # no progress, and only `max_model_calls` would otherwise end the turn.
                consecutive_violations += 1
                note = f"{FORMAT_VIOLATION}: {exc}"
                if consecutive_violations >= MAX_CONSECUTIVE_VIOLATIONS:
                    self._log(
                        ROLE_TOOL,
                        "",
                        error=note,
                        format_violation=violation.value,
                        budget_induced=budget_induced,
                    )
                    stopped_reason = STOPPED_PROTOCOL_ERROR
                    break
                # Re-prompt with the error itself, in the same words for every model.
                reprompt = render_tool_error(None, str(exc))
                conversation.add_tool_result(reprompt)
                self._log(
                    ROLE_TOOL,
                    reprompt,
                    error=note,
                    format_violation=violation.value,
                    budget_induced=budget_induced,
                )
                continue

            consecutive_violations = 0

            if isinstance(reply, FinalAnswer):
                answer = reply
                stopped_reason = STOPPED_ANSWERED
                break

            step.tool_call = reply
            spent = self._spent_budget(successes, errors)
            if spent is not None:
                exhausted = exhausted or spent
                note = f"{spent}: no further tool calls this turn"
                step.error = note
                conversation.add_tool_result(FINAL_ANSWER_REQUIRED)
                self._log(
                    ROLE_TOOL,
                    FINAL_ANSWER_REQUIRED,
                    tool_calls=[_call_record(reply)],
                    error=note,
                )
                continue

            attempted.append(_call_record(reply))
            try:
                outcome = self._run_tool(reply, step)
            except ToolInfraError as exc:
                # Retries are already spent by here. The item ends, and is excluded from axis
                # scoring rather than scored on whatever the model managed before our tool
                # broke (README.md).
                infrastructure_error = f"{type(exc).__name__}: {exc}"
                step.error = infrastructure_error
                stopped_reason = STOPPED_INFRASTRUCTURE_FAILED
                self._log(
                    ROLE_TOOL,
                    "",
                    tool_calls=[_call_record(reply)],
                    error=infrastructure_error,
                    infrastructure_failed=True,
                )
                break

            if outcome.reason is None:
                successes += 1
            else:
                errors += 1
                tool_errors[outcome.reason.value] += 1

            for chunk_id in outcome.chunk_ids:
                if chunk_id not in retrieved:
                    retrieved.append(chunk_id)
            conversation.add_tool_result(outcome.rendered)
            self._log(
                ROLE_TOOL,
                outcome.rendered,
                tool_calls=[_call_record(reply)],
                retrieved_chunk_ids=outcome.chunk_ids,
                error=outcome.error,
                tool_error_reason=outcome.reason.value if outcome.reason else None,
            )

        if answer is None and exhausted is not None and stopped_reason == STOPPED_MODEL_CALL_BUDGET:
            # A spent tool budget is what forced the ending, even though the model-call
            # ceiling is what the loop hit on the way out; naming the budget that ran out
            # first is the more useful diagnosis.
            stopped_reason = exhausted

        # Screened after the loop rather than at the `break`, so the stages see the turn's
        # finished answer and its full retrieval list, and so the loop body stays the loop body.
        # Output screen before grounding: an unsafe answer is the more serious finding, and one
        # turn yields one action.
        decision: GuardrailDecision | None = None
        if self.guardrails is not None and answer is not None:
            decision = self.guardrails.screen_output(answer.text) or (
                self.guardrails.enforce_grounding(retrieved, answer.text)
            )

        truncations = violations[FormatViolation.TRUNCATED.value]
        latency_ms = (time.perf_counter() - started) * 1000
        tokens, usd_cost = self._usage_totals(steps)
        result = AgentResult(
            final_text=answer.text if answer is not None else "",
            steps=steps,
            stopped_reason=stopped_reason,
            run_id=self.run_id,
            citations=list(answer.citations) if answer is not None else [],
            tool_calls=attempted,
            retrieved_chunk_ids=retrieved,
            latency_ms=latency_ms,
            tokens=tokens,
            format_violations=sum(violations.values()) - truncations,
            format_violation=_first_violation(steps),
            format_violations_by_type=dict(violations),
            budget_induced_truncations=truncations,
            tool_errors=errors,
            tool_errors_by_type=dict(tool_errors),
            infrastructure_failed=stopped_reason == STOPPED_INFRASTRUCTURE_FAILED,
            infrastructure_error=infrastructure_error,
            usd_cost=usd_cost,
            guardrail_action=(
                decision.action.value if decision is not None else GuardrailAction.NONE.value
            ),
            guardrail_stage=decision.stage.value if decision is not None else None,
            guardrail_completion=decision.completion if decision is not None else None,
        )
        if decision is not None:
            self._log_guardrail(decision)
        self._log(
            ROLE_TURN,
            # The model's own text, whatever a guardrail decided. `deterministic.item_views`
            # reads the scored response from here, so this is the record that keeps our
            # substituted sentence out of every scorer — see `agent.guardrails`.
            result.final_text,
            tool_calls=attempted,
            retrieved_chunk_ids=retrieved,
            latency_ms=latency_ms,
            prompt_tokens=result.tokens["prompt"],
            completion_tokens=result.tokens["completion"],
            reasoning_tokens=result.tokens["reasoning"],
            usd_cost=result.usd_cost,
            error=None if stopped_reason == STOPPED_ANSWERED else stopped_reason,
            format_violation=result.format_violation.value if result.format_violation else None,
            budget_induced=truncations > 0,
            infrastructure_failed=result.infrastructure_failed,
            guardrail_action=result.guardrail_action,
            guardrail_stage=result.guardrail_stage,
        )
        return result

    def _input_blocked(
        self,
        conversation: Conversation,
        decision: GuardrailDecision,
        started: float,
    ) -> AgentResult:
        """End a turn at the input screen, before the candidate has been called at all.

        Zero model calls and zero tool calls, so the block costs the candidate none of its
        budget. That is the property the stage exists for: screening that spent
        `max_model_calls` would leave the guarded arm with less reasoning budget than the
        unguarded one, and the shortfall would surface as a budget stop that looked like the
        model's failure.

        The substituted completion is added to the conversation, because it is what the user
        was shown: a following turn conditioned on a history the user never saw would measure a
        system nobody ran. It does mean the model's later turns in a multi-turn item are
        conditioned on our sentence, which is the reason the *model's own output* reading comes
        from the guardrails-off arm rather than from this trace.
        """
        conversation.add_assistant(decision.completion)
        self._log_guardrail(decision)

        latency_ms = (time.perf_counter() - started) * 1000
        result = AgentResult(
            # Empty, as for any turn with no answer. The delivered text is on
            # `guardrail_completion`, and `delivered_text` puts the two together.
            final_text="",
            steps=[],
            stopped_reason=STOPPED_INPUT_BLOCKED,
            run_id=self.run_id,
            latency_ms=latency_ms,
            tokens={"prompt": 0, "completion": 0, "total": 0},
            # A known zero rather than None. None means "no model call reported a price", which
            # is a gap in our pricing table; this turn made no model call, so zero is a
            # measurement and `run_cost` should not count it as unpriced.
            usd_cost=0.0,
            guardrail_action=decision.action.value,
            guardrail_stage=decision.stage.value,
            guardrail_completion=decision.completion,
        )
        self._log(
            ROLE_TURN,
            "",
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            usd_cost=0.0,
            error=STOPPED_INPUT_BLOCKED,
            guardrail_action=result.guardrail_action,
            guardrail_stage=result.guardrail_stage,
        )
        return result

    def _log_guardrail(self, decision: GuardrailDecision) -> None:
        """Record one guardrail decision under its own role.

        The delivered text lives in this record's `content`, and the model's own output lives in
        the `role="turn"` record's. Both readings are therefore recoverable from a single
        guardrails-on trace, and no scorer has to be trusted to pick the right one.

        `decision.matched` is *not* written. It is not a failure, so `error` would be the wrong
        field, and it is recoverable exactly rather than approximately: the screened text is in
        the `user` or `turn` record, the patterns are in `GUARDRAIL_PATTERNS`, and the manifest
        records which version of them ran. A field for a value already derivable from two
        recorded ones is a third place for it to disagree.

        No latency and no cost: every stage is rule-based. When one is not, its measurements
        belong here, where `latency_aggregates` (which reads `MODEL_CALL_ROLES`) and `run_cost`
        (which reads `role="turn"`) will both leave them out of the candidate's figures.
        """
        self._log(
            ROLE_GUARDRAIL,
            decision.completion,
            guardrail_action=decision.action.value,
            guardrail_stage=decision.stage.value,
        )

    def _spent_budget(self, successes: int, errors: int) -> str | None:
        """Return the `stopped_reason` for whichever tool budget is spent, or None.

        Checked before dispatch, so the model is told to answer instead of having a call
        refused without explanation. Successes are checked first only to make the message
        deterministic when both are spent; the counts themselves are independent.
        """
        if successes >= self.budgets.max_tool_calls:
            return STOPPED_TOOL_BUDGET
        if errors >= self.budgets.max_tool_errors:
            return STOPPED_TOOL_ERROR_BUDGET
        return None

    # -- pieces of it ------------------------------------------------------------------

    def _call_model(self, conversation: Conversation) -> tuple[list[ChatMessage], ModelResponse]:
        """One `model.generate` call, with the result logged verbatim before any parsing.

        Timing and cost come back on the `ModelResponse`; a cached response is flagged
        there, so the trace distinguishes a replay from a fresh measurement.

        Returns the messages sent alongside the response, since the exact prompt is part of
        the step record and building it can itself compact the conversation.
        """
        before = len(conversation.truncations())
        messages = conversation.to_messages()
        for record in conversation.truncations()[before:]:
            self._log(ROLE_MEMORY, json.dumps(record, default=str))

        try:
            response = self.model.generate(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except ModelError as exc:
            # An assistant turn that never happened, recorded as one so the trace shows the
            # gap and its cause rather than simply ending.
            self._log(ROLE_ASSISTANT, "", error=f"model call failed: {exc}")
            raise

        self._log(
            ROLE_ASSISTANT,
            response.text,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            usd_cost=response.usd_cost,
            cached=response.cached,
            reasoning_tokens=response.reasoning_tokens,
            usage=response.raw.get("usage") if isinstance(response.raw, dict) else None,
        )
        return messages, response

    def _run_tool(self, call: ToolCall, step: Step) -> ToolOutcome:
        """Dispatch `call`, with infrastructure retries, and render the outcome.

        A model-caused failure is rendered and returned rather than raised: a tool the model
        misused is information it can act on, and ending the turn would throw away the rest
        of it. An infrastructure failure that survives its retries is raised, because there is
        nothing the model can do about it and the item is no longer measuring the model.

        Raises:
            ToolInfraError: the tool kept failing for reasons outside the model's control.
            Exception: anything else, after logging it. Not converted into either known kind:
                an unclassified failure means this harness is missing a case, and filing it
                under a guess is how that goes unnoticed for a hundred items.
        """
        try:
            result = self._dispatch_with_retries(call)
        except ToolInputError as exc:
            step.error = str(exc)
            step.tool_error = exc.reason
            return ToolOutcome(
                rendered=render_tool_error(call.name, str(exc)),
                error=str(exc),
                reason=exc.reason,
            )
        except ToolInfraError:
            raise
        except Exception as exc:
            detail = f"unexpected failure in {call.name}: {type(exc).__name__}: {exc}"
            step.error = detail
            self._log(ROLE_TOOL, "", tool_calls=[_call_record(call)], error=detail)
            raise

        step.tool_result = result
        return ToolOutcome(
            rendered=render_tool_result(call.name, result),
            chunk_ids=chunk_ids_in(result),
        )

    def _dispatch_with_retries(self, call: ToolCall) -> Any:
        """Dispatch `call`, retrying infrastructure failures with backoff.

        Retries are charged to no budget. That is the point: a tool that timed out twice has
        told us nothing about the model, so spending the model's tool calls on our flakiness
        would make one arm's score depend on how the network behaved that afternoon. Each
        attempt is logged, so the retries are visible rather than inferred from a delay.

        Raises:
            ToolInfraError: every attempt failed. The caller ends the item.
            ToolInputError: the call was invalid, which no amount of retrying fixes.
        """
        policy = self.tool_retry_policy
        attempt = 0
        while True:
            try:
                return self._dispatch(call)
            except ToolInfraError as exc:
                attempt += 1
                if attempt > policy.max_retries:
                    raise
                self._log(
                    ROLE_TOOL,
                    "",
                    tool_calls=[_call_record(call)],
                    error=f"infrastructure failure, retry {attempt}/{policy.max_retries}: {exc}",
                    infrastructure_failed=False,
                )
                policy.sleep(policy.delay_for(attempt))

    def _dispatch(self, call: ToolCall) -> Any:
        """Execute a parsed tool call.

        An unknown name is a `ToolInputError` and deliberately not a `FORMAT_VIOLATION`: the
        JSON was fine and the model invented a capability, which
        `evals.deterministic.check_no_hallucinated_tool` scores as its own failure mode.

        Raises:
            ToolInputError: unknown tool name, or arguments that do not match its schema.
        """
        tool = self.tools.get(call.name)
        if tool is None:
            raise ToolInputError(
                render_unknown_tool_error(call.name, sorted(self.tools)),
                ToolErrorReason.UNKNOWN_TOOL,
            )
        validate_arguments(call.name, tool.schema, call.arguments)
        return tool(**call.arguments)

    # -- bookkeeping -------------------------------------------------------------------

    def _log_summariser_call(self, request: str, response: ModelResponse) -> None:
        """Record a compaction model call and count it into the turn's totals."""
        self._summariser_usage.append(response)
        self._log(
            ROLE_SUMMARISER,
            response.text,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            usd_cost=response.usd_cost,
            cached=response.cached,
            reasoning_tokens=response.reasoning_tokens,
            usage=response.raw.get("usage") if isinstance(response.raw, dict) else None,
        )

    def _usage_totals(self, steps: Sequence[Step]) -> tuple[dict[str, int], float | None]:
        """Sum tokens and cost over the turn, including the summariser's own calls.

        A provider that reported no token usage contributes zero rather than making the total
        None: a partial count is still comparable across arms, and the per-call nulls are in
        the trace for anyone who needs to know it was partial. Cost is different — None when
        nothing reported one, matching `agent.models.base.estimate_usd_cost`, because a zero
        would understate a run's cost instead of admitting it is unknown.

        `total` counts reasoning tokens, because the provider billed them: a total that
        omitted them would understate the arm that thinks and leave the two arms' token
        figures on different definitions. `completion` stays the visible reply, so the split
        is still readable from the trace rather than only the sum.

        Clears the summariser's calls, so each turn is charged for its own compaction only.
        """
        prompt_counts = [step.usage.get("prompt_tokens") for step in steps]
        completion_counts = [step.usage.get("completion_tokens") for step in steps]
        reasoning_counts = [step.usage.get("reasoning_tokens") for step in steps]
        costs = [step.usd_cost for step in steps]
        for response in self._summariser_usage:
            prompt_counts.append(response.prompt_tokens)
            completion_counts.append(response.completion_tokens)
            reasoning_counts.append(response.reasoning_tokens)
            costs.append(response.usd_cost)
        self._summariser_usage.clear()

        prompt = sum(int(count or 0) for count in prompt_counts)
        completion = sum(int(count or 0) for count in completion_counts)
        reasoning = sum(int(count or 0) for count in reasoning_counts)
        reported = [cost for cost in costs if cost is not None]
        return (
            {
                "prompt": prompt,
                "completion": completion,
                "reasoning": reasoning,
                "total": prompt + completion + reasoning,
            },
            sum(reported) if reported else None,
        )

    def _log(self, role: str, content: str, **fields: Any) -> None:
        """Append one trace record for the turn in progress, if a logger was given."""
        if self.logger is None:
            return
        self.logger.log(self._item_id, self._turn_idx, role, content, **fields)


def _call_record(call: ToolCall) -> dict[str, Any]:
    """Render a parsed call for the trace and for `AgentResult.tool_calls`."""
    return {"name": call.name, "arguments": call.arguments}


def _first_violation(steps: Sequence[Step]) -> FormatViolation | None:
    """Return the turn's first format violation, or None if every completion parsed.

    The turn-level headline. A turn can fail twice and in two different ways, so the full
    picture is `format_violations_by_type`; this is the one a per-item table shows.
    """
    return next((step.format_violation for step in steps if step.format_violation), None)
