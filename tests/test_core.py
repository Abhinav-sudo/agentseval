"""Tests for `agent.core`.

Three groups, in the order the loop uses them.

The parser tests are the largest group on purpose. It is the component the whole comparison
rests on: it decides what counts as a format violation, and a parser that is accidentally
lenient in one direction would flatter whichever model deviates that way. So the tolerated
deviations are pinned individually, and so are the failures.

The loop tests drive a scripted `FakeAdapter` through the paths that end a turn — an answer,
each of the three budgets, repeated unparseable output, a tool that will not work — and assert
both what the model was sent and what reached the trace. Nothing here calls a provider or an
embedding model.

The budget tests are written to fail if the three counters are ever merged back into one, and
the truncation tests to fail if a response cut off at `max_tokens` is ever counted against the
model. Both are cheap to break by accident and neither would look wrong in a report.

The last group is the harness guarantee: one loop, no branch anywhere on which model is
answering, in the spirit of the prompt-fairness tests in `test_prompts.py`.
"""

from __future__ import annotations

import inspect
import io
import json
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent import core, memory
from agent.core import (
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TOOL_ERRORS,
    FORMAT_VIOLATION,
    STOPPED_ANSWERED,
    STOPPED_INFRASTRUCTURE_FAILED,
    STOPPED_MODEL_CALL_BUDGET,
    STOPPED_PROTOCOL_ERROR,
    STOPPED_TOOL_BUDGET,
    STOPPED_TOOL_ERROR_BUDGET,
    Agent,
    Budgets,
    FinalAnswer,
    FormatViolation,
    ProtocolError,
    ToolCall,
    chunk_ids_in,
    classify_violation,
    extract_json_object,
    parse_final,
    parse_reply,
    parse_tool_call,
    strip_code_fence,
    validate_arguments,
)
from agent.memory import Conversation
from agent.models.base import FinishReason, ModelAdapter, ModelError, ModelResponse, RetryPolicy
from agent.prompts import (
    FINAL_ANSWER_REQUIRED,
    KB_TOOL,
    PROTOCOL_ERROR_PREFIX,
    TOOL_BUDGET_PREFIX,
    TOOL_RESULT_PREFIX,
    WEB_TOOL,
)
from agent.tools import ToolErrorReason, ToolInfraError, ToolInputError, registry, tool_specs
from agent.tools.lookup_kb import CITATION_FORMAT, Chunk, Hit, parse_citations
from agent.trace import TraceLogger, read_records
from tests.fakes import FakeAdapter, Reply

CHUNK_ID = "sleep-hygiene.md#2"
OTHER_CHUNK_ID = "hydration.md#4"


def tool_call(name: str = KB_TOOL, **args: Any) -> str:
    return json.dumps({"tool": name, "args": args or {"query": "sleep"}})


def final(text: str = "Sleep in a dark room.", citations: list[str] | None = None) -> str:
    return json.dumps({"final": text, "citations": citations if citations is not None else []})


def kb_payload(chunk_id: str = CHUNK_ID) -> list[dict[str, Any]]:
    """A knowledge-base result shaped like the one `lookup_kb` returns."""
    return [{"chunk_id": chunk_id, "score": 0.7, "text": "Keep the room dark and cool."}]


@dataclass
class FakeTool:
    """A tool that records its calls, and returns or raises whatever a test wants."""

    name: str
    description: str = "A tool used in tests."
    schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        }
    )
    result: Any = field(default_factory=kb_payload)
    raises: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return self.result


@dataclass
class FlakyTool(FakeTool):
    """Fails for infrastructure reasons `failures` times, then works.

    The interesting case for retries: a transient failure must not cost the model anything,
    and must not end an item that would have succeeded on the second attempt.
    """

    failures: int = 1
    attempts: int = 0

    def __call__(self, **kwargs: Any) -> Any:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ToolInfraError("index temporarily unreachable")
        return super().__call__(**kwargs)


def tools(**overrides: FakeTool) -> dict[str, FakeTool]:
    """The real tool inventory, faked. Both names are required by the system prompt."""
    inventory = {KB_TOOL: FakeTool(KB_TOOL), WEB_TOOL: FakeTool(WEB_TOOL, result={"results": []})}
    inventory.update(overrides)
    return inventory


def build(
    *completions: str | Reply | Exception,
    inventory: dict[str, FakeTool] | None = None,
    **kwargs: Any,
) -> tuple[Agent, FakeAdapter, dict[str, FakeTool]]:
    """An agent wired to a scripted adapter and fake tools.

    Infrastructure retries never sleep here: the policy's delay is exercised in
    `test_models.py`, and waiting out a real backoff would add seconds to this suite.
    """
    adapter = FakeAdapter(list(completions))
    inventory = inventory if inventory is not None else tools()
    kwargs.setdefault("tool_retry_policy", RetryPolicy(max_retries=2, jitter=False, sleep=noop))
    agent = Agent(adapter, dict(inventory), run_id="test-run", **kwargs)
    return agent, adapter, inventory


def noop(_seconds: float) -> None:
    """A `sleep` that does not."""


# --------------------------------------------------------------------------------------
# The parser: what it accepts
# --------------------------------------------------------------------------------------


def test_a_clean_tool_call_parses():
    call = parse_reply('{"tool": "lookup_kb", "args": {"query": "sleep", "top_k": 3}}')
    assert call == ToolCall(
        name="lookup_kb",
        arguments={"query": "sleep", "top_k": 3},
        raw='{"tool": "lookup_kb", "args": {"query": "sleep", "top_k": 3}}',
    )


def test_a_clean_answer_parses_with_its_citations():
    reply = parse_reply(final("Dark and cool.", [CHUNK_ID]))
    assert isinstance(reply, FinalAnswer)
    assert reply.text == "Dark and cool."
    assert reply.citations == [CHUNK_ID]


@pytest.mark.parametrize("fence", ["```json", "```JSON", "```"])
def test_a_fenced_object_is_accepted_because_the_protocol_says_so(fence):
    completion = f'{fence}\n{{"final": "hi", "citations": []}}\n```'
    assert parse_final(completion) == ("hi", [])


def test_the_fence_stripper_leaves_unfenced_text_alone():
    assert strip_code_fence('{"final": "hi"}') == '{"final": "hi"}'


def test_prose_around_the_object_is_recoverable_rather_than_a_total_failure():
    """A model that narrates before complying has deviated, not failed; scoring the two the
    same would overstate the gap between sloppy framing and unusable output."""
    completion = 'Sure, let me look that up.\n{"tool": "lookup_kb", "args": {"query": "x"}}\nDone.'
    call = parse_tool_call(completion)
    assert call is not None
    assert call.arguments == {"query": "x"}


def test_braces_in_an_answer_are_content_not_structure():
    answer = 'Reply with {"tool": ...} to call a tool. Nested {braces} are fine.'
    reply = parse_reply(json.dumps({"final": answer, "citations": []}))
    assert isinstance(reply, FinalAnswer)
    assert reply.text == answer


@pytest.mark.parametrize(
    "answer",
    [
        "Use } to close a block.",
        'It said "}" and stopped.',
        'A quote then a brace: "\\" }',
        "{ unmatched open brace",
    ],
)
def test_a_lone_brace_inside_a_string_does_not_end_the_object_early(answer):
    """The brace scanner only runs on output that needed recovering, so this goes through the
    search path deliberately: counting braces without tracking string state and backslash
    escapes stops at the first `}` inside a value, and the object then fails to parse."""
    completion = "Here you go:\n" + json.dumps({"final": answer, "citations": [CHUNK_ID]})
    reply = parse_reply(completion)

    assert isinstance(reply, FinalAnswer)
    assert reply.text == answer
    assert reply.citations == [CHUNK_ID]


def test_a_brace_in_prose_before_the_object_is_skipped():
    """The first `{` is not always the object; every candidate is tried."""
    completion = 'Using {lookup_kb} now.\n{"tool": "lookup_kb", "args": {"query": "x"}}'
    call = parse_tool_call(completion)
    assert call is not None and call.name == "lookup_kb"


def test_args_may_be_omitted_for_a_call_that_needs_none():
    assert parse_tool_call('{"tool": "lookup_kb"}') == ToolCall(
        name="lookup_kb", arguments={}, raw='{"tool": "lookup_kb"}'
    )


def test_surrounding_whitespace_is_not_a_violation():
    assert parse_final('\n\n  {"final": "hi", "citations": []}  \n') == ("hi", [])


def test_an_empty_citations_array_is_valid_and_means_nothing_was_cited():
    assert parse_final(final("I could not find that.", [])) == ("I could not find that.", [])


def test_newlines_and_unicode_survive_the_parse():
    text = "Line one.\nLine two — with an em dash."
    assert parse_final(json.dumps({"final": text, "citations": []})) == (text, [])


# --------------------------------------------------------------------------------------
# The parser: what it rejects
# --------------------------------------------------------------------------------------


#: Every rejection the parser can produce, with the message fragment a model would read and
#: the taxonomy member the report counts. Both are pinned per case: the message is what the
#: model has to recover from, and the type is what the finding is stated in. A raise site that
#: acquires the wrong member is invisible in the message and wrong in every rate derived from
#: it, which is why they are asserted together rather than in two tables.
REJECTIONS = [
    ("I will look that up for you.", "found none", FormatViolation.UNPARSEABLE_JSON),
    ("", "found none", FormatViolation.UNPARSEABLE_JSON),
    (
        '{"tool": "lookup_kb", "args": {"query": "x"}',
        "could not parse",
        FormatViolation.UNPARSEABLE_JSON,
    ),
    ('{"tool": "lookup_kb",, "args": {}}', "could not parse", FormatViolation.UNPARSEABLE_JSON),
    ("[1, 2, 3]", "expected one JSON object", FormatViolation.UNPARSEABLE_JSON),
    ('"just a string"', "expected one JSON object", FormatViolation.UNPARSEABLE_JSON),
    ('{"thought": "I should search"}', "neither", FormatViolation.UNKNOWN_TOP_LEVEL_KEY),
    (
        '{"tool": "lookup_kb", "final": "done", "citations": []}',
        "both",
        FormatViolation.WRONG_VALUE_TYPE,
    ),
    ('{"tool": "", "args": {}}', "non-empty tool name", FormatViolation.WRONG_VALUE_TYPE),
    ('{"tool": 7, "args": {}}', "non-empty tool name", FormatViolation.WRONG_VALUE_TYPE),
    (
        '{"tool": "lookup_kb", "args": "query=x"}',
        "must be a JSON object",
        FormatViolation.WRONG_VALUE_TYPE,
    ),
    (
        '{"tool": "lookup_kb", "args": null}',
        "must be a JSON object",
        FormatViolation.WRONG_VALUE_TYPE,
    ),
    ('{"final": ["done"], "citations": []}', "must be a string", FormatViolation.WRONG_VALUE_TYPE),
    ('{"final": "done"}', "required on every answer", FormatViolation.MISSING_CITATIONS_KEY),
    (
        '{"final": "done", "citations": "sleep.md#1"}',
        "array of chunk id strings",
        FormatViolation.CITATIONS_WRONG_TYPE,
    ),
    (
        '{"final": "done", "citations": [1]}',
        "array of chunk id strings",
        FormatViolation.CITATIONS_WRONG_TYPE,
    ),
]


@pytest.mark.parametrize(("completion", "expected", "violation"), REJECTIONS)
def test_output_outside_the_protocol_is_a_protocol_error(completion, expected, violation):
    with pytest.raises(ProtocolError, match=expected) as raised:
        parse_reply(completion)
    assert raised.value.violation is violation


def test_every_taxonomy_member_except_truncation_is_reachable_from_a_parse():
    """`TRUNCATED` is classified by the loop from `finish_reason`, not by the parser. Every
    other member must be produced by some real rejection, or it is a label for nothing."""
    reachable = {violation for _, _, violation in REJECTIONS}
    assert reachable == set(FormatViolation) - {FormatViolation.TRUNCATED}


def test_a_missing_citations_array_is_not_treated_as_an_empty_one():
    """`[]` says nothing was cited; a missing key says the model forgot the contract. Citation
    grounding rests on telling those apart, so they cannot be collapsed."""
    with pytest.raises(ProtocolError):
        parse_reply('{"final": "Sleep is important."}')
    assert parse_final('{"final": "Sleep is important.", "citations": []}') == (
        "Sleep is important.",
        [],
    )


def test_the_error_quotes_the_output_so_the_trace_shows_what_arrived():
    with pytest.raises(ProtocolError, match="I refuse to use JSON"):
        parse_reply("I refuse to use JSON.")


def test_a_rambling_completion_is_excerpted_rather_than_quoted_whole():
    """The error text is fed back to the model; an unbounded quote would crowd the window."""
    with pytest.raises(ProtocolError) as raised:
        parse_reply("no json here " * 500)
    assert len(str(raised.value)) < 400
    assert str(raised.value).endswith("...")


def test_extract_json_object_returns_the_object_itself():
    assert extract_json_object('```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}


def test_the_two_narrow_parsers_agree_with_the_general_one():
    assert parse_tool_call(final()) is None
    assert parse_final(tool_call()) is None


# --------------------------------------------------------------------------------------
# Truncation is ours, not the model's
# --------------------------------------------------------------------------------------


def response(text: str, finish_reason: FinishReason) -> ModelResponse:
    return ModelResponse(
        text=text,
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1.0,
        usd_cost=None,
        raw={},
        finish_reason=finish_reason,
    )


def test_a_parse_failure_on_a_truncated_response_is_reclassified_as_budget_induced():
    """The parser sees half an object and calls it unparseable, correctly, about the half it
    was given. `finish_reason` is what says the other half was cut off by our token ceiling,
    which makes it our failure and not a fact about the model."""
    error = ProtocolError("could not parse", FormatViolation.UNPARSEABLE_JSON)
    violation, budget_induced = classify_violation(error, response('{"fin', FinishReason.LENGTH))

    assert violation is FormatViolation.TRUNCATED
    assert budget_induced is True


@pytest.mark.parametrize(
    "finish_reason",
    [
        FinishReason.COMPLETE,
        FinishReason.STOP_SEQUENCE,
        FinishReason.CONTENT_FILTER,
        FinishReason.OTHER,
        FinishReason.UNKNOWN,
    ],
)
def test_only_the_length_stop_reason_earns_the_carve_out(finish_reason):
    """`UNKNOWN` included: a provider that said nothing must not be given the benefit of the
    doubt, or every adapter that forgets to map its stop values launders violations."""
    error = ProtocolError("neither", FormatViolation.UNKNOWN_TOP_LEVEL_KEY)
    violation, budget_induced = classify_violation(error, response("{}", finish_reason))

    assert violation is FormatViolation.UNKNOWN_TOP_LEVEL_KEY
    assert budget_induced is False


def test_truncation_is_not_charged_to_the_format_violation_rate():
    truncated = Reply('{"final": "Sleep in a dark roo', FinishReason.LENGTH)
    agent, _, _ = build(truncated, final("Dark and cool.", []))
    result = agent.run_turn("q")

    assert result.format_violations == 0
    assert result.budget_induced_truncations == 1
    assert result.format_violation is FormatViolation.TRUNCATED
    assert result.format_violations_by_type == {FormatViolation.TRUNCATED.value: 1}
    assert result.stopped_reason == STOPPED_ANSWERED


def test_a_truncated_reply_is_still_re_prompted_and_still_ends_a_stalled_turn():
    """Not the model's fault, but not progress either: a model truncating every reply would
    otherwise run to the model-call ceiling before anything noticed."""
    agent, adapter, _ = build(Reply('{"final": "half an ans', FinishReason.LENGTH))
    result = agent.run_turn("q")

    assert PROTOCOL_ERROR_PREFIX in adapter.prompt(1)
    assert adapter.count == 2
    assert result.stopped_reason == STOPPED_PROTOCOL_ERROR
    assert result.format_violations == 0
    assert result.budget_induced_truncations == 2


def test_a_truncation_and_a_real_violation_are_counted_separately_in_one_turn():
    agent, _, _ = build(
        Reply('{"final": "cut off', FinishReason.LENGTH),
        '{"thought": "hmm"}',
        final("Dark and cool.", []),
    )
    result = agent.run_turn("q")

    assert result.format_violations == 1
    assert result.budget_induced_truncations == 1
    assert result.format_violations_by_type == {
        FormatViolation.TRUNCATED.value: 1,
        FormatViolation.UNKNOWN_TOP_LEVEL_KEY.value: 1,
    }
    # The headline is the first of the turn; the breakdown carries the rest.
    assert result.format_violation is FormatViolation.TRUNCATED


def test_a_truncated_response_that_still_parses_is_not_a_violation_at_all():
    """A model can finish its object and be cut off in whatever followed. Nothing failed."""
    agent, _, _ = build(Reply(final("Dark and cool.", []) + " and then some", FinishReason.LENGTH))
    result = agent.run_turn("q")

    assert result.stopped_reason == STOPPED_ANSWERED
    assert result.format_violations == 0
    assert result.budget_induced_truncations == 0


def test_a_provider_that_reported_no_stop_reason_gets_no_carve_out_in_the_loop():
    """The whole-loop version of the same rule: an adapter that does not map its provider's
    stop values must not thereby launder every violation its model produces into ours."""
    agent, _, _ = build(Reply("I refuse to use JSON.", FinishReason.UNKNOWN), final("Fine.", []))
    result = agent.run_turn("q")

    assert result.format_violations == 1
    assert result.budget_induced_truncations == 0
    assert result.format_violation is FormatViolation.UNPARSEABLE_JSON


def test_the_carve_out_reads_the_finish_reason_and_not_the_shape_of_the_text():
    """Otherwise a model could earn the exemption by emitting an unbalanced brace."""
    agent, _, _ = build('{"final": "looks truncated', final("Recovered.", []))
    result = agent.run_turn("q")

    assert result.format_violations == 1
    assert result.budget_induced_truncations == 0


def failures_in(written: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-call failure records, excluding the turn's own summary of them."""
    return [r for r in written if r["format_violation"] and r["role"] != "turn"]


def test_the_trace_records_the_violation_type_and_whether_it_was_budget_induced(tmp_path):
    written = records(tmp_path, Reply('{"final": "cut', FinishReason.LENGTH), final("Fine.", []))
    (failure,) = failures_in(written)

    assert failure["format_violation"] == FormatViolation.TRUNCATED.value
    assert failure["budget_induced"] is True


def test_the_trace_types_a_real_violation_without_marking_it_budget_induced(tmp_path):
    written = records(tmp_path, '{"final": "no citations"}', final("Fine.", []))
    (failure,) = failures_in(written)

    assert failure["format_violation"] == FormatViolation.MISSING_CITATIONS_KEY.value
    assert failure["budget_induced"] is False


def test_a_turn_with_no_violation_leaves_the_typed_fields_null(tmp_path):
    """Null rather than a falsy default, so "no violation" and "a violation of type ''" cannot
    be confused by whatever aggregates these."""
    written = records(tmp_path, final("Dark and cool.", []))

    assert failures_in(written) == []
    assert all(r["format_violation"] is None for r in written)
    assert all(r["tool_error_reason"] is None for r in written)


# --------------------------------------------------------------------------------------
# Argument validation and chunk ids
# --------------------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer"},
        "min_score": {"type": "number"},
        "verbose": {"type": "boolean"},
    },
    "required": ["query"],
}


def test_valid_arguments_pass():
    validate_arguments(
        KB_TOOL, SCHEMA, {"query": "x", "top_k": 3, "min_score": 0.5, "verbose": True}
    )


def test_an_integer_argument_accepts_an_integer_but_not_a_boolean():
    """`True` is an `int` in Python, and `top_k=True` is not a search anyone asked for."""
    validate_arguments(KB_TOOL, SCHEMA, {"query": "x", "top_k": 1})
    with pytest.raises(ToolInputError, match="must be an integer, got True"):
        validate_arguments(KB_TOOL, SCHEMA, {"query": "x", "top_k": True})


def test_a_number_argument_accepts_an_integer():
    validate_arguments(KB_TOOL, SCHEMA, {"query": "x", "min_score": 1})


@pytest.mark.parametrize(
    ("arguments", "expected", "reason"),
    [
        ({}, "missing required argument 'query'", ToolErrorReason.MISSING_ARG),
        (
            {"query": "x", "k": 3},
            "'k' is not an argument of this tool; "
            "valid arguments: query, top_k, min_score, verbose",
            ToolErrorReason.SCHEMA_INVALID,
        ),
        ({"query": 3}, "'query' must be a string, got 3", ToolErrorReason.BAD_ARG_TYPE),
        (
            {"query": "x", "top_k": "three"},
            "'top_k' must be an integer, got 'three'",
            ToolErrorReason.BAD_ARG_TYPE,
        ),
        (
            {"query": "x", "verbose": "yes"},
            "'verbose' must be a boolean, got 'yes'",
            ToolErrorReason.BAD_ARG_TYPE,
        ),
    ],
)
def test_arguments_that_do_not_fit_the_schema_are_rejected(arguments, expected, reason):
    """A tool error, not a format violation: the JSON was valid and the request inside it was
    not. The typed `reason` is what the report breaks the error rate down by."""
    with pytest.raises(ToolInputError) as raised:
        validate_arguments(KB_TOOL, SCHEMA, arguments)

    assert str(raised.value) == f"{KB_TOOL}: {expected}"
    assert raised.value.reason is reason


def test_a_bad_argument_error_quotes_the_value_that_arrived():
    """`3` and `'3'` are the entire difference, and "got string" hides it."""
    with pytest.raises(ToolInputError, match=r"'top_k' must be an integer, got '3'"):
        validate_arguments(KB_TOOL, SCHEMA, {"query": "x", "top_k": "3"})


def test_an_unknown_argument_error_lists_what_the_tool_does_accept():
    with pytest.raises(ToolInputError, match="valid arguments: query, top_k, min_score, verbose"):
        validate_arguments(KB_TOOL, SCHEMA, {"query": "x", "k": 3})


def test_a_missing_argument_is_reported_before_an_unknown_one():
    """One problem at a time and always the same one, so both arms read the same sentence."""
    with pytest.raises(ToolInputError, match="missing required argument 'query'"):
        validate_arguments(KB_TOOL, SCHEMA, {"k": 3})


def test_a_schema_without_a_type_constrains_nothing():
    validate_arguments(KB_TOOL, {"properties": {"anything": {}}}, {"anything": [1, 2]})


def test_chunk_ids_are_read_from_the_same_view_the_model_was_shown():
    """`Hit` exposes its id through `to_dict()`, which is what the prompt renders."""
    hit = Hit(
        chunk=Chunk(
            chunk_id=CHUNK_ID,
            source_file="sleep-hygiene.md",
            heading_path=("Sleep",),
            ordinal=2,
            text="Keep it dark.",
            token_count=4,
        ),
        score=0.7,
    )
    assert chunk_ids_in([hit]) == [CHUNK_ID]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (kb_payload(), [CHUNK_ID]),
        ([], []),
        ({"results": []}, []),
        ("a string", []),
        ([{"chunk_id": CHUNK_ID}, {"chunk_id": CHUNK_ID}], [CHUNK_ID]),
        ([{"title": "no ids here"}], []),
    ],
)
def test_chunk_ids_tolerate_whatever_shape_a_tool_returns(result, expected):
    assert chunk_ids_in(result) == expected


# --------------------------------------------------------------------------------------
# The loop: the three paths that matter
# --------------------------------------------------------------------------------------


def test_a_clean_tool_call_runs_and_its_observation_reaches_the_model():
    agent, adapter, inventory = build(
        tool_call(KB_TOOL, query="dark bedroom"),
        final(f"Keep it dark {CITATION_FORMAT.format(chunk_id=CHUNK_ID)}.", [CHUNK_ID]),
    )
    result = agent.run_turn("How should I set up my bedroom?")

    assert inventory[KB_TOOL].calls == [{"query": "dark bedroom"}]
    assert TOOL_RESULT_PREFIX in adapter.prompt(1)
    assert CHUNK_ID in adapter.prompt(1)
    assert result.stopped_reason == STOPPED_ANSWERED
    assert result.tool_calls == [{"name": KB_TOOL, "arguments": {"query": "dark bedroom"}}]
    assert result.retrieved_chunk_ids == [CHUNK_ID]
    assert result.citations == [CHUNK_ID]
    assert result.format_violations == 0
    assert len(result.steps) == 2


def test_malformed_json_is_counted_then_the_model_is_given_the_error_and_recovers():
    agent, adapter, _ = build("Let me look that up for you.", final("Dark and cool.", []))
    result = agent.run_turn("How should I set up my bedroom?")

    assert result.format_violations == 1
    assert result.stopped_reason == STOPPED_ANSWERED
    assert result.final_text == "Dark and cool."
    reprompt = adapter.prompt(1)
    assert PROTOCOL_ERROR_PREFIX in reprompt
    assert "found none" in reprompt
    assert "exactly one JSON object" in reprompt


def test_the_tool_cap_forces_an_answer_after_three_dispatches():
    agent, adapter, inventory = build(tool_call(KB_TOOL, query="sleep"))
    result = agent.run_turn("Tell me about sleep")

    assert len(inventory[KB_TOOL].calls) == DEFAULT_MAX_TOOL_CALLS == 3
    assert len(result.tool_calls) == 3
    assert TOOL_BUDGET_PREFIX in adapter.prompt()
    assert FINAL_ANSWER_REQUIRED in adapter.prompt()
    assert result.stopped_reason == STOPPED_TOOL_BUDGET
    assert result.final_text == ""


def test_the_forced_answer_is_accepted_when_the_model_gives_one():
    agent, _, inventory = build(
        tool_call(KB_TOOL, query="a"),
        tool_call(KB_TOOL, query="b"),
        tool_call(KB_TOOL, query="c"),
        tool_call(KB_TOOL, query="d"),
        final("Here is what I found.", []),
    )
    result = agent.run_turn("Tell me about sleep")

    assert len(inventory[KB_TOOL].calls) == 3
    assert result.stopped_reason == STOPPED_ANSWERED
    assert result.final_text == "Here is what I found."


# --------------------------------------------------------------------------------------
# The loop: the other ways a turn ends
# --------------------------------------------------------------------------------------


def test_a_second_consecutive_parse_failure_ends_the_turn():
    agent, adapter, _ = build("no json", "still no json", final())
    result = agent.run_turn("q")

    assert result.stopped_reason == STOPPED_PROTOCOL_ERROR
    assert result.format_violations == 2
    assert result.final_text == ""
    assert adapter.count == 2


def test_a_recovered_failure_resets_the_consecutive_count():
    agent, _, _ = build("no json", tool_call(), "no json again", final("Found it.", []))
    result = agent.run_turn("q")

    assert result.format_violations == 2
    assert result.stopped_reason == STOPPED_ANSWERED


def test_the_model_call_ceiling_binds_before_the_tool_cap_when_it_is_lower():
    """Independent limits, and the trace should name whichever one actually stopped it."""
    agent, adapter, inventory = build(tool_call(), budgets=Budgets(max_model_calls=2))
    result = agent.run_turn("q")

    assert adapter.count == 2
    assert len(inventory[KB_TOOL].calls) == 2
    assert result.stopped_reason == STOPPED_MODEL_CALL_BUDGET


def test_an_unanswered_turn_leaves_the_answer_empty_rather_than_inventing_one():
    agent, _, _ = build(tool_call(), budgets=Budgets(max_model_calls=1))
    result = agent.run_turn("q")

    assert result.final_text == ""
    assert result.citations == []


def test_a_model_call_ceiling_below_one_is_refused_rather_than_producing_empty_answers():
    """A run of perfectly comparable zeros is worse than an error, because it looks like data."""
    with pytest.raises(ValueError, match="max_model_calls must be at least 1"):
        Budgets(max_model_calls=0)


def test_a_provider_failure_is_logged_and_then_raised(tmp_path):
    with TraceLogger("failing", runs_dir=tmp_path) as logger:
        agent, _, _ = build(ModelError("provider down"), logger=logger)
        with pytest.raises(ModelError, match="provider down"):
            agent.run_turn("q")

    errors = [r["error"] for r in read_records(logger.path) if r["error"]]
    assert any("provider down" in error for error in errors)


# --------------------------------------------------------------------------------------
# The loop: tools that go wrong
# --------------------------------------------------------------------------------------


def test_an_invented_tool_is_reported_to_the_model_but_is_not_a_format_violation():
    """Inventing a capability and emitting bad JSON are different failures, and
    `check_no_hallucinated_tool` scores the first one separately."""
    agent, adapter, _ = build(tool_call("search_the_internet", query="x"), final("Sorry.", []))
    result = agent.run_turn("q")

    assert result.format_violations == 0
    assert result.tool_errors == 1
    assert result.tool_errors_by_type == {ToolErrorReason.UNKNOWN_TOOL.value: 1}
    assert result.stopped_reason == STOPPED_ANSWERED
    assert "unknown tool 'search_the_internet'" in adapter.prompt(1)
    assert f"valid tools: {KB_TOOL}, {WEB_TOOL}" in adapter.prompt(1)


@pytest.mark.parametrize(
    ("arguments", "expected", "reason"),
    [
        ({"question": "sleep"}, "missing required argument 'query'", ToolErrorReason.MISSING_ARG),
        (
            {"query": "sleep", "limit": 3},
            "'limit' is not an argument of this tool",
            ToolErrorReason.SCHEMA_INVALID,
        ),
        (
            {"query": "sleep", "top_k": "three"},
            "'top_k' must be an integer, got 'three'",
            ToolErrorReason.BAD_ARG_TYPE,
        ),
    ],
)
def test_bad_arguments_are_explained_to_the_model_without_ending_the_turn(
    arguments, expected, reason
):
    agent, adapter, inventory = build(tool_call(KB_TOOL, **arguments), final("Sorry.", []))
    result = agent.run_turn("q")

    assert inventory[KB_TOOL].calls == []
    assert f"{KB_TOOL}: {expected}" in adapter.prompt(1)
    assert result.tool_errors_by_type == {reason.value: 1}
    assert result.stopped_reason == STOPPED_ANSWERED


def test_a_tool_input_error_is_reported_and_the_turn_continues():
    """A tool the model misused is information it can act on; the turn survives and the
    failure is charged to the error budget rather than to the model's tool calls."""
    picky = FakeTool(WEB_TOOL, raises=ToolInputError("bad query", ToolErrorReason.SCHEMA_INVALID))
    agent, adapter, _ = build(
        tool_call(WEB_TOOL, query="x"),
        final("I could not search.", []),
        inventory=tools(**{WEB_TOOL: picky}),
    )
    result = agent.run_turn("q")

    assert "bad query" in adapter.prompt(1)
    assert result.stopped_reason == STOPPED_ANSWERED
    assert result.tool_errors == 1
    assert result.infrastructure_failed is False


def test_a_tool_that_fails_for_infrastructure_reasons_is_retried_and_then_ends_the_item():
    """Retries first, because a timeout is usually transient. If it persists the item is
    excluded from scoring rather than scored on what the model managed around our outage."""
    down = FakeTool(WEB_TOOL, raises=ToolInfraError("search index unreachable"))
    agent, _, _ = build(
        tool_call(WEB_TOOL, query="x"),
        final("never reached", []),
        inventory=tools(**{WEB_TOOL: down}),
    )
    result = agent.run_turn("q")

    assert len(down.calls) == 3
    assert result.stopped_reason == STOPPED_INFRASTRUCTURE_FAILED
    assert result.infrastructure_failed is True
    assert "search index unreachable" in (result.infrastructure_error or "")
    assert result.final_text == ""


def test_an_infrastructure_failure_is_charged_to_no_budget():
    """Not to the tool calls, not to the errors, not to the format-violation rate. Our outage
    must not consume the model's room to work or show up as its failure anywhere."""
    down = FakeTool(WEB_TOOL, raises=ToolInfraError("timeout"))
    agent, _, _ = build(tool_call(WEB_TOOL, query="x"), inventory=tools(**{WEB_TOOL: down}))
    result = agent.run_turn("q")

    assert result.tool_errors == 0
    assert result.tool_errors_by_type == {}
    assert result.format_violations == 0


def test_infrastructure_retries_stop_as_soon_as_the_tool_recovers():
    flaky = FlakyTool(WEB_TOOL, failures=1)
    agent, adapter, _ = build(
        tool_call(WEB_TOOL, query="x"),
        final("Found it.", []),
        inventory=tools(**{WEB_TOOL: flaky}),
    )
    result = agent.run_turn("q")

    assert flaky.attempts == 2
    assert result.stopped_reason == STOPPED_ANSWERED
    assert TOOL_RESULT_PREFIX in adapter.prompt(1)
    assert result.tool_errors == 0


def test_each_infrastructure_retry_is_visible_in_the_trace(tmp_path):
    """A retry that leaves no record is indistinguishable from a slow call."""
    down = FakeTool(WEB_TOOL, raises=ToolInfraError("timeout"))
    written = records(tmp_path, tool_call(WEB_TOOL, query="x"), inventory=tools(**{WEB_TOOL: down}))
    retries = [r for r in written if r["error"] and "retry" in r["error"]]

    assert len(retries) == 2
    assert [r["infrastructure_failed"] for r in retries] == [False, False]
    assert any(r["infrastructure_failed"] for r in written)


def test_an_unexpected_tool_failure_is_logged_and_then_surfaces(tmp_path):
    """Neither of the two known kinds, so it is a gap in this harness rather than a fact about
    the model. Filing it under a guess is how it goes unnoticed for a hundred items."""
    broken = FakeTool(KB_TOOL, raises=RuntimeError("boom"))
    with TraceLogger("surfacing", runs_dir=tmp_path) as logger:
        agent, _, _ = build(
            tool_call(KB_TOOL, query="x"),
            inventory=tools(**{KB_TOOL: broken}),
            logger=logger,
        )
        with pytest.raises(RuntimeError, match="boom"):
            agent.run_turn("q")

    errors = [r["error"] for r in read_records(logger.path) if r["error"]]
    assert any("unexpected failure in lookup_kb" in error and "boom" in error for error in errors)


def test_an_unexpected_failure_is_not_retried():
    """Retrying a bug wastes three calls to reach the same traceback."""
    broken = FakeTool(KB_TOOL, raises=RuntimeError("boom"))
    agent, _, _ = build(tool_call(KB_TOOL, query="x"), inventory=tools(**{KB_TOOL: broken}))

    with pytest.raises(RuntimeError):
        agent.run_turn("q")
    assert len(broken.calls) == 1


# --------------------------------------------------------------------------------------
# The three budgets, counted separately
# --------------------------------------------------------------------------------------


def test_successful_calls_and_model_caused_errors_do_not_share_a_counter():
    """The point of the split: a model doing useful work and a model that cannot get its
    arguments right must not be able to exhaust the same budget."""
    agent, adapter, inventory = build(
        tool_call(KB_TOOL, query="a"),
        tool_call(KB_TOOL, top_k="three", query="b"),
        tool_call(KB_TOOL, query="c"),
        tool_call(KB_TOOL, query="d"),
        tool_call(KB_TOOL, query="e"),
        final("Here is what I found.", []),
        budgets=Budgets(max_tool_calls=3, max_tool_errors=2, max_model_calls=8),
    )
    result = agent.run_turn("q")

    # Three successes dispatched, the bad call charged elsewhere, and only the fifth attempt
    # refused: had the error consumed a tool call, "e" would have been the one turned away.
    assert [call["query"] for call in inventory[KB_TOOL].calls] == ["a", "c", "d"]
    assert result.tool_errors == 1
    assert result.stopped_reason == STOPPED_ANSWERED
    assert TOOL_BUDGET_PREFIX in adapter.prompt(5)


def test_exceeding_the_tool_call_budget_names_that_budget():
    agent, adapter, inventory = build(tool_call(KB_TOOL, query="sleep"))
    result = agent.run_turn("Tell me about sleep")

    assert len(inventory[KB_TOOL].calls) == DEFAULT_MAX_TOOL_CALLS == 3
    assert result.stopped_reason == STOPPED_TOOL_BUDGET
    assert result.tool_errors == 0


def test_exceeding_the_tool_error_budget_names_that_budget_instead():
    """A different diagnosis from a spent tool budget, and a different fix."""
    agent, adapter, inventory = build(tool_call("no_such_tool", query="x"))
    result = agent.run_turn("q")

    assert result.tool_errors == DEFAULT_MAX_TOOL_ERRORS == 2
    assert result.stopped_reason == STOPPED_TOOL_ERROR_BUDGET
    assert inventory[KB_TOOL].calls == []
    assert TOOL_BUDGET_PREFIX in adapter.prompt()


def test_the_error_budget_still_leaves_room_to_answer():
    agent, _, _ = build(
        tool_call("no_such_tool", query="x"),
        tool_call("no_such_tool", query="x"),
        final("I could not do that.", []),
    )
    result = agent.run_turn("q")

    assert result.tool_errors == 2
    assert result.stopped_reason == STOPPED_ANSWERED
    assert result.final_text == "I could not do that."


def test_the_nudge_is_the_same_message_whichever_tool_budget_ran_out():
    """Telling one model "you used your calls" and another "you made too many mistakes" would
    be different feedback at the same point in the loop, and the required reply is identical."""
    spent_calls, calls_adapter, _ = build(tool_call(KB_TOOL, query="x"))
    spent_errors, errors_adapter, _ = build(tool_call("no_such_tool", query="x"))
    spent_calls.run_turn("q")
    spent_errors.run_turn("q")

    assert FINAL_ANSWER_REQUIRED in calls_adapter.prompt()
    assert FINAL_ANSWER_REQUIRED in errors_adapter.prompt()


def test_the_model_call_ceiling_bounds_a_model_that_only_emits_bad_json():
    """Alternating garbage and valid calls resets the consecutive-failure stop, so something
    else has to end the turn."""
    agent, adapter, _ = build(
        "no json",
        tool_call(KB_TOOL, query="a"),
        "no json",
        tool_call(KB_TOOL, query="b"),
        "no json",
        tool_call(KB_TOOL, query="c"),
        budgets=Budgets(max_model_calls=6),
    )
    result = agent.run_turn("q")

    assert adapter.count == 6
    assert result.format_violations == 3
    assert result.stopped_reason in {STOPPED_MODEL_CALL_BUDGET, STOPPED_TOOL_BUDGET}


def test_budgets_reach_the_trace_as_the_reason_a_turn_stopped(tmp_path):
    written = records(tmp_path, tool_call("no_such_tool", query="x"))
    (turn,) = [record for record in written if record["role"] == "turn"]
    assert turn["error"] == STOPPED_TOOL_ERROR_BUDGET


def test_the_trace_types_each_tool_error(tmp_path):
    written = records(tmp_path, tool_call("no_such_tool", query="x"), final("Sorry.", []))
    reasons = [r["tool_error_reason"] for r in written if r["tool_error_reason"]]
    assert reasons == [ToolErrorReason.UNKNOWN_TOOL.value]


# --------------------------------------------------------------------------------------
# What the loop reports
# --------------------------------------------------------------------------------------


def test_retrieved_ids_accumulate_across_calls_without_duplicates():
    first = FakeTool(KB_TOOL, result=kb_payload(CHUNK_ID))
    agent, _, _ = build(
        tool_call(KB_TOOL, query="a"),
        tool_call(KB_TOOL, query="b"),
        final("Both.", [CHUNK_ID]),
        inventory=tools(**{KB_TOOL: first}),
    )
    result = agent.run_turn("q")
    assert result.retrieved_chunk_ids == [CHUNK_ID]


def test_declared_citations_and_inline_markers_are_both_recoverable():
    """Keeping both is what lets a check notice a declared citation that never appears
    inline, or an inline one that was never retrieved."""
    answer = f"Dark rooms help {CITATION_FORMAT.format(chunk_id=CHUNK_ID)}."
    agent, _, _ = build(final(answer, [CHUNK_ID, OTHER_CHUNK_ID]))
    result = agent.run_turn("q")

    assert result.citations == [CHUNK_ID, OTHER_CHUNK_ID]
    assert parse_citations(result.final_text) == [CHUNK_ID]


def test_tokens_and_cost_are_summed_over_every_call_in_the_turn():
    agent, adapter, _ = build(tool_call(), final())
    result = agent.run_turn("q")

    assert result.tokens == {
        "prompt": 2 * (adapter.prompt_tokens or 0),
        "completion": 2 * (adapter.completion_tokens or 0),
        "reasoning": 2 * (adapter.reasoning_tokens or 0),
        "total": 2
        * (
            (adapter.prompt_tokens or 0)
            + (adapter.completion_tokens or 0)
            + (adapter.reasoning_tokens or 0)
        ),
    }
    assert result.usd_cost == pytest.approx(2 * (adapter.usd_cost or 0))


def test_reasoning_tokens_are_summed_into_the_turn_total_not_into_completion():
    """The split has to survive aggregation: a thinking arm whose reasoning were folded into
    `completion` would report a visible reply length it never produced."""
    adapter = FakeAdapter(
        [tool_call(), final()], prompt_tokens=100, completion_tokens=20, reasoning_tokens=500
    )
    result = Agent(adapter, tools(), run_id="r").run_turn("q")

    assert result.tokens["completion"] == 40
    assert result.tokens["reasoning"] == 1000
    assert result.tokens["total"] == 200 + 40 + 1000


def test_a_provider_that_reports_no_usage_gives_zero_tokens_and_no_cost():
    """A missing count is not a zero cost; None says the number is unknown."""
    adapter = FakeAdapter(
        [final()],
        prompt_tokens=None,
        completion_tokens=None,
        reasoning_tokens=None,
        usd_cost=None,
    )
    result = Agent(adapter, tools(), run_id="r").run_turn("q")

    assert result.tokens == {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0}
    assert result.usd_cost is None


def test_each_step_records_whether_it_was_a_cache_replay():
    """An aggregate that cannot exclude replays would report disk reads as model speed."""
    adapter = FakeAdapter([final()], cached=True)
    result = Agent(adapter, tools(), run_id="r").run_turn("q")

    assert [step.cached for step in result.steps] == [True]
    assert result.steps[0].latency_ms == adapter.latency_ms


def test_a_step_keeps_the_exact_messages_it_sent():
    agent, _, _ = build(final())
    result = agent.run_turn("How should I set up my bedroom?")

    sent = result.steps[0].messages_sent
    assert sent[0]["role"] == "system"
    assert sent[-1]["content"] == "How should I set up my bedroom?"


def test_the_result_carries_the_run_id_and_a_wall_clock_latency():
    agent, _, _ = build(final())
    result = agent.run_turn("q")

    assert result.run_id == "test-run"
    assert result.latency_ms > 0


# --------------------------------------------------------------------------------------
# What reaches the trace
# --------------------------------------------------------------------------------------


def records(
    tmp_path: Path, *completions: str | Reply | Exception, **kwargs: Any
) -> list[dict[str, Any]]:
    """Run one turn against a real trace file and read the records back."""
    with TraceLogger("traced", runs_dir=tmp_path) as logger:
        agent, _, _ = build(*completions, logger=logger, **kwargs)
        agent.run_turn("How should I set up my bedroom?", item_id="case-1")
    return read_records(logger.path)


def test_every_stage_of_the_turn_appears_in_the_trace(tmp_path):
    written = records(tmp_path, tool_call(KB_TOOL, query="dark"), final("Dark.", [CHUNK_ID]))
    roles = [record["role"] for record in written]

    assert roles == ["user", "assistant", "tool", "assistant", "turn"]
    tool_record = written[2]
    assert tool_record["retrieved_chunk_ids"] == [CHUNK_ID]
    assert tool_record["tool_calls"] == [{"name": KB_TOOL, "arguments": {"query": "dark"}}]


def test_per_call_records_and_the_turn_total_are_separable(tmp_path):
    """Two roles rather than one, so summing tokens over a trace cannot double count the
    same call — the turn's aggregate is exactly the sum of the calls it covers."""
    written = records(tmp_path, tool_call(), final("Dark.", []))
    per_call = [r["prompt_tokens"] for r in written if r["role"] == "assistant"]
    (turn,) = [r for r in written if r["role"] == "turn"]

    assert len(per_call) == 2
    assert turn["prompt_tokens"] == sum(per_call)
    assert turn["latency_ms"] > 0


def test_malformed_output_is_recorded_verbatim_before_anything_tries_to_parse_it(tmp_path):
    """A model's bad JSON is data about that model; repairing or dropping it loses the
    measurement the whole comparison is built on."""
    written = records(tmp_path, "I refuse to use JSON.", final("Fine.", []))
    completions = [r["content"] for r in written if r["role"] == "assistant"]

    assert "I refuse to use JSON." in completions
    violations = [r["error"] for r in written if r["error"] and FORMAT_VIOLATION in r["error"]]
    assert len(violations) == 1


def test_the_turn_record_names_why_an_unfinished_turn_stopped(tmp_path):
    written = records(tmp_path, tool_call())
    (turn,) = [record for record in written if record["role"] == "turn"]
    assert turn["error"] == STOPPED_TOOL_BUDGET


def test_an_answered_turn_records_no_error(tmp_path):
    written = records(tmp_path, final("Dark and cool.", []))
    (turn,) = [record for record in written if record["role"] == "turn"]
    assert turn["error"] is None
    assert turn["content"] == "Dark and cool."


def test_records_carry_the_item_id_and_a_turn_index_that_restarts_per_item(tmp_path):
    with TraceLogger("multi", runs_dir=tmp_path) as logger:
        agent, _, _ = build(final(), logger=logger)
        agent.run_turn("q1", item_id="case-1")
        agent.run_turn("q2", item_id="case-1")
        agent.run_turn("q3", item_id="case-2")

    seen = {(r["item_id"], r["turn_idx"]) for r in read_records(logger.path)}
    assert seen == {("case-1", 0), ("case-1", 1), ("case-2", 0)}


def test_next_turn_idx_reports_where_the_counter_is():
    agent, _, _ = build(final())
    assert agent.next_turn_idx == 0
    agent.run_turn("q1", item_id="case-1")
    assert agent.next_turn_idx == 1


def test_resume_numbering_continues_an_existing_conversation(tmp_path):
    """A conversation carried onto a new agent is on its fifth turn, not its first."""
    with TraceLogger("resumed", runs_dir=tmp_path) as logger:
        agent, _, _ = build(final(), logger=logger)
        agent.resume_numbering("case-1", 4)
        agent.run_turn("q5", item_id="case-1")

    seen = {(r["item_id"], r["turn_idx"]) for r in read_records(logger.path)}
    assert seen == {("case-1", 4)}


def test_resume_numbering_is_overridden_by_a_different_item_id(tmp_path):
    """The counter is per conversation, so resuming one must not renumber the next."""
    with TraceLogger("elsewhere", runs_dir=tmp_path) as logger:
        agent, _, _ = build(final(), logger=logger)
        agent.resume_numbering("case-1", 4)
        agent.run_turn("q1", item_id="case-2")

    seen = {(r["item_id"], r["turn_idx"]) for r in read_records(logger.path)}
    assert seen == {("case-2", 0)}


def test_resume_numbering_keeps_counting_from_where_it_was_told(tmp_path):
    with TraceLogger("counting", runs_dir=tmp_path) as logger:
        agent, _, _ = build(final(), logger=logger)
        agent.resume_numbering("case-1", 2)
        agent.run_turn("q3", item_id="case-1")
        agent.run_turn("q4", item_id="case-1")

    seen = {(r["item_id"], r["turn_idx"]) for r in read_records(logger.path)}
    assert seen == {("case-1", 2), ("case-1", 3)}


def test_an_agent_without_a_logger_still_runs():
    """`app.py` and the tests build one; a missing logger must not be a crash."""
    agent, _, _ = build(final("Fine.", []))
    assert agent.run_turn("q").final_text == "Fine."


def test_compaction_is_logged_and_charged_to_the_turn(tmp_path):
    """The summariser is a model call like any other."""
    with TraceLogger("compacted", runs_dir=tmp_path) as logger:
        adapter = FakeAdapter([final("Answered.", [])])
        agent = Agent(adapter, tools(), run_id="r", logger=logger)
        agent.conversation.token_budget = 1
        agent.conversation.keep_last_turns = 1
        agent.run_turn("first question", item_id="case-1")
        result = agent.run_turn("second question", item_id="case-1")

    roles = [record["role"] for record in read_records(logger.path)]
    assert "summariser" in roles
    assert "memory" in roles
    # Three calls on the second turn's books: the summariser's plus the answer's.
    assert result.tokens["total"] > (adapter.prompt_tokens or 0)


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


def test_the_default_registry_is_the_real_one_and_matches_the_manifest_digest():
    agent = Agent(FakeAdapter([final()]), run_id="r")
    assert set(agent.tools) == set(registry())
    assert core.tool_specs_for(agent.tools) == tool_specs()


def test_the_system_prompt_documents_the_tools_the_loop_will_dispatch_to():
    agent, adapter, _ = build(final())
    agent.run_turn("q")

    system = adapter.messages()[0]
    assert system["role"] == "system"
    assert KB_TOOL in system["content"]
    assert WEB_TOOL in system["content"]


def test_temperature_and_token_ceiling_reach_the_provider():
    agent, adapter, _ = build(final(), temperature=0.0, max_tokens=256)
    agent.run_turn("q")

    assert adapter.calls[0]["temperature"] == 0.0
    assert adapter.calls[0]["max_tokens"] == 256


def test_the_agent_summarises_with_its_own_model():
    """A separate summariser would let a third model decide what one arm remembers."""
    agent, adapter, _ = build(final())
    assert agent.conversation.summariser is adapter


def test_its_own_conversation_carries_across_turns():
    agent, adapter, _ = build(final("First.", []), final("Second.", []))
    agent.run_turn("first question")
    agent.run_turn("second question")

    assert "first question" in adapter.prompt(1)


def test_a_supplied_conversation_keeps_cases_from_leaking_into_each_other():
    agent, adapter, _ = build(final("First.", []), final("Second.", []))
    agent.run_turn("first question", Conversation(system_prompt=agent.system_prompt))
    agent.run_turn("second question", Conversation(system_prompt=agent.system_prompt))

    assert "first question" not in adapter.prompt(1)


def test_the_scripted_adapter_satisfies_the_model_interface():
    """Otherwise these tests could pass against something the real loop cannot drive."""
    assert isinstance(FakeAdapter([final()]), ModelAdapter)


# --------------------------------------------------------------------------------------
# One harness
# --------------------------------------------------------------------------------------

#: Not bare "meta": the assertion is a substring check over module source, so it would trip on
#: any future `metadata` identifier.
MODEL_FAMILIES = (
    "claude",
    "sonnet",
    "gpt-4",
    "gpt4",
    "qwen",
    "llama",
    "meta-llama",
    "anthropic",
    "openai",
    "groq",
)


def executable_source(module) -> str:
    """A module's source with comments and string literals removed.

    Docstrings here discuss the no-branching rule at length and would match every pattern the
    test looks for, so the check has to see the code rather than the prose about it.
    """
    dropped = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    readline = io.StringIO(Path(module.__file__).read_text(encoding="utf-8")).readline
    return " ".join(
        token.string for token in tokenize.generate_tokens(readline) if token.type not in dropped
    ).lower()


def test_the_loop_never_branches_on_which_model_is_answering():
    """A fast path for the stronger model would confound model quality with harness quality
    and void every comparison built on these runs."""
    code = executable_source(core)

    for family in MODEL_FAMILIES:
        assert family not in code, f"agent.core mentions {family!r}"
    for seam in ("model . name", "model . family", ". provider", "model_id"):
        assert seam not in code, f"agent.core branches on {seam!r}"


def test_memory_never_branches_on_which_model_is_answering():
    """Compaction decides what a model still knows; per-model policy is per-model context."""
    code = executable_source(memory)

    for family in MODEL_FAMILIES:
        assert family not in code, f"agent.memory mentions {family!r}"
    for seam in ("model . name", "model . family", "summariser . name"):
        assert seam not in code, f"agent.memory branches on {seam!r}"


def test_no_public_function_in_the_loop_takes_a_model_identity():
    for name, value in vars(core).items():
        if name.startswith("_") or not callable(value) or inspect.isclass(value):
            continue
        if getattr(value, "__module__", None) != core.__name__:
            continue
        parameters = set(inspect.signature(value).parameters)
        assert parameters.isdisjoint({"model_name", "provider", "family"}), name


def test_both_arms_are_driven_through_the_identical_sequence():
    """Same script, same tools: the messages sent must not depend on the adapter's name."""
    prompts_seen = []
    for name in ("frontier", "oss"):
        adapter = FakeAdapter([tool_call(KB_TOOL, query="sleep"), final("Dark.", [])], name=name)
        agent = Agent(adapter, tools(), run_id="r")
        agent.run_turn("How should I set up my bedroom?")
        prompts_seen.append([call["messages"] for call in adapter.calls])

    assert prompts_seen[0] == prompts_seen[1]


#: Every way a model can get a tool call wrong, as it would arrive from a model.
BAD_CALLS = [
    tool_call("lookup_KB", query="sleep"),
    tool_call(KB_TOOL, question="sleep"),
    tool_call(KB_TOOL, query="sleep", top_k="3"),
    tool_call(KB_TOOL, query="sleep", limit=3),
    "I will look that up for you.",
    '{"final": "Sleep in a dark room."}',
]


@pytest.mark.parametrize("completion", BAD_CALLS)
def test_a_failure_is_explained_in_byte_identical_words_to_both_arms(completion):
    """Error text is context the next attempt is made from, so a difference in wording is a
    difference in how much help each arm got. One renderer per failure is what guarantees it,
    and this is the test that would catch a well-meant per-arm hint."""
    messages_seen = []
    for name in ("frontier", "oss"):
        adapter = FakeAdapter([completion, final("Sorry.", [])], name=name)
        Agent(adapter, tools(), run_id="r").run_turn("How should I set up my bedroom?")
        messages_seen.append(adapter.prompt(1))

    assert messages_seen[0] == messages_seen[1]


def test_the_error_renderers_are_the_only_place_these_sentences_are_written():
    """A second copy of "unknown tool" formatted somewhere else is how the two arms start
    reading different words for the same mistake."""
    sentences = ("unknown tool", "missing required argument", "is not an argument of this tool")
    for module in (core, memory):
        code = executable_source(module)
        for sentence in sentences:
            assert sentence.replace(" ", "") not in code.replace(" ", ""), (
                f"{module.__name__} writes {sentence!r} itself instead of calling agent.prompts"
            )
