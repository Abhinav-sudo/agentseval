"""Tests for `agent.memory`.

Memory policy is part of the harness, so these tests aim at the properties that make it one:
the window is the same size whoever is answering, compaction fires at the same estimated
token count for every model, and every message that stops being sent verbatim leaves a record
saying so.

The rest is failure behaviour. Compaction happens mid-turn, so a summariser that is missing,
broken, or unhelpful must cost context rather than costing the turn — and must say which
happened, so a thin later answer can be traced to lost history instead of to the model.
"""

from __future__ import annotations

import pytest

from agent.memory import (
    CHARS_PER_TOKEN,
    Conversation,
    estimate_messages_tokens,
    estimate_tokens,
)
from agent.models.base import ModelError
from agent.prompts import SUMMARY_PREFIX
from tests.fakes import FakeAdapter

SYSTEM = "SYSTEM PROMPT"

#: Long enough that a handful of turns crosses the small budgets used below.
FILLER = "x" * 200


def conversation(**kwargs) -> Conversation:
    return Conversation(system_prompt=SYSTEM, **kwargs)


def exchange(convo: Conversation, label: str, *, tool_results: int = 0) -> None:
    """Add one complete turn: a user message, optional tool results, and an answer."""
    convo.add_user(f"question {label} {FILLER}")
    for index in range(tool_results):
        convo.add_tool_result(f"TOOL RESULT ({label}.{index}) {FILLER}")
    convo.add_assistant(f"answer {label} {FILLER}")


def summariser(text: str = "Earlier: the user asked about sleep and hydration.") -> FakeAdapter:
    return FakeAdapter(completions=[text])


# --------------------------------------------------------------------------------------
# Token estimation
# --------------------------------------------------------------------------------------


def test_token_estimate_rounds_up_so_short_text_is_never_free():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * CHARS_PER_TOKEN) == 1
    assert estimate_tokens("a" * (CHARS_PER_TOKEN + 1)) == 2


def test_message_estimate_charges_for_each_message_beyond_its_content():
    """Otherwise a transcript of many tiny messages would look nearly free."""
    one = estimate_messages_tokens([{"role": "user", "content": "hi"}])
    two = estimate_messages_tokens(
        [{"role": "user", "content": "h"}, {"role": "user", "content": "i"}]
    )
    assert two > one


def test_message_estimate_survives_a_message_with_no_content():
    assert estimate_messages_tokens([{"role": "user"}]) > 0


# --------------------------------------------------------------------------------------
# The verbatim window
# --------------------------------------------------------------------------------------


def test_system_prompt_leads_and_history_follows_in_order():
    convo = conversation()
    convo.add_user("first")
    convo.add_assistant("second")
    roles = [message["role"] for message in convo.to_messages()]
    contents = [message["content"] for message in convo.to_messages()]

    assert roles == ["system", "user", "assistant"]
    assert contents == [SYSTEM, "first", "second"]


def test_tool_results_are_user_content_because_the_protocol_has_no_tool_role():
    convo = conversation()
    convo.add_user("q")
    convo.add_tool_result("TOOL RESULT (lookup_kb)")
    assert convo.to_messages()[-1] == {"role": "user", "content": "TOOL RESULT (lookup_kb)"}


def test_a_tool_result_does_not_start_a_new_turn():
    """Otherwise the window would shrink every time the model used a tool."""
    convo = conversation()
    exchange(convo, "a", tool_results=3)
    assert convo.turn_count == 1


def test_nothing_is_folded_while_the_transcript_fits():
    convo = conversation(token_budget=100_000, summariser=summariser())
    for label in "abcdef":
        exchange(convo, label)

    assert convo.summary is None
    assert convo.truncations() == []
    assert len(convo.to_messages()) == 1 + 12


def test_the_window_is_not_a_function_of_the_model():
    """Two adapters, one policy: the same transcript compacts at the same point."""
    shapes = []
    for adapter in (FakeAdapter(["S"], name="frontier"), FakeAdapter(["S"], name="oss")):
        convo = conversation(keep_last_turns=2, token_budget=200, summariser=adapter)
        for label in "abcde":
            exchange(convo, label)
        shapes.append([message["role"] for message in convo.to_messages()])
    assert shapes[0] == shapes[1]


def test_to_messages_hands_out_copies():
    """A caller mutating the messages it was given must not edit the history."""
    convo = conversation()
    convo.add_user("q")
    messages = convo.to_messages()
    messages[-1]["content"] = "tampered"
    assert convo.to_messages()[-1]["content"] == "q"


def test_a_prebuilt_message_list_is_treated_as_one_turn():
    convo = Conversation(system_prompt=SYSTEM, messages=[{"role": "user", "content": "q"}])
    assert convo.turn_count == 1


# --------------------------------------------------------------------------------------
# Compaction
# --------------------------------------------------------------------------------------


def test_older_turns_are_folded_into_a_summary_once_the_budget_is_passed():
    adapter = summariser("The user asked about sleep.")
    convo = conversation(keep_last_turns=2, token_budget=400, summariser=adapter)
    for label in "abcd":
        exchange(convo, label)

    messages = convo.to_messages()
    contents = "\n".join(message["content"] for message in messages)

    assert convo.summary == "The user asked about sleep."
    assert messages[0]["role"] == "system"
    assert messages[1]["content"].startswith(SUMMARY_PREFIX)
    assert "The user asked about sleep." in messages[1]["content"]
    # The last two turns survive verbatim; the first two are only in the summary.
    assert "question c" in contents
    assert "question d" in contents
    assert "question a" not in contents
    assert "question b" not in contents


def test_compaction_uses_the_agents_own_adapter():
    """Requirement, not detail: a shared summariser would edit one arm's context with
    another model's judgement of what mattered."""
    adapter = summariser()
    convo = conversation(keep_last_turns=1, token_budget=200, summariser=adapter)
    for label in "abc":
        exchange(convo, label)
    convo.to_messages()

    assert adapter.count == 1


def test_the_summariser_is_sent_the_folded_turns_and_no_system_prompt():
    """Sharing the agent's system prompt would have it reply with a protocol tool call."""
    adapter = summariser()
    convo = conversation(keep_last_turns=1, token_budget=200, summariser=adapter)
    exchange(convo, "a")
    exchange(convo, "b")
    convo.to_messages()

    sent = adapter.messages()
    assert [message["role"] for message in sent] == ["user"]
    assert "question a" in sent[0]["content"]
    assert SYSTEM not in sent[0]["content"]


def test_the_summariser_call_is_reported_so_it_can_be_logged():
    seen: list[tuple[str, str]] = []
    adapter = summariser("a summary")
    convo = conversation(
        keep_last_turns=1,
        token_budget=200,
        summariser=adapter,
        on_summarised=lambda request, response: seen.append((request, response.text)),
    )
    exchange(convo, "a")
    exchange(convo, "b")
    convo.to_messages()

    assert len(seen) == 1
    request, text = seen[0]
    assert "question a" in request
    assert text == "a summary"


def test_a_truncation_record_says_how_much_was_folded():
    adapter = summariser()
    convo = conversation(keep_last_turns=1, token_budget=200, summariser=adapter)
    exchange(convo, "a")
    exchange(convo, "b")
    convo.to_messages()

    (record,) = convo.truncations()
    assert record["reason"] == "token_budget"
    assert record["messages_folded"] == 2
    assert record["tokens_folded"] > 0
    assert record["summarised"] is True
    assert record["error"] is None
    assert record["at"]


def test_truncation_records_are_copies():
    adapter = summariser()
    convo = conversation(keep_last_turns=1, token_budget=200, summariser=adapter)
    exchange(convo, "a")
    exchange(convo, "b")
    convo.to_messages()

    convo.truncations()[0]["messages_folded"] = 0
    assert convo.truncations()[0]["messages_folded"] == 2


def test_the_summary_rolls_up_the_previous_summary():
    """Two summaries in the context would defeat the point of having one."""
    adapter = FakeAdapter(["first summary", "second summary"])
    convo = conversation(keep_last_turns=1, token_budget=200, summariser=adapter)
    exchange(convo, "a")
    exchange(convo, "b")
    convo.to_messages()
    exchange(convo, "c")
    convo.to_messages()

    assert convo.summary == "second summary"
    assert "first summary" in adapter.prompt()
    summaries = [m for m in convo.to_messages() if m["content"].startswith(SUMMARY_PREFIX)]
    assert len(summaries) == 1


def test_compaction_brings_the_transcript_back_under_budget():
    adapter = FakeAdapter(["tiny summary"])
    convo = conversation(keep_last_turns=2, token_budget=300, summariser=adapter)
    for label in "abcdefgh":
        exchange(convo, label)
    convo.to_messages()

    assert convo.estimated_tokens() <= 300


def test_the_newest_turn_is_never_summarised():
    """Summarising the question about to be answered would defeat the purpose."""
    adapter = summariser()
    convo = conversation(keep_last_turns=4, token_budget=10, summariser=adapter)
    exchange(convo, "a")
    exchange(convo, "b")
    convo.to_messages()

    contents = "\n".join(message["content"] for message in convo.to_messages())
    assert "question b" in contents


def test_a_single_oversized_turn_is_left_alone_and_recorded():
    """Nothing left to fold, so the provider's own limit error is the useful failure."""
    convo = conversation(token_budget=1, summariser=summariser())
    convo.add_user("a question far larger than the budget " + FILLER)
    messages = convo.to_messages()

    assert len(messages) == 2
    (record,) = convo.truncations()
    assert record["messages_folded"] == 0
    assert "nothing was folded" in str(record["error"])


def test_keeping_zero_turns_still_keeps_the_newest_one():
    convo = conversation(keep_last_turns=0, token_budget=10, summariser=summariser())
    exchange(convo, "a")
    exchange(convo, "b")
    contents = "\n".join(message["content"] for message in convo.to_messages())
    assert "question b" in contents


def test_the_message_cap_compacts_even_when_the_token_budget_is_generous():
    adapter = summariser()
    convo = conversation(
        max_messages=4,
        keep_last_turns=1,
        token_budget=1_000_000,
        summariser=adapter,
    )
    for label in "abc":
        exchange(convo, label)
    convo.to_messages()

    assert len(convo.messages) <= 4
    assert convo.truncations()[0]["reason"] == "message_cap"


# --------------------------------------------------------------------------------------
# When summarisation cannot happen
# --------------------------------------------------------------------------------------


def test_without_a_summariser_old_turns_are_dropped_and_the_loss_is_recorded():
    convo = conversation(keep_last_turns=1, token_budget=200)
    exchange(convo, "a")
    exchange(convo, "b")
    contents = "\n".join(message["content"] for message in convo.to_messages())

    assert convo.summary is None
    assert "question a" not in contents
    (record,) = convo.truncations()
    assert record["summarised"] is False
    assert "dropped" in str(record["error"])


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ([ModelError("provider down")], "summariser failed"),
        (["   "], "empty summary"),
    ],
)
def test_a_failed_summariser_costs_context_not_the_turn(script, expected):
    convo = conversation(keep_last_turns=1, token_budget=200, summariser=FakeAdapter(script))
    exchange(convo, "a")
    exchange(convo, "b")

    messages = convo.to_messages()

    assert messages[0]["content"] == SYSTEM
    assert convo.summary is None
    (record,) = convo.truncations()
    assert record["summarised"] is False
    assert expected in str(record["error"])


def test_a_failed_summariser_does_not_discard_an_earlier_good_summary():
    adapter = FakeAdapter(["a good summary", ModelError("provider down")])
    convo = conversation(keep_last_turns=1, token_budget=200, summariser=adapter)
    exchange(convo, "a")
    exchange(convo, "b")
    convo.to_messages()
    exchange(convo, "c")
    convo.to_messages()

    assert convo.summary == "a good summary"


# --------------------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------------------


def test_reset_clears_history_summary_and_records_but_keeps_the_system_prompt():
    convo = conversation(keep_last_turns=1, token_budget=200, summariser=summariser())
    exchange(convo, "a")
    exchange(convo, "b")
    convo.to_messages()
    assert convo.summary is not None

    convo.reset()

    assert convo.messages == []
    assert convo.summary is None
    assert convo.truncations() == []
    assert convo.turn_count == 0
    assert convo.to_messages() == [{"role": "system", "content": SYSTEM}]


# --------------------------------------------------------------------------------------
# Adoption: one conversation continuing under new conditions
# --------------------------------------------------------------------------------------


def test_adopt_carries_the_history_the_turn_boundaries_and_the_summary():
    source = conversation()
    exchange(source, "a")
    exchange(source, "b")
    source.summary = "what came before"

    target = Conversation(system_prompt="A DIFFERENT PROMPT")
    target.adopt(source)

    assert target.messages == source.messages
    assert target.summary == "what came before"
    assert target.turn_count == 2
    assert target.system_prompt == "A DIFFERENT PROMPT"


def test_adopt_copies_rather_than_shares_the_message_list():
    """The source belongs to a run whose trace is complete; appending to it later would edit
    history that trace already describes."""
    source = conversation()
    exchange(source, "a")

    target = conversation()
    target.adopt(source)
    exchange(target, "b")

    assert len(source.messages) == 2
    assert len(target.messages) == 4


def test_adopted_history_still_folds_at_a_turn_boundary():
    """The regression a plain assignment causes: `__post_init__` treats supplied messages as one
    opaque turn, so `_fold_boundary` returns 0 and nothing can ever be compacted."""
    source = conversation()
    for label in "abcdef":
        exchange(source, label)

    target = conversation(keep_last_turns=2, token_budget=500, summariser=summariser())
    target.adopt(source)
    target.to_messages()

    first = target.truncations()[0]
    assert first["messages_folded"] == 8
    assert first["summarised"] is True
    assert target.summary == "Earlier: the user asked about sleep and hydration."
    assert target.turn_count == 2


def test_adopt_does_not_carry_the_previous_run_s_truncation_records():
    """They describe folds performed under another manifest, and that trace holds them. A new
    trace claiming compactions it never performed would misreport its own context handling."""
    source = conversation(max_messages=1)
    exchange(source, "a")
    exchange(source, "b")
    source.to_messages()
    assert source.truncations()

    target = conversation()
    target.adopt(source)

    assert target.truncations() == []


def test_adopt_clears_truncation_records_the_target_had_of_its_own():
    """Adoption replaces the history, so records describing the replaced history would describe
    messages no longer here."""
    target = conversation(max_messages=1, summariser=summariser())
    exchange(target, "a")
    exchange(target, "b")
    target.to_messages()
    assert target.truncations()

    target.adopt(conversation())

    assert target.truncations() == []
