"""Tests for `app.py`, the chat surface — rendered headlessly, no browser and no provider.

The app holds no logic of its own (`agent.session` does), so these cover only what the surface
itself can get wrong: that the page renders, that a turn reaches the trace through it, and that
the run it *names* is the run the turn was actually logged under. That last one is not cosmetic.
Streamlit renders top to bottom, so a sidebar drawn before the turn reports the run as "not
started" for a message already on disk, which is the opposite of provenance.

Skipped unless the optional [app] extra is installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="needs the [app] extra: pip install -e '.[app]'")

from streamlit.testing.v1 import AppTest  # noqa: E402 - must follow the skip guard

import agent.models.base as models_base  # noqa: E402
from agent.trace import read_records  # noqa: E402
from app import AVATARS, STARTERS  # noqa: E402 - the labels and avatars under test
from tests.fakes import FakeAdapter  # noqa: E402
from tests.runs import chat_manifest, chat_turn, write_trace  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"
ANSWER = '{"final": "400-800 ml per hour. [hydration.md#1]", "citations": ["hydration.md#1"]}'


@pytest.fixture
def app(monkeypatch, tmp_path):
    """The app with a fake provider, writing its runs under `tmp_path`."""
    monkeypatch.setattr(
        models_base,
        "load_agent_model",
        lambda which, **kwargs: FakeAdapter([ANSWER], model_id=f"fake-{which}"),
    )
    monkeypatch.chdir(tmp_path)
    return AppTest.from_file(str(APP), default_timeout=60)


def captions(at: AppTest) -> str:
    return " ".join(caption.value for caption in at.sidebar.caption)


def test_the_page_renders_both_arms_and_a_reset(app):
    at = app.run()

    assert not at.exception
    assert [button.label for button in at.sidebar.button] == ["Reset conversation"]
    assert at.sidebar.radio[0].value == "frontier"
    assert len(at.sidebar.radio[0].options) == 2


def test_the_arm_selector_names_the_model_that_will_serve_it(app, monkeypatch):
    """The regression this exists for: the selector read "Frontier (Claude)" from a hardcoded
    label while `FRONTIER_PROVIDER=gemini` sent every turn to Google. A demo surface that names
    the wrong model is worse than one that names none, since the screenshot is the claim."""
    monkeypatch.setenv("FRONTIER_PROVIDER", "gemini")
    monkeypatch.setenv("FRONTIER_MODEL", "gemini-3.6-flash")
    at = app.run()

    labels = " ".join(at.sidebar.radio[0].options)
    assert "gemini-3.6-flash" in labels
    assert "Claude" not in labels


def test_the_badge_names_the_model_that_is_answering(app):
    """From the adapter the session is holding rather than from the environment, so the badge
    cannot outlive a `.env` edit made while the page was open."""
    at = app.run()

    assert "Frontier (fake-frontier)" in " ".join(m.value for m in at.main.markdown)


def test_an_idle_page_load_starts_no_run(app, tmp_path):
    at = app.run()

    assert "not started" in captions(at)
    assert not (tmp_path / "runs").exists()


def test_a_turn_is_answered_and_logged(app, tmp_path):
    at = app.run()
    at.chat_input[0].set_value("how much water while running?").run()

    assert not at.exception
    assert "400-800 ml per hour." in " ".join(m.value for m in at.chat_message[1].markdown)

    (trace,) = (tmp_path / "runs").glob("*.jsonl")
    records = read_records(trace)
    assert [r["role"] for r in records] == ["user", "assistant", "turn"]
    assert records[0]["content"] == "how much water while running?"


def test_the_sidebar_names_the_run_the_turn_was_logged_under(app, tmp_path):
    """The regression this file exists for: the caption is drawn after the turn, not before."""
    at = app.run()
    at.chat_input[0].set_value("how much water?").run()

    (manifest,) = (tmp_path / "runs").glob("*.manifest.json")
    run_id = manifest.name.removesuffix(".manifest.json")
    assert run_id in captions(at)
    assert f"runs/{run_id}.jsonl" in captions(at)


def test_the_turn_panel_reports_what_the_turn_cost(app):
    at = app.run()
    at.chat_input[0].set_value("how much water?").run()

    assert "Turn detail" in [expander.label for expander in at.expander]
    assert {metric.label for metric in at.metric} == {"Latency", "Tokens", "Cost", "Model calls"}
    assert "**Tool calls** (0)" in " ".join(m.value for m in at.markdown)


def test_an_empty_transcript_offers_questions_to_open_with(app):
    """An opening screen with nothing on it but an input box is how a demo goes unopened."""
    at = app.run()

    assert [button.label for button in at.main.button] == [label for label, _ in STARTERS]


def test_a_starter_sends_the_question_rather_than_its_label(app, tmp_path):
    """The buttons are labelled short to fit the column; the trace has to record what a person
    would have typed, not the caption they clicked."""
    label, question = STARTERS[0]
    at = app.run()
    (starter,) = [button for button in at.main.button if button.label == label]

    at = starter.click().run()

    assert not at.exception
    (trace,) = (tmp_path / "runs").glob("*.jsonl")
    assert [r["content"] for r in read_records(trace) if r["role"] == "user"] == [question]
    assert [button.label for button in at.main.button] == [], "the panel is an empty state"


def test_switching_arm_mints_a_new_run_and_the_sidebar_follows(app, tmp_path):
    at = app.run()
    at.chat_input[0].set_value("first").run()
    first = captions(at)

    at.sidebar.radio[0].set_value("oss").run()
    at.chat_input[0].set_value("second").run()

    assert not at.exception
    assert captions(at) != first
    manifests = sorted((tmp_path / "runs").glob("*.manifest.json"))
    assert len(manifests) == 2
    assert any(m.name.removesuffix(".manifest.json") in captions(at) for m in manifests)


def test_switching_arm_keeps_the_transcript(app):
    """The symptom this fixes: the exchange you were reading vanishing because you changed arm."""
    at = app.run()
    at.chat_input[0].set_value("how much water?").run()
    at.sidebar.radio[0].set_value("oss").run()

    assert not at.exception
    rendered = " ".join(markdown.value for markdown in at.markdown)
    assert "how much water?" in rendered
    assert "400-800 ml per hour." in rendered


def test_the_transcript_marks_where_the_model_changed(app):
    """Two arms in one transcript is only readable if it says which answer came from which."""
    at = app.run()
    at.chat_input[0].set_value("how much water?").run()
    at.sidebar.radio[0].set_value("oss").run()

    assert any("OSS (fake-oss)" in caption.value for caption in at.caption)


def test_switching_arm_before_anything_is_said_marks_nothing(app):
    """Picking an arm before typing is how the page is ordinarily started, and there is nothing
    above the marker for it to say was carried over."""
    at = app.run()
    at.sidebar.radio[0].set_value("oss").run()

    assert not at.exception
    assert not [caption for caption in at.caption if "Switched to" in caption.value]


def test_a_bubble_keeps_the_avatar_of_the_arm_that_answered_it(app):
    """The marker says where the model changed. Without a mark per bubble, every answer above it
    still looks in scroll-back like the arm now selected produced it."""
    at = app.run()
    at.chat_input[0].set_value("how much water?").run()
    at.sidebar.radio[0].set_value("oss").run()

    assert at.chat_message[1].avatar == AVATARS["frontier"]


def test_the_switch_marker_is_not_an_assistant_message(app):
    """It is the app talking. A bubble would attribute our sentence to a model, and to the arm
    that had just been selected rather than the one that produced anything."""
    at = app.run()
    at.chat_input[0].set_value("how much water?").run()
    bubbles = len(at.chat_message)
    at.sidebar.radio[0].set_value("oss").run()

    assert len(at.chat_message) == bubbles


def test_switching_arm_twice_without_sending_keeps_the_transcript(app):
    """`use` rotates lazily, so the agent still holds the history here; the display must too."""
    at = app.run()
    at.chat_input[0].set_value("how much water?").run()
    at.sidebar.radio[0].set_value("oss").run()
    at.sidebar.radio[0].set_value("frontier").run()

    assert not at.exception
    assert "how much water?" in " ".join(markdown.value for markdown in at.markdown)


def test_reset_still_clears_the_transcript(app):
    at = app.run()
    at.chat_input[0].set_value("how much water?").run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert "how much water?" not in " ".join(markdown.value for markdown in at.markdown)


def past_conversation(tmp_path, run_id="run-earlier", item_id="chat-earlier", turns=1):
    """A finished chat session on disk, as a previous run of the app would have left one."""
    chat_manifest(tmp_path, run_id)
    write_trace(
        tmp_path,
        run_id,
        [
            record
            for idx in range(turns)
            for record in chat_turn(item_id, f"earlier question {idx}", turn_idx=idx)
        ],
    )


def sidebar_button(at, label):
    """The sidebar button called `label`. By label, since there are now two of them."""
    (found,) = [button for button in at.sidebar.button if button.label == label]
    return found


def test_the_resume_control_offers_nothing_when_nothing_is_recorded(app):
    at = app.run()

    assert not at.exception
    assert any("Nothing else has been recorded" in c.value for c in at.sidebar.caption)
    assert [button.label for button in at.sidebar.button] == ["Reset conversation"]


def test_the_resume_control_excludes_the_conversation_already_open(app):
    """Resuming what is loaded would rebuild the history the live agent is holding and carry it
    onto a second copy of itself."""
    at = app.run()
    at.chat_input[0].set_value("how much water?").run()

    assert not at.exception
    assert any("Nothing else has been recorded" in c.value for c in at.sidebar.caption)


def test_the_resume_control_offers_a_past_conversation(app, tmp_path):
    past_conversation(tmp_path, turns=2)
    at = app.run()

    assert not at.exception
    (selector,) = at.sidebar.selectbox
    assert "chat-earlier — 2 turn(s)" in selector.options[0]


def test_resuming_puts_the_earlier_history_in_the_next_run(app, tmp_path):
    past_conversation(tmp_path)
    at = app.run()
    at = sidebar_button(at, "Resume").click().run()
    at.chat_input[0].set_value("and in the heat?").run()

    assert not at.exception
    traces = {path.stem: read_records(path) for path in (tmp_path / "runs").glob("*.jsonl")}
    (new_run,) = [run_id for run_id in traces if run_id != "run-earlier"]
    (carried,) = [r for r in traces[new_run] if r["role"] == "memory"]
    assert json.loads(carried["content"])["previous_run_id"] == "run-earlier"
    assert [r["turn_idx"] for r in traces[new_run] if r["role"] == "user"] == [1]


def test_resuming_says_what_was_resumed(app, tmp_path):
    past_conversation(tmp_path, turns=3)
    at = app.run()
    at = sidebar_button(at, "Resume").click().run()

    assert not at.exception
    assert any("Resumed `chat-earlier`" in caption.value for caption in at.caption)
    assert any("3 earlier turn(s)" in caption.value for caption in at.caption)


def test_resuming_writes_nothing_until_a_message_is_sent(app, tmp_path):
    past_conversation(tmp_path)
    before = sorted(path.name for path in (tmp_path / "runs").iterdir())
    at = app.run()
    at = sidebar_button(at, "Resume").click().run()

    assert not at.exception
    assert "not started" in captions(at)
    assert sorted(path.name for path in (tmp_path / "runs").iterdir()) == before


def test_a_conversation_that_moved_on_is_offered_once_at_its_latest_run(app, tmp_path):
    """It is spread over two runs, and only the later one can be picked up."""
    past_conversation(tmp_path, run_id="run-a", item_id="chat-1")
    chat_manifest(tmp_path, "run-b")
    write_trace(
        tmp_path,
        "run-b",
        [
            {
                "item_id": "chat-1",
                "turn_idx": 1,
                "role": "memory",
                "content": json.dumps({"event": "carried_over", "previous_run_id": "run-a"}),
            },
            *chat_turn("chat-1", "the later question", turn_idx=1),
        ],
    )
    at = app.run()

    (selector,) = at.sidebar.selectbox
    assert len(selector.options) == 1


def test_resuming_one_of_several_leaves_the_others_selectable(app, tmp_path):
    """The resumed conversation leaves the list, since it is now the live one, and the selector has
    to survive its own selection disappearing."""
    past_conversation(tmp_path, run_id="run-one", item_id="chat-one")
    past_conversation(tmp_path, run_id="run-two", item_id="chat-two")
    at = app.run()
    assert len(at.sidebar.selectbox[0].options) == 2

    at = sidebar_button(at, "Resume").click().run()

    assert not at.exception
    (remaining,) = at.sidebar.selectbox[0].options
    assert "chat-one" not in remaining


def test_a_broken_chain_is_an_error_rather_than_half_a_history(app, tmp_path):
    """A trace named by a carry-over record has been deleted. Half a history is a model that
    appears to have forgotten the middle of a conversation it is holding the end of."""
    chat_manifest(tmp_path, "run-orphan")
    write_trace(
        tmp_path,
        "run-orphan",
        [
            {
                "item_id": "chat-1",
                "turn_idx": 1,
                "role": "memory",
                "content": json.dumps({"event": "carried_over", "previous_run_id": "run-gone"}),
            },
            *chat_turn("chat-1", "orphaned question", turn_idx=1),
        ],
    )
    at = app.run()
    at = sidebar_button(at, "Resume").click().run()

    assert not at.exception
    assert any("run-gone" in error.value for error in at.error)


def test_a_missing_api_key_is_an_error_message_not_a_traceback(monkeypatch, tmp_path):
    """A demo surface that shows a stack trace for a missing key is not a demo surface."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FRONTIER_PROVIDER", raising=False)
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not at.exception
    assert any(".env" in error.value for error in at.error)


def test_a_misconfigured_provider_is_still_an_error_message_not_a_traceback(monkeypatch, tmp_path):
    """The arm labels resolve the same configuration the session does, and they are drawn first.
    A label that raised on a typo would pre-empt the sentence naming the variable with a
    traceback from the sidebar."""
    monkeypatch.setenv("FRONTIER_PROVIDER", "bedrock")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not at.exception
    assert any("bedrock" in error.value for error in at.error)
