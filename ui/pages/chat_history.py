"""One past chat conversation, read back out of the trace that recorded it.

The chat surface keeps its transcript in Streamlit session state, which lasts as long as the browser
tab does. The trace lasts as long as `runs/` does, and it holds strictly more: every question, the
delivered answer, what each turn cost, and whether a guardrail replaced anything. So this page needs
no new artifact — the history was already on disk, and nothing was reading it.

Two things it deliberately does not do:

* **It shows one segment per run, and names the predecessor rather than stitching the chain.** A
  conversation that crossed a model switch is spread over several runs by design, because each run
  records one set of conditions. Splicing the segments into one scroll would show two models under
  one heading and invite exactly the comparison a manifest exists to prevent. `agent.session`'s
  `conversation_from_trace` does walk the chain, because rebuilding what a model was told is a
  different job from displaying what a person saw.
* **It shows the delivered text.** On a screened turn that is the guardrail's sentence, not the
  model's own completion, which is the opposite of what every scorer wants — and `guardrail_action`
  is drawn beside it so the substitution is visible rather than silent.
"""

from __future__ import annotations

import streamlit as st

from ui.data import ChatThread, ChatTurn, RunRef, chat_runs, chat_threads
from ui.layout import READ_ONLY_NOTE, configure_page, load_runs

#: Session-state keys behind the two selectors, so a test can open a conversation without a URL
#: scheme — the reason `DETAIL_RUN_KEY` exists. Not in `ui.layout`: no other page shares these.
CHAT_RUN_KEY = "chat_run_id"
CHAT_THREAD_KEY = "chat_item_id"


def main() -> None:
    """Pick a chat run, pick a conversation in it, and render the exchange."""
    configure_page("Chat history")
    st.title("Chat history")
    st.caption(READ_ONLY_NOTE)

    runs, root = load_runs()
    sessions = chat_runs(runs)
    if not sessions:
        st.warning(
            f"No chat run under `{root}`. Chat sessions are written by `app.py` — "
            "`streamlit run app.py` — and appear here once a message has been sent."
        )
        return

    run = _pick_run(sessions)
    threads = chat_threads(run)
    if not threads:
        st.info(f"`{run.run_id}` has a manifest but no conversation in its trace yet.")
        return

    thread = _pick_thread(threads)
    _render(run, thread)


def _pick_run(sessions: list[RunRef]) -> RunRef:
    """The chat run selector, labelled by model so the ids are not chosen blind."""
    run_ids = [run.run_id for run in sessions]
    chosen = st.selectbox(
        "Chat run",
        run_ids,
        format_func=lambda run_id: _describe_run(sessions, run_id),
        key=CHAT_RUN_KEY,
    )
    return next(run for run in sessions if run.run_id == chosen)


def _describe_run(sessions: list[RunRef], run_id: str) -> str:
    run = next(one for one in sessions if one.run_id == run_id)
    return f"{run_id} — {run.manifest.model_name}, started {run.manifest.started_at}"


def _pick_thread(threads: list[ChatThread]) -> ChatThread:
    """The conversation selector. A run holds several when the chat was reset without rotating."""
    item_ids = [thread.item_id for thread in threads]
    chosen = st.selectbox(
        "Conversation",
        item_ids,
        format_func=lambda item_id: _describe_thread(threads, item_id),
        key=CHAT_THREAD_KEY,
    )
    return next(thread for thread in threads if thread.item_id == chosen)


def _describe_thread(threads: list[ChatThread], item_id: str) -> str:
    thread = next(one for one in threads if one.item_id == item_id)
    turns = len(thread.turns)
    return f"{item_id} — {turns} turn(s), opened {thread.opened_at or 'unknown'}"


def _render(run: RunRef, thread: ChatThread) -> None:
    """Draw one conversation, with its provenance above it."""
    if thread.continues_run is not None:
        st.info(
            f"This is the later part of a conversation that began in **{thread.continues_run}**. "
            "The history moved here when a condition changed — the model toggle, most likely — "
            "which minted this run so that what each model was asked stays attributable. Select "
            f"`{thread.continues_run}` above to read the earlier turns."
        )

    st.caption(
        f"`{thread.item_id}` in `{run.run_id}` — {run.manifest.model_name} "
        f"({run.manifest.provider}), temperature {run.manifest.temperature}"
    )

    for turn in thread.turns:
        with st.chat_message("user"):
            st.markdown(turn.question or "_no question recorded_")
        with st.chat_message("assistant"):
            st.markdown(turn.answer or "_no answer: the turn did not finish_")
            st.caption(_turn_note(turn))


def _turn_note(turn: ChatTurn) -> str:
    """What the turn cost, and anything that happened to it other than being answered."""
    parts = [f"turn {turn.turn_idx}"]
    if turn.latency_ms is not None:
        parts.append(f"{turn.latency_ms:.0f} ms")
    if turn.usd_cost is not None:
        parts.append(f"${turn.usd_cost:.4f}")
    if turn.guardrail_action is not None and turn.guardrail_action != "none":
        parts.append(
            f"a guardrail fired (`{turn.guardrail_action}`), so the text above is what was "
            "delivered rather than what the model wrote"
        )
    if turn.stopped_reason:
        parts.append(f"ended as `{turn.stopped_reason}`")
    return " · ".join(parts)


if __name__ == "__main__":
    main()
