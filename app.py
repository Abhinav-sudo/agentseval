"""Chat surface: `streamlit run app.py`.

A thin surface for exercising the agent by hand. The deliverable is the evals platform, not the
chatbot (PROJECT.md), so this file renders widgets and nothing else: the run lifecycle, the
manifest, and the trace belong to `agent.session`. Turns are logged to `runs/` like any other
turn, under a manifest with `run_kind="chat"` — an interactive session is still data, and if the
app and the runner drove the agent differently, what is observed here would stop predicting what
the evals measure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import streamlit as st

from agent.manifest import AgentConfig
from agent.models.base import ConfigError, ModelError, load_agent_model, load_env
from agent.models.frontier import resolved_frontier_model
from agent.models.oss import resolved_oss_model
from agent.session import ChatSession, ConversationRef, resumable_conversations, turn_detail

Arm = Literal["frontier", "oss"]

#: The role each arm plays. Deliberately no model id: which model serves an arm is configuration
#: (`FRONTIER_PROVIDER`, `FRONTIER_MODEL`, `OSS_MODEL`), and a hardcoded id here is a caption that
#: goes on being drawn after the configuration moves — this page announced "Frontier (Claude)" over
#: a Gemini run for exactly that reason. The id is resolved per render below.
ARM_ROLES: dict[Arm, str] = {"frontier": "Frontier", "oss": "OSS"}

#: Where each arm's model id comes from. The same expressions the adapters resolve their own
#: `model_id` with, so the label and the request cannot disagree.
RESOLVERS: dict[Arm, Callable[[], tuple[str, str]]] = {
    "frontier": resolved_frontier_model,
    "oss": resolved_oss_model,
}

#: One avatar per arm, carried on each assistant message. A conversation can cross a model switch,
#: and the marker below only says where that happened: without a per-bubble mark, scroll-back leaves
#: every earlier answer looking as though the arm now selected produced it.
AVATARS: dict[Arm, str] = {"frontier": ":material/bolt:", "oss": ":material/memory:"}

#: Offered on an empty transcript, as `(button, question)`. Short labels because they are stacked
#: in a centered column; the question sent is the whole sentence, so the trace records what a
#: person would have typed rather than a caption.
STARTERS: tuple[tuple[str, str], ...] = (
    ("Hydration on a long run", "How much water should I drink on a long run?"),
    ("Falling asleep faster", "What should I change to fall asleep faster?"),
    ("Rest days between sessions", "How many rest days do I need between strength sessions?"),
)

#: Carries a starter question from the click that chose it into the rerun that sends it, so that a
#: click is answered exactly as a typed message is rather than through a second code path.
PENDING_KEY = "pending_prompt"

#: Marks a transcript entry that is the app talking rather than a turn. Drawn as a rule and a
#: caption rather than a chat bubble: a model switch is a fact about the run, and dressing it as an
#: assistant message would put words in the model's mouth — including in the scroll-back of the arm
#: that never said them.
MARKER = "marker"

#: Session-state key behind the resume selector.
RESUME_KEY = "resume_choice"


def config_for(which: str) -> AgentConfig:
    """Build the config for one arm; everything else about the run is shared by construction.

    `which` comes from `ARM_ROLES`, and `load_agent_model` rejects anything else, so the cast
    cannot hide a bad value.
    """
    return AgentConfig(model=load_agent_model(cast(Arm, which)))


def arm_label(arm: Arm) -> str:
    """Label `arm` with the model id the environment says will serve it.

    Read from the environment rather than off an adapter, because the selector is drawn before any
    session exists and building an adapter to read `model_id` would demand a credential to draw a
    label. A misconfigured provider degrades to the bare role name: the same resolution runs again
    a few lines below, where the failure is caught and shown as a sentence naming the variable, and
    a label that raised would pre-empt that with a traceback from the sidebar.
    """
    try:
        return f"{ARM_ROLES[arm]} ({RESOLVERS[arm]()[1]})"
    except ConfigError:
        return ARM_ROLES[arm]


def live_label(arm: Arm, session: ChatSession) -> str:
    """Label `arm` with the model the session is actually holding for it.

    For anything drawn after the session has been built or rotated, which is where it beats
    `arm_label`: this reports the adapter that will answer, so a `.env` edited while the page was
    open cannot move the caption off the model the next turn is sent to.
    """
    return f"{ARM_ROLES[arm]} ({session.config.model.model_id})"


def render_detail(detail: dict[str, Any]) -> None:
    """Show what one turn actually did: calls, retrieval, latency, tokens, cost."""
    calls = detail["tool_calls"]
    st.markdown(f"**Tool calls** ({len(calls)})")
    if calls:
        for call in calls:
            st.code(f"{call['name']}({call['arguments']})", language="python")
    else:
        st.caption("none — answered without retrieval")

    st.markdown("**Retrieved chunks**")
    scored = dict(detail["retrieved"])
    # An id with no score came from a tool that reports none, e.g. web search.
    unscored = [c for c in detail["retrieved_chunk_ids"] if c not in scored]
    if scored:
        rows = [{"chunk_id": c, "score": round(s, 4)} for c, s in scored.items()]
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            column_config={
                "chunk_id": st.column_config.TextColumn("chunk", width="medium"),
                "score": st.column_config.NumberColumn(
                    "cosine score",
                    format="%.4f",
                    help="Similarity to the query, as the retrieval tool reported it.",
                ),
            },
        )
    if unscored or not scored:
        st.caption(", ".join(unscored) or "none")

    cost = detail["usd_cost"]
    columns = st.columns(4)
    columns[0].metric("Latency", f"{detail['latency_ms'] / 1000:.2f} s", border=True)
    columns[1].metric("Tokens", f"{detail['total_tokens'] or 0:,}", border=True)
    columns[2].metric("Cost", "unpriced" if cost is None else f"${cost:.5f}", border=True)
    columns[3].metric("Model calls", detail["model_calls"], border=True)
    cited = ", ".join(detail["citations"]) or "nothing"
    st.caption(
        f"prompt {detail['prompt_tokens'] or 0:,} + completion "
        f"{detail['completion_tokens'] or 0:,} tokens · cited {cited}"
    )

    if detail["cached"]:
        # A cache hit replays the original call's latency, so the figure above is not a
        # measurement of this call (PROJECT.md).
        st.caption("Served partly from the response cache: latency is replayed, not measured.")
    if detail["tool_errors"]:
        st.caption(f"{detail['tool_errors']} tool call(s) rejected as invalid.")
    if detail["stopped_reason"] != "answered":
        st.warning(f"Turn ended on `{detail['stopped_reason']}`.")
    if detail["format_violation"]:
        st.warning(f"Protocol violation: `{detail['format_violation']}`.")
    if detail["budget_induced_truncations"]:
        # Our token ceiling cut the reply off, which is a fact about the harness (PROJECT.md).
        st.warning(f"{detail['budget_induced_truncations']} response(s) cut off at max_tokens.")
    if detail["infrastructure_failed"]:
        st.error("A tool failed on our side; an eval would exclude this item from scoring.")


def run_turn(session: ChatSession, prompt: str, arm: Arm) -> None:
    """Send one message and render the answer with its detail panel.

    `arm` is passed in and stored on the message rather than read back from the session, so that a
    bubble keeps the avatar of the arm that answered it after the selector has moved on.
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATARS[arm]):
        status = st.status("Thinking...", type="compact")
        try:
            result = session.send(prompt)
        except ModelError as exc:
            # Drawn outside the status, which is collapsed: an error folded into a closed
            # container is an error nobody reads.
            status.update(label="The provider call failed.", state="error")
            st.error(str(exc))
            return
        detail = turn_detail(result)
        # The measured figure rather than a wall-clock one, so this line and the panel below it
        # cannot disagree — including on a cache hit, where both are replayed.
        status.update(label=f"Answered in {detail['latency_ms'] / 1000:.2f} s", state="complete")
        st.markdown(result.final_text or "_The model produced no answer._")
        with st.expander("Turn detail"):
            render_detail(detail)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.final_text, "detail": detail, "arm": arm}
    )


def choose_starter(question: str) -> None:
    """Hand a starter question to the next rerun, from the click that chose it."""
    st.session_state[PENDING_KEY] = question


def starter_prompts() -> None:
    """The empty state: what this surface is, and three questions to open it with.

    A click is handled in `choose_starter` rather than by a truthy return, because a callback runs
    before the script does: the rerun it causes finds the question already waiting and skips this
    panel, where a mid-script `st.rerun()` would leave the half-drawn panel on screen — faded, but
    on screen — for as long as the model call takes.

    Buttons rather than `st.pills`, whose selection would survive in session state and re-fire the
    moment a reset brought this panel back.
    """
    with st.container(border=True):
        st.markdown("**Ask the agent under test**")
        st.caption(
            "Every turn here is logged to `runs/` under a manifest, exactly as an eval turn is. "
            "The panel under each answer shows what the turn retrieved, what it cost, and anything "
            "that happened to it other than being answered."
        )
        for label, question in STARTERS:
            st.button(
                label,
                icon=":material/arrow_outward:",
                width="stretch",
                on_click=choose_starter,
                args=(question,),
            )


def describe_conversation(one: ConversationRef) -> str:
    """One resumable conversation, labelled so it is not chosen blind."""
    return f"{one.item_id} — {one.turns} turn(s), {one.model_name}, {one.started_at}"


def resume_control(session: ChatSession) -> None:
    """Offer a conversation recorded under `runs/` to pick up in this session.

    Only the latest segment of each conversation is offered, and the one already open here is left
    out — resuming what is already loaded would rebuild the history the live agent is holding and
    carry it onto a second copy of itself.

    Resuming does not redraw the earlier turns. They are on disk with everything that was measured
    about them, which is more than this transcript keeps, so the marker points at the Chat history
    page rather than reconstructing bubbles here.
    """
    with st.expander("Resume a past conversation"):
        found = [
            one
            for one in resumable_conversations(session.runs_dir)
            if one.item_id != session.item_id
        ]
        if not found:
            st.caption("Nothing else has been recorded under `runs/` yet.")
            return

        chosen = st.selectbox(
            "Conversation",
            found,
            format_func=describe_conversation,
            key=RESUME_KEY,
        )
        if not st.button("Resume"):
            return
        try:
            turns = session.resume(chosen.run_id, chosen.item_id)
        except ValueError as exc:
            # A broken carry-over chain, most likely a deleted trace. Said out loud rather than
            # resumed part-way: half a history is a model that appears to have forgotten the middle.
            st.error(str(exc))
            return

        st.session_state.messages = [
            {
                "kind": MARKER,
                "role": "system",
                "content": (
                    f"Resumed `{chosen.item_id}` from run `{chosen.run_id}`. Its {turns} earlier "
                    "turn(s) are in the model's context but are not redrawn here — read them on "
                    "the Chat history page (`streamlit run ui/dashboard.py`). The next message "
                    "opens a new run."
                ),
            }
        ]
        # From the top, because this list and the transcript were both drawn from the state this
        # click has just replaced: without it the selector goes on offering the conversation now
        # loaded, and clicking it again would rebuild the history the live agent is already holding.
        st.rerun()


def main() -> None:
    """Render the page: arm selector, transcript, and a detail panel per assistant turn."""
    load_env()
    # An emoji favicon rather than a Material icon, which Streamlit draws black for a favicon
    # whatever the browser's theme is and which therefore vanishes into a dark tab strip.
    st.set_page_config(page_title="AgentsEval chat", page_icon="💬", layout="centered")
    st.logo(":material/forum:", size="large")
    st.title("AgentsEval")
    st.caption("A demo surface for the agent under test. The platform is `evals/`.")

    with st.sidebar:
        st.header("Run")
        arm = st.radio(
            "Model",
            list(ARM_ROLES),
            format_func=arm_label,
            help="Both arms share one harness: same prompt, tools, budgets, and JSON protocol.",
        )
        reset = st.button("Reset conversation")

    st.session_state.setdefault("messages", [])
    try:
        if "session" not in st.session_state:
            st.session_state.session = ChatSession(config_for(arm))
            st.session_state.arm = arm
        session: ChatSession = st.session_state.session
        if st.session_state.arm != arm:
            # The session decides what this means for the run: the next message lands under a
            # new run_id and a new manifest, since a trace holding two models under one
            # manifest would be unattributable. The conversation itself continues — the history
            # moves to the new agent — so the transcript is marked rather than cleared.
            session.use(config_for(arm))
            st.session_state.arm = arm
            # Only where there is something above to carry. The rotation happens either way, but on
            # an empty transcript the sentence would be announcing the carrying of nothing — and
            # picking an arm before typing is how the page is ordinarily started.
            if st.session_state.messages:
                st.session_state.messages.append(
                    {
                        "kind": MARKER,
                        "role": "system",
                        "content": (
                            f"Switched to {live_label(arm, session)}. Everything above is carried "
                            "over; the next message opens a new run."
                        ),
                    }
                )
    except ModelError as exc:
        st.error(f"{exc}\n\nSet the provider's key in `.env` (see `.env.example`).")
        return

    if reset:
        session.reset()
        st.session_state.messages = []

    # After the session exists, since it is what gets resumed into. It reruns on a successful
    # resumption, so nothing below is drawn from the state that resumption replaced.
    with st.sidebar:
        resume_control(session)

    # The live arm, in the main pane rather than only in the sidebar radio, so a screenshot of the
    # transcript says which model produced it. The arm and not the run: the run is minted by the
    # turn below, and a badge drawn here could only report the previous one.
    st.badge(live_label(arm, session), icon=AVATARS[arm], color="violet")

    # Pinned to the bottom of the page wherever it is called from, so it can be read before the
    # transcript is drawn — which is what lets the panel below know whether a question is coming.
    typed = st.chat_input("Ask about sleep, hydration, training, stress...")
    prompt = st.session_state.pop(PENDING_KEY, None) or typed

    if not st.session_state.messages and prompt is None:
        starter_prompts()

    for message in st.session_state.messages:
        if message.get("kind") == MARKER:
            st.divider()
            st.caption(message["content"])
            continue
        # `.get` for both, since a user message carries no arm and a message recorded before this
        # field existed carries none either; None is Streamlit's own default avatar.
        with st.chat_message(message["role"], avatar=AVATARS.get(message.get("arm"))):
            st.markdown(message["content"] or "_no answer_")
            if message.get("detail"):
                with st.expander("Turn detail"):
                    render_detail(message["detail"])

    if prompt:
        run_turn(session, prompt, arm)

    # Drawn last on purpose. The run is minted by the turn above, so a sidebar written before it
    # would report "not started" for a message that had already been logged — the one thing this
    # caption exists to get right.
    with st.sidebar, st.container(border=True):
        st.caption("**Provenance**")
        st.caption(f"run_id `{session.run_id or 'not started'}` · chat `{session.item_id}`")
        st.caption(f"trace: `{session.trace_path or 'written on first message'}`")


if __name__ == "__main__":
    main()
