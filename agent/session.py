"""One interactive chat session, and the run lifecycle it implies.

The chat surface (`app.py`) is a demo, not the deliverable (PROJECT.md), so it holds no logic
of its own: it renders widgets and calls this. Everything that decides *what gets logged*
lives here, where it can be tested without a browser.

Three properties this module exists to hold:

* **One trace format, one code path.** A chat turn is logged by `agent.core.Agent` writing to
  an `agent.trace.TraceLogger`, under a manifest from `agent.manifest.build_manifest` — the
  same three objects the eval runner uses, with `run_kind="chat"` instead of `"eval"`. There is
  no chat-specific record shape and no chat-specific writer.
* **A manifest arrives before the turn it describes.** It is written lazily, on the first
  message rather than at session end: an abandoned session leaves an orphaned manifest, which
  is harmless, whereas a crashed one would otherwise leave a trace nothing can attribute. An
  idle page load writes neither file.
* **Changing a condition starts a new run, but not a new conversation.** Flipping the model
  toggle mid-session, or editing the corpus underneath it, changes what the run is measuring.
  Keeping the `run_id` would leave a trace holding two models under a manifest asserting one, and
  no downstream check could detect it — `assert_comparable` would pass, because it compares
  manifests and there would only be one. So a config change mints a new `run_id`, writes a new
  manifest, and routes subsequent turns to the new trace. What it does *not* do is discard the
  conversation: the history moves to the new agent, the `item_id` and turn numbering continue, and
  the new trace opens with a `role="memory"` record naming the run the context came from. Rotating
  the run and forgetting the conversation are separate things, and only the first is required.
  Manifests already written are never touched again.
* **A past conversation can be reopened, from the trace and nothing else.** `resume` rebuilds one
  through `conversation_from_trace` and hands it to the next agent by the same slot a switch uses,
  so reopening a conversation is a rotation whose history came off disk instead of off a live
  agent. It needs no new artifact: the trace already recorded every message, and it was the only
  thing not being read back.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent.core import Agent, AgentResult
from agent.manifest import (
    AgentConfig,
    RunKind,
    RunManifest,
    agent_config_digest,
    build_manifest,
)
from agent.memory import Conversation, estimate_messages_tokens
from agent.trace import (
    CARRY_OVER_EVENT,
    DEFAULT_RUNS_DIR,
    ROLE_ASSISTANT,
    ROLE_GUARDRAIL,
    ROLE_MEMORY,
    ROLE_TOOL,
    ROLE_USER,
    TraceLogger,
    by_conversation,
    carried_over_from,
    latest_segments,
    read_records,
    trace_path,
    utc_now_iso,
)

#: The `run_kind` a chat session's manifest carries. Named because this module both writes it and,
#: in `resumable_conversations`, reads it back to tell its own runs from an eval run's.
CHAT_RUN_KIND: RunKind = "chat"


def new_conversation_id() -> str:
    """Mint an `item_id` for one conversation within a run.

    Trace records are keyed by `(item_id, turn_idx)`, so a reset starts a new conversation id
    rather than a new run: the history the model sees is gone, but the conditions are
    unchanged, and the trace stays readable as two conversations under one manifest.
    """
    return f"chat-{uuid.uuid4().hex[:8]}"


def conversation_from_trace(
    run_id: str,
    item_id: str,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    system_prompt: str = "",
) -> Conversation:
    """Rebuild the conversation `item_id` had reached at the end of `run_id`.

    Reconstructs *the model's memory*, not the transcript a person read. Those differ on a screened
    turn, and the difference decides what the resumed model is conditioned on:

    * On an output substitution, `Agent.run_turn` screens after the loop, so the model's own
      completion is already in the conversation and our replacement sentence never enters it. So the
      `assistant` record is preferred and the `guardrail` record ignored.
    * On an input block there is no completion — the turn never reached the model — and the safe
      answer *is* what memory holds, so the `guardrail` record is the only source for that turn.

    Preferring the completion and falling back is what handles both without asking which happened.

    Walks the carry-over chain backward to the run the conversation began in and replays the
    segments head-first, because a conversation that crossed a model switch is spread over several
    traces by design; reading only the run named here would rebuild its last segment and restart
    numbering low enough to collide with records already written.

    Args:
        system_prompt: Belongs to whoever will speak next, so the default is empty and the caller
            that has an agent supplies it. `Conversation.adopt` does not carry one either.

    Returns:
        The full verbatim history, not the compacted view. The trace holds every message ever sent,
        and the resuming agent re-compacts it under its own budget and summariser.

    Raises:
        ValueError: The chain is broken — a trace named by a carry-over record is missing — or it
            loops. Either way the history is incomplete, and a partial one silently returned is a
            model appearing to forget the middle of a conversation it is holding the end of.
    """
    segments: list[list[dict[str, Any]]] = []
    seen: list[str] = []
    current: str | None = run_id
    while current is not None:
        if current in seen:
            chain = " -> ".join([*seen, current])
            raise ValueError(f"carry-over chain loops: {chain}")
        seen.append(current)

        path = trace_path(current, runs_dir)
        if not path.exists():
            raise ValueError(
                f"cannot rebuild {item_id}: {path} is missing, and it holds the earlier part of "
                f"this conversation (reached from {seen[0]})"
            )
        records = [record for record in read_records(path) if record.get("item_id") == item_id]
        segments.append(records)
        current = carried_over_from(records)

    conversation = Conversation(system_prompt=system_prompt)
    for records in reversed(segments):
        _replay(conversation, records)
    return conversation


def _replay(conversation: Conversation, records: Sequence[Mapping[str, Any]]) -> None:
    """Add one segment's messages to `conversation`, in the order they were written.

    A `role="assistant"` record carrying an `error` is not a message: `_call_model` logs the gap
    and then raises, before anything reaches memory. Replaying it would put an empty assistant
    message in a history that never held one.
    """
    spoken = {
        _turn_key(record)
        for record in records
        if record.get("role") == ROLE_ASSISTANT and not record.get("error")
    }
    for record in records:
        role = record.get("role")
        content = str(record.get("content") or "")
        if role == ROLE_USER:
            conversation.add_user(content)
        elif role == ROLE_TOOL:
            conversation.add_tool_result(content)
        elif _is_remembered_answer(record, spoken):
            conversation.add_assistant(content)


def _is_remembered_answer(record: Mapping[str, Any], spoken: set[tuple[Any, Any]]) -> bool:
    """Whether `record` is the assistant message this turn left in the conversation.

    The model's own completion when the turn produced one, and a guardrail's substitute only for a
    turn that produced none. See `conversation_from_trace` for why that is the right way round.
    """
    role = record.get("role")
    if role == ROLE_ASSISTANT:
        return not record.get("error")
    return role == ROLE_GUARDRAIL and _turn_key(record) not in spoken


def _turn_key(record: Mapping[str, Any]) -> tuple[Any, Any]:
    """What identifies the turn a record belongs to."""
    return record.get("item_id"), record.get("turn_idx")


@dataclass(frozen=True)
class ConversationRef:
    """One conversation found on disk, enough to label it and to resume it.

    Attributes:
        run_id: The run holding its latest segment, which is the one `resume` is given.
        started_at: When that run started, not when the conversation did. A conversation that
            crossed a switch began in an earlier run, named by `continues_run`.
        turns: Questions asked across the whole conversation, not just the segment in `run_id`.
            The count is what identifies a conversation in a picker, and one that said "1 turn" for
            a conversation the model is holding six turns of would be worse than no label.
    """

    run_id: str
    item_id: str
    model_name: str
    started_at: str
    turns: int
    continues_run: str | None = None


def resumable_conversations(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[ConversationRef]:
    """Every chat conversation under `runs_dir` that can be picked up, newest run first.

    `runs_dir` itself rather than its subdirectories, because a `ChatSession` writes its manifest
    and its trace side by side there and `resume` reads them back from the same one directory. A
    chat run filed under `runs/somewhere/` is browsable on the history page and is not offered here.

    Superseded segments are left out — see `latest_segments` — and each survivor's `turns` counts
    the whole chain behind it rather than its own segment. A run with a manifest and no trace, or a
    trace holding no question, contributes nothing rather than an unresumable entry.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []

    found: list[ConversationRef] = []
    for path in sorted(runs_dir.glob("*.manifest.json")):
        try:
            manifest = RunManifest.read(path)
        except (OSError, ValueError):
            # A manifest that will not parse is a problem for the browser to report, and this is a
            # picker: an entry nothing could resume is worse here than an omission.
            continue
        if manifest.run_kind != CHAT_RUN_KIND:
            continue

        trace = trace_path(manifest.run_id, runs_dir)
        if not trace.exists():
            continue
        for item_id, records in by_conversation(read_records(trace)).items():
            turns = sum(1 for record in records if record.get("role") == ROLE_USER)
            if not turns:
                continue
            found.append(
                ConversationRef(
                    run_id=manifest.run_id,
                    item_id=item_id,
                    model_name=manifest.model_name,
                    started_at=manifest.started_at,
                    turns=turns,
                    continues_run=carried_over_from(records),
                )
            )

    tips = latest_segments((one.run_id, one.item_id, one.continues_run) for one in found)
    segments = {(one.run_id, one.item_id): one for one in found}
    resumable = [
        replace(one, turns=_turns_behind(one, segments))
        for one in found
        if (one.run_id, one.item_id) in tips
    ]
    return sorted(resumable, key=lambda one: one.started_at, reverse=True)


def _turns_behind(tip: ConversationRef, segments: Mapping[tuple[str, str], ConversationRef]) -> int:
    """Questions asked across `tip`'s whole chain, counting the segments already read.

    From `segments` rather than by reading the earlier traces again, since the caller has just read
    every one of them. A chain that leaves what it has been given — a run whose trace has since been
    deleted — stops there and returns the part it can see: this labels a picker, and `resume` is the
    one that has to refuse an incomplete history.
    """
    total = 0
    at: ConversationRef | None = tip
    seen: set[tuple[str, str]] = set()
    while at is not None and (at.run_id, at.item_id) not in seen:
        seen.add((at.run_id, at.item_id))
        total += at.turns
        at = segments.get((at.continues_run, at.item_id)) if at.continues_run else None
    return total


def retrieved_with_scores(result: AgentResult) -> list[tuple[str, float]]:
    """Return `(chunk_id, score)` for every chunk retrieved this turn, best first per call.

    Scores live on the `lookup_kb.Hit` objects the tool returned, which `AgentResult` carries
    on `steps[].tool_result`; the trace records chunk ids without them. Deduplicated in
    first-seen order, matching `AgentResult.retrieved_chunk_ids`, so the display cannot imply
    the agent saw a passage twice.

    A tool result that carries no scores — a web search, an empty lookup — contributes nothing
    rather than raising.
    """
    found: dict[str, float] = {}
    for step in result.steps:
        payload = step.tool_result
        if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
            continue
        for hit in payload:
            chunk = getattr(hit, "chunk", None)
            chunk_id = getattr(chunk, "chunk_id", None)
            score = getattr(hit, "score", None)
            if isinstance(chunk_id, str) and chunk_id and isinstance(score, (int, float)):
                found.setdefault(chunk_id, float(score))
    return list(found.items())


class ChatSession:
    """A chat session: one config at a time, one run per config, every turn logged.

    Holds the current `run_id`, the manifest describing it, the trace logger writing it, and
    the agent (and therefore the conversation memory) producing it. Rebuilt as a group when the
    config changes, because they are only meaningful together.
    """

    def __init__(self, config: AgentConfig, runs_dir: Path = DEFAULT_RUNS_DIR) -> None:
        self.runs_dir = Path(runs_dir)
        self.config = config
        self.manifest: RunManifest | None = None
        self.item_id = new_conversation_id()
        self._logger: TraceLogger | None = None
        self._agent: Agent | None = None
        # History waiting to be handed to the next agent, with the run it came from. Set when a
        # changed condition is about to discard the agent holding it, and consumed by `_open`. A
        # slot rather than an argument because the two producers are far apart: a rotation reads it
        # off the outgoing agent, and there is no agent in sight when one is read off a trace.
        self._carry: Conversation | None = None
        self._carried_from: str | None = None
        # Turns sent under the current `item_id`, so a carried conversation's numbering can
        # continue on the new agent. The agent's own counter cannot serve: it is rebuilt with the
        # agent, and `Conversation.turn_count` counts only what survived compaction.
        self._turns_sent = 0

    # -- the config ---------------------------------------------------------------------

    @property
    def run_id(self) -> str | None:
        """The active run, or None before the first message has been sent."""
        return self.manifest.run_id if self.manifest is not None else None

    @property
    def trace_path(self) -> Path | None:
        """Where this session's turns are being written, or None before the first message."""
        return self.manifest.trace_path(self.runs_dir) if self.manifest is not None else None

    def use(self, config: AgentConfig) -> None:
        """Adopt `config` for subsequent turns.

        Deliberately does not mint anything: the new run is created when a message actually
        arrives, so toggling a control twice and sending one message produces one run rather
        than three manifests for two abandoned configurations.
        """
        self.config = config

    def send(self, message: str) -> AgentResult:
        """Run one turn, minting or rotating the run first as needed.

        The manifest is written before the agent appends anything, so no record can exist
        without the conditions that produced it being on disk already.
        """
        agent = self._ensure_run()
        # Counted before the call, not after: `run_turn` consumes the index at its top and a
        # provider failure does not give it back, so a counter that only advanced on success would
        # hand out an index the failed turn had already written records under.
        self._turns_sent += 1
        return agent.run_turn(message, item_id=self.item_id)

    def resume(self, run_id: str, item_id: str) -> int:
        """Pick up conversation `item_id`, last seen in `run_id`, in this session.

        The history is rebuilt from the traces on disk and set aside for the next agent, exactly as
        a model switch sets one aside — resumption differs only in where the conversation was read
        from. So the next message mints a run under *this* session's conditions and opens its trace
        with a carry-over record naming `run_id`, and a resumption nobody follows up on writes
        nothing at all.

        The current run is closed first, because the conversation on the current agent is not this
        one and must not be the thing that gets carried.

        Returns:
            The number of turns rebuilt, which is where this session's numbering will continue from.

        Raises:
            ValueError: The conversation could not be rebuilt — see `conversation_from_trace` — or
                it has no messages. An empty history is a resumption of nothing, and treating it as
                one would claim a continuity the trace does not support.
        """
        history = conversation_from_trace(run_id, item_id, self.runs_dir)
        if not history.messages:
            raise ValueError(f"{item_id} has no messages in {run_id}: nothing to resume")

        self.close()
        self.manifest = None
        self.item_id = item_id
        self._turns_sent = history.turn_count
        self._carry = history
        self._carried_from = run_id
        return history.turn_count

    def reset(self) -> None:
        """Start a new conversation, keeping the run.

        A reset changes no condition, so it needs no new manifest; it gets a fresh `item_id`
        and clears the memory the model sees. The discarded turns stay in the trace, which is
        the point of logging them.

        Any history set aside for a pending rotation goes too. A reset means forget this, and
        carrying it onto the next agent anyway would make the button a no-op in exactly the case
        someone reached for it: after switching model.
        """
        self.item_id = new_conversation_id()
        self._turns_sent = 0
        self._carry = None
        self._carried_from = None
        if self._agent is not None:
            self._agent.conversation.reset()

    # -- the run ------------------------------------------------------------------------

    def _ensure_run(self) -> Agent:
        """Open a run if there is none, or rotate to a new one if a condition has changed.

        The check is a digest over the manifest's whole agent-config group rather than a
        comparison of the fields this app happens to expose. The model toggle is the control a
        user will reach for, but a corpus edited between two messages changes the same thing,
        and enumerating the causes by hand is how one gets missed.

        A rotation changes the conditions, not the conversation: the history is set aside for the
        new agent, which continues it under a new `run_id` and a new manifest. An *empty* history
        is not set aside — a conversation that has been reset, or never spoken in, is nothing to
        carry, and a carry-over record describing none of it would claim a continuity that is not
        there.
        """
        pending = build_manifest(self.config, run_kind=CHAT_RUN_KIND)
        current = self.manifest
        if current is None or agent_config_digest(pending) != agent_config_digest(current):
            if self._agent is not None and self._agent.conversation.messages:
                self._carry = self._agent.conversation
                self._carried_from = self.run_id
            return self._start(pending)
        if self._agent is None:
            # Reopened after `close`. The conditions have not changed, so this is the same run
            # and the trace is appended to; the manifest on disk still describes it and is
            # left exactly as it was written. The agent went with the close and its memory with
            # it, so there is nothing to carry and nothing is claimed.
            return self._open(current)
        return self._agent

    def _start(self, manifest: RunManifest) -> Agent:
        """Write `manifest` and point the logger and agent at its run.

        The previous logger is closed rather than left open: its file is complete, and the
        manifest describing it stays exactly as it was written.
        """
        self.close()
        manifest.write(self.runs_dir)
        self.manifest = manifest
        return self._open(manifest)

    def _open(self, manifest: RunManifest) -> Agent:
        """Open the trace for `manifest`'s run and build the agent that writes to it.

        With nothing set aside to carry, the new agent's memory is empty, so this is a new
        conversation and gets a new `item_id`: reusing it would present two disjoint histories as
        one continuous conversation.

        With a carried history the opposite holds — it *is* one conversation, so the id and the
        turn numbering continue, and the new trace opens with a record saying where it came from.
        """
        self._logger = TraceLogger(manifest.run_id, self.runs_dir)
        agent = self.config.build_agent(self._logger)

        carry, source = self._carry, self._carried_from
        self._carry, self._carried_from = None, None
        if carry is None:
            self.item_id = new_conversation_id()
            self._turns_sent = 0
        else:
            agent.conversation.adopt(carry)
            agent.resume_numbering(self.item_id, self._turns_sent)
            self._log_carry_over(source, carry)

        self._agent = agent
        return agent

    def _log_carry_over(self, source: str | None, carry: Conversation) -> None:
        """Record, in the new trace, the history its first turn will be conditioned on.

        Under `role="memory"`, the role `agent.core` already uses for compaction, because this is
        the same kind of fact: an account of what the model was shown and why this file does not
        contain it. Without the record, the run's first turn shows a model knowing things nothing
        here explains, and `previous_run_id` is what makes the chain followable.
        """
        if self._logger is None:
            return
        self._logger.log(
            self.item_id,
            self._turns_sent,
            ROLE_MEMORY,
            json.dumps(
                {
                    "at": utc_now_iso(),
                    "event": CARRY_OVER_EVENT,
                    "previous_run_id": source,
                    "messages_carried": len(carry.messages),
                    "tokens_carried": estimate_messages_tokens(carry.messages),
                    "summary_carried": carry.summary is not None,
                }
            ),
        )

    def close(self) -> None:
        """Close the trace file, if one is open. Idempotent."""
        if self._logger is not None:
            self._logger.close()
            self._logger = None
        self._agent = None

    def __enter__(self) -> ChatSession:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def turn_detail(result: AgentResult) -> dict[str, Any]:
    """Summarise one turn for display: what it called, what it retrieved, what it cost.

    Assembled here rather than in the app so that the panel a human reads and the record the
    trace holds are derived from the same `AgentResult`. `cached` matters enough to surface:
    a replayed latency is not a measurement (PROJECT.md), and a panel reporting 12 ms without
    saying so would be misleading.
    """
    tokens: Mapping[str, int] = result.tokens or {}
    return {
        "tool_calls": list(result.tool_calls),
        "retrieved": retrieved_with_scores(result),
        "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
        "latency_ms": result.latency_ms,
        "prompt_tokens": tokens.get("prompt"),
        "completion_tokens": tokens.get("completion"),
        "total_tokens": tokens.get("total"),
        "usd_cost": result.usd_cost,
        "cached": any(step.cached for step in result.steps),
        "model_calls": len(result.steps),
        "stopped_reason": result.stopped_reason,
        "format_violation": (
            result.format_violation.value if result.format_violation is not None else None
        ),
        "budget_induced_truncations": result.budget_induced_truncations,
        "tool_errors": result.tool_errors,
        "infrastructure_failed": result.infrastructure_failed,
        "citations": list(result.citations),
    }
