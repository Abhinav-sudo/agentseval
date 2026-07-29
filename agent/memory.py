"""Conversation state across turns.

Holds the message list handed to the model on each call: system prompt, user turns,
assistant turns, and rendered tool results (which enter as `user` content, since the JSON
protocol has no tool role).

Memory policy is part of the harness and therefore uniform: both agents get the same
history window and the same truncation rule. Tuning context handling per model would mean
the arms no longer share a harness (PROJECT.md).

The policy has two parts. The most recent `keep_last_turns` turns are kept verbatim, and
once the estimated token total exceeds `token_budget` everything older is folded into a
single rolling summary produced by `summariser` — in a real run, the agent's own adapter, so
the arms are not summarised by models of differing quality.

Any truncation or summarisation is recorded in the trace, so a run's context can be
reconstructed exactly rather than inferred: `truncations()` says how much was folded and
whether a summary replaced it, and `on_summarised` hands the summariser's own model call to
whatever is writing the trace.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from agent.models.base import (
    DEFAULT_TEMPERATURE,
    ChatMessage,
    ModelAdapter,
    ModelError,
    ModelResponse,
)
from agent.prompts import render_summary, render_summary_request
from agent.trace import utc_now_iso

#: Turns kept verbatim. A turn is a user message plus everything the loop produced in
#: response to it, so a tool-using turn occupies several messages.
DEFAULT_KEEP_LAST_TURNS = 4

#: Estimated prompt tokens above which older turns are folded into the summary. Sized for
#: the smallest context window in play, leaving room for the completion, rather than for the
#: largest: a budget only the frontier model could satisfy would compact the two arms at
#: different points, which is the harness variation PROJECT.md forbids.
DEFAULT_TOKEN_BUDGET = 6000

#: Ceiling for the summariser's own completion. The prompt asks for under 200 words; this
#: leaves headroom without letting a runaway summary cost more than the turns it replaced.
SUMMARY_MAX_TOKENS = 512

#: Deliberately crude token estimation, and deliberately the same for every model. A real
#: provider tokenizer would have the two arms compacting at different points on identical
#: transcripts, turning memory policy into a variable; this threshold only needs to be
#: roughly right, and being uniformly wrong is the property that matters.
CHARS_PER_TOKEN = 4

#: Per-message allowance for the role and whatever framing a provider adds around it.
PER_MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count of `text`, rounding up.

    An approximation by construction — see `CHARS_PER_TOKEN`.
    """
    return -(-len(text) // CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: Sequence[ChatMessage]) -> int:
    """Estimate the prompt tokens a message list would cost."""
    return sum(
        estimate_tokens(message.get("content", "")) + PER_MESSAGE_OVERHEAD_TOKENS
        for message in messages
    )


@dataclass
class Conversation:
    """An ordered message list plus the bookkeeping needed to replay it.

    Attributes:
        messages: History excluding the system prompt, in wire format. Append through
            `add_user`, `add_assistant`, and `add_tool_result`, which also track where each
            turn begins; a list supplied at construction is treated as one opaque turn.
        max_messages: Hard ceiling on message count, applied alongside the token budget.
            None disables it. Useful when a transcript is many short messages, where the
            token estimate stays low but the message list still grows without end.
        summariser: Adapter used to compact older turns. When None, folded messages are
            dropped instead of summarised and the loss is recorded in `truncations`.
        summary: The rolling summary, or None before the first compaction. Each compaction
            resummarises the previous summary together with the newly folded turns, so it
            stays one message rather than a growing pile of them.
        on_summarised: Called with `(request, response)` for each summariser call, so the
            caller can log it. It is a model call like any other and belongs in the trace
            with its tokens counted.
    """

    system_prompt: str
    messages: list[ChatMessage] = field(default_factory=list)
    max_messages: int | None = None
    keep_last_turns: int = DEFAULT_KEEP_LAST_TURNS
    token_budget: int = DEFAULT_TOKEN_BUDGET
    summariser: ModelAdapter | None = None
    summary: str | None = None
    on_summarised: Callable[[str, ModelResponse], None] | None = None

    def __post_init__(self) -> None:
        #: Index into `messages` where each user turn starts, maintained by `add_user` and
        #: rebased on compaction. Tracked rather than inferred from message content: a user
        #: is free to type something that looks exactly like a rendered tool result.
        self._turn_starts: list[int] = [0] if self.messages else []
        self._truncations: list[dict[str, object]] = []

    # -- appending ---------------------------------------------------------------------

    def add_user(self, content: str) -> None:
        """Append a user turn."""
        self._turn_starts.append(len(self.messages))
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Append an assistant turn, stored raw (tool-call JSON included)."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, rendered: str) -> None:
        """Append a rendered tool result as user content, per the protocol.

        Part of the turn already in progress, not a new one: the model asked for it, so
        counting it as a turn would shrink the verbatim window every time a tool ran.
        """
        self.messages.append({"role": "user", "content": rendered})

    # -- reading -----------------------------------------------------------------------

    def to_messages(self) -> list[ChatMessage]:
        """Materialise the message list for a model call, system prompt first.

        Applies the truncation policy; anything dropped is reported via `truncations`. This
        may therefore call the summariser, which is why it happens here — one place, just
        before the messages are used, rather than at every append site.
        """
        self._enforce_budget()
        return [dict(message) for message in self._materialise()]

    def estimated_tokens(self) -> int:
        """Estimated prompt cost of the current message list, summary included."""
        return estimate_messages_tokens(self._materialise())

    @property
    def turn_count(self) -> int:
        """Number of user turns currently held verbatim."""
        return len(self._turn_starts)

    def truncations(self) -> list[dict[str, object]]:
        """Describe messages dropped by the truncation policy, for the trace."""
        return [dict(record) for record in self._truncations]

    def reset(self) -> None:
        """Clear history, keeping the system prompt.

        Truncation records go too: they describe the history that was just discarded, and
        the trace already holds them.
        """
        self.messages.clear()
        self._turn_starts.clear()
        self._truncations.clear()
        self.summary = None

    def adopt(self, other: Conversation) -> None:
        """Take over `other`'s history: messages, turn boundaries, and rolling summary.

        For a conversation that continues under new conditions — the chat surface's model toggle,
        which mints a new run rather than a new conversation (`agent.session`). The system prompt
        is not taken: it belongs to the agent doing the adopting, whose tool inventory may differ.

        Copied rather than shared, since `other` belongs to a run whose trace is now complete. Its
        truncation records are not taken either, and this conversation's are cleared: they describe
        folds performed under another manifest, and that trace already holds them.

        The turn boundaries are the part that is easy to lose. `__post_init__` treats a message
        list supplied at construction as one opaque turn, so assigning `messages` alone would
        leave a history `_fold_boundary` can never fold — it would grow until the provider
        refused it.
        """
        self.messages = [dict(message) for message in other.messages]
        self._turn_starts = list(other._turn_starts)
        self._truncations.clear()
        self.summary = other.summary

    # -- the policy --------------------------------------------------------------------

    def _materialise(self) -> list[ChatMessage]:
        """Assemble system prompt, rolling summary, and verbatim history, in that order."""
        assembled: list[ChatMessage] = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            # Harness-supplied context arrives as user content, like a tool result: the
            # protocol has no role for "the harness is telling you something".
            assembled.append({"role": "user", "content": render_summary(self.summary)})
        assembled.extend(self.messages)
        return assembled

    def _exceeded(self) -> str | None:
        """Return which limit is exceeded, or None if the message list fits."""
        if self.max_messages is not None and len(self.messages) > self.max_messages:
            return "message_cap"
        if self.estimated_tokens() > self.token_budget:
            return "token_budget"
        return None

    def _fold_boundary(self) -> int:
        """Return the index before which messages should be folded, or 0 for nothing to do."""
        keep = max(1, self.keep_last_turns)
        starts = self._turn_starts
        if len(starts) > keep:
            return starts[-keep]
        if len(starts) > 1:
            # The window itself is over budget, so shrink it — but never past the newest
            # turn. Summarising the question about to be answered defeats the purpose.
            return starts[-1]
        return 0

    def _enforce_budget(self) -> None:
        """Fold older turns into the summary until the message list fits, or cannot shrink.

        Terminates: each pass folds at least one turn into a bounded summary, and the loop
        stops once only the newest turn remains. A single turn larger than the budget is
        left oversized rather than mangled — the provider's error naming its own limit is
        more useful than a silently truncated question, and `truncations` records that the
        floor was hit.
        """
        while (reason := self._exceeded()) is not None:
            cut = self._fold_boundary()
            if cut == 0:
                self._truncations.append(
                    {
                        "at": utc_now_iso(),
                        "reason": reason,
                        "messages_folded": 0,
                        "tokens_folded": 0,
                        "summarised": False,
                        "error": "the newest turn alone exceeds the budget; nothing was folded",
                    }
                )
                return
            self._fold(cut, reason)

    def _fold(self, cut: int, reason: str) -> None:
        """Replace `messages[:cut]` with a summary covering them and the previous summary."""
        folded = self.messages[:cut]
        to_summarise: list[ChatMessage] = list(folded)
        if self.summary:
            to_summarise.insert(0, {"role": "summary", "content": self.summary})

        summary, error = self._summarise(to_summarise)
        self._truncations.append(
            {
                "at": utc_now_iso(),
                "reason": reason,
                "messages_folded": len(folded),
                "tokens_folded": estimate_messages_tokens(folded),
                "summarised": summary is not None,
                "error": error,
            }
        )

        del self.messages[:cut]
        self._turn_starts = [start - cut for start in self._turn_starts if start >= cut]
        if summary is not None:
            self.summary = summary

    def _summarise(self, messages: Sequence[ChatMessage]) -> tuple[str | None, str | None]:
        """Summarise `messages`, returning `(summary, error)` with exactly one set.

        A failure here degrades to dropping the folded messages rather than aborting: losing
        old context costs answer quality on later turns, while raising would cost the turn
        outright. Either way the loss is recorded, so it cannot be mistaken for a model that
        simply forgot.

        The request goes out as a standalone user message with no system prompt. Sharing the
        agent's would have the summariser reply with a protocol tool call, since that is
        what that prompt demands every turn.
        """
        if self.summariser is None:
            return None, "no summariser configured; folded messages were dropped"

        request = render_summary_request(messages)
        try:
            response = self.summariser.generate(
                [{"role": "user", "content": request}],
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=SUMMARY_MAX_TOKENS,
            )
        except ModelError as exc:
            return None, f"summariser failed: {exc}"

        if self.on_summarised is not None:
            self.on_summarised(request, response)
        summary = response.text.strip()
        if not summary:
            return None, "summariser returned an empty summary"
        return summary, None
