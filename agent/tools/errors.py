"""The two ways a tool call can fail, as types rather than as a judgement at the catch site.

`ToolInputError` means the model asked for something impossible: a tool that does not exist,
arguments that do not fit the schema. It is charged to the model's error budget and reported
as its failure. `ToolInfraError` means our side broke — a timeout, an unreachable index, an
unbuilt dependency — and is charged to nothing, retried, and if it persists, excludes the item
from scoring rather than scoring the model down for our outage. Booking the second as the
first is how a flaky network turns into a finding about a model.

Re-exported from `agent.tools`, which is where callers should import them from. They live in
their own module because the tool modules raise them and `agent.tools.__init__` imports the
tool modules: a shared leaf keeps that dependency running one way only.

Nothing here imports `agent.prompts`. The model-facing wording of these errors is that
module's job; carrying it is this one's.
"""

from __future__ import annotations

from enum import StrEnum


class ToolErrorReason(StrEnum):
    """Why a tool call was rejected before it ran, for aggregation by type.

    Kept separate from the message text because the message is written for the model and the
    reason is written for the report: `tool_call_error_rate` is broken down by these, and
    computing that by pattern-matching English would break the first time the wording improved.
    """

    UNKNOWN_TOOL = "unknown_tool"
    MISSING_ARG = "missing_arg"
    BAD_ARG_TYPE = "bad_arg_type"
    SCHEMA_INVALID = "schema_invalid"


class ToolError(Exception):
    """Base for the two failure kinds, so one clause can name both."""


class ToolInputError(ToolError):
    """The model's call could not be dispatched: wrong name, or wrong arguments.

    Attributable to the model, so it is charged to `max_tool_errors` and counted in the
    per-arm error rate. The message is rendered straight back to the model, in the same
    words for every arm.
    """

    def __init__(self, message: str, reason: ToolErrorReason) -> None:
        super().__init__(message)
        self.reason = reason


class ToolInfraError(ToolError):
    """A tool failed for a reason the model did not cause and cannot fix.

    Raised by tool code so the loop can retry it and, if it keeps failing, mark the item
    `infrastructure_failed` instead of letting our outage depress the model's score.
    """
