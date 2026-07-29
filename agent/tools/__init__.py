"""Tools available to the agent.

Both agents get the same registry with the same schemas and the same documentation text.
Tools are described to models only through the prompt-based JSON protocol in
`agent.prompts` — never through a provider's native tool-calling parameters (PROJECT.md).

Each tool is a module exporting `name`, `description`, `schema`, and a callable of the same
name. `RegisteredTool` binds those four into one object, so `registry()` (what the loop
dispatches through) and `tool_specs()` (what the prompt documents) are built from the same
source and cannot describe different tools.

A tool failure is one of two things, and `agent.tools.errors` makes the distinction a type
rather than a judgement call at the catch site: `ToolInputError` for a call the model got
wrong, `ToolInfraError` for our side breaking. Both are re-exported here, which is where
callers should import them from.

Two kinds of argument reach a tool, and they arrive by different routes. The model supplies
what the schema documents — `query`, `top_k` — and the run supplies its own conditions, the
corpus directory and the retrieval floor, which the model must not see and cannot be trusted to
set. `registry()` binds neither and leaves every run-level default in place; `bound_registry()`
binds the second kind from `AgentConfig`, which is what a graded run uses so that the manifest's
`retrieval_config_sha256` describes the retrieval that actually happened.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Protocol

from agent.tools import lookup_kb as lookup_kb_module
from agent.tools import search_web as search_web_module
from agent.tools.errors import ToolError, ToolErrorReason, ToolInfraError, ToolInputError
from agent.tools.lookup_kb import DEFAULT_KB_DIR, DEFAULT_MIN_SCORE

__all__ = [
    "CONFIG_BOUND_PARAMS",
    "TOOL_MODULES",
    "RegisteredTool",
    "Tool",
    "ToolError",
    "ToolErrorReason",
    "ToolInfraError",
    "ToolInputError",
    "bound_registry",
    "registry",
    "tool_specs",
]


class Tool(Protocol):
    """A callable tool with a name, JSON argument schema, and docs for the prompt."""

    name: str
    description: str
    schema: dict[str, Any]

    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass
class RegisteredTool:
    """One tool: its prompt-facing metadata and the function that runs it.

    Satisfies `Tool`. Built from a module rather than restating its metadata, so a schema
    edit reaches the prompt, the argument validation in `agent.core`, and the running code
    in one move.
    """

    name: str
    description: str
    schema: dict[str, Any]
    func: Any

    @classmethod
    def from_module(cls, module: ModuleType) -> RegisteredTool:
        return cls(
            name=module.name,
            description=module.description,
            schema=module.schema,
            func=getattr(module, module.name),
        )

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def spec(self) -> dict[str, Any]:
        """Render this tool for `prompts.build_system_prompt`."""
        return {"name": self.name, "description": self.description, "schema": self.schema}


#: The inventory, in the order the prompt documents it. Order is load-bearing: the rendered
#: tool docs are hashed into `prompts.system_prompt_sha256`, which the run manifest records,
#: so reordering this tuple changes the digest and stops two runs being comparable across
#: the change. Add to the end.
TOOL_MODULES: tuple[ModuleType, ...] = (lookup_kb_module, search_web_module)


def registry() -> dict[str, Tool]:
    """Return the tool registry, identical for every agent.

    A fresh dict per call, so a caller that adds or removes a tool for a test cannot leak
    that change into the next run.
    """
    return {tool.name: tool for tool in (RegisteredTool.from_module(m) for m in TOOL_MODULES)}


#: Which run-level settings each tool takes, by tool name. Only `lookup_kb` reads the corpus,
#: so only `lookup_kb` is listed; `search_web` has no corpus and no floor, and inventing
#: parameters for it to ignore would put two names in a table that describes one thing.
#:
#: An explicit table rather than inspecting each signature for parameters that happen to share
#: these names. Binding by coincidence of naming is how a renamed parameter turns into a run
#: that silently uses the default while the manifest records the configured value — the exact
#: failure this seam exists to close.
CONFIG_BOUND_PARAMS: Mapping[str, frozenset[str]] = MappingProxyType(
    {"lookup_kb": frozenset({"kb_dir", "min_score"})}
)


def bound_registry(
    *,
    kb_dir: Path = DEFAULT_KB_DIR,
    min_score: float = DEFAULT_MIN_SCORE,
) -> dict[str, Tool]:
    """Return the registry with a run's retrieval settings bound into the tool functions.

    `registry()` hands the loop the bare module functions, so a run configured with a fixture
    corpus or a confidence floor got neither: `lookup_kb` fell back to `DEFAULT_KB_DIR` and
    `DEFAULT_MIN_SCORE` no matter what the run was launched with, while `build_manifest`
    recorded the configured values into `kb_sha256` and `retrieval_config_sha256`. A manifest
    describing conditions that did not hold is worse than one that omits them, because
    `assert_comparable` then certifies as comparable two runs that read different corpora.

    The model-facing `schema` is untouched, and deliberately: these are properties of the run,
    not arguments the model gets to choose. That also makes the binding tamper-proof rather than
    merely undocumented — `core.validate_arguments` rejects any argument absent from
    `properties`, so a model naming `min_score` in a tool call gets a schema error instead of a
    lower floor. Leaving the schema alone keeps `prompts.system_prompt_sha256` still, so a run
    with a floor stays comparable with one without on every axis except the floor itself.

    Args:
        kb_dir: Corpus directory for `lookup_kb`.
        min_score: Cosine floor below which hits are dropped, so "retrieved nothing" and
            "retrieved nothing convincing" become the same observable — which is what lets
            `guardrails.enforce_grounding` read the floor off the result rather than
            re-deriving it.

    Raises:
        RuntimeError: a tool named in `CONFIG_BOUND_PARAMS` does not accept a parameter the
            table binds. Loud at construction, because the alternative is a silent fallback to
            the defaults and a manifest that misdescribes the run.
    """
    values: dict[str, Any] = {"kb_dir": Path(kb_dir), "min_score": float(min_score)}
    tools = registry()
    for name, bound in CONFIG_BOUND_PARAMS.items():
        tool = tools.get(name)
        if tool is None:  # pragma: no cover - a table entry for a removed tool
            raise RuntimeError(
                f"CONFIG_BOUND_PARAMS names {name!r}, which is not in the registry"
            )
        assert isinstance(tool, RegisteredTool)
        accepted = inspect.signature(tool.func).parameters
        missing = sorted(param for param in bound if param not in accepted)
        if missing:
            raise RuntimeError(
                f"tool {name!r} does not accept {', '.join(missing)}; either the parameter was "
                f"renamed or CONFIG_BOUND_PARAMS is stale. Left unbound, the run would use the "
                f"tool defaults while the manifest recorded the configured values"
            )
        tools[name] = replace(
            tool, func=partial(tool.func, **{param: values[param] for param in sorted(bound)})
        )
    return tools


def tool_specs() -> list[dict[str, Any]]:
    """Return name/description/schema for each tool, for `prompts.build_system_prompt`.

    Generated from the tools themselves so prompt docs cannot drift from behaviour.
    """
    return [RegisteredTool.from_module(module).spec() for module in TOOL_MODULES]
