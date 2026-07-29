"""Run manifests and the comparability guard.

A manifest records the conditions one run was executed under, so that a number can be traced
back to what produced it (PROJECT.md). `runs/{run_id}.manifest.json` sits beside the
`runs/{run_id}.jsonl` trace that `agent.trace` writes, joined on `run_id`.

**Manifests are an agent-layer concern.** A dataset belongs to an eval run, not to the agent,
so it is a nullable tail on one manifest rather than the reason for a second kind of manifest.
The same goes for a judge run's rubric and input file. `build_manifest` is the only function in
the project that builds one: `evals.runner` calls it with `run_kind="eval"` and a `DatasetRef`,
`agent.session` calls it with `run_kind="chat"`, `evals.judge` calls it with `run_kind="judge"`
and a `JudgeRef`, and none of them adds a field of its own. That keeps the dependency direction
one-way — `evals/` imports `agent/`, never the reverse — so the chat surface does not depend on
the evaluation harness.

This module sits one layer above `agent.trace`: assembling a manifest means knowing the
rendered prompt, the tool inventory, the corpus, and the price table, whereas writing a trace
needs none of them. The split is what keeps the writer standard-library only.

`assert_comparable` is what turns the manifest from documentation into a guarantee. It refuses
to let two runs be compared unless every recorded condition matches except the model identity
itself — the executable form of the claim that the frontier and OSS arms differed only in
model weights, which is what the rest of the project rests on.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from agent.core import DEFAULT_TEMPERATURE, Agent, Budgets, tool_specs_for
from agent.guardrails import Guardrails
from agent.models.base import DEFAULT_MAX_TOKENS, PRICING, ModelAdapter
from agent.prompts import judge_pair_template_sha256, system_prompt_sha256
from agent.tools import Tool, bound_registry
from agent.tools.lookup_kb import (
    DEFAULT_KB_DIR,
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    MAX_TOKENS,
    MIN_TOKENS,
    corpus_files,
)
from agent.trace import (
    DEFAULT_RUNS_DIR,
    TraceLogger,
    code_version,
    git_dirty,
    git_sha,
    manifest_path,
    sha256_of_paths,
    sha256_text,
    trace_path,
    utc_now_iso,
)

#: What a run is. `chat` is an interactive session, `eval` a dataset run, `judge` a scoring pass
#: over `(prompt, response)` pairs. The distinction is recorded because no two of them are arms
#: of the same experiment, and `assert_comparable` refuses to compare across it.
RunKind = Literal["chat", "eval", "judge"]

#: Length of a minted `run_id`. Matches `agent.core.Agent`, which mints one the same way when
#: it is given no logger.
RUN_ID_CHARS = 12


def new_run_id() -> str:
    """Mint a run id.

    One per manifest, minted where the manifest is built: a new manifest *is* a new run, and
    letting the two be created separately is how a trace ends up under a manifest describing
    different conditions.
    """
    return uuid.uuid4().hex[:RUN_ID_CHARS]


def _canonical(payload: Mapping[str, Any]) -> str:
    """Serialise `payload` so that equal configurations always produce equal bytes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------------------
# What a run is configured with
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    """Everything that decides what a run measures.

    One object builds both the agent and the manifest, via `build_agent` and
    `build_manifest`. That is the point of it: a manifest assembled from configuration
    defaults rather than from the object the `Agent` was constructed with would describe a run
    that did not happen, and `assert_comparable` would then be checking a fiction.

    Attributes:
        model: The adapter that will answer. Its `model_id` is what the manifest records.
        budgets: Handed to the `Agent` whole, so the three budgets in the manifest are the
            ones that bounded the turn.
        top_k: Retrieval breadth recorded for the run. Still not bound into the tool: the
            model chooses `top_k` per call within the schema, and taking that choice away is a
            change to what the agent may do rather than to how the run is configured, so
            setting this to anything but the tool default would misdescribe the run.
        min_score: Cosine floor, bound into `lookup_kb` by `inventory` and digested into
            `retrieval_config_sha256`. Unlike `top_k` this is not the model's to choose — it is
            a condition of the run, and the schema does not mention it.
        kb_dir: Corpus directory. Its contents are digested into `kb_sha256`, and it is bound
            into `lookup_kb` alongside the floor.
        guardrails: Whether the three screening stages run. The condition the ablation varies,
            and the only one it varies: `min_score` is held fixed across both arms, because
            moving the floor and its enforcement together would leave the delta attributable to
            neither. Identical for both models, always — nothing here branches on the model.
        tools: The inventory the loop dispatches through, and the one the prompt documents.
            None means "the standard registry with this config's retrieval settings bound in",
            which is what a graded run wants; `inventory` resolves it. An explicit mapping is
            used verbatim, since a caller substituting a tool has taken responsibility for what
            it does.
    """

    model: ModelAdapter
    budgets: Budgets = field(default_factory=Budgets)
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    kb_dir: Path = DEFAULT_KB_DIR
    guardrails: bool = False
    tools: Mapping[str, Tool] | None = None

    def build_guardrails(self) -> Guardrails | None:
        """The `Guardrails` this config describes, or None when they are off.

        A method rather than a field holding a `Guardrails` instance, so that `guardrails`
        stays a plain boolean condition the manifest can record and `assert_comparable` can
        compare. An object in the config would be a condition whose identity is a memory
        address.
        """
        return Guardrails() if self.guardrails else None

    def guardrails_digest(self) -> str | None:
        """`guardrails_sha256()` when guardrails are on, None when they are off.

        None rather than the digest for an off run, and the asymmetry is deliberate: a run with
        no guardrail has no guardrail configuration, so recording one would make two
        guardrails-off runs refuse each other after a pattern edit that changed neither.
        """
        screens = self.build_guardrails()
        return screens.digest() if screens is not None else None

    def inventory(self) -> Mapping[str, Tool]:
        """The tools this config runs, with its retrieval settings bound in.

        Resolved on each call rather than cached into the field at construction, and that is
        the whole point of the field defaulting to None. A `dataclasses.replace` carries
        existing field values forward, so an inventory bound once at construction would survive
        `replace(cfg, min_score=0.37)` unchanged — leaving a config whose manifest records one
        floor and whose tools apply another, which is the exact mismatch `bound_registry` exists
        to remove.
        """
        if self.tools is not None:
            return self.tools
        return bound_registry(kb_dir=self.kb_dir, min_score=self.min_score)

    def build_agent(self, logger: TraceLogger) -> Agent:
        """Construct the agent this config describes, logging to `logger`.

        The agent takes its `run_id` from the logger, so the trace, the manifest, and the
        records the agent writes all carry the same one.
        """
        return Agent(
            self.model,
            dict(self.inventory()),
            budgets=self.budgets,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logger=logger,
            guardrails=self.build_guardrails(),
        )

    def tool_specs(self) -> list[dict[str, Any]]:
        """Render the prompt specs for this inventory, as `agent.core` renders them.

        Reads the resolved inventory, so the prompt documents the tools that will run. The
        specs are unaffected by the binding — `bound_registry` leaves every `schema` alone —
        which is why `system_prompt_sha256` does not move when a floor is configured.
        """
        return tool_specs_for(self.inventory())

    def provider(self) -> str:
        """Return the provider name for the manifest.

        Falls back to the model family when an adapter declares no provider: `ModelAdapter`
        requires `name`, `model_id`, and `family`, and `provider` is a `ChatAdapter` addition,
        so a conforming adapter without one must still yield a recorded value rather than
        `None`.
        """
        return str(getattr(self.model, "provider", "") or self.model.family)


@dataclass(frozen=True)
class DatasetRef:
    """The eval set a run was executed over.

    `sha256` is of the file's bytes, and it is the field that matters: `path` is a name, and
    two runs pointing at the same name after the file was edited between them are not
    comparable. Only the hash catches that, which is why `assert_comparable` compares the hash
    and treats the path as informational.

    Attributes:
        n_items: Cases in the set, so a truncated dataset is visible in the manifest.
        seeds: Random seeds used. None throughout this project so far — runs are at
            temperature 0 and nothing samples — but recorded rather than assumed absent.
    """

    path: Path
    sha256: str
    n_items: int
    seeds: list[int] | None = None

    @classmethod
    def for_file(cls, path: Path, n_items: int, seeds: list[int] | None = None) -> DatasetRef:
        """Build a reference to `path`, digesting its bytes.

        Bytes rather than parsed contents: a reformatted dataset is a different file, and
        deciding it is equivalent is exactly the judgement this hash exists to avoid.
        """
        path = Path(path)
        return cls(
            path=path,
            sha256=sha256_of_paths([path], root=path.parent) or "",
            n_items=n_items,
            seeds=list(seeds) if seeds is not None else None,
        )


@dataclass(frozen=True)
class JudgeRef:
    """The conditions of a judge run: which judge answered, and under which rubric.

    A judge run has no agent, no corpus, and no tools, so it has none of the conditions an
    `AgentConfig` describes. What it does have is the pair here, and both halves matter: scores
    from two rubrics are not comparable, and neither are scores from two judge models.

    `model_name` and `provider` reach `RunManifest.model_name`/`provider` as well, since those
    fields name whichever model actually ran. They are repeated as `judge_model` and
    `judge_provider` because `COMPARABLE_EXEMPT` excuses a difference in `model_name` — which is
    right for two arms of an agent A/B and wrong for two judge runs, where a different judge is
    a different measurement. The duplicate is the field the guard checks.

    Attributes:
        temperature: 0 for every graded verdict. Recorded rather than assumed, because the
            stability sampling path deliberately runs hotter and must not be mistaken for one.
        rubric_sha256: `prompts.judge_rubric_sha256()`, covering every rubric file's bytes and
            the schema rendered into them.
        rubric_names: Which rubrics that digest covers, so a reader knows what was hashed.
        pairs_path: Informational, like `DatasetRef.path`. For `score_run` this is the trace
            that was scored.
        pairs_sha256: Of the scored file's bytes. The field that decides whether two judge runs
            read the same pairs.
        pair_template_sha256: `prompts.judge_pair_template_sha256()`, over the pair-rendering
            template's headings, canonical block order, and separator. Covers what
            `rubric_sha256` does not: rewording `### Response` changes every judge message while
            leaving every rubric byte-identical. Optional here only so a caller that predates the
            field is not silently given a wrong digest — `build_manifest` fills it in for every
            judge run it writes.
    """

    model_name: str
    provider: str
    rubric_sha256: str
    rubric_names: list[str]
    pairs_path: Path
    pairs_sha256: str
    n_pairs: int
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    pair_template_sha256: str | None = None

    @classmethod
    def for_file(
        cls,
        path: Path,
        n_pairs: int,
        *,
        model_name: str,
        provider: str,
        rubric_sha256: str,
        rubric_names: Sequence[str],
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        pair_template_sha256: str | None = None,
    ) -> JudgeRef:
        """Build a reference to the pairs in `path`, digesting its bytes.

        Bytes rather than parsed pairs, for the reason `DatasetRef.for_file` uses them: a
        re-saved file is a different file, and deciding otherwise is the judgement the hash
        exists to avoid.

        `pair_template_sha256` defaults to the current template's digest rather than to None: a
        judge run happening now read the template that exists now, and defaulting to "unknown"
        would write a manifest claiming otherwise. `None` is for a manifest that predates the
        field, which is a thing read off disk and never a thing built here.
        """
        path = Path(path)
        return cls(
            model_name=model_name,
            provider=provider,
            rubric_sha256=rubric_sha256,
            rubric_names=list(rubric_names),
            pairs_path=path,
            pairs_sha256=sha256_of_paths([path], root=path.parent) or "",
            n_pairs=n_pairs,
            temperature=temperature,
            max_tokens=max_tokens,
            pair_template_sha256=(
                pair_template_sha256
                if pair_template_sha256 is not None
                else judge_pair_template_sha256()
            ),
        )


# --------------------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------------------


@dataclass
class RunManifest:
    """The conditions one run was executed under, in four groups.

    Every field is something that could change a result. Adding one automatically strengthens
    `assert_comparable`, since the guard iterates the dataclass rather than a hand-maintained
    list.

    **Identity** — `run_id`, `started_at`, `run_kind`. These name the run rather than describe
    a condition, and the guard drops them; `run_kind` is checked separately and more strictly,
    because a chat session, an eval run, and a judge run are not arms of one experiment.

    **Agent config** — everything from `model_name` to `retrieval_stack_version`. These must match
    between two comparable runs except for the model itself (`COMPARABLE_EXEMPT`). The ones
    describing an agent's prompt, tools, retrieval, and guardrails are None on a judge run,
    which has none of them.

    **Eval-only** — `dataset_path`, `dataset_sha256`, `n_items`, `seeds`. All None on a chat
    run, which is why they are nullable; `build_manifest` refuses an eval run without them
    rather than writing a manifest with holes where the dataset should be.

    **Judge-only** — `judge_model`, `judge_provider`, `judge_rubric_sha256`, `judge_rubrics`,
    `pairs_path`, `pairs_sha256`, `n_pairs`, `judge_pair_template_sha256`. All None on every other
    kind, and a judge run without a `JudgeRef` is refused for the reason an eval run without a
    dataset is.

    Attributes:
        started_at: UTC, ISO-8601 with an explicit offset.
        model_name: The provider-side model id, dated where the provider dates it (e.g.
            `claude-sonnet-4-20250514`). It is therefore also the model version, and there is
            no second field for one.
        max_tokens: None when the provider default is used. A condition, never exempt: it is
            what makes a reply truncate.
        chunk_size: Token ceiling used when merging paragraphs into chunks, normally
            `lookup_kb.MAX_TOKENS`. Recorded because it changes what retrieval returns, so two
            runs chunked differently are not comparable. None if no corpus was used.
        retrieval_config_sha256: Digest of the retrieval settings in force — embedding model,
            `top_k`, score floor, and the chunking token band. It covers the two settings that
            have no field of their own; `top_k` and `chunk_size` remain separate fields because
            a digest can say a condition drifted and cannot say which one.
        retrieval_stack_version: Installed versions of the libraries that compute the embeddings,
            as `numpy==2.4.4 sentence-transformers==5.6.1 ...`. The companion to
            `retrieval_config_sha256`: that field records the encoder a run asked for, this one
            records the code that ran it, because the same model id under a different `torch` can
            return different vectors. None when the run used no corpus, for the reason
            `chunk_size` is — there was no retrieval to describe — and **None also means the
            manifest predates this field: pre-instrumentation, unknown.** Never backfilled, since
            a guess here would assert that an old run embedded under today's libraries.
        system_prompt_sha256: Digest of the rendered prompt, tool docs included, so a tool
            added, removed, or re-documented changes it.
        kb_sha256: Digest of the `kb/` corpus. None when the run used no corpus. Changing the
            corpus changes retrieval, which is why this is a manifest field rather than
            something inferred from the trace.
        pricing_version: Digest of `models.base.PRICING`, despite the name. A version integer
            is a promise to remember to bump it; a digest changes whether or not anyone
            remembered.
        git_sha: None when there is no commit to point at.
        git_dirty: True if uncommitted or untracked changes were present, in which case
            `git_sha` does not fully identify the code that ran.
        guardrails: Whether `agent.guardrails` screened this run. None on a judge run, which
            has no agent to screen. A condition and not a flag: without it in the manifest, two
            runs differing only in guardrails would produce identical manifests and
            `assert_comparable` would pass on a comparison of different conditions — silently,
            which is the one failure the manifest exists to prevent. It is deliberately *not*
            in `COMPARABLE_EXEMPT`: guardrails-on against guardrails-off is an ablation, and
            `assert_ablation_comparable` is the guard for it.
        guardrails_sha256: `guardrails.guardrails_sha256()` — the patterns, the delivered
            completions, the screening budget, and any screening model id. Recorded when
            guardrails are on and **None when they are off**, because a run with no guardrail
            has no guardrail configuration to describe: recording the digest anyway would make
            two guardrails-off runs refuse each other after a pattern edit that changed nothing
            about either of them. The floor is not in here — `retrieval_config_sha256` already
            covers `min_score`, and digesting it twice would leave a reader unable to say which
            condition moved.
        max_tool_calls: Successful tool calls allowed per turn.
        max_tool_errors: Model-caused tool errors allowed per turn.
        max_model_calls: Hard ceiling on model calls per turn.
        dataset_path: Informational. Compared by name nowhere — `dataset_sha256` is the field
            that decides whether two runs saw the same data.
        judge_model: The judge that produced the scores, repeated from `model_name` because
            `COMPARABLE_EXEMPT` excuses that field and two judge runs scored by two different
            judges are not one measurement. See `JudgeRef`.
        judge_rubric_sha256: Digest of every rubric file, rendered. A rubric edit between two
            judge runs makes their scores incomparable and nothing else would catch it.
        judge_pair_template_sha256: Digest of the pair-rendering template — the block headings,
            the canonical block order, and the separator. The other half of what the judge read:
            `judge_rubric_sha256` covers the system message and this covers the user one, so
            rewording `### Response` no longer leaves two runs comparing as equal. **None means
            the manifest predates this field: pre-instrumentation, unknown.** Never backfilled —
            see `POST_HOC_OPTIONAL_FIELDS` — and not exempt from `assert_comparable`, because
            "unknown" is not evidence that two runs read the same template.
        pairs_path: Informational, like `dataset_path`; `pairs_sha256` is the deciding field.

    The three budgets are here, and not in `COMPARABLE_EXEMPT`, because they decide how much
    room a model had to recover from its own mistakes. An arm given six model calls against one
    given four is a different experiment, and the difference would show up as a quality gap.
    """

    # -- identity: named by IDENTITY_FIELDS, dropped by the guard ------------------------
    run_id: str
    started_at: str
    run_kind: RunKind

    # -- agent config: must match between arms, except the model itself -----------------
    # The six that describe an agent's prompt, tools, and retrieval are None on a judge run,
    # which has none of them. `build_manifest` refuses a None on a chat or eval run, where each
    # of them is a real condition.
    model_name: str
    provider: str
    temperature: float
    max_tokens: int | None
    top_k: int | None
    chunk_size: int | None
    retrieval_config_sha256: str | None
    system_prompt_sha256: str | None
    kb_sha256: str | None
    pricing_version: str
    max_tool_calls: int | None
    max_tool_errors: int | None
    max_model_calls: int | None
    git_sha: str | None
    code_version: str
    git_dirty: bool = False
    guardrails: bool | None = None
    guardrails_sha256: str | None = None
    retrieval_stack_version: str | None = None

    # -- eval-only: None on a chat run --------------------------------------------------
    dataset_path: str | None = None
    dataset_sha256: str | None = None
    n_items: int | None = None
    seeds: list[int] | None = None

    # -- judge-only: None on every other kind -------------------------------------------
    judge_model: str | None = None
    judge_provider: str | None = None
    judge_rubric_sha256: str | None = None
    judge_rubrics: list[str] | None = None
    pairs_path: str | None = None
    pairs_sha256: str | None = None
    n_pairs: int | None = None
    judge_pair_template_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a plain dict, in field order."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        """Build a manifest from a mapping, rejecting an inexact field set.

        Strict on purpose: silently tolerating a missing or unknown key would let a manifest
        written by a different version of this code load as though it were comparable to a
        current one.

        The single exception is `POST_HOC_OPTIONAL_FIELDS`, an explicit allowlist rather than a
        relaxation. An unknown key is still fatal, and so is every other absent field.

        Raises:
            ValueError: `data` does not have exactly the manifest's fields, allowing for the
                post-hoc group.
        """
        expected = {f.name for f in fields(cls)}
        given = set(data)
        missing = sorted(expected - given - POST_HOC_OPTIONAL_FIELDS)
        unexpected = sorted(given - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {missing}")
            if unexpected:
                details.append(f"unexpected {unexpected}")
            raise ValueError(f"Not a valid run manifest: {', '.join(details)}")
        # An absent post-hoc field loads as None, which reads as "pre-instrumentation, unknown".
        # Not a default and not a guess: `assert_comparable` refuses to pair a None against a
        # digest, so the manifest's age surfaces at the comparison rather than being smoothed over
        # at the load.
        return cls(**{name: data[name] for name in expected if name in given})

    def trace_path(self, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
        """Return the path of the trace this manifest describes."""
        return trace_path(self.run_id, runs_dir)

    def path(self, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
        """Return the path this manifest writes to."""
        return manifest_path(self.run_id, runs_dir)

    def write(self, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
        """Write `runs/{run_id}.manifest.json` and return its path.

        Written to a temporary file and moved into place, so an interrupted write leaves the
        previous manifest intact rather than a truncated file that looks valid.

        A manifest is written once and never edited afterwards. Conditions that change
        mid-session mint a new `run_id` and a new manifest (see `agent.session`); rewriting
        this one would leave the trace it already describes unattributable.
        """
        runs_dir = Path(runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        target = self.path(runs_dir)

        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(runs_dir), suffix=".manifest.tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def read(cls, path: Path) -> RunManifest:
        """Read a manifest from `path`.

        Raises:
            ValueError: the file is not JSON, or its field set is not exactly the manifest's.
        """
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ValueError(f"{path} does not contain a JSON object")
        return cls.from_dict(data)

    @classmethod
    def load(cls, run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> RunManifest:
        """Read the manifest for `run_id`."""
        return cls.read(manifest_path(run_id, runs_dir))


# --------------------------------------------------------------------------------------
# Building one
# --------------------------------------------------------------------------------------


def retrieval_config_sha256(
    *,
    embedding_model: str = EMBEDDING_MODEL,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Digest the retrieval settings that decide what a lookup returns.

    Covers the embedding model and the score floor, neither of which has a manifest field of
    its own, alongside the breadth and token band that do. A different encoder over the same
    corpus retrieves different chunks, and without this the two runs would look identical.
    """
    return sha256_text(
        _canonical(
            {
                "embedding_model": embedding_model,
                "top_k": top_k,
                "min_score": min_score,
                "min_tokens": min_tokens,
                "max_tokens": max_tokens,
            }
        )
    )


#: The packages whose code turns a document into a vector, and so decide what a lookup returns.
#: `sentence-transformers` runs the encoder, `transformers` and `torch` are what it runs the
#: encoder on, and `numpy` holds the matrix the cosine similarity is computed over. Not a general
#: environment dump: a version of `rich` cannot change a retrieved chunk, and recording it would
#: make two runs refuse each other over a console-formatting upgrade.
RETRIEVAL_STACK_PACKAGES = ("numpy", "sentence-transformers", "torch", "transformers")

#: What `retrieval_stack_version` records for a package that is not installed. A fact about the
#: environment rather than an error: a missing encoder fails loudly at the lookup, and a manifest
#: that refused to be written would lose the run instead of describing it.
ABSENT_PACKAGE = "(absent)"


def _installed_version(package: str) -> str:
    """Return `package`'s installed version, or `ABSENT_PACKAGE`.

    Read from distribution metadata rather than by importing the module, which keeps this cheap
    enough to call on every run: importing `torch` costs seconds, reading its version costs
    milliseconds.
    """
    try:
        return version(package)
    except PackageNotFoundError:
        return ABSENT_PACKAGE


def retrieval_stack_version(packages: Sequence[str] = RETRIEVAL_STACK_PACKAGES) -> str:
    """Return the installed versions of the libraries that compute the embeddings.

    `retrieval_config_sha256` records which encoder was *asked* for; this records the code that
    ran it. The same model id under a different `torch` can return different vectors, so without
    this two runs whose retrieval behaved differently would produce identical manifests — the
    silent failure the manifest exists to prevent.

    Readable rather than a digest, unlike the other derived conditions here. A digest could say
    the stack drifted and not which library moved, which is the objection
    `retrieval_config_sha256` already records against folding `top_k` into a hash; four version
    strings are short enough to compare by eye. Sorted, so the value does not depend on the order
    the packages were named in.
    """
    return " ".join(f"{name}=={_installed_version(name)}" for name in sorted(packages))


def pricing_sha256() -> str:
    """Digest the price table, which is what `pricing_version` records.

    A digest rather than a hand-bumped integer, for the reason `prompt_version` does not
    exist: a version string is a promise to remember, whereas this changes whether or not
    anyone remembered. Re-pricing a model changes every cost figure derived from it, so two
    runs priced differently are not comparable.
    """
    return sha256_text(_canonical({model: list(rates) for model, rates in PRICING.items()}))


def _check_arguments(
    run_kind: RunKind,
    cfg: AgentConfig | None,
    dataset: DatasetRef | None,
    judge: JudgeRef | None,
) -> None:
    """Raise unless the arguments describe exactly one kind of run.

    Each refusal is a manifest that would otherwise be written with a hole or a fiction in it:
    a graded run whose data cannot be identified, an argument accepted and then dropped because
    the kind has nowhere to put it, or an agent's configuration recorded for a run in which no
    agent took part.
    """
    if run_kind == "judge":
        if judge is None:
            raise ValueError(
                "a judge run needs a JudgeRef: the judge model and the rubric digest are its "
                "conditions, and scores from another rubric are not comparable to these"
            )
        if cfg is not None:
            raise ValueError(
                "a judge run has no agent, but an AgentConfig was given; its prompt, tools, and "
                "retrieval settings describe a run that did not happen"
            )
        if dataset is not None:
            raise ValueError(
                f"a judge run scores a pairs file, not a dataset, but one was given "
                f"({dataset.path}); pass it as JudgeRef.pairs_path"
            )
        return

    if cfg is None:
        raise ValueError(f"a {run_kind} run needs an AgentConfig: it is what the agent ran under")
    if judge is not None:
        raise ValueError(
            f"a {run_kind} run has no judge, but a JudgeRef was given "
            f"({judge.pairs_path}); pass run_kind='judge' to record it"
        )
    if run_kind == "eval" and dataset is None:
        raise ValueError("an eval run needs a DatasetRef: its dataset is part of its conditions")
    if run_kind == "chat" and dataset is not None:
        raise ValueError(
            f"a chat run has no dataset, but one was given ({dataset.path}); "
            "pass run_kind='eval' to record it"
        )


def _judge_manifest(judge: JudgeRef) -> RunManifest:
    """Assemble the manifest for a judge run.

    The agent-config fields describing a prompt, an inventory, a corpus, and guardrails are None
    rather than filled with plausible defaults: a judge reads none of them, and a fabricated
    `top_k` would sit exactly where `assert_comparable` trusts a fact. `pricing_version` is not
    among them —
    the judge's cost is computed from the same price table, so re-pricing it changes a figure
    this run reports.
    """
    return RunManifest(
        run_id=new_run_id(),
        started_at=utc_now_iso(),
        run_kind="judge",
        model_name=judge.model_name,
        provider=judge.provider,
        temperature=judge.temperature,
        max_tokens=judge.max_tokens,
        top_k=None,
        chunk_size=None,
        retrieval_config_sha256=None,
        retrieval_stack_version=None,
        system_prompt_sha256=None,
        kb_sha256=None,
        pricing_version=pricing_sha256(),
        max_tool_calls=None,
        max_tool_errors=None,
        max_model_calls=None,
        git_sha=git_sha(),
        code_version=code_version(),
        git_dirty=git_dirty(),
        # None, not False: a judge run has no agent to screen, so "guardrails were off" would be
        # a claim about a stage that never had anything to act on.
        guardrails=None,
        guardrails_sha256=None,
        judge_model=judge.model_name,
        judge_provider=judge.provider,
        judge_rubric_sha256=judge.rubric_sha256,
        judge_rubrics=list(judge.rubric_names),
        pairs_path=str(judge.pairs_path),
        pairs_sha256=judge.pairs_sha256,
        n_pairs=judge.n_pairs,
        judge_pair_template_sha256=judge.pair_template_sha256,
    )


def build_manifest(
    cfg: AgentConfig | None = None,
    *,
    run_kind: RunKind,
    dataset: DatasetRef | None = None,
    judge: JudgeRef | None = None,
) -> RunManifest:
    """Assemble the manifest for a run, minting its `run_id`.

    The single builder for all three kinds of run. Every field is derived from `cfg` — the
    object that also constructs the agent — rather than from configuration defaults, since the
    manifest is what `assert_comparable` later trusts. A judge run has no agent and is built
    from its `JudgeRef` instead.

    Args:
        cfg: The configuration the run will actually execute under. Required for a chat or eval
            run, forbidden for a judge one.
        run_kind: `chat` for an interactive session, `eval` for a dataset run, `judge` for a
            scoring pass.
        dataset: Required for an eval run, forbidden otherwise.
        judge: Required for a judge run, forbidden otherwise.

    Raises:
        ValueError: the arguments do not describe exactly one kind of run, or a condition that
            a chat or eval run must have is missing. See `_check_arguments`.
    """
    _check_arguments(run_kind, cfg, dataset, judge)
    if judge is not None:
        return _judge_manifest(judge)
    if cfg is None:  # pragma: no cover - _check_arguments has already refused this
        raise ValueError(f"a {run_kind} run needs an AgentConfig")

    kb_dir = Path(cfg.kb_dir)
    kb_digest = sha256_of_paths(corpus_files(kb_dir), root=kb_dir)

    return RunManifest(
        run_id=new_run_id(),
        started_at=utc_now_iso(),
        run_kind=run_kind,
        model_name=cfg.model.model_id,
        provider=cfg.provider(),
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        top_k=cfg.top_k,
        # The chunking ceiling describes a corpus; with no corpus there is nothing to describe.
        chunk_size=MAX_TOKENS if kb_digest is not None else None,
        retrieval_config_sha256=retrieval_config_sha256(top_k=cfg.top_k, min_score=cfg.min_score),
        # Like `chunk_size`: the encoder's version describes an embedded corpus, and with no
        # corpus there is nothing for a `torch` upgrade to have changed.
        retrieval_stack_version=retrieval_stack_version() if kb_digest is not None else None,
        system_prompt_sha256=system_prompt_sha256(cfg.tool_specs()),
        kb_sha256=kb_digest,
        pricing_version=pricing_sha256(),
        max_tool_calls=cfg.budgets.max_tool_calls,
        max_tool_errors=cfg.budgets.max_tool_errors,
        max_model_calls=cfg.budgets.max_model_calls,
        git_sha=git_sha(),
        code_version=code_version(),
        git_dirty=git_dirty(),
        guardrails=cfg.guardrails,
        guardrails_sha256=cfg.guardrails_digest(),
        dataset_path=str(dataset.path) if dataset is not None else None,
        dataset_sha256=dataset.sha256 if dataset is not None else None,
        n_items=dataset.n_items if dataset is not None else None,
        seeds=list(dataset.seeds) if dataset is not None and dataset.seeds is not None else None,
    )


# --------------------------------------------------------------------------------------
# The comparability guard
# --------------------------------------------------------------------------------------

#: Fields that identify a run instance rather than its conditions. `run_id` and `started_at`
#: differ between any two runs by construction; `run_kind` is here because it is checked on
#: its own terms, before the field diff, and more strictly than any diff could express.
IDENTITY_FIELDS = frozenset({"run_id", "started_at", "run_kind"})

#: The only conditions permitted to differ between two comparable runs. This set is the whole
#: claim: the arms differed in which model answered and what it cost, and in nothing else.
#: Widening it weakens every comparison built on top of it, so it is a PROJECT.md-level change.
COMPARABLE_EXEMPT = frozenset({"model_name", "provider", "usd_cost"})

#: Fields that describe an eval set, and are None on a chat run. Named as a group because
#: `agent_config_fields` is defined by subtraction: what is left once identity, the dataset, and
#: the judge are removed is exactly what has to hold still for a run to keep its identity.
EVAL_ONLY_FIELDS = frozenset({"dataset_path", "dataset_sha256", "n_items", "seeds"})

#: Fields that describe a judge run, and are None on every other kind. Excluded from
#: `agent_config_fields` for the same reason the eval-only group is: they are not conditions an
#: agent ran under, so a chat session comparing its own configuration must not see them move.
JUDGE_ONLY_FIELDS = frozenset(
    {
        "judge_model",
        "judge_provider",
        "judge_rubric_sha256",
        "judge_rubrics",
        "pairs_path",
        "pairs_sha256",
        "n_pairs",
        "judge_pair_template_sha256",
    }
)

#: Fields added after the manifest format already existed on disk. Absent from an older file means
#: **pre-instrumentation, unknown** — never a backfilled guess, because a guessed digest would
#: assert that an old run read today's template, which is exactly the claim nobody can make.
#:
#: This is an allowlist, not a relaxation. `from_dict` still rejects every unknown key and every
#: other missing field, and nothing here is exempt from `assert_comparable`: a None paired against
#: a digest is refused, since "unknown" is not evidence of sameness. Adding a name here is a
#: decision that a manifest written before the field is still worth loading, and it is the only
#: way this set grows.
POST_HOC_OPTIONAL_FIELDS = frozenset({"judge_pair_template_sha256", "retrieval_stack_version"})

#: The ablations this project knows how to compare: the named condition, mapped to every field
#: that condition owns. `guardrails_sha256` brings `guardrails` with it, because turning the
#: screens off removes the configuration as well as the behaviour — the digest is None on an off
#: run — and a guard that permitted only the digest to move would refuse every on/off pair it
#: exists to accept.
#:
#: A registry rather than a free-form field name at the call site. An ablation over an
#: unregistered field is refused, so "vary this one field" cannot become "exempt this field" by
#: someone passing a name that looked reasonable, and adding an ablation is a decision recorded
#: here where the whole set is visible at once.
ABLATION_CONDITIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {"guardrails_sha256": frozenset({"guardrails_sha256", "guardrails"})}
)

#: What an ablation permits to differ whatever it varies. This is `COMPARABLE_EXEMPT` minus the
#: model identity: an ablation is *one model in two settings*, so `model_name` and `provider`
#: are ordinary conditions here and a pair differing in them is refused. Cost stays exempt for
#: the reason it is exempt there — it is an outcome of the run, not a condition on it, and a
#: guardrailed arm that made fewer model calls is *supposed* to cost less.
#:
#: Written out rather than computed from `COMPARABLE_EXEMPT`, so that widening one set cannot
#: silently widen the other. `COMPARABLE_EXEMPT` is the arm-comparison claim and this is the
#: ablation claim; they are two claims that happen to overlap.
ABLATION_EXEMPT = frozenset({"usd_cost"})

#: Fields that name a condition without being one. `dataset_path` is a filename; two runs over
#: `datasets/core.jsonl` before and after an edit differ in no path and in every byte, so
#: `dataset_sha256` is what must match. `pairs_path` is the same for a judge run. This is not a
#: relaxation of anything previously checked — a path has never been a condition.
INFORMATIONAL_FIELDS = frozenset({"dataset_path", "pairs_path"})

ManifestLike = RunManifest | Mapping[str, Any]


class NotComparableError(ValueError):
    """Two runs differ in a condition that must be held constant to compare them."""


def _as_field_map(manifest: ManifestLike) -> dict[str, Any]:
    """Normalise a manifest or manifest-shaped mapping to a plain dict."""
    if dataclasses.is_dataclass(manifest) and not isinstance(manifest, type):
        return {f.name: getattr(manifest, f.name) for f in fields(manifest)}
    if isinstance(manifest, Mapping):
        return dict(manifest)
    raise TypeError(f"Expected a RunManifest or a mapping, got {type(manifest).__name__}")


def _require_same_schema(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    """Raise unless both manifests describe exactly the same set of conditions.

    Comparing manifests with different field sets is the failure mode that would quietly
    hollow out this guard: a field present in only one of them cannot be checked, so the
    comparison would pass while an unknown condition went unexamined. Refuse instead.
    """
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a or only_b:
        details = []
        if only_a:
            details.append(f"only in first: {only_a}")
        if only_b:
            details.append(f"only in second: {only_b}")
        raise ValueError(
            f"Manifests do not share a schema and cannot be compared ({'; '.join(details)})"
        )


def compare_manifests(a: ManifestLike, b: ManifestLike) -> list[str]:
    """Return the sorted names of fields whose values differ between `a` and `b`.

    Reports every difference, including identity and informational fields — those often
    differ, and an honest diff should say so. `assert_comparable` is the function that knows
    which differences are acceptable.

    Raises:
        ValueError: the two manifests do not have the same field set.
    """
    a_map = _as_field_map(a)
    b_map = _as_field_map(b)
    _require_same_schema(a_map, b_map)
    return sorted(name for name, value in a_map.items() if value != b_map[name])


def assert_comparable(a: ManifestLike, b: ManifestLike) -> None:
    """Raise unless `a` and `b` differ only in model identity.

    This is the guard behind the central claim of the project: that the two agents were
    measured under one harness, so a score gap is attributable to the models rather than to
    their conditions. Everything in the manifest must match except `COMPARABLE_EXEMPT`, with
    run identity and `INFORMATIONAL_FIELDS` ignored.

    Two runs of different `run_kind` are refused before any field is examined. A chat session,
    an eval run, and a judge run are not arms of an experiment, and reporting the difference as
    a list of drifted conditions would invite someone to reconcile them one field at a time.
    Two judge runs are compared on their own terms: `pairs_sha256` and `judge_rubric_sha256`
    are ordinary conditions here, so a rubric edit or a different pairs file is refused, while
    `pairs_path` is informational for the reason `dataset_path` is.

    Raises:
        NotComparableError: the runs are of different kinds, or some other condition differs.
            In the latter case the message names every offending field with both values, so
            one call reports the full extent of the drift instead of the first instance of it.
        ValueError: the two manifests do not have the same field set.
    """
    a_map = _as_field_map(a)
    b_map = _as_field_map(b)
    _require_same_schema(a_map, b_map)

    a_kind, b_kind = a_map.get("run_kind"), b_map.get("run_kind")
    if a_kind != b_kind:
        raise NotComparableError(
            f"Runs are of different kinds ({a_kind!r} != {b_kind!r}) and are not two arms of "
            "one experiment"
        )

    ignored = IDENTITY_FIELDS | COMPARABLE_EXEMPT | INFORMATIONAL_FIELDS
    offending = [name for name in compare_manifests(a_map, b_map) if name not in ignored]
    if not offending:
        return

    lines = [f"  {name}: {a_map[name]!r} != {b_map[name]!r}" for name in offending]
    allowed = ", ".join(sorted(COMPARABLE_EXEMPT))
    message = (
        f"Runs are not comparable: {len(offending)} condition(s) differ beyond "
        f"{{{allowed}}}:\n" + "\n".join(lines)
    )
    unknown = _predates_field(a_map, b_map, offending)
    if unknown:
        message += (
            "\n\n"
            + "\n".join(
                f"One manifest predates {name}: it is None there and recorded here, so that run "
                f"was written before the field existed. The value is unknown rather than "
                f"different, and unknown is not evidence of sameness — re-score under the current "
                f"code, or compare the two runs on the conditions both of them recorded."
                for name in unknown
            )
        )
    raise NotComparableError(message)


def assert_ablation_comparable(
    a: ManifestLike, b: ManifestLike, *, varying: str = "guardrails_sha256"
) -> None:
    """Raise unless `a` and `b` differ only in the one condition named by `varying`.

    The sibling of `assert_comparable`, with the same one-variable discipline and a different
    variable. `assert_comparable` will correctly refuse a guardrails-on/guardrails-off pair once
    `guardrails` is a manifest field, and that refusal is right rather than an obstacle: the two
    runs are not two arms of one experiment, they are one arm under two settings. This is the
    guard for that question.

    **`COMPARABLE_EXEMPT` is not widened, and must not be.** That set is the arm-comparison
    claim — the arms differed in which model answered and in nothing else — and loosening it to
    accommodate an ablation would weaken every comparison in the project, including the ones
    nobody re-reads. A second guard costs a function; a wider exemption costs the claim.

    Unlike the arm comparison, `model_name` and `provider` are ordinary conditions here: an
    ablation is one model in two settings. The four-run design is therefore a 2×2 whose *edges*
    are comparable and whose diagonal is not. Frontier-with-guardrails against
    OSS-without differs in two conditions at once, and it is refused by both guards — by this
    one on the model, and by `assert_comparable` on the guardrails.

    The condition must actually have varied. Two runs identical in `varying` are refused with a
    pointer to `assert_comparable`, because a report row labelled as a guardrails delta between
    two runs configured the same way is a number someone will quote.

    Args:
        varying: A key of `ABLATION_CONDITIONS`. Anything else is refused rather than treated as
            a field name to exempt.

    Raises:
        NotComparableError: the runs are of different kinds, the named condition did not vary, or
            some other condition did.
        ValueError: `varying` names no registered ablation, or the two manifests do not have the
            same field set.
    """
    if varying not in ABLATION_CONDITIONS:
        known = ", ".join(sorted(ABLATION_CONDITIONS))
        raise ValueError(
            f"{varying!r} is not a registered ablation condition; known: {known}. Add it to "
            f"ABLATION_CONDITIONS with the fields it owns, so that varying a condition stays a "
            f"decision recorded in one place rather than an exemption granted at a call site"
        )

    a_map = _as_field_map(a)
    b_map = _as_field_map(b)
    _require_same_schema(a_map, b_map)

    a_kind, b_kind = a_map.get("run_kind"), b_map.get("run_kind")
    if a_kind != b_kind:
        raise NotComparableError(
            f"Runs are of different kinds ({a_kind!r} != {b_kind!r}) and are not one run under "
            "two settings"
        )

    condition = ABLATION_CONDITIONS[varying]
    differing = compare_manifests(a_map, b_map)
    if not (condition & set(differing)):
        raise NotComparableError(
            f"Runs do not differ in {varying} ({a_map.get(varying)!r} on both sides), so there "
            f"is no ablation between them. Two runs of one configuration are compared with "
            f"assert_comparable, which is the guard for a replicate"
        )

    ignored = IDENTITY_FIELDS | ABLATION_EXEMPT | INFORMATIONAL_FIELDS | condition
    offending = [name for name in differing if name not in ignored]
    if not offending:
        return

    lines = [f"  {name}: {a_map[name]!r} != {b_map[name]!r}" for name in offending]
    allowed = ", ".join(sorted(condition | ABLATION_EXEMPT))
    raise NotComparableError(
        f"Runs are not a {varying} ablation: {len(offending)} condition(s) differ beyond "
        f"{{{allowed}}}:\n" + "\n".join(lines) + "\n\n"
        "An ablation holds everything still but one condition, model included — a pair "
        "differing in both the model and the setting yields a delta attributable to neither."
    )


def _predates_field(
    a: Mapping[str, Any], b: Mapping[str, Any], offending: Sequence[str]
) -> list[str]:
    """Return the post-hoc fields whose difference is a None on exactly one side.

    A refusal either way — the field is not exempt — but a message saying "None != 'a1b2...'" sends
    a reader looking for a condition that changed, when what happened is that one manifest is older
    than the field. Naming that is the difference between a guard that teaches and one that
    annoys.

    One caveat this cannot see: `retrieval_stack_version` is also None on a run that used no
    corpus, so a corpus run against a corpus-free one is reported as an age gap when it is really
    a difference in whether retrieval happened at all. `kb_sha256` and `chunk_size` differ in the
    same pair and are named alongside it, which is what tells the two cases apart.
    """
    return [
        name
        for name in offending
        if name in POST_HOC_OPTIONAL_FIELDS and (a.get(name) is None) != (b.get(name) is None)
    ]


def agent_config_fields() -> tuple[str, ...]:
    """Return the manifest's agent-config field names, in declaration order.

    The conditions that must hold still for a run to keep its identity: everything that is
    neither run identity, eval-only, nor judge-only. `agent.session` compares exactly this group
    to decide whether a mid-session change has made the current manifest a description of a
    different run.
    """
    skipped = IDENTITY_FIELDS | EVAL_ONLY_FIELDS | JUDGE_ONLY_FIELDS
    return tuple(f.name for f in fields(RunManifest) if f.name not in skipped)


def agent_config_digest(manifest: ManifestLike) -> str:
    """Digest the agent-config group of `manifest`.

    One value answering "is this still the same run?", so a caller does not compare a dozen
    fields by hand and forget the one added last month.
    """
    field_map = _as_field_map(manifest)
    return sha256_text(_canonical({name: field_map[name] for name in agent_config_fields()}))
