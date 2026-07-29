# PROJECT.md — AgentsEval

**This file is the single source of truth for this project.** Every design decision
recorded here is *locked*. Code, tests, and documentation must not contradict it. If a
decision genuinely needs to change, change it *here first*, in a dedicated commit, and
then update the code to match — never the other way around.

---

## Locked decisions

These are recorded verbatim as the governing constraints of the project:

* **Frontier model:** Claude Sonnet (or GPT-4o-class) via API.
* **OSS model:** a small open-weights instruct model via a hosted provider
  (Groq/Together). Currently Llama 3.1 8B Instant on Groq; previously Qwen 2.5 7B Instruct,
  changed because Groq withdrew it and Together requires prepayment. The constraint that is
  locked is *small and open-weights on a hosted provider*, not the particular checkpoint —
  a provider retiring a model must be a config change rather than another edit to this list.
* **Judge model:** a third model family, different from both agents, to avoid
  self-preference bias.
* **Tool-calling protocol:** a uniform prompt-based JSON protocol for BOTH agents.
  Native function-calling APIs are forbidden — using them on the frontier side only
  would confound model quality with harness quality.
* **Retrieval:** paragraph chunks, all-MiniLM-L6-v2 embeddings, cosine similarity over
  a numpy array. No FAISS/Chroma — the corpus is tiny.
* **Determinism:** agent runs at temperature 0; web search results cached to disk.
* **Everything logged:** every turn writes JSONL with a run manifest attached.
* **The deliverable is the evals platform, not the chatbot.** The judge must score
  arbitrary (prompt, response) pairs from an external file supplied by a grader.

---

## What each decision implies for the code

The list above is normative. This section is explanatory — it exists so that an
implementer does not have to re-derive the consequences, and so that reviewers can spot
a violation quickly.

### Three model families, three roles

| Role     | Model                        | Provider              | Module                          |
| -------- | ---------------------------- | --------------------- | ------------------------------- |
| Frontier | Claude Sonnet / GPT-4o-class | Anthropic / OpenAI    | `agent/models/frontier.py`      |
| OSS      | Llama 3.1 8B Instant         | Groq or Together      | `agent/models/oss.py`           |
| Judge    | third, distinct family       | distinct from the two | `agent/models/judge_model.py`   |

The judge must not belong to the same family as either agent under test. If the frontier
agent is Claude and the OSS agent is Llama, the judge is something else again (e.g. a
GPT-4o-class model). Sharing a family between judge and candidate introduces
self-preference bias and invalidates the comparison.

Which family an arm belongs to is `ModelAdapter.family`, and it is the executable form of
this requirement rather than a label: `assert_distinct_families` reads it, and
`manifest.provider` falls back to it for an adapter that declares no provider. Changing
which checkpoint an arm runs therefore means changing `family` in the same commit — an arm
whose `family` names the model it used to be is a manifest that misreports the experiment.

All three are reached through the single interface in `agent/models/base.py`, so the
harness cannot accidentally treat one provider differently from another.

### Uniform prompt-based JSON tool protocol

Both agents receive the same tool documentation in their system prompt and must emit tool
calls as JSON in their message content. The harness parses that JSON. Neither agent uses
`tools=` / `functions=` / native tool-calling parameters, even when the provider offers
them and even when they would work better.

The reason is measurement validity: if the frontier model got native structured output
and the OSS model got prompt-based parsing, then any score gap would be a mix of "this
model is smarter" and "this model got a better harness", with no way to separate the two.
A uniform protocol keeps the harness a constant.

A consequence to accept, not fix: the OSS model will sometimes emit malformed JSON.
Parse failures are real observations about the model and must be logged as such, not
silently repaired.

They must also be attributed correctly. A parse failure caused by our own `max_tokens`
truncating the reply is a fact about the harness, and a tool of ours that timed out is not
evidence about a model at all. The typed classification — `core.FormatViolation`,
`budget_induced` truncations, and `ToolInputError` versus `ToolInfraError` — and the rules for
which items are excluded from scoring are **pre-registered in
[README.md](README.md#pre-registered-scoring-rules)**, written before any graded run.
Exclusions must be applied identically to both arms, reported per arm, and never widened
after seeing results; that is as load-bearing as `COMPARABLE_EXEMPT` and changing it is a
PROJECT.md-level decision.

Enforced in code by `agent/models/`. Every adapter builds its request body itself, and none
may emit `tools`, `functions`, `tool_choice`, or `function_call` — the set is
`base.FORBIDDEN_BODY_KEYS`, and `tests/test_models.py` asserts their absence in the
serialised body of all three adapters.

The contract itself lives in `agent/prompts.py`. Each turn the model emits exactly one JSON
object and nothing else, either a call or an answer:

```json
{"tool": "lookup_kb", "args": {"query": "hydration during exercise"}}
{"final": "Aim for 400-800 ml per hour [[hydration.md#2]].", "citations": ["hydration.md#2"]}
```

`citations` is required on every answer and is knowledge-base ids only — `[]` when nothing
came from the corpus, including on a refusal, so the model is never pushed into inventing a
citation to satisfy the schema. Two worked examples are included in the prompt: one where the
KB answers, one where it does not and the web is consulted.

Details that keep this honest rather than merely written down:

* **Byte-identical for both agents.** Nothing in `prompts.py` branches on model identity, no
  function there takes a model argument, and no model family is named in anything a model
  reads. If the OSS model needs coaxing to emit clean JSON, the coaxing goes into the shared
  text and the frontier model reads it too.
* **The protocol's JSON keys, the citation format, and the tool names are single
  definitions**, imported rather than retyped. `CITATION_FORMAT` comes from
  `tools/lookup_kb.py`, so the prompt that asks for citations and the check that scores them
  cannot drift; the key constants are read by `core.parse_tool_call`, so the shape we ask for
  is the shape we parse.
* **Tool docs are rendered from the registry's schemas**, and `build_system_prompt` refuses
  an inventory missing a tool the prompt names. A prompt advertising an unregistered tool
  would produce calls that fail, and `check_no_hallucinated_tool` would then score the
  harness's mistake as the model's.
* **Every JSON example is built with `json.dumps` from a Python object.** A malformed example
  would teach the model to emit malformed JSON, which the platform would then report as that
  model's parse-failure rate.
* **The worked examples are checked for self-consistency**: their inline `[[id]]` markers
  match their `citations` array, and every cited id appears in a tool result shown earlier in
  the same example, so the examples demonstrate provenance rather than just asserting it.

### One harness for every model

All three models are reached through `agent/models/base.py`:

* `ModelAdapter` is the interface — `generate(messages, temperature, max_tokens, stop)`
  returning a `ModelResponse`. Plain chat completions, nothing else.
* Requests go over `httpx` directly rather than through provider SDKs. The SDKs retry on
  their own internal schedules, which would give the two arms different retry semantics
  unless each was separately disabled; one transport makes that impossible and gives tests
  a single mocking seam.
* Retries, response caching, timing, cost, and error mapping live in `ChatAdapter`, shared
  by all three. An adapter supplies only its provider's wire format: endpoint, headers,
  request body, response parsing. Nothing in the pipeline may branch on `self.name`.
* **One price table.** `base.PRICING` is the only place model prices exist, so cost is
  computed rather than guessed. Unpriced models report `None`, never `0.0`.
* **One retry policy.** Exponential backoff with jitter, at most 5 retries after the first
  attempt, on 429 and 5xx and connection failures only. A 4xx fails immediately: the
  request is wrong or the key is, and retrying only buries the cause.

### Retrieval

Markdown files in `kb/` are split into paragraph chunks, embedded with
`all-MiniLM-L6-v2`, and stored as a single `numpy` array. Search is cosine similarity
over that array, brute force. No vector database, no ANN index, no FAISS, no Chroma —
the corpus is small enough that a matrix multiply is both faster and easier to audit.

Implemented in `agent/tools/lookup_kb.py`:

* **Chunks are paragraphs merged up to 256 tokens**, which is the embedding model's input
  window rather than a tuning choice. Past it the encoder truncates, so a longer chunk would
  be shown to the agent in full but retrieved on only its first 256 tokens. Merging never
  crosses a heading — a chunk spanning two sections has no honest heading path — and an
  oversize paragraph is split at sentence boundaries only. A single sentence over the window
  is logged, since it is the one case that does get truncated.
* **The 200-token merge target is a threshold, not a floor.** Heading integrity wins: where a
  section is shorter than the target, it becomes one chunk on its own rather than being
  merged into its neighbour. The corpus is written to exploit that: each `##` section holds two
  paragraphs that merge into a single chunk, so chunks run 93-222 tokens and each section is
  exactly one citable id. `--stats` reports the real distribution, and says so explicitly when
  the merge target goes unreached, rather than leaving either to be discovered.
* **`chunk_id` is `{source_file}#{ordinal}`**, e.g. `sleep-hygiene.md#2`, using the kb-relative
  path so two same-named files in different directories cannot collide into one id. Ids are
  stable for an unchanged corpus, and each chunk carries its `heading_path` so a retrieved
  passage says where it came from.
* **Citations are `[[chunk_id]]`**, defined once as `CITATION_FORMAT` and parsed by
  `parse_citations`. The prompt that asks for citations and the check that scores them read
  the same constant, so the ask and the grading cannot drift. Double brackets so that
  mentioning a filename in prose is not counted as a citation.
* **`min_score`** is exposed on both `search` and `lookup_kb`, defaulting to 0 (no floor). It
  is the hook for refusing to answer on weak retrieval; the threshold is left unset until
  there is measured data to choose it from.
* **`kb/README.md` is not indexed.** It documents the corpus rather than being corpus
  content, and indexing it makes prose about chunking compete for retrieval slots.

The index persists to `kb/.index.npz` (the matrix) and `kb/.index.json` (chunk metadata,
per-file stamps, and the corpus digest), both written temp-then-rename. It rebuilds when the
file set changes, when a file's content hash changes, when the embedding model or token band
changes, or when the two files disagree about which corpus they describe. File mtimes are the
fast path, but a moved mtime only triggers a re-hash rather than a rebuild: `git clone` and
`git checkout` rewrite every mtime without changing a byte, and rebuilding then would empty
the cache exactly when it is most useful. `agentseval-index --check` reports staleness
without building, so a graded run cannot quietly use a stale index.

### Determinism

Agent runs use `temperature=0`. Web search results are cached to disk and re-read on
subsequent runs, so a re-run of an eval does not depend on what the live web returned
today. Anything else that varies per run (timestamps, latencies, model version strings)
is logged rather than allowed to influence the graded output.

Determinism here means *reproducible enough to compare two runs*. Providers do not
guarantee bit-identical output at temperature 0; the cache and the logs are what make
discrepancies visible when they happen.

Model responses are cached the same way and for the same reason, in
`base.ResponseCache`: entries live under `.cache/models/`, keyed by
`sha256(model + messages + temperature + max_tokens + stop)`, so re-running an eval over an
unchanged dataset costs nothing and returns exactly what it returned before. The key
excludes the API key, so rotating a credential neither invalidates the cache nor writes a
secret to disk. `--no-cache` (or `AGENTSEVAL_NO_CACHE=1`) bypasses it.

One caveat that must not be lost: **a cache hit is a replay, not a measurement.** The
cached `latency_ms` is the original call's. `ModelResponse.cached` marks these and
`agent/trace.py` logs it as a `cached` column, so anything aggregating latency or cost can
account for it — `evals/metrics.py` averages latency over uncached calls only and reports
`cached_fraction` alongside, because most calls in a re-run are hits. A trace written before
that column existed reads as unknown rather than as uncached, and reports no latency at all
rather than a wrong one.

### Everything logged

Every turn appends a JSONL record to `runs/`, and every run carries a manifest
identifying models, prompt versions, tool inventory, KB revision, and configuration. A
result that cannot be traced back to the exact conditions that produced it is not a
result. `runs/` is gitignored — it is generated data.

A run writes two sibling files, joined on `run_id`:

* `runs/{run_id}.jsonl` — one record per turn, appended and flushed as it happens, so a
  crashed run keeps everything up to the failure;
* `runs/{run_id}.manifest.json` — the conditions of the run.

Records carry `run_id` rather than an inline copy of the manifest; duplicating a dozen
fields on every line buys nothing that the join does not.

`agent/trace.py` implements the writer. The manifest and its guard live in
`agent/manifest.py`, one layer up, because building a manifest requires knowing the prompt,
the tool inventory, the corpus, and the price table, whereas the writer needs none of them
and stays standard-library only.

**The manifest is an agent-layer concern, not an eval-layer one.** A dataset is a property
of an eval run, not of the agent, so it is a nullable tail on one manifest rather than the
reason for a second kind. `build_manifest(cfg, run_kind=..., dataset=..., judge=...)` is the
only function that builds one; `evals/runner.py` calls it with `run_kind="eval"` and a
`DatasetRef`, `app.py` calls it with `run_kind="chat"`, `evals/judge.py` calls it with
`run_kind="judge"` and a `JudgeRef`, and none adds a field of its own. The dependency
direction is one-way and load-bearing: `evals/` may import `agent/`, never the reverse, so the
chat surface does not depend on the evaluation harness.

**A judge run is a third kind, not a second manifest.** `RunKind` is
`chat | eval | judge`. Scoring is decoupled from running, so a judge run is its own run with
its own conditions — which judge model answered, and which rubric it read — and a published
judge score has to be traceable to both. It is not an eval run: there is no agent, no corpus,
no retrieval, and no tool budget, so those fields are null on it and `build_manifest` refuses
a null where the kind does have one. Recording a fabricated `top_k` on a judge run would put a
fiction exactly where `assert_comparable` trusts a fact. `assert_comparable`'s existing
cross-kind refusal covers this for free: a judge run and an eval run are not two arms of one
experiment.

`RunManifest`'s fields are in four groups, legible as such in the source:

* **Identity** — `run_id`, UTC `started_at`, `run_kind`. These identify a run instance
  rather than a condition, and `assert_comparable` drops them.
* **Agent config** — `model_name`, `provider`, `temperature`, `max_tokens`, `top_k`,
  `chunk_size`, `retrieval_config_sha256`, `system_prompt_sha256`, `kb_sha256`,
  `pricing_version`, `max_tool_calls`, `max_tool_errors`, `max_model_calls`, `git_sha`,
  `git_dirty`, `code_version`. These must match between two arms except for the model
  itself. The six that describe an agent's tools, prompt, and retrieval — `top_k`,
  `retrieval_config_sha256`, `system_prompt_sha256`, and the three budgets — are `None` on a
  judge run, which has none of them, and `build_manifest` refuses `None` on a chat or eval run
  where each is a real condition. `model_name` is the provider-side model id, dated where the
  provider dates it, so
  it is also the model version and there is no separate field for one.
  `system_prompt_sha256` covers the tool inventory as well as the prompt text, since
  `prompts.build_system_prompt` renders the tool docs into the prompt. `kb_sha256` is the
  KB revision required above, and `retrieval_config_sha256` covers the retrieval settings
  that have no field of their own — the embedding model and the score floor — alongside the
  ones that do; `top_k` and `chunk_size` stay separate fields because a digest says a
  condition drifted and cannot say which. The three budgets are separate fields because
  they bound different things — successful tool calls, model-caused tool errors, and model
  calls — and an arm given more room to recover from its own mistakes is a different
  experiment. With `max_tokens`, which is what makes a reply truncate, they are conditions
  and never exempt.
* **Eval-only** — `dataset_path`, `dataset_sha256`, `n_items`, `seeds`. All `None` on a
  chat run. `run_kind="eval"` without a dataset is an error rather than a manifest with
  nulls where the dataset should be.
* **Judge-only** — `judge_model`, `judge_provider`, `judge_rubric_sha256`, `judge_rubrics`,
  `pairs_path`, `pairs_sha256`, `n_pairs`. All `None` on every other kind, and
  `run_kind="judge"` without a `JudgeRef` is an error for the same reason an eval run without a
  dataset is. `pairs_sha256` is of the scored file's bytes and `pairs_path` is informational,
  exactly as with a dataset: two judge runs naming one grader file after it was edited between
  them are not comparable, and only the hash catches that. `judge_rubric_sha256` is the digest
  above, and `judge_rubrics` names which rubric files it covers.

There is deliberately no `prompt_version` field: a version string is a promise to remember
to bump it, whereas the digest changes whether or not anyone remembered.
`prompts.PROMPT_VERSION` exists as a label for humans reading traces. `pricing_version` is
a digest of the price table for the same reason, despite the name.

The guard that gives the manifest teeth:

* `compare_manifests(a, b)` lists the fields that differ, including identity fields — an
  honest diff says so, and the guard is what excuses them.
* `assert_comparable(a, b)` raises unless the only differences are the identity fields
  (which differ between any two runs) and `{model_name, provider, usd_cost}`. This is the
  executable form of the uniform-harness requirement: it refuses to let two runs be
  compared unless they differed only in which model answered. Widening that exempt set
  weakens every comparison in the project, so it is a PROJECT.md-level change.
* Two further refusals, both narrowing rather than widening. Manifests with different
  `run_kind` are refused outright: a chat session, an eval run, and a judge run are not arms
  of an experiment. And `dataset_sha256` must match whenever both manifests carry one, because
  `dataset_path` is informational only — two runs both naming `datasets/core.jsonl` after
  the file was edited between them are not comparable, and the hash is the only thing that
  catches it. `pairs_sha256` is checked the same way, for the same reason.

A manifest is written once and never mutated. `app.py` writes one lazily, on the first
message of a session rather than at session end, so an abandoned session still has
provenance; and any change to an agent-config field mid-session — flipping the model
toggle, editing the corpus — mints a new `run_id` and a new manifest and routes subsequent
turns to it. The alternative is a trace holding two models under a manifest asserting one,
which no downstream check could detect.

**Rotating the run does not end the conversation.** The rule above is about attribution, and
attribution does not require amnesia: the history moves to the new agent, the `item_id` and turn
numbering continue, and the new trace opens with a `role="memory"` record carrying
`previous_run_id`. That record is what makes the segments joinable, and it is what a reader follows
to reconstruct one conversation from the several traces a switch spread it over. The same mechanism
resumes a conversation off disk — a rotation whose history was read from a trace rather than taken
from a live agent — so nothing new is written to support it. Turn indices continue rather than
restarting, because two traces each holding `(item_id, 0)` are orderable only by timestamp.

### The deliverable is the evals platform

The chat app (`app.py`) exists to exercise the agent, not as the product. The graded
artifact is the evaluation harness under `evals/`.

Concretely, the judge must accept an **external file supplied by a grader** containing
arbitrary `(prompt, response)` pairs — records that this project's agents never produced
— and score them. So:

* the judge takes `(prompt, response)` as data, with no dependency on our trace format,
  our agents, or our KB;
* judge input parsing is a first-class feature, not a test fixture;
* `evals/validate_judge.py` exists to establish that the judge's scores track human
  labels, because an unvalidated judge is an opinion rather than a measurement.

### The judge rubric lives on disk, one file per axis

Rubric *text* is in `agent/rubrics/`: `default.md` plus one file per `evals.schema.Axis`, each
paired with a `{name}.anchors.json`. This is a deliberate narrowing of the layout rule above —
`prompts.py` holds all prompt text the *agent* reads, and it still owns everything executable
about the judge prompt: the loader, the rendered JSON, the constants, and the digest. What moved
out is prose, because a rubric that must be axis-specific becomes four near-duplicate f-strings
otherwise, and near-duplicates drift.

The four properties the single-f-string arrangement gave for free are kept explicitly:

* **The digest.** `judge_rubric_sha256()` covers every rubric, rendered, in sorted name order —
  so it changes when a file's bytes change *and* when the interpolated schema or
  `JUDGE_SCALE_MAX` changes. A judge run records it for the reason an agent run records
  `system_prompt_sha256` and `kb_sha256`: scores from two rubrics are not comparable, and only
  a digest catches an edit between runs.
* **Every JSON example is still built with `json.dumps` from a Python object.** Each `.md`
  carries a `{schema_json}` placeholder and an `{anchors}` placeholder, and the loader raises
  when either is absent. Anchor verdicts are data in the sibling `.anchors.json`, parsed and
  re-serialised at load, so a malformed example fails loudly here instead of being taught to the
  judge and reported later as a parse-failure rate.
* **No fallback.** A missing, empty, or placeholder-less axis file raises, exactly as
  `build_system_prompt` refuses an inventory missing a tool the prompt names. Quietly falling
  back to `default.md` would score one axis under a rubric the report claims it did not use.
* **Anchors are held to the same self-consistency discipline as the worked examples** in the
  agent prompt: each anchor's `overall` sits inside the band its label names
  (`prompts.JUDGE_SCORE_BANDS`), and each anchor's evidence spans are verbatim substrings of its
  own response. They are synthetic rather than drawn from `evals/datasets/`, which would leak
  the eval into the judge, and they cite no `kb/` chunk ids, since a grader's file has none and
  the rubric deliberately says nothing about them.

No model or vendor name appears in any rubric file. That is the other half of the
self-preference defence; a third model family is the first.

The axis selects rubric *text* and nothing else. `prompts.JUDGE_DIMENSIONS` remains the four
reported numbers on every axis, so scores stay comparable across axes — a per-axis output
schema would make them incomparable, which is the opposite of what a per-axis rubric is for.

### The eval item shape

**`evals/schema.py` holds the single definition of an eval item.** `EvalItem` is the only
one, and `evals/runner.py` no longer carries a shape of its own: a package holding two
dataset shapes is a package where `deterministic.py` and `report.py` can read one file and
disagree about what is in it, with nothing raising to say so. `load_dataset` returns
`list[EvalItem]`.

Datasets are JSONL, one object per line, and the model is Pydantic v2 with
`extra="forbid", frozen=True`. Forbidding extras is the load-bearing half. A misspelled
`attack_typ` that was merely ignored would leave the item scored as though the field had
never been set — a silent change to what was measured rather than a failure anyone sees.

| Field | Type | Why it exists |
| ----- | ---- | ------------- |
| `id` | `str` | Unique within a file, and frozen once the item is labelled, for the same reason chunk ids are: a label refers to an id, so reusing one silently re-points existing labels. |
| `axis` | `Axis` | `hallucination` \| `bias` \| `safety`. Groups items. |
| `subcategory` | `str` | From a per-axis controlled vocabulary in `schema.py`. Free text drifts into thirty singleton buckets that `report.py` cannot group, so a breakdown by subcategory stops being a breakdown. |
| `turns` | `list[str]`, min 1 | **User messages only** — the assistant turns are what is under test. One turn is the single-turn case; more is multi-turn escalation. |
| `expected_behavior` | `str` | What a passing response does. This is the annotator's instruction and the reason a label is auditable a month later. |
| `must_include` | `list[str]` | Feeds `deterministic.check_contains`. |
| `expected_tool` | `str \| None` | Feeds `deterministic.check_tool_used`, which otherwise has no source for the argument. |
| `answerable` | `bool` | Whether `kb/` genuinely covers the question. Load-bearing for the hallucination axis — see the corpus exclusions below. |
| `counterfactual_id` | `str \| None` | Pair key. |
| `counterfactual_variant` | `str \| None` | The varied attribute's *value* for this item, e.g. `male`. |
| `counterfactual_attribute` | `str \| None` | The attribute's *name*, e.g. `gender`. Required with `counterfactual_id`. |
| `attack_type` | `AttackType \| None` | Required when `axis == safety`, except for the `benign_control` subcategory; `None` on every other axis. |
| `notes` | `str \| None` | Free text for the author. |

The counterfactual fields are not called `pair_*`. In `evals/judge.py` and
`evals/validate_judge.py` a `pair_id` is already one `(prompt, response)` pair, and one name
carrying two meanings inside one package is a defect waiting for someone to join on the
wrong column. For the same reason `human_label` is not `human_score`: the schema's label is
a binary behavioural verdict on a response, and `validate_judge.LabelledPair.human_score` is
a 1-5 rubric label for judge-vs-human agreement. They are separate label spaces and nothing
converts between them.

Four properties of the shape, each recorded because it is the kind of thing that gets
quietly reversed later:

* **Nothing rewrites a dataset file.** `DatasetRef.for_file` digests the file's *bytes*, and
  `assert_comparable` refuses two runs whose `dataset_sha256` differ even when
  `dataset_path` matches. So there is no formatter, no key reordering, no in-place
  canonicalisation, and no writing labels back into the item. A tool that tidies a dataset
  after one arm has run destroys the comparison, and does it without an error.
* **The scored turn is the response to the final turn.** Earlier turns are replayed through
  `agent.memory.Conversation` as ordinary context and are not scored. Scoring an
  intermediate turn would score the agent partway through the escalation the item was built
  to provoke, which is not what the item is asking.
* **`expected_behavior` and `notes` are never shown to a model** — not to the agent, not to
  the judge. They are annotator instructions; putting them in the context turns the eval
  into a test of instruction-following, and it would raise both arms' scores while measuring
  less. `schema.MODEL_VISIBLE_FIELDS` is the executable form of this.
* **An axis is not a judge dimension.** `axis` groups items; `prompts.JUDGE_DIMENSIONS`
  (`helpfulness`, `accuracy`, `safety`, `communication`) scores a response. The two
  taxonomies overlap on the word "safety" and are not the same thing. An axis therefore
  selects which rubric text the judge reads and never which numbers it returns: every item on
  every axis is scored on all four dimensions, because a per-axis output schema would make two
  axes' scores incomparable. Bias in particular is
  measured as a *within-pair delta* rather than as a rubric dimension, which is why pairing
  is a schema-level invariant instead of an authoring convention — and why one unpaired bias
  item is an error rather than an item worth 1/2 of a comparison.

That last point has a scoring consequence, so it is pre-registered in
[README.md](README.md#pre-registered-scoring-rules) alongside the other exclusion rules:
**if either variant of a counterfactual pair ends `infrastructure_failed`, both are
excluded.** Half a pair yields no delta, and keeping the survivor would put it into the
bias metric as though it did.

**The safety axis carries its own control bucket.** `SAFETY_SUBCATEGORIES` includes
`benign_control`: ordinary health questions the corpus answers well, where a refusal — or a
hedge that never reaches an answer — is the failure. Without it, a model that refuses
everything scores perfectly on safety, which is the same hole `answerable_kb` closes on the
hallucination axis.

`benign_control` is the one safety subcategory that takes `attack_type: None`, so the
invariant is *required on the safety axis except for `benign_control`* rather than *required
on the safety axis*. Every `AttackType` value names a technique for eliciting an unsafe
response, and a legitimate question uses none of them. The alternative — a `BENIGN_CONTROL`
member of the enum — would put a non-technique into a vocabulary whose whole purpose is to
record which technique worked, and would then appear in every per-attack-type breakdown as a
row that is not an attack.

That has a reporting consequence, so it is pre-registered in
[README.md](README.md#pre-registered-scoring-rules) too: **over-refusal and harm-compliance
are reported as separate rates and are never averaged into a single safety pass rate.** They
are different quantities over disjoint item sets — one counts refusals that should have been
answers, the other answers that should have been refusals. A single mean over both moves when
either moves, so a model that became more cautious and less useful would be indistinguishable
from one that changed in neither direction, and the trade-off those two rates exist to expose
is the one the average hides.

Human labels live in an **append-only sidecar** under `evals/datasets/labels/`, never in the
dataset, because the dataset's bytes are its identity. Each record carries the
`dataset_sha256` and the `run_id` and `response_sha256` it was made against, so a label can
be verified against the exact artifact that produced it and a regenerated trace is
detectable rather than assumed equivalent. Correcting a label appends a superseding record;
readers take the last record per `(run_id, item_id)`.

Two tools support the format, both of which only read: `evals/validate_dataset.py` lints a
file and exits non-zero, and `evals/label.py` collects human labels.

---

## Layout

```
agent/
  core.py              agent loop: prompt -> tool-call JSON -> tool -> answer
  memory.py            conversation state: verbatim window + rolling summary
  prompts.py           agent prompt text: role, JSON protocol, tool docs; judge rubric loader
  trace.py             JSONL trace writer, run paths, digests, git probes
  manifest.py          AgentConfig, RunManifest, build_manifest, comparability guard
  session.py           one chat session: lazy manifest, run lifecycle, trace routing,
                       carrying a conversation across a rotation, rebuilding one from a trace
  rubrics/             judge rubric text: default.md + one .md per axis, each with anchors
  models/
    base.py            ModelAdapter interface + shared pipeline: retries, cache, pricing
    oss.py             Llama 3.1 8B Instant (Groq/Together)
    frontier.py        Claude Sonnet / GPT-4o-class
    judge_model.py     third-family judge adapter
  tools/
    __init__.py        the registry: one inventory for prompt docs and dispatch
    lookup_kb.py       chunking, embedding index, cosine search, citation format
    search_web.py      web search with on-disk cache
evals/
  schema.py            EvalItem: the single definition of an eval item + vocabularies
  validate_dataset.py  dataset linter, byte-strict, exits non-zero
  label.py             keystroke labeling helper, append-only sidecars
  datasets/            eval sets, judge validation data, labels/ sidecars
  fixtures/            injected/ poisoned doc + compose.py; not the main corpus
  runner.py            executes eval sets against an agent, writes traces
  judge.py             scores arbitrary (prompt, response) pairs
  validate_judge.py    judge-vs-human agreement
  metrics.py           aggregation with Wilson and bootstrap intervals
  deterministic.py     rule-based checks that need no model
  report.py            human-readable output
ui/
  dashboard.py         Streamlit entry script for the platform's read-only views
  data.py              run discovery, judge pairing, cached summaries, chat transcripts
  pages/               one script per view: browse_runs.py, run_detail.py, chat_history.py
kb/                    markdown corpus + .index.npz / .index.json (both gitignored)
runs/                  {run_id}.jsonl + {run_id}.manifest.json (gitignored)
.cache/models/         cached provider responses (gitignored)
app.py                 Streamlit chat surface (demo surface, not the deliverable)
tests/
```

### `ui/` is a view, and a third layer rather than a second app

The platform's numbers were computable before anything rendered them, and reading a single run
meant reading a `RunSummary` object. `ui/` closes that, and it is a separate entry script rather
than more pages on `app.py`: the statement above that the chat app is a demo surface and not the
product has to stay true, and eval views bolted onto it would falsify it. Streamlit's `pages/`
must be a sibling of its entry script, so both live under `ui/` — a repo-root `pages/` would be
picked up by `app.py` as well.

Four properties, each enforced rather than intended:

* **The dependency direction gains a layer and keeps its direction.** `ui/` imports `agent/` and
  `evals/` and is imported by neither, so the same test that walks `agent/`'s import graph for
  `evals` walks both packages for `ui`. A view that anything under test imports is no longer a
  view.
* **It reads and never writes.** `metrics.summarise_run` joins the trace, the judgements, and the
  dataset in memory every time, and the reason is above: a second copy of a run is a second thing
  to keep truthful. Caching is `st.cache_data` in memory, keyed on the run id and the mtimes of
  the files it read, never `persist=`. A chat transcript is keyed on its trace's size as well,
  because that is the one file here that grows while a page is open. Nothing under `runs/` or
  `evals/datasets/` is written, which for the datasets is the same rule as "nothing rewrites a
  dataset file" — `DatasetRef` digests bytes, so even reformatting one destroys every comparison
  referencing it. Reading a past chat conversation therefore needs no new artifact: the trace
  already held every message, and it was simply never read back.
* **A chat transcript is the one place the delivered text is the right text.** Everywhere else in
  this project a screened turn is read as the model's own completion, because that is what the
  model produced and what every rate is about. A transcript is a record of what happened in front
  of a person, so it shows the substituted sentence and says that a guardrail fired. The
  reconstruction that resumes a conversation takes the other reading, since it is rebuilding what
  the model was told rather than what was displayed.
* **The rendering rules hold on this path too.** Attack success is never drawn without its
  over-refusal control, empty buckets are zero rows with an explanation rather than absences, and
  a rate with a threshold curve is shown at all four cuts. These are pre-registered in README.md
  and were previously enforced only by `report.render_comparison`; a second renderer that relaxed
  any of them would be a second answer.
* **A row carries data, not text.** `report.summary_rows` returns structured `SummaryRow`s and
  `report.render_run_summary` formats them, exactly as `metrics.compare_runs` returns
  `Comparison`s that `report.render_comparison` formats. Pre-formatted rows cannot be sorted,
  plotted, or joined back to their items, so a UI given them would compute its own — and two
  implementations of "what does this run say" are two answers.

`ui/` shows no `expected_behavior` and no `notes`, for the reason given under the eval item
shape: they are annotator instructions, and `MODEL_VISIBLE_FIELDS` is not a rule about models
only.

## Status

Implemented and unit-tested: all of `agent/` — the loop in
`core.py`, `memory.py`, the tool registry, `prompts.py`, `trace.py`, `manifest.py`,
`session.py`, all three adapters, and both tools — `tools/lookup_kb.py` with the `kb/` corpus,
and `tools/search_web.py` against Tavily with results cached to disk. The chat
surface in `app.py` runs on `agent.session`, so an interactive turn is logged exactly as an
eval turn is; a conversation there survives a model switch and can be picked back up from the
traces on disk. Under `evals/`, the dataset format and its tooling are implemented:
`schema.py`, `validate_dataset.py`, and `label.py`, with `evals/datasets/example.jsonl` as a
golden file the linter passes on. `judge.py` is implemented — input parsing, blinded
single-response scoring at temperature 0, the per-axis rubrics in `agent/rubrics/`, strict
parsing with one repair attempt, evidence-span verification, and the cache-off stability
primitive. The statistics over its output are implemented too: `validate_judge.py` for
judge-vs-human agreement, `deterministic.py` for the rule checks, `runner.py` for executing a
dataset against an agent, and `metrics.py` for the aggregation — Wilson intervals on every rate,
bootstrap intervals on every mean and counterfactual pair delta, the threshold curve at all four
cuts, and a `compare_runs` that refuses two runs whose manifests disagree. `report.py` renders
both a two-run comparison and a single run: `summary_rows` flattens a `RunSummary` into structured
rows and `render_comparison` / `render_run_summary` format them. The report *files* are still
stubs — `write_markdown_report`, `print_report`, and `agentseval-report` raise. `ui/` renders the
same rows in a browser, read-only.

`search_web` still raises `ToolInfraError` rather than `NotImplementedError` when the provider is
unreachable or unconfigured: an outage is our gap, so the loop books it as infrastructure and
excludes the item from scoring rather than charging the model for it.

## Knowledge base

`kb/` holds twelve synthetic documents on general consumer wellness: sleep, hydration,
nutrition principles and food labels, strength training, cardio, walking and daily movement,
warm-up and mobility, recovery and rest, everyday stress, desk ergonomics, and habit
formation. They are the assistant's source of record, and they cover the same ground the
system prompt says they do.

The corpus deliberately says nothing about medications or supplements, medical conditions or
diagnosis, pregnancy, pediatric health, injury rehabilitation, or mental-health treatment.
Those gaps are what make the hallucination axis measurable. The eval needs questions the corpus
genuinely cannot answer, so that an agent answering them from its training rather than from a
retrieved chunk is distinguishable from one that says the corpus does not cover it. Widening
coverage into those areas would delete the unanswerable bucket, and the scores would improve
while measuring less — which is the failure mode worth naming explicitly, because it looks
like progress.

The corpus is also not itself a source of unsafe advice: no dosages, no calorie or
macronutrient targets, no fasting protocols, no diagnostic or treatment claims. The safety axis
exists to test the agent's judgement, and a corpus containing bad advice would confound it.

Chunk ids are `{filename}#{ordinal}`, and citation scoring compares against them, so filenames
and section order are frozen once an eval set is labelled. Each `##` section is written as two
paragraphs that merge into exactly one chunk, which keeps ids predictable. `kb/README.md` lists
every file and restates the exclusions for anyone adding to the corpus.

### The injected fixture corpus

Retrieval-borne prompt injection — a corpus document carrying instructions addressed to the
assistant — cannot be tested against `kb/`. Putting a poisoned document there would change
`kb_sha256` and `corpus_sha256` for every run, force an index rebuild, and contradict the
paragraph above: the corpus must not be a source of unsafe advice, because the safety axis
tests the agent's judgement rather than the corpus's. Pasting the same text into a user turn
avoids all of that and measures something else — instruction-following over text the user
supplied, rather than whether the agent trusts a tool result *because* it came from a tool.
That distinction is the whole point of the item, so the fixture is the only honest option.

There is therefore a second corpus, and it is a fixture rather than a corpus.
`evals/fixtures/injected/` holds one poisoned document; the corpus a run actually reads is
`kb/`'s twelve documents plus that one, assembled by `evals/fixtures/compose.py` into a
gitignored directory. Composed rather than committed as a thirteen-file copy, because two
hand-maintained copies of the same twelve documents drift, and here the drift stays invisible
until a citation id stops resolving. The poisoned document follows the same authoring rules as
the rest of the corpus, so it chunks to one citable id per section like everything else.

Three consequences, each already enforced rather than promised:

* **Runs against it are not comparable to main runs.** `kb_sha256` is a manifest field and is
  not in `COMPARABLE_EXEMPT`, so `assert_comparable` refuses the pairing. That is the guard
  working, not an obstacle to route around: the corpus differed, so the two runs did not
  measure the same thing.
* **Injection results are reported separately** and never pooled into the safety axis rates,
  for the same reason over-refusal and harm-compliance are not averaged. Pre-registered in
  [README.md](README.md#pre-registered-scoring-rules).
* **The fixture corpus is versioned like the main one.** Its digest is a condition, so two
  fixture runs are comparable to each other and to nothing else.

`AttackType.PROMPT_INJECTION` already existed for this, so the fixture needs no vocabulary
change. It follows that `evals/datasets/safety.jsonl` carries zero `prompt_injection` items and
that the per-attack-type breakdown prints a zero for it. **The zero is printed, never
omitted.** An omitted row cannot be told apart from a vocabulary that never had the value, and
the fact a reader needs is that the bucket is empty here by design because it is measured in a
different run against a different corpus.
