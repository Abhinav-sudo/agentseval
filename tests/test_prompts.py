"""Tests for `agent.prompts`.

Prompt text is hard to test and easy to break, so these tests aim at the properties that
actually matter rather than at wording:

* the prompt is byte-identical for both agents, and nothing in the module can branch on
  which model is running — the uniform-harness requirement in PROJECT.md;
* every JSON object shown to the model parses, since an invalid example would teach the
  model to emit invalid JSON and the harness would then record that as the model's parse
  failure rate;
* the worked examples are self-consistent: their inline citations match their `citations`
  array, and every cited id appears in a tool result shown earlier in the same example;
* the format documented in the protocol is the format the harness actually sends;
* the digest covers the tool inventory, which is what lets a manifest prove both agents saw
  the same instructions.

The six requirements the system prompt has to state are asserted individually, so dropping
one during a rewrite fails a named test.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from types import MappingProxyType

import pytest

from agent import prompts
from agent.prompts import (
    ARGS_KEY,
    CITATIONS_KEY,
    FINAL_ANSWER_REQUIRED,
    FINAL_KEY,
    JUDGE_DIMENSIONS,
    JUDGE_EVIDENCE_KEY,
    JUDGE_SCALE_MAX,
    KB_TOOL,
    PROTOCOL_ERROR_PREFIX,
    REQUIRED_TOOLS,
    SUMMARY_PREFIX,
    SYSTEM_PROMPT,
    TOOL_BUDGET_PREFIX,
    TOOL_ERROR_PREFIX,
    TOOL_KEY,
    TOOL_PROTOCOL,
    TOOL_RESULT_PREFIX,
    WEB_TOOL,
    build_system_prompt,
    judge_rubric_names,
    judge_rubric_prompt,
    judge_rubric_sha256,
    render_bad_argument_type_error,
    render_judge_pair,
    render_missing_argument_error,
    render_summary,
    render_summary_request,
    render_tool_docs,
    render_tool_error,
    render_tool_result,
    render_unknown_argument_error,
    render_unknown_tool_error,
    system_prompt_sha256,
)
from agent.tools import lookup_kb, search_web, tool_specs
from agent.tools.lookup_kb import CITATION_FORMAT, Chunk, Hit, parse_citations
from agent.tools.search_web import SearchResult
from agent.trace import sha256_text

TOOL_MODULES = (lookup_kb, search_web)

#: The real corpus, located from this file rather than the working directory.
REAL_KB = Path(__file__).resolve().parent.parent / "kb"


@pytest.fixture
def specs():
    """Tool specs as `agent.tools.tool_specs()` will produce them."""
    return [
        {"name": module.name, "description": module.description, "schema": module.schema}
        for module in TOOL_MODULES
    ]


@pytest.fixture
def prompt(specs):
    return build_system_prompt(specs)


def json_objects(text: str) -> list[dict]:
    """Parse every JSON object the prompt presents on a line of its own."""
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            found.append(json.loads(stripped))
    return found


def flat(text: str) -> str:
    """Collapse whitespace, so a phrase assertion survives the prompt being re-wrapped."""
    return " ".join(text.split())


def worked_examples(prompt: str) -> list[dict]:
    """JSON objects from the worked examples only, excluding the shape templates."""
    _, _, examples = prompt.partition("**Example ")
    return json_objects(examples)


# --------------------------------------------------------------------------------------
# One harness: no model-specific anything
# --------------------------------------------------------------------------------------


#: Not bare "meta": the core-module check is a substring search over source, where it would
#: trip on any future `metadata` identifier.
MODEL_FAMILIES = (
    "claude",
    "sonnet",
    "gpt-4",
    "gpt4",
    "qwen",
    "llama",
    "meta-llama",
    "anthropic",
    "openai",
    "groq",
)


def test_no_model_family_is_named_in_anything_the_model_reads(prompt):
    """Coaxing aimed at one model would make the harness a variable rather than a constant.

    Checked on every rendered artefact, because text a model never sees cannot affect it —
    the module's own docstring is free to discuss the policy.
    """
    rendered = "\n".join(
        [
            prompt,
            judge_rubric_prompt(),
            render_judge_pair("q", "a", "r"),
            render_tool_result(KB_TOOL, [hit()]),
            render_tool_error(KB_TOOL, "boom"),
            render_tool_error(None, "boom"),
            FINAL_ANSWER_REQUIRED,
            render_summary_request([{"role": "user", "content": "q"}]),
            render_summary("a summary"),
        ]
    ).lower()
    for family in MODEL_FAMILIES:
        assert family not in rendered, f"prompt text mentions {family!r}"


def test_no_public_function_accepts_a_model(specs):
    """There is no seam through which a per-model prompt could be requested."""
    for name, value in vars(prompts).items():
        if name.startswith("_") or not callable(value) or inspect.isclass(value):
            continue
        if getattr(value, "__module__", None) != prompts.__name__:
            continue
        parameters = set(inspect.signature(value).parameters)
        assert parameters.isdisjoint({"model", "model_name", "provider", "family"}), name


def test_the_real_registry_produces_the_specs_these_tests_assume(specs):
    """Otherwise this file could pass on a tool inventory the agent never sees."""
    assert tool_specs() == specs


def test_rendering_is_deterministic(specs):
    assert build_system_prompt(specs) == build_system_prompt(list(specs))
    assert system_prompt_sha256(specs) == system_prompt_sha256(list(specs))


# --------------------------------------------------------------------------------------
# The six things the system prompt must say
# --------------------------------------------------------------------------------------

REQUIREMENTS = [
    ("a: prefer the KB", [rf"`{KB_TOOL}` first"]),
    ("b: web only as fallback", [rf"`{WEB_TOOL}` only when", r"insufficient"]),
    (
        "c: cite chunk ids inline",
        [re.escape(CITATION_FORMAT.format(chunk_id="chunk_id")), r"inline", r"verbatim"],
    ),
    ("d: say you don't know", [r"do not know, say so", r"guess"]),
    (
        "e: refuse harm, still answer health questions",
        [r"Refuse these", r"self-harm", r"legitimate health questions"],
    ),
    ("f: never claim to be a doctor", [r"not a doctor", r"[Nn]ever say or imply"]),
]


@pytest.mark.parametrize(
    ("requirement", "patterns"), REQUIREMENTS, ids=[r[0] for r in REQUIREMENTS]
)
def test_system_prompt_states_requirement(requirement, patterns):
    """Matched against whitespace-collapsed text, so re-wrapping a line is not a failure."""
    for pattern in patterns:
        assert re.search(pattern, flat(SYSTEM_PROMPT)), f"{requirement}: missing {pattern!r}"


def test_refusal_is_partial_not_wholesale():
    """Over-refusal is a failure mode too, so the prompt has to say to answer the rest."""
    assert "Refuse the harmful part specifically and answer the rest" in flat(SYSTEM_PROMPT)


def test_emergencies_come_first():
    assert "emergency care now, and say it first" in flat(SYSTEM_PROMPT)


def test_disclaimers_are_not_requested_on_every_answer():
    """Reflexive disclaimers are the standard way this instruction goes wrong."""
    assert "Do not append it to every answer" in flat(SYSTEM_PROMPT)


def test_system_prompt_does_not_hardcode_a_regional_crisis_number():
    """A US hotline shown to a user elsewhere is worse than pointing at local services."""
    assert not re.search(r"\b\d{3}[- ]?\d{3,4}\b", SYSTEM_PROMPT)
    assert "crisis line" in SYSTEM_PROMPT


# --------------------------------------------------------------------------------------
# Protocol shape
# --------------------------------------------------------------------------------------


def test_protocol_documents_exactly_the_two_shapes():
    assert "exactly one JSON object" in TOOL_PROTOCOL
    assert f'"{TOOL_KEY}"' in TOOL_PROTOCOL
    assert f'"{ARGS_KEY}"' in TOOL_PROTOCOL
    assert f'"{FINAL_KEY}"' in TOOL_PROTOCOL
    assert f'"{CITATIONS_KEY}"' in TOOL_PROTOCOL


def test_protocol_forbids_mixing_or_repeating_objects():
    assert "One object per turn" in TOOL_PROTOCOL
    assert "never two tool calls" in TOOL_PROTOCOL


def test_protocol_requires_a_citations_list_even_when_empty():
    """`[]` has to be legal, or the model invents a citation to satisfy the schema."""
    assert f"`{CITATIONS_KEY}` is required" in TOOL_PROTOCOL
    assert "Use `[]` when no claim came from" in TOOL_PROTOCOL


def test_every_json_object_in_the_prompt_parses(prompt):
    """An invalid example would be recorded later as the model's parse failure."""
    objects = json_objects(prompt)
    assert len(objects) >= 8
    for obj in objects:
        assert isinstance(obj, dict)


def test_every_prompt_object_is_a_tool_call_a_final_answer_or_a_tool_result(prompt):
    """No fourth shape is demonstrated, since the parser only accepts these."""
    allowed = [{TOOL_KEY, ARGS_KEY}, {FINAL_KEY, CITATIONS_KEY}, {TOOL_KEY, "result"}]
    for obj in json_objects(prompt):
        assert set(obj) in allowed, obj


def test_protocol_includes_two_worked_examples():
    assert TOOL_PROTOCOL.count("**Example ") == 2


def test_examples_show_both_a_kb_hit_and_a_web_fallback():
    assert f'"{TOOL_KEY}": "{KB_TOOL}"' in TOOL_PROTOCOL
    assert f'"{TOOL_KEY}": "{WEB_TOOL}"' in TOOL_PROTOCOL


def test_worked_example_citations_match_their_inline_markers(prompt):
    """The examples must demonstrate the contract, not merely state it."""
    finals = [obj for obj in worked_examples(prompt) if FINAL_KEY in obj]
    assert len(finals) == 2
    for final in finals:
        assert parse_citations(final[FINAL_KEY]) == final[CITATIONS_KEY]


def test_worked_example_only_cites_ids_it_was_shown(prompt):
    """Every cited id appears in a tool result earlier in the prompt: provenance, taught."""
    for final in [obj for obj in worked_examples(prompt) if FINAL_KEY in obj]:
        for chunk_id in final[CITATIONS_KEY]:
            before = prompt.split(json.dumps(final[FINAL_KEY], ensure_ascii=False))[0]
            assert f'"chunk_id": "{chunk_id}"' in before


def test_web_example_cites_nothing_and_names_its_source(prompt):
    """A web claim gets a URL, not a chunk id — the citations list is KB-only."""
    web_final = [obj for obj in worked_examples(prompt) if FINAL_KEY in obj][-1]
    assert web_final[CITATIONS_KEY] == []
    assert "https://" in web_final[FINAL_KEY]


def test_protocol_documents_the_format_the_harness_actually_sends():
    """Guards against the examples drifting from `render_tool_result`."""
    assert render_tool_result(KB_TOOL, []) in TOOL_PROTOCOL


def test_protocol_tolerates_a_fence_uniformly():
    """Whatever shape is permitted is permitted for both models, and it is stated."""
    assert "```json fence around the object is accepted" in flat(TOOL_PROTOCOL)


def test_protocol_mentions_the_step_budget():
    assert "step budget" in TOOL_PROTOCOL


# --------------------------------------------------------------------------------------
# Tool documentation is generated, not retyped
# --------------------------------------------------------------------------------------


def test_tool_docs_render_names_descriptions_and_arguments(specs):
    docs = render_tool_docs(specs)
    for spec in specs:
        assert f"### {spec['name']}" in docs
        assert spec["description"] in docs
        for argument in spec["schema"]["properties"]:
            assert f"`{argument}`" in docs


def test_tool_docs_mark_required_and_optional_arguments(specs):
    docs = render_tool_docs(specs)
    assert "`query` (string, required)" in docs
    assert "`top_k` (integer, optional)" in docs


def test_tool_docs_handle_a_tool_without_arguments():
    docs = render_tool_docs([{"name": "ping", "description": "Ping.", "schema": {}}])
    assert "- none" in docs


def test_prompt_documents_every_tool_and_no_others(prompt, specs):
    assert prompt.count("### ") == len(specs)


def test_build_refuses_an_inventory_missing_a_tool_the_prompt_names(specs):
    """Otherwise the model calls a tool that is not registered, and the hallucinated-tool
    metric blames the model for the harness's mistake."""
    with pytest.raises(ValueError, match=WEB_TOOL):
        build_system_prompt([spec for spec in specs if spec["name"] != WEB_TOOL])


def test_build_refuses_an_empty_inventory():
    with pytest.raises(ValueError):
        build_system_prompt([])


def test_required_tools_exist_as_real_tools():
    """The names in the prompt text are the names the modules export."""
    assert set(REQUIRED_TOOLS) == {module.name for module in TOOL_MODULES}


def test_citation_format_is_the_retrieval_modules_definition(prompt):
    """One definition, so the ask and the citation scoring cannot drift.

    The strong half is the round trip: the markers the prompt teaches are markers
    `parse_citations` recognises, which is the function that later scores them.
    """
    assert CITATION_FORMAT is lookup_kb.CITATION_FORMAT
    assert parse_citations(prompt) == ["sleep-hygiene.md#2", "hydration.md#4"]


def test_prompt_only_cites_files_that_exist_in_the_corpus(prompt):
    """A worked example naming a document we deleted teaches the model a fabricated id.

    Only the filename half is checked: the ordinal depends on which tokenizer chunked the
    corpus, and this suite deliberately runs a word counter rather than the model's.
    """
    corpus = {path.name for path in lookup_kb.corpus_files(REAL_KB)}
    assert corpus, "no corpus found; this test would pass vacuously"
    for chunk_id in parse_citations(prompt):
        assert chunk_id.split("#")[0] in corpus, f"{chunk_id} names no file in kb/"


# --------------------------------------------------------------------------------------
# Assembly and digest
# --------------------------------------------------------------------------------------


def test_prompt_is_role_then_tools_then_protocol(prompt):
    assert prompt.index("wellness assistant") < prompt.index("## Tools")
    assert prompt.index("## Tools") < prompt.index("## Tool protocol")


def test_digest_is_of_the_rendered_prompt(specs):
    assert system_prompt_sha256(specs) == sha256_text(build_system_prompt(specs))


def test_digest_covers_the_tool_inventory(specs):
    """PROJECT.md requires this: a tool added or re-documented changes the digest."""
    baseline = system_prompt_sha256(specs)

    reordered = system_prompt_sha256(list(reversed(specs)))
    assert reordered != baseline

    edited = [dict(spec) for spec in specs]
    edited[0]["description"] = "Something else entirely."
    assert system_prompt_sha256(edited) != baseline

    retyped = [dict(spec) for spec in specs]
    retyped[0] = {**retyped[0], "schema": {"type": "object", "properties": {}}}
    assert system_prompt_sha256(retyped) != baseline


def test_digest_changes_when_the_prompt_text_changes(specs, monkeypatch):
    baseline = system_prompt_sha256(specs)
    monkeypatch.setattr(prompts, "SYSTEM_PROMPT", SYSTEM_PROMPT + "\nOne more rule.\n")
    assert system_prompt_sha256(specs) != baseline


def test_prompt_stays_within_a_sane_size(prompt):
    """Every call pays for this prompt, so growth should be deliberate."""
    assert len(prompt) < 12_000, f"system prompt is {len(prompt)} chars"


# --------------------------------------------------------------------------------------
# Rendering results back to the model
# --------------------------------------------------------------------------------------


def hit(chunk_id: str = "sleep-hygiene.md#1", score: float = 0.5) -> Hit:
    chunk = Chunk(
        chunk_id=chunk_id,
        source_file=chunk_id.split("#")[0],
        heading_path=("Sleep Hygiene", "How much sleep adults need"),
        ordinal=int(chunk_id.split("#")[1]),
        text="Most adults need seven to nine hours.",
        token_count=8,
    )
    return Hit(chunk=chunk, score=score)


def test_tool_result_is_a_header_then_one_json_line_then_the_reminder():
    rendered = render_tool_result(KB_TOOL, [hit()])
    header, payload, reminder = rendered.split("\n")

    assert header == f"{TOOL_RESULT_PREFIX} ({KB_TOOL})"
    assert json.loads(payload)[TOOL_KEY] == KB_TOOL
    assert "exactly one JSON object" in reminder


def test_tool_result_leads_with_the_id_the_model_must_cite():
    payload = render_tool_result(KB_TOOL, [hit("sleep-hygiene.md#3")]).split("\n")[1]
    first_chunk = json.loads(payload)["result"][0]
    assert next(iter(first_chunk)) == "chunk_id"
    assert first_chunk["chunk_id"] == "sleep-hygiene.md#3"
    assert first_chunk["citation"] == CITATION_FORMAT.format(chunk_id="sleep-hygiene.md#3")


def test_tool_result_envelope_is_the_same_for_every_tool():
    """No per-tool special casing, so a new tool needs no prompt change."""
    kb = json.loads(render_tool_result(KB_TOOL, [hit()]).split("\n")[1])
    web = json.loads(render_tool_result(WEB_TOOL, {"cached": True, "results": []}).split("\n")[1])
    assert set(kb) == set(web) == {TOOL_KEY, "result"}


def test_tool_result_serialises_dataclasses_without_a_to_dict():
    result = [SearchResult(title="T", url="https://e.example", snippet="S")]
    payload = json.loads(render_tool_result(WEB_TOOL, result).split("\n")[1])
    assert payload["result"][0]["url"] == "https://e.example"


def test_tool_result_survives_an_unserialisable_value():
    """A tool returning something odd should degrade, not crash the turn."""
    payload = json.loads(render_tool_result(KB_TOOL, {"when": object()}).split("\n")[1])
    assert isinstance(payload["result"]["when"], str)


def test_empty_result_is_rendered_as_empty_not_omitted():
    """The model needs to see that the KB had nothing, which is example 2's whole point."""
    payload = json.loads(render_tool_result(KB_TOOL, []).split("\n")[1])
    assert payload["result"] == []


def test_tool_error_names_the_tool_and_restates_the_contract():
    rendered = render_tool_error(KB_TOOL, "unknown argument 'k'")
    assert rendered.startswith(f"{TOOL_ERROR_PREFIX} ({KB_TOOL})")
    assert "unknown argument 'k'" in rendered
    assert "exactly one JSON object" in rendered


def test_protocol_error_has_no_tool_name():
    rendered = render_tool_error(None, "expected one JSON object, found prose")
    assert rendered.startswith(PROTOCOL_ERROR_PREFIX)
    assert "exactly one JSON object" in rendered


def test_parse_failure_wording_does_not_depend_on_the_model():
    """The same failure is described identically however it arose."""
    assert render_tool_error(None, "bad json") == render_tool_error(None, "bad json")


# --------------------------------------------------------------------------------------
# Why a tool call was rejected
# --------------------------------------------------------------------------------------
#
# Exact strings, because the model has to fix its call from this sentence alone and both arms
# read the same one. Pinning the bytes is what makes a reworded error a deliberate change
# rather than a silent asymmetry.


def test_an_unknown_tool_error_names_the_invention_and_lists_the_real_tools():
    """A near-miss — wrong case, a plausible invention — is recoverable from the correct
    spelling and not otherwise."""
    assert (
        render_unknown_tool_error("lookup_KB", [KB_TOOL, WEB_TOOL])
        == "unknown tool 'lookup_KB'; valid tools: lookup_kb, search_web"
    )


def test_a_missing_argument_error_names_the_argument():
    assert (
        render_missing_argument_error(KB_TOOL, "query")
        == "lookup_kb: missing required argument 'query'"
    )


def test_a_bad_type_error_names_the_type_and_quotes_the_value_received():
    """`3` and `'3'` are the entire difference between a working call and a rejected one, and
    "got string" hides exactly that."""
    assert (
        render_bad_argument_type_error(KB_TOOL, "top_k", "integer", "3")
        == "lookup_kb: 'top_k' must be an integer, got '3'"
    )


def test_an_unknown_argument_error_lists_the_arguments_the_tool_takes():
    assert render_unknown_argument_error(KB_TOOL, "limit", ["query", "top_k"]) == (
        "lookup_kb: 'limit' is not an argument of this tool; valid arguments: query, top_k"
    )


@pytest.mark.parametrize(
    ("expected", "phrase"),
    [
        ("integer", "must be an integer"),
        ("array", "must be an array"),
        ("object", "must be an object"),
        ("string", "must be a string"),
        ("number", "must be a number"),
        ("boolean", "must be a boolean"),
    ],
)
def test_the_article_is_right_for_every_json_type_name(expected: str, phrase: str):
    """Not cosmetic: a model reads this as instructions, and "must be a integer" is the kind of
    sloppiness that makes a harness look unreliable to the person auditing it."""
    assert phrase in render_bad_argument_type_error(KB_TOOL, "arg", expected, 1)


def test_an_empty_inventory_is_said_rather_than_left_blank():
    """A trailing colon with nothing after it reads as a truncated message, and the model
    cannot tell whether the list was empty or the harness broke."""
    assert render_unknown_tool_error("x", []).endswith("valid tools: none")
    assert render_unknown_argument_error(KB_TOOL, "x", []).endswith("valid arguments: none")


@pytest.mark.parametrize(
    "render",
    [
        lambda: render_unknown_tool_error("lookup_KB", [KB_TOOL]),
        lambda: render_missing_argument_error(KB_TOOL, "query"),
        lambda: render_unknown_argument_error(KB_TOOL, "limit", ["query"]),
        lambda: render_bad_argument_type_error(KB_TOOL, "top_k", "integer", "3"),
    ],
)
def test_a_rejection_reads_the_same_every_time_it_is_rendered(render):
    """These are the only producers of these sentences, so identical inputs must give
    identical bytes — that is what makes the two arms' feedback comparable."""
    assert render() == render()


# --------------------------------------------------------------------------------------
# Budget exhaustion and summarisation
# --------------------------------------------------------------------------------------


def test_budget_nudge_is_labelled_and_asks_only_for_an_answer():
    """Offering a further tool call here would invite a call the loop has to refuse."""
    assert FINAL_ANSWER_REQUIRED.startswith(TOOL_BUDGET_PREFIX)
    assert FINAL_KEY in FINAL_ANSWER_REQUIRED
    assert CITATIONS_KEY in FINAL_ANSWER_REQUIRED
    assert "further tool call" not in FINAL_ANSWER_REQUIRED


def test_budget_nudge_does_not_licence_answering_from_memory():
    """The point of the cap is a bounded turn, not permission to start inventing."""
    assert "guess" in FINAL_ANSWER_REQUIRED
    assert "cite only ids that appeared" in FINAL_ANSWER_REQUIRED


@pytest.mark.parametrize("word", ["error", "mistake", "invalid", "wrong", "too many"])
def test_budget_nudge_does_not_say_which_budget_ran_out(word: str):
    """Sent when the successful-call budget is spent and when the error budget is, and the
    required reply is the same either way. Telling one model "you used your calls" and another
    "you made too many mistakes" would be different feedback at the same point in the loop —
    which is a harness difference, and the thing this project exists to avoid."""
    assert word not in FINAL_ANSWER_REQUIRED.lower()


def test_loop_messages_are_outside_the_system_prompt_digest(specs):
    """They are separate messages, so adding them must not invalidate existing manifests."""
    prompt = build_system_prompt(specs)
    assert FINAL_ANSWER_REQUIRED not in prompt
    assert render_summary_request([{"role": "user", "content": "q"}]) not in prompt


def test_summary_request_carries_the_transcript_it_is_summarising():
    request = render_summary_request(
        [
            {"role": "user", "content": "how much water"},
            {"role": "assistant", "content": "it depends"},
        ]
    )
    assert "how much water" in request
    assert "it depends" in request
    assert request.index("how much water") < request.index("it depends")


def test_summary_request_asks_for_prose_because_it_shares_the_agents_adapter():
    """A JSON demand would come back as a protocol object from a model primed for one."""
    request = render_summary_request([{"role": "user", "content": "how much water"}])
    assert "not JSON" in request
    assert json_objects(request) == []


def test_summary_request_forbids_adding_to_what_it_was_given():
    """A summariser that answers open questions launders invention into later turns."""
    request = render_summary_request([{"role": "user", "content": "q"}])
    assert "Do not add facts" in request


def test_summary_request_survives_a_message_missing_its_fields():
    """Compaction runs mid-turn; a malformed entry must not take the turn down with it."""
    assert "unknown" in render_summary_request([{}])


def test_summary_is_labelled_so_it_is_not_read_as_the_user_speaking():
    assert render_summary("  they asked about sleep  ").startswith(SUMMARY_PREFIX)
    assert "they asked about sleep" in render_summary("  they asked about sleep  ")


def test_summarisation_text_is_deterministic():
    messages = [{"role": "user", "content": "q"}]
    assert render_summary_request(messages) == render_summary_request(list(messages))


# --------------------------------------------------------------------------------------
# Judge rubric
# --------------------------------------------------------------------------------------


# Every assertion here runs against each rubric on disk, not just the default one: with one
# file per axis, a rule enforced on `default.md` alone is a rule three files can quietly drop.
# The loader's own refusals are covered in tests/test_judge.py.
RUBRICS = pytest.mark.parametrize("rubric_name", judge_rubric_names())


@RUBRICS
def test_rubric_scores_arbitrary_pairs_without_assuming_our_harness(rubric_name):
    """A grader's file has no chunk ids and no tool protocol; penalising their absence
    would score conformity to our harness instead of answer quality."""
    rubric = judge_rubric_prompt(rubric_name).lower()
    for ours in ("chunk_id", "chunk id", KB_TOOL, WEB_TOOL, "knowledge base", "[[", "trace"):
        assert ours.lower() not in rubric


@RUBRICS
def test_rubric_hides_which_system_produced_the_response(rubric_name):
    """Half the self-preference defence; a third model family is the other half."""
    assert "not told which system produced it" in judge_rubric_prompt(rubric_name)


@RUBRICS
def test_rubric_asks_for_every_dimension_plus_a_rationale(rubric_name):
    """`evidence` is part of the schema because the spans are verified against the response.

    Asserted over every JSON object in the rubric, which is the schema block plus the anchors'
    verdicts: an anchor demonstrating a different shape from the one being asked for is how a
    parse-failure rate gets manufactured.
    """
    rubric = judge_rubric_prompt(rubric_name)
    expected = {*JUDGE_DIMENSIONS, "rationale", "overall", JUDGE_EVIDENCE_KEY}
    objects = json_objects(rubric)
    assert objects
    for obj in objects:
        assert set(obj) == expected

    for dimension in JUDGE_DIMENSIONS:
        assert f"- {dimension}:" in rubric


@RUBRICS
def test_rubric_asks_for_the_rationale_before_the_scores(rubric_name):
    """Scoring after reasoning is more reliable than reasoning after a committed number."""
    rubric = judge_rubric_prompt(rubric_name)
    assert "rationale before the scores" in flat(rubric)
    schema = json.dumps(json_objects(rubric)[0])
    assert schema.index("rationale") < schema.index(JUDGE_DIMENSIONS[0])


@RUBRICS
def test_rubric_treats_over_refusal_as_a_failure(rubric_name):
    """Otherwise the highest-scoring safe response is one that declines everything."""
    assert "unnecessary refusal is a failure" in flat(judge_rubric_prompt(rubric_name))


@RUBRICS
def test_rubric_penalises_invented_specifics_above_vagueness(rubric_name):
    assert "Penalise invented specifics" in flat(judge_rubric_prompt(rubric_name))


@RUBRICS
def test_rubric_json_example_parses(rubric_name):
    assert json_objects(judge_rubric_prompt(rubric_name))


@RUBRICS
def test_rubric_states_the_scale_from_the_constant(rubric_name):
    """A rubric asking for 1-5 while the code expects another range would collect scores the
    parser then rejects, so the bound is interpolated rather than typed into the file."""
    assert f"from 1 to {JUDGE_SCALE_MAX}" in judge_rubric_prompt(rubric_name)


@RUBRICS
def test_rubric_frames_a_reference_as_one_acceptable_answer(rubric_name):
    """The slot external ground truth arrives in; treated as the only answer, it penalises
    a correct response for being worded differently."""
    rubric = flat(judge_rubric_prompt(rubric_name))
    assert "one acceptable answer rather than the only one" in rubric


def test_judge_pair_includes_a_reference_only_when_given():
    with_reference = render_judge_pair("Q", "A", reference="R")
    assert "### Reference answer" in with_reference

    for blank in (None, "", "   "):
        assert "### Reference" not in render_judge_pair("Q", "A", reference=blank)


def test_judge_pair_keeps_prompt_and_response_separate():
    rendered = render_judge_pair("The question", "The answer")
    assert rendered.index("### Prompt") < rendered.index("### Response")
    assert "The question" in rendered
    assert "The answer" in rendered


# --------------------------------------------------------------------------------------
# block_order: byte-identical by default, a permutation or nothing
# --------------------------------------------------------------------------------------

#: What `render_judge_pair` produced before `block_order` existed, character for character.
#: Frozen here rather than derived, because a golden string computed from the same constants the
#: renderer reads would agree with any change either of them made.
GOLDEN_PAIR = (
    "### Prompt\n"
    "The question\n"
    "\n"
    "### Response\n"
    "The answer\n"
    "\n"
    "### Reference answer (one acceptable answer, not the only one)\n"
    "One acceptable answer"
)


def test_the_default_rendering_is_byte_identical_to_the_frozen_golden_string():
    """`block_order`'s default must not move the output by a single character. A drift of one
    newline would re-key every `ResponseCache` entry and orphan every judgement already recorded
    against the old text, while every digest in the manifest stayed the same."""
    assert (
        render_judge_pair("The question", "The answer", "One acceptable answer") == GOLDEN_PAIR
    )


def test_passing_the_canonical_order_explicitly_changes_nothing():
    assert (
        render_judge_pair(
            "The question",
            "The answer",
            "One acceptable answer",
            block_order=prompts.CANONICAL_BLOCK_ORDER,
        )
        == GOLDEN_PAIR
    )


def test_a_reordering_moves_the_blocks_and_nothing_else():
    reordered = render_judge_pair(
        "The question",
        "The answer",
        "One acceptable answer",
        block_order=("response", "prompt", "reference"),
    )

    assert reordered.index("### Response") < reordered.index("### Prompt")
    assert reordered != GOLDEN_PAIR
    # Same blocks, same bodies: only their order differs, which is what makes a measured drift a
    # drift from ordering rather than from a changed message.
    assert sorted(reordered.split("\n\n")) == sorted(GOLDEN_PAIR.split("\n\n"))


def test_a_reordering_omits_an_absent_reference_as_the_default_does():
    reordered = render_judge_pair(
        "Q", "A", block_order=("reference", "response", "prompt")
    )

    assert "### Reference" not in reordered
    assert reordered.index("### Response") < reordered.index("### Prompt")


@pytest.mark.parametrize(
    "order",
    [
        ("prompt", "response"),
        ("prompt", "response", "reference", "rationale"),
        ("prompt", "prompt", "response"),
        ("prompt", "response", "Reference"),
        (),
    ],
)
def test_a_block_order_that_is_not_a_permutation_is_refused(order):
    """A dropped block or an ignored unknown one would change what the judge reads, and the drift
    measured against it would not be a drift from reordering."""
    with pytest.raises(ValueError, match="not a permutation"):
        render_judge_pair("Q", "A", "R", block_order=order)


def test_the_pair_template_digest_is_stable_and_sensitive(monkeypatch):
    """Covers what `judge_rubric_sha256` deliberately does not: rewording a block heading changes
    every judge message while leaving every rubric identical."""
    baseline = prompts.judge_pair_template_sha256()
    assert prompts.judge_pair_template_sha256() == baseline

    reworded = {**prompts.JUDGE_PAIR_HEADINGS, "response": "### Candidate response"}
    monkeypatch.setattr(prompts, "JUDGE_PAIR_HEADINGS", MappingProxyType(reworded))

    assert prompts.judge_pair_template_sha256() != baseline


def test_the_pair_template_digest_does_not_cover_the_pair_itself():
    """The prompt and the response are the data being scored. A digest over them would be a
    per-pair value, and a manifest records one number for a whole run."""
    first = prompts.judge_pair_template_sha256()
    render_judge_pair("a different question", "a different answer")

    assert prompts.judge_pair_template_sha256() == first


def test_the_rubric_digest_does_not_move_when_the_pair_template_does(monkeypatch):
    """Kept separate on purpose. Folding this into `judge_rubric_sha256` would move it for every
    judge manifest already written, orphaning runs whose rubrics never changed."""
    baseline = judge_rubric_sha256()

    reworded = {**prompts.JUDGE_PAIR_HEADINGS, "prompt": "### The prompt"}
    monkeypatch.setattr(prompts, "JUDGE_PAIR_HEADINGS", MappingProxyType(reworded))

    assert judge_rubric_sha256() == baseline


def test_rubric_digest_is_stable_and_sensitive(monkeypatch):
    baseline = judge_rubric_sha256()
    assert judge_rubric_sha256() == baseline
    monkeypatch.setattr(prompts, "JUDGE_SCALE_MAX", 7)
    assert judge_rubric_sha256() != baseline
