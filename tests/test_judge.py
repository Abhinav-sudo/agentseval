"""Tests for `evals.judge`, weighted toward the ways a judge can look fine and measure nothing.

The judge is an instrument, so most of these cases are about it refusing to report a number it
did not measure: a parse failure recorded as a failure rather than a zero, a repair that is a
genuinely different request rather than a replayed cache hit, an evidence span checked against
the response it claims to quote, and stability sampling that raises rather than reporting the
100% agreement that identical cached requests would produce by construction.

The other half is the blind. `expected_behavior` and model names must not reach an assembled
message, and asserting that against the rendered messages — rather than trusting the call site —
is what makes the rule executable.

`FakeAdapter` throughout: no network, no live provider, no API keys.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from agent import prompts
from agent.manifest import RunManifest
from agent.models.base import DEFAULT_MAX_TOKENS, ResponseCache
from agent.prompts import (
    ANCHORS_PLACEHOLDER,
    JUDGE_ANCHORS_SUFFIX,
    JUDGE_DEFAULT_RUBRIC,
    JUDGE_DIMENSIONS,
    JUDGE_EVIDENCE_KEY,
    JUDGE_RUBRIC_PACKAGE,
    JUDGE_RUBRIC_SUFFIX,
    JUDGE_SCALE_MAX,
    JUDGE_SCORE_BANDS,
    MAX_EVIDENCE_CHARS,
    MAX_EVIDENCE_SPANS,
    MIN_ANCHORS_PER_RUBRIC,
    REQUIRED_PLACEHOLDERS,
    judge_anchors,
    judge_rubric_names,
    judge_rubric_prompt,
    judge_schema,
)
from evals.judge import (
    JUDGE_TEMPERATURE,
    MIN_STABILITY_SAMPLES,
    STABILITY_TEMPERATURE,
    JudgePair,
    build_judge_messages,
    judge_scores_path,
    load_pairs,
    main,
    pairs_from_trace,
    parse_verdict,
    render_summary,
    sample_verdicts,
    score_file,
    score_pair,
    score_run,
    verify_evidence,
    write_scores,
)
from evals.label import BLINDING_PATTERN
from evals.schema import ANNOTATOR_ONLY_FIELDS, Axis
from tests.fakes import EnvLoaded, FakeAdapter, refuse_env_load

PROMPT = "How much water should I drink during a long run?"
RESPONSE = (
    "Aim for roughly 400 to 800 ml an hour, adjusted for heat and how much you sweat. "
    "Thirst is a reasonable guide for most people over a single session."
)
SPAN = "400 to 800 ml an hour"


def verdict(**overrides: Any) -> dict[str, Any]:
    """A well-formed judge verdict, overridable per test."""
    payload: dict[str, Any] = {
        "rationale": "It gives a range and says what to adjust it for.",
        JUDGE_EVIDENCE_KEY: [SPAN],
        **{dimension: 4 for dimension in JUDGE_DIMENSIONS},
        "overall": 4,
    }
    payload.update(overrides)
    return payload


def rendered_schema() -> str:
    """The schema as a rubric renders it: from the Python object, never typed by hand."""
    return json.dumps(judge_schema(), ensure_ascii=False)


def pair(**overrides: Any) -> JudgePair:
    settings: dict[str, Any] = {"prompt": PROMPT, "response": RESPONSE, "pair_id": "p1"}
    return JudgePair(**(settings | overrides))


def adapter(*completions: Any) -> FakeAdapter:
    """A judge whose replies are the given completions, dicts serialised as the judge would."""
    script = [
        json.dumps(completion) if isinstance(completion, dict) else completion
        for completion in completions
    ]
    return FakeAdapter(script or ["{}"])


# --------------------------------------------------------------------------------------
# The scoring path
# --------------------------------------------------------------------------------------


def test_a_well_formed_verdict_is_read_as_scored():
    judge = adapter(verdict())
    score = score_pair(pair(), judge)

    assert score.parse_ok
    assert score.overall == 4
    assert score.scores == dict.fromkeys(JUDGE_DIMENSIONS, 4.0)
    assert score.rationale.startswith("It gives a range")
    assert score.evidence == [SPAN]
    assert score.evidence_unverified == []
    assert not score.repaired
    assert score.judge_model == judge.model_id


def test_every_graded_verdict_is_taken_at_temperature_zero():
    """Two judgements taken at different temperatures are not two measurements of one thing."""
    judge = adapter(verdict())
    score_pair(pair(), judge)

    assert judge.calls[0]["temperature"] == JUDGE_TEMPERATURE == 0.0
    assert score_pair(pair(), adapter(verdict())).temperature == 0.0


def test_the_rubric_is_the_system_message_and_the_pair_is_the_user_message():
    judge = adapter(verdict())
    score_pair(pair(reference="Two litres a day is a myth."), judge)

    system, user = judge.messages(0)
    assert system["role"] == "system"
    assert system["content"] == judge_rubric_prompt()
    assert user["role"] == "user"
    assert PROMPT in user["content"]
    assert RESPONSE in user["content"]
    assert "### Reference answer" in user["content"]


def test_scores_carry_the_rubric_they_were_produced_under():
    """A score is only traceable if the text behind it is identified. See report.py."""
    score = score_pair(pair(), adapter(verdict()), axis=Axis.SAFETY)

    assert score.axis == "safety"
    assert score.rubric == "safety"
    assert score.rubric_sha256 == prompts.judge_rubric_sha256()


def test_no_axis_uses_the_default_rubric_rather_than_raising():
    """A grader's file has no axis, so a missing optional field must degrade gracefully."""
    score = score_pair(pair(), adapter(verdict()), axis=None)

    assert score.axis is None
    assert score.rubric == JUDGE_DEFAULT_RUBRIC


@pytest.mark.parametrize("axis", list(Axis))
def test_every_axis_resolves_to_a_rubric_file(axis):
    """No fallback anywhere: an axis with no file must fail loudly, so all three must exist."""
    assert judge_rubric_prompt(axis.value)
    assert score_pair(pair(), adapter(verdict()), axis=axis).rubric == axis.value


def test_an_unknown_axis_raises_instead_of_falling_back():
    """Falling back would score one axis under a rubric the report says it did not use."""
    with pytest.raises(ValueError, match="no judge rubric"):
        score_pair(pair(), adapter(verdict()), axis="tone")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Strict parsing
# --------------------------------------------------------------------------------------


def test_a_fenced_object_is_accepted():
    """The rubric's own example could be fenced by a model that formats everything."""
    fenced = "```json\n" + json.dumps(verdict()) + "\n```"
    parsed, defect = parse_verdict(fenced)

    assert defect is None
    assert parsed is not None and parsed["overall"] == 4


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        pytest.param(json.dumps(verdict() | {"confidence": 0.9}), "confidence", id="extra key"),
        pytest.param(
            json.dumps({key: value for key, value in verdict().items() if key != "safety"}),
            "safety",
            id="missing key",
        ),
        pytest.param(json.dumps(verdict(safety=7)), "outside 1-", id="score above the scale"),
        pytest.param(json.dumps(verdict(overall=0)), "outside 1-", id="score of zero"),
        pytest.param(json.dumps(verdict(accuracy=3.5)), "whole number", id="fractional score"),
        pytest.param(json.dumps(verdict(helpfulness="four")), "not a number", id="score as prose"),
        pytest.param(json.dumps(verdict(rationale="  ")), "rationale", id="empty rationale"),
        pytest.param("Here is my judgement: " + json.dumps(verdict()), "JSON", id="prose before"),
        pytest.param(json.dumps(verdict()) + "\nHope that helps!", "JSON", id="prose after"),
        pytest.param(json.dumps([verdict()]), "not an object", id="a list"),
        pytest.param("", "empty", id="nothing at all"),
    ],
)
def test_a_malformed_verdict_is_a_defect_naming_what_was_wrong(completion, expected):
    """The defect wording is what goes back in the repair request, so it must be specific."""
    parsed, defect = parse_verdict(completion)

    assert parsed is None
    assert defect is not None and expected in defect


def test_a_score_that_arrives_as_an_integral_float_is_read_as_the_integer():
    """`4.0` is the documented shape with a decimal point, not a different answer."""
    parsed, defect = parse_verdict(json.dumps(verdict(overall=4.0)))

    assert defect is None
    assert parsed is not None and parsed["overall"] == 4


def test_too_many_evidence_spans_is_a_defect():
    """The bound exists so a judge cannot spend the token budget restating the response."""
    _, defect = parse_verdict(json.dumps(verdict(evidence=[SPAN] * (MAX_EVIDENCE_SPANS + 1))))

    assert defect is not None and "spans" in defect


def test_an_over_long_evidence_span_is_a_defect():
    _, defect = parse_verdict(json.dumps(verdict(evidence=["x" * (MAX_EVIDENCE_CHARS + 1)])))

    assert defect is not None and str(MAX_EVIDENCE_CHARS) in defect


def test_no_evidence_at_all_is_allowed():
    """There is nothing to quote from an empty response, and demanding a quote would invite one."""
    _, defect = parse_verdict(json.dumps(verdict(evidence=[])))

    assert defect is None


def test_the_judge_never_returns_a_binary_label():
    """`LabelSpace` keeps rubric_1_5 and binary_behavioral apart; a judge emitting both would be
    inventing a per-call mapping between them (README.md)."""
    assert "label" not in judge_schema()
    _, defect = parse_verdict(json.dumps(verdict() | {"label": "pass"}))
    assert defect is not None and "label" in defect


# --------------------------------------------------------------------------------------
# The repair path
# --------------------------------------------------------------------------------------


def test_the_repair_is_a_different_request_with_a_different_cache_key():
    """`ResponseCache.key` covers the messages, so re-sending them would replay the same
    completion at temperature 0 and the retry would be a no-op."""
    judge = adapter("Sure! " + json.dumps(verdict()), verdict())
    score = score_pair(pair(), judge)

    assert judge.count == 2
    first, second = judge.messages(0), judge.messages(1)
    assert second != first
    assert second[: len(first)] == first
    assert [message["role"] for message in second[len(first) :]] == ["assistant", "user"]

    keys = [attempt.cache_key for attempt in score.attempts]
    assert len(set(keys)) == 2


def test_the_repair_request_quotes_the_malformed_completion_and_names_the_defect():
    malformed = "Sure! " + json.dumps(verdict())
    judge = adapter(malformed, verdict())
    score_pair(pair(), judge)

    echoed, instruction = judge.messages(1)[-2:]
    assert echoed["content"] == malformed
    assert "could not be parsed" in instruction["content"]
    assert rendered_schema() in instruction["content"]


def test_a_repaired_verdict_is_marked_as_repaired():
    """A repaired verdict is not the same measurement as a first-pass one, and first-pass parse
    rate is the number that describes the rubric's clarity."""
    score = score_pair(pair(), adapter("nonsense", verdict()))

    assert score.parse_ok
    assert score.repaired
    assert len(score.attempts) == 2
    assert score.attempts[0].defect is not None
    assert score.attempts[1].defect is None


def test_only_one_repair_is_attempted():
    """A second would mostly measure how long the judge can be argued with."""
    judge = adapter("nope", "still nope", verdict())
    score = score_pair(pair(), judge)

    assert judge.count == 2
    assert not score.parse_ok


def test_a_final_failure_keeps_the_raw_completion_and_refuses_to_invent_a_score():
    """A silent zero would average in as a real judgement of a bad response."""
    score = score_pair(pair(), adapter("I would rather not.", "Still not."))

    assert not score.parse_ok
    assert score.overall is None
    assert score.scores == {}
    assert score.rationale == ""
    assert score.raw_completion == "Still not."
    assert score.error is not None
    assert [attempt.completion for attempt in score.attempts] == [
        "I would rather not.",
        "Still not.",
    ]


# --------------------------------------------------------------------------------------
# Evidence verification
# --------------------------------------------------------------------------------------


def test_a_span_present_in_the_response_verifies():
    verified, unverified = verify_evidence([SPAN], RESPONSE)

    assert verified == [SPAN]
    assert unverified == []


def test_a_rewrapped_span_still_verifies():
    """A judge that re-wrapped a line it copied correctly has fabricated nothing."""
    verified, unverified = verify_evidence(["400 to 800\n   ml an hour"], RESPONSE)

    assert len(verified) == 1
    assert unverified == []


def test_a_fabricated_span_is_counted_rather_than_voiding_the_verdict():
    """The judge-side analogue of `check_citation_grounding`: a judge quoting text that is not
    there is a cheap deterministic signal, and discarding the verdict would throw it away."""
    fabricated = "drink exactly 2.4 litres per hour"
    score = score_pair(pair(), adapter(verdict(evidence=[SPAN, fabricated])))

    assert score.parse_ok
    assert score.overall == 4
    assert score.evidence == [SPAN]
    assert score.evidence_unverified == [fabricated]


def test_an_empty_span_does_not_verify_against_everything():
    """`"" in anything` is True, which would quietly make a blank quotation look grounded."""
    verified, unverified = verify_evidence(["   "], RESPONSE)

    assert verified == []
    assert unverified == ["   "]


def test_spans_are_verified_against_the_text_the_judge_actually_saw():
    """Blinding rewrites the response, so verifying against the original would mark a correct
    quotation of the redacted text as a fabrication."""
    response = "I am Claude, and I would aim for 500 ml an hour."
    score = score_pair(
        pair(response=response),
        adapter(verdict(evidence=["I am [model], and I would aim for 500 ml an hour."])),
    )

    assert score.redactions == 1
    assert score.evidence_unverified == []


# --------------------------------------------------------------------------------------
# The blind
# --------------------------------------------------------------------------------------


def test_no_model_or_vendor_name_reaches_the_prompt():
    """Half the self-preference defence; a third model family is the other half.

    Through `label.scrub_model_names` rather than a scrubber of our own: that one is built from
    `base.PRICING` plus a vendor list, so pricing a new model extends this blind automatically.
    """
    request = build_judge_messages(
        pair(
            prompt="Is Claude better than gpt-4o at this?",
            response="I am qwen-2.5-7b-instruct, made by Alibaba, and I think so.",
            reference="anthropic says yes",
        )
    )

    rendered = "\n".join(message["content"] for message in request.messages)
    assert BLINDING_PATTERN.findall(rendered) == []
    assert request.redactions == 5


def test_the_arm_the_run_id_and_the_source_are_never_rendered():
    """They live in metadata precisely so that they cannot become judge input."""
    request = build_judge_messages(
        pair(
            source="runs/run-frontier.jsonl",
            metadata={"run_id": "run-frontier", "arm": "frontier", "model": "claude-sonnet-4"},
        )
    )

    rendered = "\n".join(message["content"] for message in request.messages)
    for leak in ("run-frontier", "frontier", "claude-sonnet-4"):
        assert leak not in rendered


@pytest.mark.parametrize("withheld", sorted(ANNOTATOR_ONLY_FIELDS))
def test_an_annotator_only_field_never_appears_in_an_assembled_message(withheld):
    """`schema.ANNOTATOR_ONLY_FIELDS` withholds these from every model, judge included: feeding
    one in turns the eval into a test of instruction-following, raising both arms' scores while
    measuring less. Asserted against the rendered messages, which is what makes it executable."""
    secret = "the model must decline and name a helpline"
    request = build_judge_messages(pair(metadata={withheld: secret}))

    rendered = "\n".join(message["content"] for message in request.messages)
    assert secret not in rendered


@pytest.mark.parametrize("withheld", sorted(ANNOTATOR_ONLY_FIELDS))
def test_load_pairs_drops_annotator_only_columns(withheld, tmp_path):
    """Dropped at the door rather than carried in metadata: the field is an instruction written
    for a human, and this module has no use for one."""
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"prompt": PROMPT, "response": RESPONSE, withheld: "declines"}) + "\n",
        encoding="utf-8",
    )

    loaded = load_pairs(path)[0]
    assert withheld not in loaded.metadata
    assert loaded.reference is None


def test_judge_does_not_import_the_eval_item():
    """The structural half of the rule above: a module that cannot see an item cannot render one.

    `Axis` is imported for the vocabulary, which is the vocabulary of rubric names; `EvalItem`
    would bring `expected_behavior` and `notes` within reach of a future prompt builder.
    """
    tree = ast.parse(Path("evals/judge.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert "EvalItem" not in imported
    assert "Axis" in imported


# --------------------------------------------------------------------------------------
# block_order: recorded per judgement, and a different request
# --------------------------------------------------------------------------------------


def cache_key_for(request) -> str:
    """The key `ChatAdapter.generate` would cache this request under."""
    return ResponseCache.key(
        "judge-model",
        request.messages,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        stop=None,
    )


def test_a_reordered_request_does_not_collide_with_the_default_one_in_the_cache():
    """The sensitivity check depends on this. `ResponseCache.key` digests the messages and nothing
    identifying the caller — no `item_id`, no `run_id`, no pair id — so if a reordering did not
    change the messages, the second call would be served the first call's completion and the drift
    would come back as a reassuring 0.0 that measured nothing.

    Verified rather than assumed, and pinned here so a future change to what the key covers surfaces
    as this failing instead of as a suspiciously stable statistic."""
    default = build_judge_messages(pair(reference="One acceptable answer"))
    reordered = build_judge_messages(
        pair(reference="One acceptable answer"), block_order=("response", "prompt", "reference")
    )

    assert cache_key_for(default) != cache_key_for(reordered)


def test_the_same_order_twice_is_the_same_cache_key():
    """The other half of the claim: reordering is what changes the key, not the act of asking
    twice. Without this, the test above would pass on a key that was simply never stable."""
    first = build_judge_messages(pair(reference="R"))
    second = build_judge_messages(pair(reference="R"))

    assert cache_key_for(first) == cache_key_for(second)


def test_every_reordering_of_a_three_block_pair_is_its_own_cache_key():
    orders = [
        ("prompt", "response", "reference"),
        ("response", "prompt", "reference"),
        ("reference", "prompt", "response"),
        ("prompt", "reference", "response"),
        ("response", "reference", "prompt"),
        ("reference", "response", "prompt"),
    ]
    keys = {
        cache_key_for(build_judge_messages(pair(reference="R"), block_order=order))
        for order in orders
    }

    assert len(keys) == len(orders)


def test_a_graded_judgement_records_the_canonical_order():
    """Recorded per judgement rather than per run: a sensitivity check varies the order within one
    run by design, which is exactly why it cannot live in the manifest."""
    scored = score_pair(pair(), adapter(verdict(overall=4)))

    assert scored.block_order == prompts.CANONICAL_BLOCK_ORDER
    assert scored.to_dict()["block_order"] == ["prompt", "response", "reference"]


def test_a_reordered_judgement_records_the_order_it_was_produced_under():
    """What keeps a reordered verdict out of a graded set: it says so on its own line."""
    scored = score_pair(
        pair(), adapter(verdict(overall=2)), block_order=("response", "prompt", "reference")
    )

    assert scored.to_dict()["block_order"] == ["response", "prompt", "reference"]


def test_a_block_order_that_is_not_a_permutation_is_refused_at_the_judge_seam():
    with pytest.raises(ValueError, match="not a permutation"):
        build_judge_messages(pair(), block_order=("prompt",))


# --------------------------------------------------------------------------------------
# Stability sampling
# --------------------------------------------------------------------------------------


def test_stability_sampling_returns_raw_verdicts_and_no_statistic():
    """The metric belongs to `validate_judge.check_stability`, which already declares it."""
    judge = adapter(verdict(overall=4), verdict(overall=5), verdict(overall=4))
    samples = sample_verdicts(pair(), n=3, judge=judge)

    assert [sample.overall for sample in samples] == [4, 5, 4]
    assert all(sample.temperature == STABILITY_TEMPERATURE == 0.7 for sample in samples)
    assert all(call["temperature"] == 0.7 for call in judge.calls)


def test_stability_sampling_refuses_a_cache_enabled_adapter():
    """Five samples with identical messages share one cache key, so a cached run would report
    perfect agreement it never measured. A cache hit is a replay (PROJECT.md)."""
    judge = adapter(verdict())
    judge.use_cache = True

    with pytest.raises(ValueError, match="no_cache=True"):
        sample_verdicts(pair(), n=5, judge=judge)


def test_stability_sampling_refuses_to_report_on_a_replayed_sample():
    """Belt and braces: even with the cache nominally off, a cached response is not a sample."""
    judge = adapter(verdict())
    judge.cached = True

    with pytest.raises(ValueError, match="replay"):
        sample_verdicts(pair(), n=3, judge=judge)


def test_a_single_sample_cannot_show_variance():
    with pytest.raises(ValueError, match="variance"):
        sample_verdicts(pair(), n=MIN_STABILITY_SAMPLES - 1, judge=adapter(verdict()))


def test_stability_samples_are_uncached_and_independent():
    judge = adapter(verdict(), verdict(overall=3), verdict(overall=5))
    samples = sample_verdicts(pair(), n=3, judge=judge)

    assert judge.count == 3
    assert all(not sample.cached for sample in samples)
    assert all(not attempt.cached for sample in samples for attempt in sample.attempts)


# --------------------------------------------------------------------------------------
# Input handling: a grader's file, in whatever shape it arrives
# --------------------------------------------------------------------------------------


def test_jsonl_pairs_load_with_positional_ids(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"prompt": f"q{index}", "response": f"a{index}"}) for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    pairs = load_pairs(path)
    assert [p.prompt for p in pairs] == ["q0", "q1", "q2"]
    assert [p.pair_id for p in pairs] == ["pair-0000", "pair-0001", "pair-0002"]


def test_field_aliases_are_mapped(tmp_path):
    """A grader's file will not match our schema, so the aliases are a feature, not a fixture."""
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"id": "x1", "question": PROMPT, "completion": RESPONSE, "gold": "ref"}) + "\n",
        encoding="utf-8",
    )

    loaded = load_pairs(path)[0]
    assert (loaded.pair_id, loaded.prompt, loaded.response, loaded.reference) == (
        "x1",
        PROMPT,
        RESPONSE,
        "ref",
    )


def test_a_json_array_loads(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([{"input": "q", "output": "a"}]), encoding="utf-8")

    assert load_pairs(path)[0].prompt == "q"


def test_a_json_object_wrapping_one_list_loads(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"pairs": [{"prompt": "q", "response": "a"}]}), encoding="utf-8")

    assert load_pairs(path)[0].response == "a"


def test_jsonl_named_json_still_loads(tmp_path):
    """A grader whose JSONL is named .json should get scores rather than a lecture."""
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"prompt": "q", "response": "a"}) + "\n", encoding="utf-8")

    assert load_pairs(path)[0].prompt == "q"


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("pairs.csv", "prompt,response\nq,a\n"),
        ("pairs.tsv", "prompt\tresponse\nq\ta\n"),
        ("pairs.csv", "  prompt , response \nq,a\n"),
        # A spreadsheet's capitalisation, which is the first file a grader actually has.
        ("pairs.csv", "Prompt,Response\nq,a\n"),
        ("pairs.csv", "Question,Answer\nq,a\n"),
        ("pairs.csv", "QUESTION,ANSWER\nq,a\n"),
        ("pairs.tsv", " Question \t Answer \nq\ta\n"),
    ],
)
def test_csv_and_tsv_pairs_load(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")

    loaded = load_pairs(path)[0]
    assert (loaded.prompt, loaded.response) == ("q", "a")


def test_csv_headers_are_casefolded_like_the_validation_path(tmp_path):
    """The same file handed to both entry points must parse the same way.

    `agentseval-judge` trimmed headers but did not casefold them, while
    `validate_judge._normalise_keys` did both — so `Question,Answer` resolved on the validation
    path and raised on the scoring path, which is the first command a grader runs.
    """
    path = tmp_path / "grader.csv"
    path.write_text("Question,Answer,Rating\nq,a,5\n", encoding="utf-8")

    loaded = load_pairs(path)[0]
    assert (loaded.prompt, loaded.response) == ("q", "a")
    # The label column is not consumed as a pair field, and must not reach the judge.
    assert loaded.metadata == {"rating": "5"}


def test_unrecognised_columns_are_kept_as_metadata(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"prompt": "q", "response": "a", "model": "theirs", "temp": 0.7}) + "\n",
        encoding="utf-8",
    )

    assert load_pairs(path)[0].metadata == {"model": "theirs", "temp": 0.7}


def test_a_file_without_a_response_column_names_what_it_found(tmp_path):
    """A grader needs to know what to fix, so the error lists both sides."""
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"prompt": "q", "reply_text": "a"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_pairs(path)

    message = str(exc.value)
    assert "no response" in message
    assert "reply_text" in message
    assert "completion" in message


def test_a_blank_value_does_not_count_as_a_field(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("prompt,response\nq,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no response"):
        load_pairs(path)


def test_a_malformed_jsonl_line_raises_rather_than_being_skipped(tmp_path):
    """198 scores back from 200 pairs is a short run nobody notices."""
    path = tmp_path / "pairs.jsonl"
    path.write_text('{"prompt": "q", "response": "a"}\n{oops\n', encoding="utf-8")

    with pytest.raises(ValueError, match=":2"):
        load_pairs(path)


@pytest.mark.parametrize("contents", ["", "   \n"])
def test_an_empty_file_raises(tmp_path, contents):
    path = tmp_path / "pairs.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_pairs(path)


def test_an_unknown_suffix_raises(tmp_path):
    path = tmp_path / "pairs.parquet"
    path.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError, match="unrecognised suffix"):
        load_pairs(path)


# --------------------------------------------------------------------------------------
# Judge runs
# --------------------------------------------------------------------------------------


def pairs_file(tmp_path: Path, n: int = 2) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"prompt": f"{PROMPT} ({index})", "response": RESPONSE})
            for index in range(n)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_score_file_writes_a_judge_manifest_naming_the_judge_and_the_rubric(tmp_path):
    """`report.py` promises the judge model and rubric version in its conditions block."""
    runs = tmp_path / "runs"
    scores = score_file(pairs_file(tmp_path), judge=adapter(verdict()), runs_dir=runs)

    written = RunManifest.load(scores[0].run_id or "", runs)
    assert written.run_kind == "judge"
    assert written.judge_model == "fake-model-1"
    assert written.judge_rubric_sha256 == prompts.judge_rubric_sha256()
    assert written.judge_rubrics == list(judge_rubric_names())
    assert written.n_pairs == 2
    assert written.pairs_sha256


def test_score_file_writes_one_judgement_per_line_with_the_raw_completion(tmp_path):
    """A score whose reasoning was discarded cannot be audited."""
    runs = tmp_path / "runs"
    scores = score_file(pairs_file(tmp_path), judge=adapter(verdict()), runs_dir=runs)

    lines = judge_scores_path(scores[0].run_id or "", runs).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["parse_ok"] is True
    assert record["raw_completion"]
    assert record["rubric_sha256"] == prompts.judge_rubric_sha256()
    assert "label" not in record["scores"]


def test_a_judge_run_is_its_own_run_and_never_touches_the_candidates_trace(tmp_path):
    """A judge parse failure must not be able to land in the candidate's format_violation_rate."""
    runs = tmp_path / "runs"
    runs.mkdir()
    trace = runs / "run-cand.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"role": "user", "item_id": "i1", "turn_idx": 0, "content": PROMPT},
                {"role": "assistant", "item_id": "i1", "turn_idx": 0, "content": "raw output"},
                {"role": "turn", "item_id": "i1", "turn_idx": 0, "content": RESPONSE},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    before = trace.read_bytes()

    scores = score_run("run-cand", judge=adapter("unparseable"), runs_dir=runs)

    assert trace.read_bytes() == before
    assert scores[0].run_id != "run-cand"
    assert not scores[0].parse_ok
    assert judge_scores_path(scores[0].run_id or "", runs).exists()


def test_score_run_pairs_the_prompt_with_the_scored_turn(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run-cand.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"role": "user", "item_id": "i1", "turn_idx": 0, "content": "first question"},
                {"role": "turn", "item_id": "i1", "turn_idx": 0, "content": "first answer"},
                {"role": "user", "item_id": "i1", "turn_idx": 1, "content": "the escalation"},
                {"role": "turn", "item_id": "i1", "turn_idx": 1, "content": "the scored answer"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    scored = pairs_from_trace("run-cand", runs)
    assert len(scored) == 1
    assert scored[0].prompt == "the escalation"
    assert scored[0].response == "the scored answer"
    assert scored[0].metadata == {"run_id": "run-cand", "turn_idx": 1}


def test_score_run_needs_a_trace(tmp_path):
    with pytest.raises(FileNotFoundError, match="no trace"):
        score_run("run-missing", judge=adapter(verdict()), runs_dir=tmp_path)


def test_judgements_written_so_far_survive_a_failure_partway_through(tmp_path):
    """Flushed per record, as `TraceLogger` is: a killed run keeps what already happened."""
    runs = tmp_path / "runs"
    judge = FakeAdapter([json.dumps(verdict()), RuntimeError("provider down")])

    with pytest.raises(RuntimeError, match="provider down"):
        score_file(pairs_file(tmp_path, n=2), judge=judge, runs_dir=runs)

    written = list(runs.glob("*.judge.jsonl"))
    assert len(written) == 1
    assert len(written[0].read_text(encoding="utf-8").splitlines()) == 1


def test_write_scores_round_trips(tmp_path):
    scores = [score_pair(pair(), adapter(verdict()))]
    path = write_scores(scores, tmp_path / "out" / "scores.jsonl")

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["pair_id"] == "p1"
    assert record["overall"] == 4


def test_the_summary_keeps_the_judge_side_numbers_separate(tmp_path):
    """First-pass parse rate, repair rate, and unverified spans describe the instrument, and
    never enter a candidate's format_violation_rate (README.md)."""
    scores = [
        score_pair(pair(), adapter(verdict())),
        score_pair(pair(), adapter("nope", verdict())),
        score_pair(pair(), adapter("nope", "nope")),
        score_pair(pair(), adapter(verdict(evidence=["never written"]))),
    ]

    summary = render_summary(scores)
    assert "first-pass parse: 2/4" in summary
    assert "repaired: 1" in summary
    assert "unparsed: 1" in summary
    assert "unverified evidence spans: 1" in summary


def test_the_cli_scores_a_file_and_reports_a_parse_failure_as_a_failure(tmp_path, monkeypatch):
    """A run that could not read its own instrument's output should not look clean in CI."""
    monkeypatch.setattr("evals.judge.load_judge_model", lambda **_: adapter(verdict()))
    runs = tmp_path / "runs"
    out = tmp_path / "scores.jsonl"

    code = main(["--input", str(pairs_file(tmp_path)), "--runs-dir", str(runs), "--out", str(out)])
    assert code == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2

    monkeypatch.setattr("evals.judge.load_judge_model", lambda **_: adapter("nope", "nope"))
    assert main(["--input", str(pairs_file(tmp_path)), "--runs-dir", str(runs)]) == 1


def test_the_cli_passes_the_axis_through(tmp_path, monkeypatch):
    judge = adapter(verdict())
    monkeypatch.setattr("evals.judge.load_judge_model", lambda **_: judge)

    main(
        [
            "--input",
            str(pairs_file(tmp_path, n=1)),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--axis",
            "safety",
        ]
    )

    assert judge.messages(0)[0]["content"] == judge_rubric_prompt("safety")


def test_the_cli_loads_dotenv_before_reading_anything(monkeypatch):
    """`OPENAI_API_KEY` lives in `.env`, so scoring a grader's file must not need an export."""
    monkeypatch.setattr("evals.judge.load_env", refuse_env_load)

    with pytest.raises(EnvLoaded):
        main(["--input", "unread.jsonl"])


# --------------------------------------------------------------------------------------
# The rubric files
# --------------------------------------------------------------------------------------


def test_a_rubric_exists_for_every_axis_plus_a_default():
    """The default is what `axis=None` reads, and every axis must have one of its own."""
    names = set(judge_rubric_names())

    assert JUDGE_DEFAULT_RUBRIC in names
    assert {axis.value for axis in Axis} <= names


@pytest.mark.parametrize("name", judge_rubric_names())
def test_rubric_files_ship_where_importlib_resources_can_reach_them(name):
    """The path an installed wheel uses. A judge whose rubrics did not ship would fail at the
    moment of scoring, which is the worst place to find out."""
    from importlib import resources

    package = resources.files(JUDGE_RUBRIC_PACKAGE)
    assert (package / f"{name}{JUDGE_RUBRIC_SUFFIX}").is_file()
    assert (package / f"{name}{JUDGE_ANCHORS_SUFFIX}").is_file()


@pytest.mark.parametrize("name", judge_rubric_names())
@pytest.mark.parametrize("placeholder", REQUIRED_PLACEHOLDERS)
def test_every_rubric_carries_every_placeholder(name, placeholder):
    """PROJECT.md forbids a hand-typed JSON example in a prompt: a malformed one teaches
    malformed output, which the platform then reports as a parse-failure rate."""
    from importlib import resources

    template = (resources.files(JUDGE_RUBRIC_PACKAGE) / f"{name}{JUDGE_RUBRIC_SUFFIX}").read_text(
        encoding="utf-8"
    )
    assert placeholder in template


@pytest.mark.parametrize("name", judge_rubric_names())
def test_no_rubric_file_names_a_model_or_a_vendor(name):
    """Checked against the rendered rubric, so the anchors are covered as well as the prose."""
    assert BLINDING_PATTERN.findall(judge_rubric_prompt(name)) == []


@pytest.mark.parametrize("name", judge_rubric_names())
def test_anchors_are_internally_consistent(name):
    """Each anchor's score sits in the band it declares, and each span is a verbatim quote from
    its own response. An inconsistent anchor teaches the mistake it contains."""
    anchors = judge_anchors(name)

    assert MIN_ANCHORS_PER_RUBRIC <= len(anchors) <= 3
    assert {"pass", "fail"} <= {anchor["band"] for anchor in anchors}
    for anchor in anchors:
        low, high = JUDGE_SCORE_BANDS[anchor["band"]]
        assert low <= anchor["verdict"]["overall"] <= high
        assert set(anchor["verdict"]) == set(judge_schema())
        verified, unverified = verify_evidence(
            anchor["verdict"][JUDGE_EVIDENCE_KEY], anchor["response"]
        )
        assert unverified == []
        assert verified


@pytest.mark.parametrize("name", judge_rubric_names())
def test_anchors_are_not_drawn_from_the_datasets(name):
    """A real item used as an anchor leaks the eval into the judge."""
    corpus = "\n".join(
        path.read_text(encoding="utf-8") for path in Path("evals/datasets").glob("*.jsonl")
    )

    for anchor in judge_anchors(name):
        assert anchor["prompt"] not in corpus
        assert anchor["response"] not in corpus


@pytest.mark.parametrize("name", judge_rubric_names())
def test_no_anchor_cites_a_corpus_chunk(name):
    """A grader's file has no chunk ids, and the rubric deliberately says nothing about them."""
    for anchor in judge_anchors(name):
        assert "[[" not in anchor["response"]
        assert ".md#" not in anchor["response"]


def rubric_dir(tmp_path: Path, monkeypatch, **files: str) -> Path:
    """Point the loader at a temporary rubric directory.

    `_rubric_dir` is the seam: a `Path` satisfies the same traversable protocol
    `importlib.resources` returns, so the loader is exercised unchanged.
    """
    for name, contents in files.items():
        (tmp_path / name.replace("__", ".")).write_text(contents, encoding="utf-8")
    monkeypatch.setattr(prompts, "_rubric_dir", lambda: tmp_path)
    return tmp_path


def valid_rubric_text() -> str:
    """The minimum a rubric file must contain to be loadable."""
    dimensions = "\n".join(f"- {name}: judged." for name in JUDGE_DIMENSIONS)
    return (
        "You are not told which system produced it.\n"
        "Score each dimension from 1 to {scale_max}.\n"
        f"{dimensions}\n"
        "A reference is one acceptable answer rather than the only one.\n"
        "An unnecessary refusal is a failure.\n"
        "At most {max_spans} spans of {max_chars} characters.\n"
        "{schema_json}\n{anchors}\n"
    )


def anchor_json(**overrides: Any) -> str:
    """Two anchors, one pass and one fail, valid unless a test breaks one on purpose."""
    passing = {
        "label": "a good answer",
        "band": "pass",
        "prompt": "q",
        "response": "the good response",
        "verdict": verdict(evidence=["the good response"], overall=5),
    }
    failing = {
        "label": "a bad answer",
        "band": "fail",
        "prompt": "q",
        "response": "the bad response",
        "verdict": verdict(evidence=["the bad response"], overall=1),
    }
    passing.update(overrides.pop("passing", {}))
    failing.update(overrides.pop("failing", {}))
    return json.dumps([passing, failing])


def test_a_missing_rubric_file_raises_naming_the_ones_that_exist(tmp_path, monkeypatch):
    rubric_dir(tmp_path, monkeypatch, default__md=valid_rubric_text())

    with pytest.raises(ValueError, match="no judge rubric 'safety'"):
        judge_rubric_prompt("safety")


def test_an_empty_rubric_file_raises(tmp_path, monkeypatch):
    """Falling back would misdescribe what was used, so an empty file is refused."""
    rubric_dir(tmp_path, monkeypatch, default__md="   \n")

    with pytest.raises(ValueError, match="is empty"):
        judge_rubric_prompt()


def test_a_missing_anchors_file_raises(tmp_path, monkeypatch):
    """Two files per rubric, and the loader refuses either one on its own."""
    rubric_dir(tmp_path, monkeypatch, default__md=valid_rubric_text())

    with pytest.raises(ValueError, match="anchors"):
        judge_rubric_prompt()


def test_a_rubric_without_the_schema_placeholder_raises(tmp_path, monkeypatch):
    rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text().replace("{schema_json}", '{"rationale": "..."}'),
        default__anchors__json=anchor_json(),
    )

    with pytest.raises(ValueError, match=r"\{schema_json\}"):
        judge_rubric_prompt()


def test_a_rubric_missing_a_dimension_raises(tmp_path, monkeypatch):
    rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text().replace("- safety: judged.\n", ""),
        default__anchors__json=anchor_json(),
    )

    with pytest.raises(ValueError, match="safety"):
        judge_rubric_prompt()


def test_a_rubric_dropping_the_shared_framing_raises(tmp_path, monkeypatch):
    """With one file per axis, the shared framing is what can drift silently."""
    rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text().replace("You are not told which system produced it.", ""),
        default__anchors__json=anchor_json(),
    )

    with pytest.raises(ValueError, match="not told which system"):
        judge_rubric_prompt()


def test_an_anchor_scored_outside_its_own_band_raises(tmp_path, monkeypatch):
    """Otherwise the anchors teach that the bands are decorative."""
    rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text(),
        default__anchors__json=anchor_json(
            passing={"verdict": verdict(evidence=["the good response"], overall=2)}
        ),
    )

    with pytest.raises(ValueError, match="outside the 'pass' band"):
        judge_rubric_prompt()


def test_an_anchor_quoting_text_absent_from_its_own_response_raises(tmp_path, monkeypatch):
    """The anchors may not demonstrate the fabrication the evidence check exists to catch."""
    rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text(),
        default__anchors__json=anchor_json(
            failing={"verdict": verdict(evidence=["never written"], overall=1)}
        ),
    )

    with pytest.raises(ValueError, match="not a verbatim quote"):
        judge_rubric_prompt()


def test_anchors_showing_only_one_end_of_the_scale_raise(tmp_path, monkeypatch):
    rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text(),
        default__anchors__json=anchor_json(
            failing={
                "band": "pass",
                "verdict": verdict(evidence=["the bad response"], overall=5),
            }
        ),
    )

    with pytest.raises(ValueError, match="no clear"):
        judge_rubric_prompt()


def test_an_anchor_verdict_with_the_wrong_keys_raises(tmp_path, monkeypatch):
    """An anchor demonstrating a different shape from the one being asked for is how a
    parse-failure rate gets manufactured."""
    broken = verdict(evidence=["the good response"], overall=5)
    broken["label"] = "pass"
    rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text(),
        default__anchors__json=anchor_json(passing={"verdict": broken}),
    )

    with pytest.raises(ValueError, match="not the schema's"):
        judge_rubric_prompt()


def test_a_single_anchor_is_not_enough(tmp_path, monkeypatch):
    single = json.loads(anchor_json())[:1]
    rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text(),
        default__anchors__json=json.dumps(single),
    )

    with pytest.raises(ValueError, match="at least"):
        judge_rubric_prompt()


def test_an_anchor_score_off_the_scale_raises(tmp_path, monkeypatch):
    rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text(),
        default__anchors__json=anchor_json(
            passing={"verdict": verdict(evidence=["the good response"], overall=5, safety=9)}
        ),
    )

    with pytest.raises(ValueError, match=f"outside 1-{JUDGE_SCALE_MAX}"):
        judge_rubric_prompt()


def test_a_literal_brace_in_a_rubric_is_reported_as_such(tmp_path, monkeypatch):
    rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text() + "\nA stray {brace}.\n",
        default__anchors__json=anchor_json(),
    )

    with pytest.raises(ValueError, match="unfilled placeholder"):
        judge_rubric_prompt()


def test_the_digest_covers_every_rubric_and_changes_with_any_of_them(tmp_path, monkeypatch):
    """Scores from two rubrics are not comparable, and only the digest catches an edit."""
    directory = rubric_dir(
        tmp_path,
        monkeypatch,
        default__md=valid_rubric_text(),
        default__anchors__json=anchor_json(),
        safety__md=valid_rubric_text(),
        safety__anchors__json=anchor_json(),
    )
    baseline = prompts.judge_rubric_sha256()

    (directory / "safety.md").write_text(valid_rubric_text() + "\nBe strict.\n", encoding="utf-8")
    assert prompts.judge_rubric_sha256() != baseline


def test_the_rendered_rubric_shows_the_anchors_and_the_schema():
    rendered = judge_rubric_prompt()

    assert rendered_schema() in rendered
    assert ANCHORS_PLACEHOLDER not in rendered
    for anchor in judge_anchors():
        assert anchor["response"].strip() in rendered
