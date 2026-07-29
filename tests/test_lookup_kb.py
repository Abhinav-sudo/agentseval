"""Tests for retrieval over `kb/`.

Every test here runs offline and without torch. `LexicalEmbedder` stands in for MiniLM: it
is a real TF-IDF retriever rather than a stub, so assertions about ranking are testing the
chunker and the search maths rather than a fixture that was rigged to agree with them. The
one test that needs the actual model is marked `slow` and skipped unless
`AGENTSEVAL_TEST_REAL_EMBEDDINGS=1`, since it downloads about 90MB.

The properties worth guarding here are provenance and determinism: chunk ids must be stable
and unique because they travel into the agent's answer and are later scored as citations,
and an index that rebuilds when it should not, or fails to rebuild when it should, quietly
changes what every eval retrieves.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from agent.tools import lookup_kb as mod
from agent.tools.lookup_kb import (
    CITATION_RE,
    INDEX_EXCLUDE,
    MAX_TOKENS,
    MIN_TOKENS,
    MODEL_WINDOW_TOKENS,
    Chunk,
    Hit,
    KnowledgeBase,
    chunk_markdown,
    corpus_files,
    format_citation,
    iter_blocks,
    l2_normalise,
    load_corpus,
    lookup_kb,
    parse_citations,
    reset_default_kb,
    split_sentences,
)
from agent.trace import sha256_of_paths

REAL_KB = Path(__file__).resolve().parent.parent / "kb"


def words(text: str) -> int:
    """Whitespace token counter, standing in for the model's tokenizer."""
    return len(text.split())


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class LexicalEmbedder:
    """TF-IDF over a fixed vocabulary: deterministic, offline, and genuinely topical.

    Records `calls` so tests can assert that a warm index performs no embedding at all,
    which is the whole point of persisting one.
    """

    def __init__(self, documents: list[str], model_name: str = "test-lexical-v1") -> None:
        self.model_name = model_name
        self.calls = 0
        self.encoded: list[str] = []
        tokenised = [tokenise(doc) for doc in documents]
        vocabulary = sorted({word for doc in tokenised for word in doc})
        self.vocabulary = {word: i for i, word in enumerate(vocabulary)}
        frequencies = Counter(word for doc in tokenised for word in set(doc))
        total = len(tokenised) or 1
        self.idf = np.zeros(len(self.vocabulary), dtype=np.float32)
        for word, i in self.vocabulary.items():
            self.idf[i] = math.log((1 + total) / (1 + frequencies[word])) + 1.0

    @classmethod
    def from_dir(cls, kb_dir: Path, **kwargs: object) -> LexicalEmbedder:
        texts = [path.read_text(encoding="utf-8") for path in corpus_files(kb_dir)]
        return cls(texts or ["placeholder"], **kwargs)  # type: ignore[arg-type]

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.encoded.extend(texts)
        matrix = np.zeros((len(texts), max(len(self.vocabulary), 1)), dtype=np.float32)
        for row, text in enumerate(texts):
            for word, count in Counter(tokenise(text)).items():
                index = self.vocabulary.get(word)
                if index is not None:
                    matrix[row, index] = count * self.idf[index]
        return matrix

    def count_tokens(self, text: str) -> int:
        return words(text)


@pytest.fixture
def kb_factory(tmp_path):
    """Build a KnowledgeBase over a temp corpus, keeping the index out of the repo."""

    def build(corpus_dir: Path, *, embedder=None, index_dir: Path | None = None, **kwargs):
        embedder = embedder or LexicalEmbedder.from_dir(corpus_dir)
        return KnowledgeBase(
            corpus_dir,
            cache_dir=index_dir if index_dir is not None else tmp_path / "index",
            embedder=embedder,
            **kwargs,
        )

    return build


@pytest.fixture
def corpus(tmp_path):
    """A small two-file corpus with headings, plus a README that must be ignored."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.md").write_text(
        "# Alpha\n\n"
        "Alpha handles widget calibration for the northern plant.\n\n"
        "## Calibration\n\n"
        "Recalibrate widgets every fourteen days using the bench jig.\n\n"
        "## Faults\n\n"
        "A flashing amber lamp means the encoder has lost its reference.\n",
        encoding="utf-8",
    )
    (root / "beta.md").write_text(
        "# Beta\n\nBeta covers invoicing and the monthly reconciliation run.\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Notes\n\nThis file documents the corpus.\n", encoding="utf-8"
    )
    return root


def paragraphs(count: int, size: int, *, sentence_words: int = 10) -> str:
    """Build `count` paragraphs of roughly `size` words each, in whole sentences."""
    blocks = []
    for p in range(count):
        sentences = [
            " ".join(f"word{p}x{s}y{w}" for w in range(sentence_words - 1)).capitalize() + " end."
            for s in range(max(1, size // sentence_words))
        ]
        blocks.append(" ".join(sentences))
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------------------
# Block parsing and chunking
# --------------------------------------------------------------------------------------


def test_headings_become_context_not_content():
    blocks = list(iter_blocks("# Title\n\nBody text here.\n\n## Section\n\nMore text.\n"))
    assert [b.text for b in blocks] == ["Body text here.", "More text."]
    assert [b.heading_path for b in blocks] == [("Title",), ("Title", "Section")]


def test_heading_path_tracks_nesting_and_siblings():
    markdown = "# Top\n\nA.\n\n## One\n\nB.\n\n### Deep\n\nC.\n\n## Two\n\nD.\n\n# Other\n\nE.\n"
    paths = [b.heading_path for b in iter_blocks(markdown)]
    assert paths == [
        ("Top",),
        ("Top", "One"),
        ("Top", "One", "Deep"),
        ("Top", "Two"),
        ("Other",),
    ]


def test_code_fence_is_one_block_despite_blank_lines():
    markdown = "# T\n\nIntro.\n\n```bash\nfirst --flag\n\nsecond --flag\n```\n\nAfter.\n"
    texts = [b.text for b in iter_blocks(markdown)]
    assert len(texts) == 3
    assert texts[1].startswith("```bash")
    assert texts[1].endswith("```")
    assert "second --flag" in texts[1]


def test_unterminated_fence_is_emitted_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        texts = [b.text for b in iter_blocks("# T\n\n```bash\nno close fence\n")]
    assert any("no close fence" in t for t in texts)
    assert "unterminated code fence" in caplog.text


def test_merging_reaches_the_token_band():
    markdown = "# T\n\n" + paragraphs(6, 70)
    chunks = chunk_markdown(markdown, "t.md", words)
    assert len(chunks) > 1
    # Every chunk but the trailing remainder should have reached the minimum.
    assert all(chunk.token_count >= MIN_TOKENS for chunk in chunks[:-1])


def test_the_ceiling_respects_the_models_window():
    """Pinned to the literal window, not to `MAX_TOKENS` itself.

    Every other size assertion here is written against `MAX_TOKENS`, so raising it would
    merely move the goalposts. all-MiniLM-L6-v2 accepts 256 word pieces; a band above that
    embeds a prefix and discards the rest, which looks like nothing at all going wrong.
    """
    assert MODEL_WINDOW_TOKENS == 256
    assert MAX_TOKENS <= MODEL_WINDOW_TOKENS
    assert MIN_TOKENS <= MAX_TOKENS


def test_no_chunk_exceeds_the_ceiling():
    """The ceiling is the embedding window; over it, a chunk is retrieved on a prefix."""
    markdown = "# T\n\n" + paragraphs(12, 70)
    chunks = chunk_markdown(markdown, "t.md", words)
    assert chunks
    assert all(chunk.token_count <= MAX_TOKENS for chunk in chunks)


def test_merging_stops_at_heading_boundaries():
    markdown = "# T\n\n## One\n\nShort text.\n\n## Two\n\nOther text.\n"
    chunks = chunk_markdown(markdown, "t.md", words)
    assert len(chunks) == 2
    assert chunks[0].heading_path == ("T", "One")
    assert chunks[1].heading_path == ("T", "Two")


def test_real_corpus_stays_inside_the_ceiling():
    chunks = load_corpus(REAL_KB, words)
    assert chunks
    assert all(chunk.token_count <= MAX_TOKENS for chunk in chunks)


def test_oversize_paragraph_splits_only_at_sentence_boundaries():
    long_paragraph = " ".join(
        f"Sentence number {i} carries a few extra words here." for i in range(60)
    )
    chunks = chunk_markdown(f"# T\n\n{long_paragraph}\n", "t.md", words)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= MAX_TOKENS
        assert chunk.text.rstrip().endswith(".")


def test_single_oversize_sentence_is_kept_whole_and_warned(caplog):
    """The one case the encoder truncates, so it must be visible in the log."""
    runaway = " ".join(f"word{i}" for i in range(MAX_TOKENS + 50))
    with caplog.at_level("WARNING"):
        chunks = chunk_markdown(f"# T\n\n{runaway}\n", "t.md", words)

    assert len(chunks) == 1
    assert chunks[0].token_count > MAX_TOKENS
    assert "truncated when embedded" in caplog.text


def test_split_sentences_keeps_sentences_intact():
    assert split_sentences("One two. Three four! Five six?") == [
        "One two.",
        "Three four!",
        "Five six?",
    ]


def test_split_sentences_does_not_break_decimals_or_lowercase_continuations():
    text = "The label lists 1,000 milligrams of sodium. see the note on 2.5 gram servings."
    assert len(split_sentences(text)) == 1


# --------------------------------------------------------------------------------------
# Chunk ids
# --------------------------------------------------------------------------------------


def test_chunk_id_is_source_file_and_ordinal():
    chunks = chunk_markdown("# T\n\nOne.\n\n## S\n\nTwo.\n", "sleep-hygiene.md", words)
    assert [c.chunk_id for c in chunks] == ["sleep-hygiene.md#0", "sleep-hygiene.md#1"]
    assert [c.ordinal for c in chunks] == [0, 1]


def test_chunk_ids_are_unique_across_the_real_corpus():
    chunks = load_corpus(REAL_KB, words)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_are_stable_across_reloads():
    """Citations scored against a later run are worthless if ids drift."""
    first = [c.chunk_id for c in load_corpus(REAL_KB, words)]
    second = [c.chunk_id for c in load_corpus(REAL_KB, words)]
    assert first == second


def test_nested_files_do_not_collide_on_basename(tmp_path):
    root = tmp_path / "kb"
    (root / "one").mkdir(parents=True)
    (root / "two").mkdir(parents=True)
    (root / "one" / "policy.md").write_text("# A\n\nFirst policy.\n", encoding="utf-8")
    (root / "two" / "policy.md").write_text("# B\n\nSecond policy.\n", encoding="utf-8")

    ids = [chunk.chunk_id for chunk in load_corpus(root, words)]
    assert sorted(ids) == ["one/policy.md#0", "two/policy.md#0"]


def test_readme_is_not_indexed():
    indexed = {path.name for path in corpus_files(REAL_KB)}
    assert indexed
    assert indexed.isdisjoint(INDEX_EXCLUDE)


def test_corpus_dir_that_does_not_exist_is_empty(tmp_path):
    assert corpus_files(tmp_path / "nope") == []


# --------------------------------------------------------------------------------------
# Index persistence and staleness
# --------------------------------------------------------------------------------------


def test_build_writes_both_index_files(corpus, kb_factory):
    kb = kb_factory(corpus)
    assert kb.matrix_path.exists()
    assert kb.sidecar_path.exists()

    sidecar = json.loads(kb.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["embedding_model"] == "test-lexical-v1"
    assert len(sidecar["chunks"]) == len(kb.chunks)
    assert sidecar["corpus_sha256"] == kb.corpus_fingerprint()
    assert {entry["path"] for entry in sidecar["files"]} == {"alpha.md", "beta.md"}


def test_index_defaults_to_living_beside_the_corpus(corpus):
    kb = KnowledgeBase(corpus, embedder=LexicalEmbedder.from_dir(corpus))
    assert kb.matrix_path.parent == corpus
    assert (corpus / ".index.npz").exists()
    assert (corpus / ".index.json").exists()


def test_no_temporary_files_are_left_behind(corpus, kb_factory):
    kb = kb_factory(corpus)
    assert list(kb.index_dir.glob("*.tmp")) == []


def test_reopening_a_fresh_index_does_not_embed(corpus, kb_factory, tmp_path):
    first = kb_factory(corpus)
    warm = LexicalEmbedder.from_dir(corpus)
    second = kb_factory(corpus, embedder=warm)

    assert warm.calls == 0
    assert second.staleness_reason() is None
    assert [c.chunk_id for c in second.chunks] == [c.chunk_id for c in first.chunks]
    assert second.embeddings.shape == first.embeddings.shape


def test_touching_a_file_without_editing_it_does_not_re_embed(corpus, kb_factory):
    """git clone and git checkout rewrite mtimes; that must not cost a rebuild."""
    kb_factory(corpus)
    target = corpus / "alpha.md"
    os.utime(target, (1_000_000, 1_000_000))

    warm = LexicalEmbedder.from_dir(corpus)
    kb = kb_factory(corpus, embedder=warm)

    assert warm.calls == 0
    assert kb.staleness_reason() is None


def test_a_refreshed_stamp_is_persisted(corpus, kb_factory):
    """After a mtime-only change the new stamp is written, so the next open is cheap."""
    kb = kb_factory(corpus)
    os.utime(corpus / "alpha.md", (1_000_000, 1_000_000))
    kb_factory(corpus, embedder=LexicalEmbedder.from_dir(corpus))

    sidecar = json.loads(kb.sidecar_path.read_text(encoding="utf-8"))
    stamps = {entry["path"]: entry["mtime_ns"] for entry in sidecar["files"]}
    assert stamps["alpha.md"] == 1_000_000 * 10**9


def test_editing_a_file_rebuilds(corpus, kb_factory):
    kb_factory(corpus)
    (corpus / "alpha.md").write_text("# Alpha\n\nCompletely different content.\n", encoding="utf-8")

    warm = LexicalEmbedder.from_dir(corpus)
    kb = kb_factory(corpus, embedder=warm)

    assert warm.calls == 1
    assert any("different content" in chunk.text for chunk in kb.chunks)


def test_adding_a_file_rebuilds(corpus, kb_factory):
    kb_factory(corpus)
    (corpus / "gamma.md").write_text("# Gamma\n\nNew document.\n", encoding="utf-8")

    warm = LexicalEmbedder.from_dir(corpus)
    kb = kb_factory(corpus, embedder=warm)

    assert warm.calls == 1
    assert "gamma.md" in {chunk.source_file for chunk in kb.chunks}


def test_removing_a_file_rebuilds(corpus, kb_factory):
    kb_factory(corpus)
    (corpus / "beta.md").unlink()

    warm = LexicalEmbedder.from_dir(corpus)
    kb = kb_factory(corpus, embedder=warm)

    assert warm.calls == 1
    assert "beta.md" not in {chunk.source_file for chunk in kb.chunks}


def test_changing_the_embedding_model_rebuilds(corpus, kb_factory):
    kb_factory(corpus)
    other = LexicalEmbedder.from_dir(corpus, model_name="different-model")
    kb = kb_factory(corpus, embedder=other, auto_build=False)

    assert "different-model" in (kb.staleness_reason() or "")
    assert kb.chunks == []


def test_changed_token_band_rebuilds(corpus, kb_factory, monkeypatch):
    kb = kb_factory(corpus)
    sidecar = json.loads(kb.sidecar_path.read_text(encoding="utf-8"))
    sidecar["max_tokens"] = MAX_TOKENS + 64
    kb.sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    fresh = kb_factory(corpus, embedder=LexicalEmbedder.from_dir(corpus), auto_build=False)
    assert fresh.staleness_reason() == "token band changed"


def test_missing_matrix_rebuilds(corpus, kb_factory):
    kb = kb_factory(corpus)
    kb.matrix_path.unlink()

    warm = LexicalEmbedder.from_dir(corpus)
    rebuilt = kb_factory(corpus, embedder=warm)
    assert warm.calls == 1
    assert rebuilt.matrix_path.exists()


def test_truncated_matrix_degrades_to_a_rebuild(corpus, kb_factory, caplog):
    kb = kb_factory(corpus)
    kb.matrix_path.write_bytes(b"PK\x03\x04 not really a zip")

    warm = LexicalEmbedder.from_dir(corpus)
    with caplog.at_level("WARNING"):
        rebuilt = kb_factory(corpus, embedder=warm)

    assert warm.calls == 1
    assert rebuilt.chunks
    assert "rebuilding" in caplog.text


def test_corrupt_sidecar_degrades_to_a_rebuild(corpus, kb_factory):
    kb = kb_factory(corpus)
    kb.sidecar_path.write_text("{ truncated", encoding="utf-8")

    warm = LexicalEmbedder.from_dir(corpus)
    rebuilt = kb_factory(corpus, embedder=warm)
    assert warm.calls == 1
    assert rebuilt.chunks


def test_matrix_and_sidecar_from_different_corpora_rebuild(corpus, kb_factory, caplog):
    """Guards the window between the two writes: never pair one corpus's vectors with
    another's text."""
    kb = kb_factory(corpus)
    sidecar = json.loads(kb.sidecar_path.read_text(encoding="utf-8"))
    sidecar["corpus_sha256"] = "0" * 64
    kb.sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    warm = LexicalEmbedder.from_dir(corpus)
    with caplog.at_level("WARNING"):
        kb_factory(corpus, embedder=warm)
    assert warm.calls == 1
    assert "different corpora" in caplog.text


def test_auto_build_false_reports_without_building(corpus, kb_factory):
    kb = kb_factory(corpus, auto_build=False)
    assert kb.chunks == []
    assert kb.staleness_reason() == "no index on disk"
    assert not kb.matrix_path.exists()


def test_empty_corpus_builds_an_empty_index(tmp_path, caplog):
    empty = tmp_path / "empty"
    empty.mkdir()
    with caplog.at_level("WARNING"):
        kb = KnowledgeBase(empty, cache_dir=tmp_path / "idx", embedder=LexicalEmbedder(["x"]))

    assert kb.chunks == []
    assert kb.search("anything") == []
    assert "no markdown to index" in caplog.text


# --------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------


def test_top_k_limits_and_orders_results(kb_factory):
    kb = kb_factory(REAL_KB)
    hits = kb.search("progressive overload and rest between sessions", top_k=3)

    assert len(hits) == 3
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_scores_are_cosine_similarities(kb_factory):
    kb = kb_factory(REAL_KB)
    for hit in kb.search("paced breathing and everyday stress", top_k=5):
        assert -1.0 - 1e-6 <= hit.score <= 1.0 + 1e-6


def test_identical_text_scores_near_one(kb_factory):
    kb = kb_factory(REAL_KB)
    chunk = kb.chunks[0]
    top = kb.search(chunk.text, top_k=1)[0]
    assert top.chunk.chunk_id == chunk.chunk_id
    assert top.score == pytest.approx(1.0, abs=1e-5)


def test_ties_break_on_chunk_id(tmp_path):
    """Deterministic ordering: two runs of an unchanged corpus must agree."""
    root = tmp_path / "kb"
    root.mkdir()
    for name in ("zulu.md", "alpha.md", "mike.md"):
        (root / name).write_text("# H\n\nIdentical body text.\n", encoding="utf-8")

    kb = KnowledgeBase(root, cache_dir=tmp_path / "idx", embedder=LexicalEmbedder.from_dir(root))
    hits = kb.search("identical body text", top_k=3)

    assert len({round(hit.score, 6) for hit in hits}) == 1
    assert [hit.chunk.chunk_id for hit in hits] == ["alpha.md#0", "mike.md#0", "zulu.md#0"]


def test_min_score_filters_weak_matches(kb_factory):
    kb = kb_factory(REAL_KB)
    query = "bedroom temperature and darkness before bed"

    assert kb.search(query, top_k=4, min_score=0.0)
    assert kb.search(query, top_k=4, min_score=0.99) == []


def test_min_score_returns_an_empty_list_rather_than_a_bad_guess(kb_factory):
    """The signal a caller turns into a refusal instead of an invented answer."""
    kb = kb_factory(REAL_KB)
    assert kb.search("quantum chromodynamics lattice gauge", top_k=4, min_score=0.5) == []


def test_top_k_zero_returns_nothing(kb_factory):
    assert kb_factory(REAL_KB).search("anything", top_k=0) == []


def test_unknown_vocabulary_query_does_not_crash(kb_factory):
    """A query sharing no vocabulary embeds to a zero vector; it must score, not divide."""
    kb = kb_factory(REAL_KB)
    hits = kb.search("zzzqqq wwwvvv", top_k=3)
    assert all(hit.score == 0.0 for hit in hits)


def test_l2_normalise_handles_zero_rows():
    matrix = l2_normalise(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    assert np.allclose(matrix[0], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(matrix[1]), 1.0)


def test_l2_normalise_of_empty_matrix():
    assert l2_normalise(np.zeros((0, 0), dtype=np.float32)).size == 0


# --------------------------------------------------------------------------------------
# Known queries: the corpus answers what it should
# --------------------------------------------------------------------------------------

KNOWN_QUERIES = [
    ("how many hours of sleep do adults need", "sleep-hygiene.md"),
    ("is the eight glasses of water a day rule true", "hydration.md"),
    ("what should a balanced plate look like", "nutrition-principles.md"),
    ("how many sets and repetitions should a beginner lift", "strength-training-basics.md"),
    ("how many minutes of moderate activity are recommended weekly", "cardio-basics.md"),
    ("breathing exercise to calm down when stressed", "stress-management.md"),
    ("how long does muscle soreness last after a hard session", "recovery-and-rest-days.md"),
    ("where should my monitor sit relative to my eyes", "desk-ergonomics.md"),
    ("do I really need ten thousand steps a day", "walking-and-daily-movement.md"),
    ("should I stretch before or after exercise", "warm-up-and-mobility.md"),
    ("how long does it take for a new habit to stick", "sustainable-habits.md"),
    ("what does percent daily value mean on a food label", "reading-nutrition-labels.md"),
]


@pytest.mark.parametrize(("query", "expected_file"), KNOWN_QUERIES)
def test_known_query_retrieves_the_expected_source_file(kb_factory, query, expected_file):
    """A bag-of-words retriever earns top-3; the real model is held to top-1 below."""
    hits = kb_factory(REAL_KB).search(query, top_k=3)
    assert expected_file in [hit.chunk.source_file for hit in hits]


real_embeddings = pytest.mark.skipif(
    os.environ.get("AGENTSEVAL_TEST_REAL_EMBEDDINGS") != "1",
    reason="needs a ~90MB model download; set AGENTSEVAL_TEST_REAL_EMBEDDINGS=1",
)


@pytest.mark.slow
@real_embeddings
@pytest.mark.parametrize(("query", "expected_file"), KNOWN_QUERIES)
def test_known_query_with_real_minilm(tmp_path, query, expected_file):
    """The test that validates semantic retrieval rather than plumbing."""
    pytest.importorskip("sentence_transformers")
    kb = KnowledgeBase(REAL_KB, cache_dir=tmp_path / "idx")
    hits = kb.search(query, top_k=3)
    assert hits[0].chunk.source_file == expected_file


@pytest.mark.slow
@real_embeddings
def test_declared_window_matches_the_real_model():
    """`MODEL_WINDOW_TOKENS` is a copied fact; confirm it against the model itself."""
    pytest.importorskip("sentence_transformers")
    from agent.tools.lookup_kb import MiniLMEmbedder

    assert MiniLMEmbedder().model.max_seq_length == MODEL_WINDOW_TOKENS


# --------------------------------------------------------------------------------------
# Tool entry point and serialisation
# --------------------------------------------------------------------------------------


def test_lookup_kb_returns_hits_with_scores(corpus, monkeypatch):
    monkeypatch.setattr(mod, "default_embedder", lambda: LexicalEmbedder.from_dir(corpus))
    reset_default_kb()
    try:
        hits = lookup_kb("widget calibration bench jig", top_k=2, kb_dir=corpus)
        assert hits
        assert all(isinstance(hit, Hit) for hit in hits)
        assert hits[0].chunk.source_file == "alpha.md"
        assert hits[0].score > 0
    finally:
        reset_default_kb()


def test_lookup_kb_reuses_one_knowledge_base(corpus, monkeypatch):
    """Rebuilding per tool call would dominate the latency it is supposed to measure."""
    embedder = LexicalEmbedder.from_dir(corpus)
    monkeypatch.setattr(mod, "default_embedder", lambda: embedder)
    reset_default_kb()
    try:
        lookup_kb("widget", kb_dir=corpus)
        after_first = embedder.calls
        lookup_kb("invoicing", kb_dir=corpus)
        assert embedder.calls == after_first + 1  # one query embed, no rebuild
    finally:
        reset_default_kb()


def test_hit_to_dict_is_json_serialisable_and_leads_with_the_id(kb_factory):
    hit = kb_factory(REAL_KB).search("hydration", top_k=1)[0]
    payload = hit.to_dict()

    assert next(iter(payload)) == "chunk_id"
    assert payload["citation"] == format_citation(hit.chunk.chunk_id)
    assert json.loads(json.dumps(payload))["chunk_id"] == hit.chunk.chunk_id


def test_chunk_round_trips_through_dict():
    chunk = Chunk("a.md#1", "a.md", ("A", "B"), 1, "text", 7)
    assert Chunk.from_dict(chunk.to_dict()) == chunk


def test_tool_schema_declares_query_and_top_k():
    assert mod.name == "lookup_kb"
    assert mod.schema["required"] == ["query"]
    assert set(mod.schema["properties"]) == {"query", "top_k"}


# --------------------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------------------


def test_citation_round_trips():
    text = f"A fixed wake time is the strongest lever {format_citation('sleep-hygiene.md#2')}."
    assert parse_citations(text) == ["sleep-hygiene.md#2"]


def test_bare_filename_is_not_a_citation():
    """Otherwise mentioning a document would score as citing it."""
    assert parse_citations("See sleep-hygiene.md#2 for details") == []
    assert parse_citations("As sleep-hygiene.md explains") == []


def test_citations_are_deduplicated_in_order():
    text = "[[b.md#1]] then [[a.md#0]] and [[b.md#1]] again"
    assert parse_citations(text) == ["b.md#1", "a.md#0"]


def test_citation_pattern_requires_an_ordinal():
    assert CITATION_RE.search("[[sleep-hygiene.md]]") is None
    assert CITATION_RE.search("[[sleep-hygiene.md#12]]") is not None


def test_every_real_chunk_id_round_trips_as_a_citation():
    """The end-to-end promise: any retrieved id can be cited and parsed back."""
    ids = [chunk.chunk_id for chunk in load_corpus(REAL_KB, words)]
    rendered = " ".join(format_citation(chunk_id) for chunk_id in ids)
    assert parse_citations(rendered) == ids


# --------------------------------------------------------------------------------------
# Fingerprint
# --------------------------------------------------------------------------------------


def test_fingerprint_matches_the_manifest_helper(corpus, kb_factory):
    kb = kb_factory(corpus)
    assert kb.corpus_fingerprint() == sha256_of_paths(corpus_files(corpus), root=corpus)


def test_fingerprint_changes_when_the_corpus_changes(corpus, kb_factory):
    kb = kb_factory(corpus)
    before = kb.corpus_fingerprint()
    (corpus / "alpha.md").write_text("# Alpha\n\nEdited.\n", encoding="utf-8")
    assert kb.corpus_fingerprint() != before


# --------------------------------------------------------------------------------------
# Build script
# --------------------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch, corpus):
    """Run main() against the temp corpus with the lexical embedder."""
    monkeypatch.setattr(mod, "default_embedder", lambda: LexicalEmbedder.from_dir(corpus))
    return lambda *args: mod.main([*args, "--kb-dir", str(corpus)])


def test_cli_builds_then_reports_up_to_date(cli, capsys):
    assert cli() == 0
    first = capsys.readouterr().out
    assert "building index (no index on disk)" in first
    assert "indexed:     2 files" in first
    assert "skipped:     README.md" in first

    assert cli() == 0
    assert "up to date" in capsys.readouterr().out


def test_cli_rebuild_forces_a_build(cli, capsys):
    cli()
    capsys.readouterr()
    assert cli("--rebuild") == 0
    assert "forced rebuild" in capsys.readouterr().out


def test_cli_check_fails_on_a_stale_index(cli, capsys, corpus):
    assert cli("--check") == 1
    assert "stale" in capsys.readouterr().err

    cli()
    capsys.readouterr()
    assert cli("--check") == 0
    assert "up to date" in capsys.readouterr().out

    (corpus / "alpha.md").write_text("# Alpha\n\nEdited after indexing.\n", encoding="utf-8")
    assert cli("--check") == 1
    assert "alpha.md changed" in capsys.readouterr().err


def test_cli_check_does_not_write_an_index(cli, corpus):
    assert cli("--check") == 1
    assert not (corpus / ".index.npz").exists()
    assert not (corpus / ".index.json").exists()


def test_cli_stats_reports_the_token_band(cli, capsys):
    cli()
    capsys.readouterr()
    assert cli("--stats") == 0
    out = capsys.readouterr().out
    assert f"merge to {MIN_TOKENS}, ceiling {MAX_TOKENS}" in out
    assert "corpus hash:" in out


def test_cli_says_so_when_sections_are_smaller_than_the_merge_target(cli, capsys):
    """The fixture corpus has tiny sections; the report should not leave that implicit."""
    cli()
    assert "every section is under" in capsys.readouterr().out
