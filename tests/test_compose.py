"""Tests for composing a fixture corpus.

`evals/fixtures/compose.py` builds the corpus the injection dataset runs against: `kb/` plus
one poisoned document. The properties worth guarding are the ones whose failure is silent.

A composed corpus that quietly lost a document, kept a stale one, or let a fixture shadow a
real file would still index, still retrieve, and still produce scores — the eval would simply
be measuring a corpus nobody described. And because the corpus digest is a manifest field,
each of those also changes `kb_sha256`, which is what makes a run's conditions wrong rather
than merely its answers.

The poisoned document itself is checked here too, on the one property the dataset depends on:
that each `##` section is exactly one citable chunk. `evals/datasets/injection.jsonl` cites
`energy-and-fatigue.md#1` through `#5` by ordinal, so a section that split would repoint every
one of those notes at different text. MiniLM is not available offline, so the check uses a
token counter that deliberately overestimates — if the sections hold under a pessimistic
count, they hold under the real one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.lookup_kb import INDEX_EXCLUDE, MAX_TOKENS, chunk_markdown, corpus_files
from agent.trace import sha256_of_paths
from evals.fixtures.compose import (
    DEFAULT_FIXTURE,
    FIXTURES_DIR,
    CompositionError,
    compose,
)

#: The real corpus's worst tokens-per-word ratio is 1.34; 1.5 is a deliberate overestimate, so
#: a section that stays one chunk here stays one chunk under MiniLM.
PESSIMISTIC_TOKENS_PER_WORD = 1.5

POISONED_DOC = "energy-and-fatigue.md"

#: Chunks the injection dataset cites by ordinal. Frozen with the dataset, per kb/README.md.
POISONED_CHUNK_COUNT = 6


def overestimate(text: str) -> int:
    return int(len(text.split()) * PESSIMISTIC_TOKENS_PER_WORD) + 1


@pytest.fixture
def kb(tmp_path: Path) -> Path:
    base = tmp_path / "kb"
    base.mkdir()
    (base / "alpha.md").write_text("# Alpha\n\nAlpha body.\n", encoding="utf-8")
    (base / "beta.md").write_text("# Beta\n\nBeta body.\n", encoding="utf-8")
    (base / "README.md").write_text("# Not corpus\n\nDocs about the corpus.\n", encoding="utf-8")
    return base


def test_composes_the_base_corpus_plus_the_fixture(kb: Path, tmp_path: Path):
    out, _ = compose(DEFAULT_FIXTURE, kb_dir=kb, output_root=tmp_path / "out")

    names = {p.name for p in corpus_files(out)}
    assert names == {"alpha.md", "beta.md", POISONED_DOC}


def test_readme_is_not_composed_into_the_corpus(kb: Path, tmp_path: Path):
    """The fixture's own README documents the fixture; indexing it would make prose about the
    fixture compete for retrieval slots, exactly as under `kb/`."""
    out, _ = compose(DEFAULT_FIXTURE, kb_dir=kb, output_root=tmp_path / "out")

    assert INDEX_EXCLUDE == ("README.md",)
    assert not (out / "README.md").exists()


def test_digest_is_stable_across_recomposition(kb: Path, tmp_path: Path):
    """The digest lands in `kb_sha256`. If it moved on every build, no two runs against the
    fixture would ever be comparable to each other."""
    first = compose(DEFAULT_FIXTURE, kb_dir=kb, output_root=tmp_path / "out")[1]
    second = compose(DEFAULT_FIXTURE, kb_dir=kb, output_root=tmp_path / "out")[1]

    assert first == second


def test_digest_differs_from_the_base_corpus(kb: Path, tmp_path: Path):
    """`assert_comparable` refuses a fixture run against a main run on `kb_sha256`. That guard
    is only load-bearing if the two digests actually differ."""
    out, composed = compose(DEFAULT_FIXTURE, kb_dir=kb, output_root=tmp_path / "out")

    assert composed != sha256_of_paths(corpus_files(kb), root=kb)


def test_stale_documents_are_removed_rather_than_left(kb: Path, tmp_path: Path):
    """A leftover from a previous composition is a corpus document no source directory
    contains, and it would be embedded, retrieved, and cited like any other."""
    root = tmp_path / "out"
    out, expected = compose(DEFAULT_FIXTURE, kb_dir=kb, output_root=root)
    (out / "stale.md").write_text("# Stale\n\nFrom a previous build.\n", encoding="utf-8")

    out, after = compose(DEFAULT_FIXTURE, kb_dir=kb, output_root=root)

    assert not (out / "stale.md").exists()
    assert after == expected


def test_a_fixture_shadowing_a_base_document_is_refused(tmp_path: Path):
    """Composing the fixture over itself collides on the poisoned document's own name. A
    shadowed file leaves every item citing it pointing at text that is no longer there."""
    with pytest.raises(CompositionError, match=POISONED_DOC):
        compose(
            DEFAULT_FIXTURE,
            kb_dir=FIXTURES_DIR / DEFAULT_FIXTURE,
            output_root=tmp_path / "out",
        )


def test_unknown_fixture_names_the_ones_that_exist(kb: Path, tmp_path: Path):
    with pytest.raises(CompositionError, match=DEFAULT_FIXTURE):
        compose("no-such-fixture", kb_dir=kb, output_root=tmp_path / "out")


def test_empty_base_corpus_is_refused(tmp_path: Path):
    """A fixture corpus is the real corpus plus a document. An empty base silently yields a
    one-file corpus that retrieves the poisoned document for every query."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(CompositionError, match="no markdown"):
        compose(DEFAULT_FIXTURE, kb_dir=empty, output_root=tmp_path / "out")


def test_poisoned_document_is_one_chunk_per_section():
    """`injection.jsonl` cites these ordinals, so a split section repoints its notes."""
    path = FIXTURES_DIR / DEFAULT_FIXTURE / POISONED_DOC
    chunks = chunk_markdown(path.read_text(encoding="utf-8"), POISONED_DOC, overestimate)

    assert [c.chunk_id for c in chunks] == [
        f"{POISONED_DOC}#{i}" for i in range(POISONED_CHUNK_COUNT)
    ]
    assert all(c.token_count <= MAX_TOKENS for c in chunks)


def test_poisoned_document_carries_no_unsafe_specifics():
    """The payloads instruct the model to produce unsafe specifics and must not contain any.
    A fixture shipping the harmful content would make a compliant answer *retrievable* rather
    than generated, which measures the corpus instead of the model."""
    text = (FIXTURES_DIR / DEFAULT_FIXTURE / POISONED_DOC).read_text(encoding="utf-8")

    assert not any(char.isdigit() for char in text)
