"""Retrieval over the markdown corpus in `kb/`.

Locked design (PROJECT.md): paragraph chunks, `all-MiniLM-L6-v2` embeddings, cosine
similarity over a numpy array. No FAISS, no Chroma, no vector service — the corpus is
tiny, so a single matrix multiply is faster than an index and far easier to audit. Do not
add an ANN dependency here; if the corpus ever outgrows brute force, amend PROJECT.md
first.

Determinism: embeddings are computed once and persisted to `kb/.index.npz` plus a
`kb/.index.json` sidecar, so repeated runs retrieve identical chunks. Every lookup logs its
query, the chunk ids returned, and their scores into the trace.

Chunk ids are the unit of provenance. They are stable for an unchanged corpus, they travel
into the agent's answer as citations, and `evals/deterministic.py` scores those citations
against the ids actually retrieved — so `CITATION_FORMAT` here is the one definition of the
citation syntax, read by both the prompt that asks for it and the check that grades it.

Adjacent paragraphs are merged up to `MAX_TOKENS`, which is the embedding model's input
window rather than an arbitrary size. A longer chunk would be shown to the agent in full but
embedded only up to the window, so its tail would silently stop influencing retrieval.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from agent.trace import sha256_of_paths, sha256_text, utc_now_iso

logger = logging.getLogger(__name__)

#: Counts tokens the way the embedding model will, normally `Embedder.count_tokens`.
TokenCounter = Callable[[str], int]

DEFAULT_KB_DIR = Path("kb")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4

#: No confidence floor by default. A threshold picked before the eval set exists would be
#: guesswork; expose the knob, calibrate it against measured retrieval, then set it. Around
#: 0.3 is a sensible starting point for MiniLM cosine scores.
DEFAULT_MIN_SCORE = 0.0

#: The pre-registered floor for grounding enforcement, and the only floor any graded run uses.
#: Derived by `agentseval-calibrate-retrieval` under the rule in `evals.calibrate_retrieval`
#: — the 5th percentile of the top-1 cosine score over items the eval sets mark `answerable`,
#: rounded down — and recorded in `runs/retrieval_calibration.json`. Measured from the
#: answerable side alone, so it cannot have been fitted to whatever the guardrail later scored.
#:
#: Separate from `DEFAULT_MIN_SCORE` rather than replacing it. This is a condition of the
#: ablation runs, passed in through `AgentConfig.min_score` and digested into
#: `retrieval_config_sha256`, whereas the default above is what the library does when nobody has
#: chosen — and PROJECT.md's "defaulting to 0 (no floor)" is a locked statement about that.
#:
#: A number in source rather than one read from the artifact at import: a floor that reloaded
#: itself from a file would change what a run measured whenever the file was regenerated, which
#: is the opposite of pre-registration. Moving it is a commit, and `retrieval_config_sha256`
#: makes runs on either side of that commit incomparable, which is correct.
#:
#: As measured over the 180 items of the three main eval sets: answerable items reach a mean
#: top-1 of 0.558 and unanswerable ones 0.399, and this floor sits below 95.6% of the answerable
#: distribution and above 48.5% of the unanswerable one. The second figure is what the grounding
#: stage can catch; the first is what it costs in over-refusal, and both are reported rather
#: than traded off in the choice.
GROUNDING_MIN_SCORE = 0.37

#: all-MiniLM-L6-v2's input window, in word pieces. A property of the model, not a choice:
#: text beyond it is dropped by the encoder. `MiniLMEmbedder` checks this against the loaded
#: model, and a test pins MAX_TOKENS at or below it, so raising the band without changing
#: model fails loudly instead of silently embedding prefixes.
MODEL_WINDOW_TOKENS = 256

#: Merge adjacent paragraphs until at least MIN_TOKENS, never past MAX_TOKENS. The ceiling
#: is the window above: a longer chunk would reach the agent in full but be retrieved on a
#: prefix of itself.
MIN_TOKENS = 200
MAX_TOKENS = MODEL_WINDOW_TOKENS

#: Files under `kb/` that are documentation *about* the corpus rather than corpus content.
#: Indexing them makes prose about chunking compete with the corpus for retrieval slots.
INDEX_EXCLUDE = ("README.md",)

INDEX_MATRIX_NAME = ".index.npz"
INDEX_SIDECAR_NAME = ".index.json"

#: Bumped when the sidecar layout changes, so an old index rebuilds instead of being
#: misread by newer code.
INDEX_VERSION = 1

#: How the agent cites a chunk, and how `parse_citations` finds one. Double brackets rather
#: than a bare id so that mentioning a filename in prose is not mistaken for a citation.
CITATION_FORMAT = "[[{chunk_id}]]"
CITATION_RE = re.compile(r"\[\[([^\[\]]+?\.md#\d+)\]\]")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

#: Sentence boundary: terminator, whitespace, then something that can start a sentence.
#: Deliberately conservative — a missed boundary merely yields a larger unit, whereas a
#: false one would cut a sentence in half, which is the thing we promise not to do.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[`])")


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, with enough provenance to cite it.

    Attributes:
        chunk_id: `{source_file}#{ordinal}`, e.g. `sleep-hygiene.md#2`. Stable for an
            unchanged corpus and quoted verbatim in the agent's citations.
        source_file: Path relative to the kb directory, POSIX separators. The full relative
            path rather than the bare filename, so two `README.md` files in different
            subdirectories cannot collide into one id and silently merge provenance.
        heading_path: Enclosing markdown headings, outermost first. Retrieved chunks arrive
            without their document, so this is what tells the model where the text is from.
        ordinal: 0-based position within `source_file`.
        token_count: Length in the embedding model's tokens, including its special tokens.
    """

    chunk_id: str
    source_file: str
    heading_path: tuple[str, ...]
    ordinal: int
    text: str
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "heading_path": list(self.heading_path),
            "ordinal": self.ordinal,
            "text": self.text,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=data["chunk_id"],
            source_file=data["source_file"],
            heading_path=tuple(data["heading_path"]),
            ordinal=data["ordinal"],
            text=data["text"],
            token_count=data["token_count"],
        )

    def citation(self) -> str:
        """Render this chunk's citation token, the form the agent must reproduce."""
        return CITATION_FORMAT.format(chunk_id=self.chunk_id)


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk and its cosine similarity to the query.

    Score lives here rather than on `Chunk` because it is a property of one query: a cached
    corpus chunk carrying a score field would be describing a search that already ended.
    """

    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Flatten for the tool protocol, `chunk_id` first so the model sees what to cite."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "score": round(self.score, 4),
            "source_file": self.chunk.source_file,
            "heading_path": list(self.chunk.heading_path),
            "citation": self.chunk.citation(),
            "text": self.chunk.text,
        }


# --------------------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------------------


class Embedder(Protocol):
    """Turns text into vectors, and counts tokens the same way it does when embedding.

    Both operations belong to one object because they are the same model's opinion: chunk
    boundaries are set by the tokenizer that will later truncate the text, so a token count
    from anywhere else could let a chunk overflow the window unnoticed.
    """

    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one row vector per text."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return the token length of `text` as the encoder will see it."""
        ...


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 via sentence-transformers, loaded on first use.

    Lazy because importing this module must stay cheap: the CLI's staleness check, the
    chunk metadata, and every test but one need no model at all, and loading one costs a
    couple of seconds and a few hundred megabytes.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise RuntimeError(
                    "sentence-transformers is required to build the KB index; "
                    "install it or pass an explicit embedder"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            window = getattr(self._model, "max_seq_length", None)
            if window and window < MAX_TOKENS:
                logger.warning(
                    "%s accepts %d tokens but chunks are built up to %d; chunk tails will "
                    "be truncated at embedding time and will not affect retrieval. Lower "
                    "MAX_TOKENS or change MODEL_WINDOW_TOKENS to match the model.",
                    self.model_name,
                    window,
                    MAX_TOKENS,
                )
        return self._model

    @property
    def version(self) -> str | None:
        """sentence-transformers version, recorded so an index diff can be explained."""
        try:
            import sentence_transformers

            return str(sentence_transformers.__version__)
        except Exception:  # pragma: no cover - version metadata is best-effort
            return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(matrix, dtype=np.float32)

    def count_tokens(self, text: str) -> int:
        """Count word pieces including [CLS]/[SEP], since those consume the window too."""
        return len(self.model.tokenizer.encode(text, add_special_tokens=True))


def default_embedder() -> Embedder:
    """Return the project's embedder. Constructing it loads no model."""
    return MiniLMEmbedder()


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so cosine similarity is a plain dot product.

    Zero rows are left alone rather than producing NaN; they simply score 0 against
    everything, which is the honest answer for a vector carrying no signal.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def embed(texts: list[str]) -> np.ndarray:
    """Embed `texts` with all-MiniLM-L6-v2, returning L2-normalised row vectors.

    Normalising at embed time makes cosine similarity a plain dot product.
    """
    return l2_normalise(default_embedder().encode(texts))


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Block:
    """One markdown block (paragraph, list, or fenced code) with its heading context."""

    text: str
    heading_path: tuple[str, ...]


def iter_blocks(markdown: str) -> Iterator[_Block]:
    """Split markdown into blocks on blank lines, tracking the enclosing headings.

    Heading lines are context rather than content: they set `heading_path` and are not
    emitted, so a chunk is prose the model can use plus a trail saying where it came from.
    Fenced code blocks are never split, whatever blank lines they contain.
    """
    heading_path: tuple[str, ...] = ()
    buffer: list[str] = []
    fence: str | None = None

    def flush() -> Iterator[_Block]:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            yield _Block(text=text, heading_path=heading_path)

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        fence_match = _FENCE_RE.match(line)

        if fence is not None:
            buffer.append(line)
            if fence_match and fence_match.group(1) == fence:
                fence = None
            continue

        if fence_match:
            yield from flush()
            fence = fence_match.group(1)
            buffer.append(line)
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            yield from flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_path = heading_path[: level - 1] + (title,)
            continue

        if not line.strip():
            yield from flush()
            continue

        buffer.append(line)

    if fence is not None:
        # Unterminated fence: emit what we have rather than dropping the tail silently.
        logger.warning("unterminated code fence; emitting the remainder as one block")
    yield from flush()


def split_sentences(text: str) -> list[str]:
    """Split `text` at sentence boundaries, preserving the original substrings."""
    parts = [part.strip() for part in _SENTENCE_RE.split(text)]
    return [part for part in parts if part]


def _split_oversize(
    block: _Block, count: TokenCounter, max_tokens: int, source_file: str
) -> Iterator[_Block]:
    """Break a block that exceeds the window into sentence-bounded pieces.

    Never splits mid-sentence. A single sentence longer than the window is emitted alone and
    warned about: it is the one case the encoder will truncate, and it should be visible in
    the log rather than discovered as mysteriously poor retrieval.
    """
    sentences = split_sentences(block.text)
    if len(sentences) <= 1:
        logger.warning(
            "%s: a single sentence is %d tokens, over the %d-token window; it will be "
            "truncated when embedded",
            source_file,
            count(block.text),
            max_tokens,
        )
        yield block
        return

    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence])
        if current and count(candidate) > max_tokens:
            yield _Block(text=" ".join(current), heading_path=block.heading_path)
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        text = " ".join(current)
        if count(text) > max_tokens:
            logger.warning(
                "%s: a single sentence is %d tokens, over the %d-token window; it will be "
                "truncated when embedded",
                source_file,
                count(text),
                max_tokens,
            )
        yield _Block(text=text, heading_path=block.heading_path)


def chunk_markdown(
    markdown: str,
    source_file: str,
    counter: TokenCounter,
    *,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[Chunk]:
    """Chunk one document: merge paragraphs into the token band, then assign ids.

    Merging stops at a heading boundary. A chunk spanning two sections would have no single
    honest `heading_path`, and citing it would point the reader at the wrong part of the
    document.

    Args:
        markdown: File contents.
        source_file: Path relative to the kb directory, used in `chunk_id`.
        counter: Token counter, normally `Embedder.count_tokens`.
        min_tokens: Flush a chunk once it reaches this size.
        max_tokens: Never exceed this; it is the embedding model's window.
    """
    units: list[_Block] = []
    for block in iter_blocks(markdown):
        if counter(block.text) > max_tokens:
            units.extend(_split_oversize(block, counter, max_tokens, source_file))
        else:
            units.append(block)

    chunks: list[Chunk] = []
    pending: list[str] = []
    pending_heading: tuple[str, ...] = ()

    def flush() -> None:
        if not pending:
            return
        text = "\n\n".join(pending)
        chunks.append(
            Chunk(
                chunk_id=f"{source_file}#{len(chunks)}",
                source_file=source_file,
                heading_path=pending_heading,
                ordinal=len(chunks),
                text=text,
                token_count=counter(text),
            )
        )
        pending.clear()

    for unit in units:
        if pending:
            merged = "\n\n".join([*pending, unit.text])
            if unit.heading_path != pending_heading or counter(merged) > max_tokens:
                flush()
        if not pending:
            pending_heading = unit.heading_path
        pending.append(unit.text)
        if counter("\n\n".join(pending)) >= min_tokens:
            flush()

    flush()
    return chunks


def corpus_files(kb_dir: Path = DEFAULT_KB_DIR) -> list[Path]:
    """Return the markdown files to index, sorted, excluding `INDEX_EXCLUDE`."""
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        return []
    return sorted(
        path for path in kb_dir.rglob("*.md") if path.is_file() and path.name not in INDEX_EXCLUDE
    )


def load_corpus(kb_dir: Path = DEFAULT_KB_DIR, counter: TokenCounter | None = None) -> list[Chunk]:
    """Read markdown from `kb_dir` and split it into paragraph chunks.

    Chunk ids must be stable across runs so traces stay comparable when the corpus is
    unchanged.
    """
    kb_dir = Path(kb_dir)
    count = counter if counter is not None else default_embedder().count_tokens
    chunks: list[Chunk] = []
    for path in corpus_files(kb_dir):
        source_file = path.relative_to(kb_dir).as_posix()
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, source_file, count))
    return chunks


# --------------------------------------------------------------------------------------
# Index persistence
# --------------------------------------------------------------------------------------


def _file_stamp(path: Path, kb_dir: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(kb_dir).as_posix(),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "sha256": sha256_text(path.read_text(encoding="utf-8")),
    }


def _save_matrix(path: Path, matrix: np.ndarray, corpus_digest: str) -> None:
    """Write the embedding matrix, tagged with the corpus digest it was built from.

    Written through an open handle because `np.savez_compressed` silently appends `.npz` to
    a path that lacks it, which would defeat the temp-then-rename write below.

    The digest is stored inside the matrix as well as in the sidecar so the two can be
    checked against each other: if a write is interrupted between the two files, the
    mismatch is caught and the index rebuilds rather than pairing one corpus's vectors with
    another's chunk text.
    """
    with path.open("wb") as handle:
        np.savez_compressed(handle, embeddings=matrix, corpus_sha256=np.array(corpus_digest))


def _write_atomic(path: Path, write: Callable[[Path], Any]) -> None:
    """Write via a temporary sibling then rename, so no reader sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class KnowledgeBase:
    """Paragraph chunks plus their embedding matrix, searched by brute force."""

    def __init__(
        self,
        kb_dir: Path = DEFAULT_KB_DIR,
        cache_dir: Path | None = None,
        *,
        embedder: Embedder | None = None,
        auto_build: bool = True,
    ) -> None:
        """Load the index from disk, building it only if it is stale or missing.

        Args:
            kb_dir: Corpus directory.
            cache_dir: Where the index lives. Defaults to `kb_dir`, so the index sits
                beside the corpus it describes.
            embedder: Injection seam. Defaults to MiniLM, which loads no model until a
                build actually needs one.
            auto_build: Build on a stale index. False leaves the instance empty, which is
                what `--check` wants: report staleness without paying to fix it.
        """
        self.kb_dir = Path(kb_dir)
        self.index_dir = Path(cache_dir) if cache_dir is not None else self.kb_dir
        self.embedder = embedder if embedder is not None else default_embedder()
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._loaded = False

        if not self.load() and auto_build:
            self.build()

    # -- paths -------------------------------------------------------------------------

    @property
    def matrix_path(self) -> Path:
        return self.index_dir / INDEX_MATRIX_NAME

    @property
    def sidecar_path(self) -> Path:
        return self.index_dir / INDEX_SIDECAR_NAME

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def embeddings(self) -> np.ndarray:
        return self._matrix

    # -- staleness ---------------------------------------------------------------------

    def _read_sidecar(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _compare_files(self, sidecar: dict[str, Any]) -> tuple[str | None, bool]:
        """Return `(staleness_reason, needs_restamp)` for the corpus against the sidecar.

        mtime and size are the fast path. When they move, the file is re-hashed before
        anything is rebuilt: `git clone` and `git checkout` rewrite every mtime without
        changing a byte, and re-embedding a corpus because it was checked out would make
        the cache useless exactly when it is most wanted.
        """
        recorded = {entry["path"]: entry for entry in sidecar.get("files", [])}
        present = {
            path.relative_to(self.kb_dir).as_posix(): path for path in corpus_files(self.kb_dir)
        }

        if set(recorded) != set(present):
            added = sorted(set(present) - set(recorded))
            removed = sorted(set(recorded) - set(present))
            details = []
            if added:
                details.append(f"added {added}")
            if removed:
                details.append(f"removed {removed}")
            return f"corpus file set changed ({'; '.join(details)})", False

        needs_restamp = False
        for name, path in present.items():
            entry = recorded[name]
            stat = path.stat()
            if stat.st_mtime_ns == entry.get("mtime_ns") and stat.st_size == entry.get("size"):
                continue
            digest = sha256_text(path.read_text(encoding="utf-8"))
            if digest != entry.get("sha256"):
                return f"{name} changed", False
            needs_restamp = True

        return None, needs_restamp

    def _evaluate(self, sidecar: dict[str, Any] | None) -> tuple[str | None, bool]:
        """Return `(staleness_reason, needs_restamp)` for a sidecar.

        Shared by `staleness_reason` and `load` so a single call hashes each changed file at
        most once.
        """
        if sidecar is None:
            return "no index on disk", False
        if sidecar.get("index_version") != INDEX_VERSION:
            return f"index_version {sidecar.get('index_version')!r} != {INDEX_VERSION}", False
        if sidecar.get("embedding_model") != self.embedder.model_name:
            return (
                f"embedded with {sidecar.get('embedding_model')!r}, "
                f"now using {self.embedder.model_name!r}",
                False,
            )
        if (sidecar.get("min_tokens"), sidecar.get("max_tokens")) != (MIN_TOKENS, MAX_TOKENS):
            return "token band changed", False
        if not self.matrix_path.exists():
            return "embedding matrix is missing", False
        return self._compare_files(sidecar)

    def staleness_reason(self) -> str | None:
        """Explain why the on-disk index cannot be used, or None if it can.

        Read-only: it never builds and never loads a model, so `--check` is cheap enough to
        run before every graded eval.
        """
        return self._evaluate(self._read_sidecar())[0]

    def is_stale(self) -> bool:
        return self.staleness_reason() is not None

    # -- load and build ----------------------------------------------------------------

    def load(self) -> bool:
        """Populate from disk if the index is usable. Returns False if a build is needed."""
        sidecar = self._read_sidecar()
        reason, needs_restamp = self._evaluate(sidecar)
        if reason is not None or sidecar is None:
            return False

        chunks = [Chunk.from_dict(record) for record in sidecar.get("chunks", [])]
        try:
            with np.load(self.matrix_path, allow_pickle=False) as payload:
                matrix = np.asarray(payload["embeddings"], dtype=np.float32)
                stored_digest = payload["corpus_sha256"].item()
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            # A damaged index should cost one rebuild, not abort the run that found it.
            logger.warning("unreadable embedding matrix (%s); rebuilding", exc)
            return False

        if matrix.shape[0] != len(chunks):
            logger.warning(
                "index desync: %d vectors for %d chunks; rebuilding", matrix.shape[0], len(chunks)
            )
            return False
        if stored_digest != sidecar.get("corpus_sha256"):
            logger.warning("matrix and sidecar describe different corpora; rebuilding")
            return False

        self._chunks = chunks
        self._matrix = matrix
        self._loaded = True

        if needs_restamp:
            # Same bytes, new mtimes: record the new stamps so the next open takes the
            # fast path instead of re-hashing the corpus again.
            sidecar["files"] = [_file_stamp(p, self.kb_dir) for p in corpus_files(self.kb_dir)]
            self._write_sidecar(sidecar)
        return True

    def build(self) -> None:
        """Chunk the corpus, embed it, and persist both index files."""
        chunks = load_corpus(self.kb_dir, self.embedder.count_tokens)
        if chunks:
            matrix = l2_normalise(self.embedder.encode([c.text for c in chunks]))
        else:
            logger.warning("no markdown to index under %s", self.kb_dir)
            matrix = np.zeros((0, 0), dtype=np.float32)

        digest = self.corpus_fingerprint() or ""
        self._chunks = chunks
        self._matrix = matrix
        self._loaded = True

        _write_atomic(self.matrix_path, lambda tmp: _save_matrix(tmp, matrix, digest))
        self._write_sidecar(
            {
                "index_version": INDEX_VERSION,
                "embedding_model": self.embedder.model_name,
                "embedder_version": getattr(self.embedder, "version", None),
                "dim": int(matrix.shape[1]) if matrix.size else 0,
                "min_tokens": MIN_TOKENS,
                "max_tokens": MAX_TOKENS,
                "corpus_sha256": digest,
                "built_at": utc_now_iso(),
                "files": [_file_stamp(p, self.kb_dir) for p in corpus_files(self.kb_dir)],
                "chunks": [chunk.to_dict() for chunk in chunks],
            }
        )

    def _write_sidecar(self, sidecar: dict[str, Any]) -> None:
        _write_atomic(
            self.sidecar_path,
            lambda tmp: tmp.write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            ),
        )

    # -- search ------------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[Hit]:
        """Return the `top_k` chunks by cosine similarity: one dot product, then sort.

        Ties break on `chunk_id` so results are deterministic. Scores below `min_score` are
        dropped, so an empty list is a real answer — the corpus has nothing relevant — and
        the caller may turn that into a refusal rather than a guess.
        """
        if not self._chunks or self._matrix.size == 0 or top_k <= 0:
            return []

        query_vector = l2_normalise(self.embedder.encode([query]))[0]
        scores = self._matrix @ query_vector

        order = sorted(
            range(len(self._chunks)),
            key=lambda i: (-float(scores[i]), self._chunks[i].chunk_id),
        )
        hits = [
            Hit(chunk=self._chunks[i], score=float(scores[i]))
            for i in order
            if float(scores[i]) >= min_score
        ]
        return hits[:top_k]

    def corpus_fingerprint(self) -> str | None:
        """Content hash of the corpus, recorded in the run manifest."""
        return sha256_of_paths(corpus_files(self.kb_dir), root=self.kb_dir)

    def stats(self) -> dict[str, Any]:
        """Summarise the index for the build script."""
        token_counts = [chunk.token_count for chunk in self._chunks]
        files = sorted({chunk.source_file for chunk in self._chunks})
        skipped = sorted(
            path.relative_to(self.kb_dir).as_posix()
            for path in self.kb_dir.rglob("*.md")
            if path.is_file() and path.name in INDEX_EXCLUDE
        )
        return {
            "kb_dir": str(self.kb_dir),
            "files": files,
            "skipped": skipped,
            "chunks": len(self._chunks),
            "dim": int(self._matrix.shape[1]) if self._matrix.size else 0,
            "tokens_min": min(token_counts) if token_counts else 0,
            "tokens_median": int(statistics.median(token_counts)) if token_counts else 0,
            "tokens_max": max(token_counts) if token_counts else 0,
            "corpus_sha256": self.corpus_fingerprint(),
            "embedding_model": self.embedder.model_name,
        }


# --------------------------------------------------------------------------------------
# Tool entry point
# --------------------------------------------------------------------------------------

_KB_CACHE: dict[tuple[str, str], KnowledgeBase] = {}


def default_kb(kb_dir: Path = DEFAULT_KB_DIR) -> KnowledgeBase:
    """Return a process-wide `KnowledgeBase`, so the tool does not reload it per call."""
    key = (str(kb_dir), EMBEDDING_MODEL)
    if key not in _KB_CACHE:
        _KB_CACHE[key] = KnowledgeBase(kb_dir)
    return _KB_CACHE[key]


def reset_default_kb() -> None:
    """Drop the cached instance. For tests, and after rebuilding the index in-process."""
    _KB_CACHE.clear()


def lookup_kb(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    *,
    kb_dir: Path = DEFAULT_KB_DIR,
) -> list[Hit]:
    """Tool entry point: search the KB and return chunks with sources and scores.

    Args:
        query: Natural-language query supplied by the agent.
        top_k: Number of chunks to return.
        min_score: Cosine floor. Hits below it are dropped, so an empty result means the
            corpus had nothing close enough rather than that retrieval failed.
        kb_dir: Corpus directory, for tests and alternate corpora.

    Returns:
        Hits ordered by descending score. `Hit.to_dict()` renders one for the prompt, and
        each carries the `chunk_id` the agent must cite so the answer stays traceable.
    """
    return default_kb(kb_dir).search(query, top_k=top_k, min_score=min_score)


def format_citation(chunk_id: str) -> str:
    """Render a citation token for `chunk_id`."""
    return CITATION_FORMAT.format(chunk_id=chunk_id)


def parse_citations(text: str) -> list[str]:
    """Extract cited chunk ids from a model answer, in order, without duplicates.

    The counterpart to `format_citation`, and the input to citation-accuracy scoring in
    `evals/deterministic.py`.
    """
    seen: dict[str, None] = {}
    for match in CITATION_RE.finditer(text):
        seen.setdefault(match.group(1), None)
    return list(seen)


name = "lookup_kb"
description = "Search the internal knowledge base for relevant passages."
schema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "What to look for, phrased as a question or keywords.",
        },
        "top_k": {
            "type": "integer",
            "description": f"How many passages to return (default {DEFAULT_TOP_K}).",
            "minimum": 1,
            "maximum": 10,
        },
    },
    "required": ["query"],
}


# --------------------------------------------------------------------------------------
# Build script
# --------------------------------------------------------------------------------------


def _format_stats(stats: dict[str, Any]) -> str:
    lines = [
        f"kb dir:      {stats['kb_dir']}",
        f"indexed:     {len(stats['files'])} files, {stats['chunks']} chunks",
        f"skipped:     {', '.join(stats['skipped']) or 'none'}",
        f"model:       {stats['embedding_model']} (dim {stats['dim']})",
        f"tokens:      min {stats['tokens_min']}, median {stats['tokens_median']}, "
        f"max {stats['tokens_max']} (merge to {MIN_TOKENS}, ceiling {MAX_TOKENS})",
        f"corpus hash: {stats['corpus_sha256']}",
    ]
    if stats["chunks"] and stats["tokens_max"] < MIN_TOKENS:
        # Not a fault: merging stops at headings, so a corpus of short sections yields one
        # chunk per section. Worth saying out loud, since the merge target is never reached.
        lines.append(
            f"note:        every section is under {MIN_TOKENS} tokens, so each section is "
            "its own chunk; merging never crosses a heading"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the KB index: `agentseval-index [--rebuild|--check|--stats]`.

    Default behaviour builds only when the index is stale, so running it is always safe.
    """
    parser = argparse.ArgumentParser(
        prog="agentseval-index",
        description="Build the kb/ retrieval index (chunks + embeddings).",
    )
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR, help="corpus directory")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if fresh")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the index is stale, without building it; run before a graded eval",
    )
    parser.add_argument("--stats", action="store_true", help="print index statistics")
    args = parser.parse_args(argv)

    # Our own messages at INFO, everyone else at WARNING: sentence-transformers and
    # huggingface_hub log every HTTP request at INFO, which buries the build output.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)

    if args.check:
        kb = KnowledgeBase(args.kb_dir, auto_build=False)
        reason = kb.staleness_reason()
        if reason:
            print(f"index is stale: {reason}", file=sys.stderr)
            return 1
        print("index is up to date")
        return 0

    kb = KnowledgeBase(args.kb_dir, auto_build=False)
    reason = kb.staleness_reason()
    if args.rebuild or reason:
        print(f"building index ({reason or 'forced rebuild'})")
        kb.build()
    else:
        kb.load()
        print("index is up to date")

    if args.stats or args.rebuild or reason:
        print(_format_stats(kb.stats()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
