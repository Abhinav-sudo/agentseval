"""Compose a fixture corpus: the real `kb/` plus one fixture's extra documents.

The injection dataset needs a corpus containing a poisoned document, and `kb/` must not be
that corpus — see PROJECT.md § "The injected fixture corpus". This builds the alternative into
a throwaway directory that a run points `--kb-dir` at.

**Composed rather than committed.** A thirteen-file copy of `kb/` checked in beside the real
one is two hand-maintained copies of twelve documents, and they drift. The drift is invisible
too: nothing fails until an eval item cites a chunk id whose ordinal moved in one copy and not
the other, at which point the citation check reports a model error for an authoring mistake.
Copying at build time makes the shared half derived, so it cannot disagree with its source.

**The output directory is rebuilt, not updated.** A stale document left behind from a previous
composition is a file in the corpus that no source directory contains, and it would be
embedded, retrieved, and cited like any other. Since the corpus digest is a manifest field,
that also means a run's conditions would depend on what happened to be on disk beforehand.

**Filename collisions are refused.** A fixture document sharing a name with a `kb/` document
would shadow it, silently removing a document the eval sets cite while leaving the item that
cites it looking merely wrong.

The composed corpus's digest is printed because it is the value that ties a fixture run to the
text it read: `assert_comparable` will refuse a fixture run against a main one on `kb_sha256`,
and this is the number that explains why.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from agent.tools.lookup_kb import DEFAULT_KB_DIR, corpus_files
from agent.trace import sha256_of_paths

#: Fixture source directories live here, one subdirectory each.
FIXTURES_DIR = Path(__file__).resolve().parent

#: Where composed corpora are written. Gitignored: this is derived data, rebuilt on demand,
#: and checking it in would restore the duplicate-copy problem the composition exists to avoid.
DEFAULT_OUTPUT_ROOT = FIXTURES_DIR / ".composed"

#: The only fixture so far. Named so the CLI's default needs no argument for the common case.
DEFAULT_FIXTURE = "injected"

EXIT_OK = 0
EXIT_FAILED = 1


class CompositionError(RuntimeError):
    """A fixture cannot be composed as specified."""


def fixture_files(fixture_dir: Path) -> list[Path]:
    """The fixture's own documents, on the same terms as `corpus_files` reads `kb/`.

    Same helper rather than a local glob so that `INDEX_EXCLUDE` applies identically: the
    fixture's `README.md` documents the fixture and must no more be indexed than `kb/README.md`
    is. A second implementation here would be a second place for that rule to be forgotten.
    """
    return corpus_files(fixture_dir)


def compose(
    fixture: str = DEFAULT_FIXTURE,
    *,
    kb_dir: Path = DEFAULT_KB_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, str]:
    """Build `output_root/fixture` from `kb_dir` plus the fixture's documents.

    Returns the composed directory and its corpus digest — the same digest
    `build_manifest` will record as `kb_sha256` for a run pointed at it.

    Raises:
        CompositionError: the fixture directory is missing, holds no documents, or names a
            document that already exists in `kb_dir`.
    """
    fixture_dir = FIXTURES_DIR / fixture
    if not fixture_dir.is_dir():
        available = sorted(
            p.name for p in FIXTURES_DIR.iterdir() if p.is_dir() and p.name[0] not in "._"
        )
        raise CompositionError(
            f"no fixture named {fixture!r} under {FIXTURES_DIR}; available: "
            f"{', '.join(available) or '(none)'}"
        )

    base = corpus_files(kb_dir)
    if not base:
        raise CompositionError(
            f"no markdown to compose from {kb_dir}; a fixture corpus is the real corpus plus "
            "an extra document, so an empty base is a misconfiguration rather than a corpus "
            "of one file"
        )

    extra = fixture_files(fixture_dir)
    if not extra:
        raise CompositionError(
            f"fixture {fixture!r} contributes no documents (README.md is excluded from the "
            "index, so a fixture holding only one is empty for this purpose)"
        )

    collisions = sorted({p.name for p in base} & {p.name for p in extra})
    if collisions:
        raise CompositionError(
            f"fixture {fixture!r} would shadow {len(collisions)} document(s) from {kb_dir}: "
            f"{', '.join(collisions)}. Rename the fixture document; a shadowed file leaves "
            "every eval item citing it pointing at text that is no longer there"
        )

    out_dir = Path(output_root) / fixture
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for source in [*base, *extra]:
        shutil.copyfile(source, out_dir / source.name)

    digest = sha256_of_paths(corpus_files(out_dir), root=out_dir) or ""
    return out_dir, digest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `agentseval-compose-fixture [FIXTURE] [--kb-dir DIR] [--output-root DIR]`."""
    parser = argparse.ArgumentParser(
        prog="agentseval-compose-fixture",
        description="Compose a fixture corpus (kb/ plus a fixture's documents) for an eval run.",
    )
    parser.add_argument(
        "fixture",
        nargs="?",
        default=DEFAULT_FIXTURE,
        help=f"fixture directory under {FIXTURES_DIR.name}/ (default: {DEFAULT_FIXTURE})",
    )
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR, help="the base corpus")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="where composed corpora are written",
    )
    args = parser.parse_args(argv)

    try:
        out_dir, digest = compose(
            args.fixture, kb_dir=args.kb_dir, output_root=args.output_root
        )
    except CompositionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED

    files = corpus_files(out_dir)
    print(f"composed {len(files)} documents into {out_dir}")
    print(f"corpus sha256: {digest}")
    print(f"next: agentseval-index --kb-dir {out_dir}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
