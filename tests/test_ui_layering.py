"""The layering around `ui/`, asserted rather than remembered.

Modelled on `test_guardrails.py::test_no_agent_module_imports_anything_from_evals`, and here for
the same kind of reason. `ui/` is a view: it imports `agent/` and `evals/` and nothing imports it.
The moment something under `evals/` imports a page — to reuse a formatter, most likely — the
platform's numbers depend on a rendering layer, `pytest` starts needing Streamlit installed, and a
figure could differ between the terminal and the browser because one of them went through a widget.

Deliberately free of `pytest.importorskip`: this file must run, and fail, in a suite with no
Streamlit installed. A layering test that skipped itself in the environment where the layering
matters most would be worse than no test, because it would report a pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Everything that must not import `ui`: the two library layers, plus the chat demo. `app.py` is on
#: the list because PROJECT.md's claim that it is a demo surface and not the product is only true
#: while it knows nothing about the eval views.
LOWER_LAYERS: tuple[Path, ...] = (
    *sorted((REPO_ROOT / "agent").rglob("*.py")),
    *sorted((REPO_ROOT / "evals").rglob("*.py")),
    REPO_ROOT / "app.py",
)

#: `ui/` may import these and nothing else of ours. Listed so that a page reaching into `tests/`,
#: or a future `scripts/`, is a failure rather than a surprise.
ALLOWED_FIRST_PARTY: frozenset[str] = frozenset({"agent", "evals", "ui"})


def imported_modules(path: Path) -> set[str]:
    """Every module name `path` imports, from its AST rather than from its text.

    An AST walk rather than a regex over the source, so that `ui` inside a docstring or a comment —
    this module's own docstrings mention it repeatedly — is not mistaken for an import.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and not node.level:
            names.add(node.module)
    return names


def first_party(names: set[str]) -> set[str]:
    """The top-level package of each import that is one of ours."""
    tops = {name.split(".")[0] for name in names}
    return {top for top in tops if (REPO_ROOT / top / "__init__.py").exists()}


@pytest.mark.parametrize(
    "path", LOWER_LAYERS, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_nothing_below_the_view_imports_it(path: Path) -> None:
    """`ui/` is imported by neither library layer nor by the chat demo."""
    offending = {name for name in imported_modules(path) if name.split(".")[0] == "ui"}

    assert offending == set(), (
        f"{path.relative_to(REPO_ROOT)} imports {sorted(offending)}. ui/ is a view: it imports "
        "agent/ and evals/ and is imported by neither. An import in this direction makes the "
        "platform's numbers depend on a rendering layer, and makes Streamlit a requirement for "
        "computing them"
    )


@pytest.mark.parametrize(
    "path",
    sorted((REPO_ROOT / "ui").rglob("*.py")),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_the_view_imports_only_the_layers_below_it(path: Path) -> None:
    """A page reads `agent/` and `evals/`. Anything else of ours is not a layer below it."""
    imported = first_party(imported_modules(path))

    assert imported <= ALLOWED_FIRST_PARTY, (
        f"{path.relative_to(REPO_ROOT)} imports {sorted(imported - ALLOWED_FIRST_PARTY)}, which "
        f"is outside {sorted(ALLOWED_FIRST_PARTY)}"
    )


def test_the_pages_directory_holds_importable_module_names() -> None:
    """`1_Runs.py` is a valid Streamlit page and not a valid module, so `mypy` could not read it."""
    for path in sorted((REPO_ROOT / "ui" / "pages").glob("*.py")):
        assert path.stem.isidentifier(), (
            f"{path.name} is not an importable module name, so it is invisible to mypy and to "
            "the import-graph tests in this file"
        )
