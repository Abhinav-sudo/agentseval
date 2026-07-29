"""Suite-wide isolation: no test reads the developer's `.env`.

`base.load_env` walks up from the working directory, which during a test run is the repository
root. Without this fixture every test that calls a CLI `main` would load real credentials into
the process, and each later test would see them — so a suite that passes on a machine with no
`.env` could fail on one that has it, or worse, pass for the wrong reason. `test_models.py`
asserts a missing key raises; that assertion must not depend on whose laptop it runs on.

Discovery is neutralised at `find_dotenv` rather than at `load_env`, because each entry point
imported `load_env` by value and patching the name in `base` would not reach them. It is done
here rather than in each CLI test so that a new entry point calling `load_env` is isolated by
default instead of by someone remembering.

The tests covering `load_env` itself opt back in; see the `env` fixture in `test_models.py`.
"""

from __future__ import annotations

import pytest

from agent.models import base


@pytest.fixture(autouse=True)
def no_ambient_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `.env` discovery find nothing, for every test."""
    monkeypatch.setattr(base, "find_dotenv", lambda **_: "")
