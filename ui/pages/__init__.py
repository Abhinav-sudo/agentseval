"""One script per view, discovered by Streamlit as the entry script's `pages/` directory.

Streamlit's page discovery skips `__init__.py`, so this file makes the directory a package for
`mypy` and the import-graph test without becoming a page. The filenames carry no leading digit for
the same reason: `1_Runs.py` is not a valid module name, and these are type-checked alongside the
rest of the project rather than exempted from it.
"""
