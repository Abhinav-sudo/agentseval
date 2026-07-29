"""Fixture corpora: alternate knowledge bases that exist to be measured against.

Only one so far, `injected/`, which holds the poisoned document behind the retrieval-borne
prompt-injection dataset. Nothing here is part of `kb/`, and nothing here is read by a run
unless that run was pointed at a composed fixture directory on purpose.

`compose.py` builds those directories. See PROJECT.md § "The injected fixture corpus".
"""
