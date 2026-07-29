"""Test suite.

`test_manifest.py` covers `agent.manifest`, weighted toward `assert_comparable` — the guard
that two runs differed only in model weights — and toward `build_manifest` being one builder
for chat and eval runs. `test_trace.py` covers the JSONL writer that sits under it, weighted
toward records reaching disk before a run ends and never being dropped. `test_session.py`
covers the chat run lifecycle: a manifest written before the turn it describes, and a changed
condition minting a new run rather than continuing one under a manifest that no longer
describes it. `test_app.py` renders the chat surface headlessly and skips unless the optional
[app] extra is installed. `test_models.py` covers the three adapters with the
HTTP layer mocked, weighted toward the no-native-tool-calling guard, the shared retry policy,
and the cache key. `test_lookup_kb.py` covers chunking, the embedding index, and search, with
a TF-IDF stand-in for MiniLM so the suite stays offline and torch-free; the one test that
needs the real model is opt-in. `test_prompts.py` covers the shared prompt text, weighted
toward the properties that make it a constant across arms rather than toward wording.
`test_schema.py` covers the one definition of an eval item, weighted toward a wrong item being
rejected rather than loaded with a field quietly unset — the failure a dataset actually has.
`test_validate_dataset.py` covers the linter, weighted toward every diagnostic reporting the
right line and the right code, since a finding without a location is one an author works around
by guessing. `test_label.py` covers the labeling helper, weighted toward the properties that
protect work nothing can regenerate: the dataset's bytes unchanged by a session, corrections
appended rather than edited, and counterfactual variants never presented close together.
`test_compose.py` covers the fixture-corpus builder behind the injection dataset, weighted
toward the ways a composed corpus goes wrong without failing — a shadowed document, a stale
one left behind, a digest that moves between builds — and toward the poisoned document
chunking to the ordinals that dataset cites.

The tests that matter most are the ones guarding locked decisions in PROJECT.md, because
those are the failures that silently invalidate results rather than raising:

* tool-call parsing is identical for both agents, including on malformed JSON;
* both agents receive byte-identical prompt text, and every JSON example in it parses;
* no adapter passes native function-calling parameters to a provider;
* the judge scores an external (prompt, response) file with no reference to our agents,
  trace format, or knowledge base;
* the judge's family differs from both agents';
* retrieval is deterministic for a fixed corpus, and cosine search is brute force;
* chunk ids are stable and unique, since citation accuracy is scored against them;
* no chunk exceeds the embedding model's window, which would truncate it invisibly;
* every turn produces a JSONL record under a run manifest, interactive turns included, and
  the manifest is written before the turn rather than after it;
* a mid-session change of conditions starts a new run instead of leaving a trace that holds
  two models under a manifest asserting one;
* a cache hit is marked as a replay rather than passing as a fresh measurement;
* there is exactly one eval-item shape, an unknown field in a dataset is an error rather than a
  silently ignored key, and a bias item without a counterfactual pair is refused;
* no tool rewrites a dataset file, since the manifest digests its bytes and a tidy-up after one
  arm has run would void the comparison without failing.
"""
