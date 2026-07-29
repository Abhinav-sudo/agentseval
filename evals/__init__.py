"""The evaluation platform — the deliverable of this project (PROJECT.md).

The agent and the chat app exist to be measured; this package does the measuring.

* `runner`         executes eval sets against an agent and writes JSONL traces.
* `judge`          scores arbitrary (prompt, response) pairs, including pairs from an
                   external file supplied by a grader that our agents never produced.
* `validate_judge` measures judge-vs-human agreement, so judge scores can be trusted.
* `deterministic`  rule-based checks that need no model.
* `metrics`        aggregation across records and runs.
* `report`         human-readable output.
"""
