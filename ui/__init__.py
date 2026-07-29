"""Read-only web views over `runs/`.

A third layer above `agent/` and `evals/`: this package imports both and is imported by neither,
so a test can walk the import graph and say so (PROJECT.md, "`ui/` is a view"). Nothing here
computes a number of its own — `evals.metrics` computes and `evals.report` shapes, and a view that
did its own arithmetic would be a second answer to the same question.

Nothing here writes. `metrics.summarise_run` joins the trace, the judgements, and the dataset in
memory every time it is asked, because a second copy of a run is a second thing to keep truthful.
"""
