# Knowledge base

Markdown corpus retrieved by `agent/tools/lookup_kb.py`. Drop `.md` files here.

The contents are twelve synthetic documents on general consumer wellness, written for this
project. They match the assistant's role in `agent/prompts.py`, and they are deliberately
bounded — see "Not covered" below, which is load-bearing for the eval rather than an apology
for missing content.

## Files

* `sleep-hygiene.md` — sleep need, consistent wake time, light and bedroom conditions,
  wind-down routines, caffeine and alcohol in the evening.
* `hydration.md` — how much fluid people need, the eight-glasses myth, practical signs of
  adequate hydration, drinking around exercise and in heat, what counts as fluid.
* `nutrition-principles.md` — the plate model, whole and minimally processed foods, fibre and
  plant variety, protein across meals, consistency over perfection.
* `strength-training-basics.md` — why strength work matters, training frequency, movement
  patterns, sets and repetitions, progressive overload, form and starting weight.
* `cardio-basics.md` — weekly activity guidance, judging intensity, building up gradually,
  steady versus harder efforts, symptoms that mean stop.
* `stress-management.md` — how ongoing stress presents, paced breathing, movement and sleep,
  structure and boundaries, social connection and when to seek help.
* `recovery-and-rest-days.md` — why adaptation happens during recovery, ordinary muscle
  soreness, rest and active recovery, sleep and food, signs of too much load.
* `desk-ergonomics.md` — chair and seated position, monitor placement, keyboard and mouse,
  laptops and improvised setups, movement breaks and eye strain.
* `walking-and-daily-movement.md` — why incidental movement counts, the ten-thousand-step
  figure, breaking up sitting, making walking a habit, pace and terrain.
* `warm-up-and-mobility.md` — what a warm-up does, a general warm-up structure, dynamic versus
  static stretching, mobility and range of motion, cooling down.
* `sustainable-habits.md` — starting small, anchoring to existing routines, the twenty-one-day
  myth, handling missed days, tracking and adding habits one at a time.
* `reading-nutrition-labels.md` — serving sizes, percent Daily Value, the ingredients list,
  sugars and sodium and fats, front-of-pack claims.

## Not covered

The corpus deliberately says nothing about:

* medications or supplements
* medical conditions, symptoms as diagnosis, or interpreting test results
* pregnancy
* pediatric or infant health
* rehabilitation of an injury
* treatment of mental-health conditions

This is a design constraint, not a backlog. The hallucination axis of the eval needs questions
the knowledge base genuinely cannot answer, so that an agent which answers them anyway — from
its own training rather than from a retrieved source — is measurably distinguishable from one
that says the corpus does not cover it. Letting coverage creep into these areas would quietly
delete that axis, and the deletion would look like an improvement, because retrieval scores
would rise.

Where these topics touch the corpus at all, they appear only as boundaries: a sentence saying
that persistent pain, disordered eating, or a possible emergency is a clinician's question. That
is the correct amount. A file that starts explaining what to do about an injury has broken the
constraint even if every sentence in it is true.

The corpus is also not itself a source of unsafe advice. It contains no dosages, no calorie or
macronutrient targets, no fasting protocols, and no diagnostic or treatment claims. The safety
axis tests the agent's judgement, so the corpus must not hand it a bad answer to retrieve.

## Filenames are citation ids

A chunk id is `{filename}#{ordinal}`, and those ids are what the agent cites and what
`evals/deterministic.py` scores citations against. Renaming a file or reordering its sections
changes the ids, which invalidates any labelled dataset that referenced them. Names and section
order are therefore frozen once an eval set is labelled; add a new file rather than
restructuring an existing one.

**This file is not indexed.** `INDEX_EXCLUDE` in `lookup_kb.py` skips `README.md`, because it
documents the corpus rather than being part of it — otherwise prose about chunking competes
with the corpus for retrieval slots. The build script prints what it skipped.

## Indexing

Indexing (locked in PROJECT.md): documents are split into **paragraph chunks**, embedded with
**all-MiniLM-L6-v2**, and searched by **cosine similarity over a numpy array**. No FAISS or
Chroma — the corpus is tiny, so brute force is both faster and easier to audit.

Adjacent paragraphs are merged up to 256 tokens, which is the embedding model's input window;
anything longer would be shown to the agent in full but embedded only up to that point.
Merging never crosses a heading, and a paragraph over the limit is split at sentence
boundaries.

Because merging stops at headings, a section shorter than the 200-token merge target stays a
chunk on its own. The corpus is written to exploit this: each `##` section holds two paragraphs
that merge into a single chunk, so every section is exactly one chunk and one citable id. The
current index is 12 files and 72 chunks — the six chunks per file being an introduction plus
five sections — with chunk sizes from 93 to 222 tokens and a median of 195.
`agentseval-index --stats` reports the actual distribution.

## Notes for authors

* Separate paragraphs with a blank line; the paragraph is the retrieval unit, so one idea
  per paragraph retrieves better than one long block.
* Two paragraphs of roughly 90 to 110 words per `##` section keeps the merged chunk just under
  the ceiling. A third paragraph in a section usually spills into a second, smaller chunk.
* A paragraph should stand alone. Retrieved chunks reach the model without their
  surrounding document, so a paragraph that only makes sense after the one above it will
  read as a non-answer.
* Prefer specific, checkable facts. "Five percent Daily Value is low and twenty percent is
  high" can be scored; "read the label carefully" cannot. The false-premise questions in the
  eval need concrete claims to misquote.
* No tables and no code blocks. Both survive chunking badly and neither belongs in this corpus.
* Headings are retrieval context, not chunk content: a chunk carries its heading path, so
  `## Paced breathing` under `# Everyday Stress Management` tells the model where the text
  came from.
* Editing the corpus changes its fingerprint, which is recorded in every run manifest.
  Runs against different corpus revisions are not comparable. Run `agentseval-index` after
  editing, or the next run will rebuild the index itself.
