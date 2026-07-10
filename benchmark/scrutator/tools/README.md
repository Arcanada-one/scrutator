# Golden-set growth tooling

Candidate-generation + two-LLM-agreement-filter tooling for growing the golden set
(v0 → v0.1 → v0.2 → v1, per the plan's Fork A 3-batch cadence). **Code only** — this task
does not run these tools to produce v0.1/v0.2/v1 data, and does not fabricate labels.

## What this does

`generate_candidates.py` implements the PRD's Fork 1 governance mechanism:

1. **Pass 1** — an LLM reads a source document and proposes candidate questions
   (`class` + `answer` + `gold_source_paths`).
2. **Pass 2** — a second, independent LLM call re-answers each candidate's question from
   the same document, blind to pass 1's stated answer.
3. **Judge pass** — a third LLM call decides whether pass 1's and pass 2's answers
   materially agree. Disagreement ⇒ the candidate is rejected before a human ever sees it.

Output is a `*-candidates.jsonl` file — **not** a golden set. Every row's `review_status` is
`"candidate"`, never `"gold"`.

## What this does NOT do (human-annotation gate)

Per the PRD's Fork 1 ("the two-LLM-pass disagreement filter is the stand-in for a second
human reviewer, not a replacement for the one human review that matters"), promoting a
candidate to `gold` requires **one human (operator) verification pass per row**: confirm the
document says what the label claims, confirm the path is still live, then promote. This is
not automatable and this task does not attempt it. `golden-arcanada-v0.jsonl` remains the
only shipped golden set after this task; `v0.1`/`v0.2`/`v1` are not created here.

## Invocation (once the operator authorizes running a growth batch)

```bash
# Explicit --connector required — no silent Model Connector provider-default fallback.
python benchmark/scrutator/tools/generate_candidates.py \
    --docs-file batch1-doc-list.txt \
    --corpus-root /home/dev/arcanada \
    --connector claude-code --model haiku \
    --api-key "$MC_API_KEY" \
    --n-per-doc 4 \
    --out benchmark/scrutator/golden/batch1-candidates.jsonl
```

Then, manually (operator, not this tooling):

1. Read `batch1-candidates.jsonl`. For each row: confirm the document actually says what
   `answer` claims, confirm `gold_source_paths` still exist, and either promote the row
   (append to the next `golden-arcanada-v0.N.jsonl`, `review_status: "gold"`, set
   `corpus_pinned_at` to today) or discard it.
2. Append one line per batch to `golden/review-log.md` documenting the promotion decision.
3. Repeat for batch 2 and batch 3 (Fork A) until the cumulative set lands in [100, 150]
   rows (V-AC-01).
