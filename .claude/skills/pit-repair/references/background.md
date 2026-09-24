# Where the pit-repair protocol comes from

This skill repackages a loop validated by a differential-execution harness
plus a repair loop, run against the Olist e-commerce corpus. The numbers
below back the caveats in `SKILL.md`; full write-ups live in
`docs/results.md`. `scripts/checker_core.py` is a standalone
implementation inside this skill, with no dependency on the path to any
external repository or directory such as `prestudy/`.

| Measurement | What | Result |
|---|---|---|
| Direct-invocation repair loop | fixed repair instruction, `z-ai/glm-5.3-flash`, up to 5 iterations, 76 programs | CLEAN@5 = 75/76 = 98.7%; CLEAN@1 = 65/76 |
| Repeatability of the repair loop | same loop, 3 independent repeats, 62 programs | CLEAN@5 = 182/186 = 97.8%, cluster bootstrap 95% CI [95.2%, 100%] |
| Code-reading detection | model reads the code and reports a verdict, same fixed instruction | recall 97.3%, false-positive rate 12.0% |
| Prediction-time density and canary false positives | how many tested moments are needed, and how often canary fires on clean code | witness false-positive rate 0/30; canary false-positive rate 4/30 = 13.3% |
| Self-check vs. named error class | "check your own code" vs. being told the error class | self-check 1/12 CLEAN, named class 11/12 |

## What `pit_check.py` actually checks

The numbers in the table above were measured on the Olist availability map
(`references/example-availability-map.olist.json`, the `time_cols`/
`self_availability_cols`/`gatekeeper_cols` values originally worked out
while developing the protocol). The code itself
(`scripts/checker_core.py`) does not bake the map in -- it's an input
parameter, `--availability-map`; below is what that map meant on the
corpus this was validated against:

- `checker_core.truncate(db, t, time_cols)`: keeps rows with time ≤ t per
  the `time_cols` columns (in the example -- `orders.order_purchase_timestamp`,
  `order_items.ts`, `reviews.review_creation_date`, `payments.ts`).
- `checker_core.perturb_canary(db, t, time_cols, ...)`: in rows with time
  > t, replaces values with an opaque sentinel (numbers +7919, dates +10
  years past the horizon, categories → `__canary_sentinel__`); in
  self-availability channels (in the example -- delivery dates, review
  response) replaces values not yet available at t.
- `checker_core.frames_equal`: per-column comparison with 1e-9 tolerance
  and NaN==NaN.
- Entities: 15 products (`--n-entities`), `RandomState(0)` via
  `sample_entities(db, entity_table, entity_column, n)`; 30s timeout per
  call; sandbox: only `pandas`/`numpy`, file/network/system calls
  forbidden.

## The one holdout in the direct-invocation run

One program (author model `deepseek-v4-flash-0731`, program uid
`f1cade690ca333e8`): after five repair iterations the witness level stays
silent, but canary consistently flags a delivery-recency feature at every
one of the six tested prediction times.

## Where the model's code reading disagreed with the map

Measured while validating code-reading detection against the same map:

- a mutable order-status field (4 cases);
- delayed payment availability (3 cases);
- a review-response timestamp read where the map declares the
  review-creation timestamp (2 cases).

## Safety discipline

In the underlying research protocol, the network step (getting the model's
response) and the step that executes the generated code were separate
processes; the provider key was removed from the environment before
execution. `pit_check.py` reproduces the second half of that rule: it
refuses to start if a provider key is present in the environment.
