---
name: pit-repair
description: Detect and repair temporal (point-in-time) data leakage in a feature function get_features(db, entity_ids, seed_time) by differential execution (witness + canary), then re-verify each repair on held-out cutoffs. Works on any relational database given a hand-written availability map -- not tied to one dataset. Use this whenever a feature-engineering function over event tables must only use information available at seed_time, when someone asks to check code for future leakage / look-ahead / point-in-time correctness, or to fix such a leak. Also use it when the request only asks to check features for temporal correctness without naming leakage explicitly.
---

# pit-repair: detect and repair leakage from the future

This skill packages a validated loop: differential execution finds a leak,
the model repairs the code against a fixed feedback instruction, execution
confirms the repair on held-out prediction times. None of this is replaced
by reading the code by eye: without an executable check, a model finds a
leak in only 1 of 12 programs on its own; given the named error class, it
repairs 11 of 12.

The skill isn't tied to one database: which column of which table carries
time, and which column becomes available only later, is something you
declare through an availability map (`--availability-map`), not something
baked into the code. Map format and how to write one:
`references/availability-map.md`. The validation numbers below were
measured on the Olist e-commerce corpus
(`references/example-availability-map.olist.json` is that same map, kept as
a worked example) -- they support the protocol, they don't restrict which
databases the skill can run on.

## Interface of the checked function

```python
def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame
```

`db`: a dict of pandas tables, one per `*.csv` in `--db-dir` (table name =
filename without extension); `entity_ids`: a list of entities; `seed_time`:
a `pandas.Timestamp`. The result is indexed by `entity_ids`. Only `pd` and
`np` are available. If the real code is structured differently, wrap it in
a function with this signature; the wrapper may reorder calls but must not
add feature logic.

## Steps

### 0. Preconditions

- Python with pandas 2.2.x and numpy 2.2.x (this project's numbers were
  produced on 2.2.3 / 2.2.6).
- The database -- a directory of `*.csv` files, one per table (`--db-dir`).
- The availability map -- a JSON file (`--availability-map`,
  `references/availability-map.md`).
- The table and column that test entities are sampled from
  (`--entity-table`, `--entity-column`) -- any id column in the database.
- The check command: `pit-check` (this skill's `bin/pit-check` wrapper; if
  it isn't on PATH, call `python <skill dir>/scripts/pit_check.py`
  directly). The only dependency is `scripts/checker_core.py` next to it --
  no external repository root is required.
- No LLM provider key may be present in the checking process's
  environment (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`). The script executes generated code and refuses to
  run if a key is present. Don't export keys and don't work around the
  refusal: the network step and the execution step must be separate
  processes.

### 1. Detection

```bash
pit-check --code get_features.py \
  --db-dir <path to a directory of *.csv tables> \
  --availability-map <path to the JSON map> \
  --entity-table <table> --entity-column <column> \
  --dev-seeds 2018-01-01,2018-04-01,2018-07-01 \
  --held-out-seeds 2017-10-01,2018-02-01,2018-06-01
```

`--dev-seeds`/`--held-out-seeds` are required and must be chosen for your
own database (a date range the data actually covers) -- there is no
universal default sensible for every database.

The script calls the function three times per prediction time: on the full
database, on the database truncated at `seed_time`
(`checker_core.truncate`), and on the database with the standard canary
perturbation of late-available fields (`checker_core.perturb_canary`) --
both functions work strictly from your map, guessing nothing from column
names beyond `key_hints`.

Two lines in the output:

- `witness`: columns that diverged between the full and truncated database.
  This is proof: the function read a row timestamped later than
  `seed_time`.
- `canary`: columns that diverged between the full and perturbed database.
  The function read a field that becomes available not at the row's own
  creation time but later (see the map's `self_availability_cols`).

The verdict is `CLEAN` only if both levels stay silent at every tested time
(dev + held-out). A `CLEAN` verdict at iteration 0 means there's nothing to
repair: report that and stop. An `ERROR` verdict means the candidate didn't
execute in the sandbox (a runtime error or a forbidden construct, e.g.
`locals()`, or file/network access): fix the cause and, in the same pass,
still run the repair step, because the leak hasn't gone anywhere.

### 2. Repair

On a `LEAK` verdict, re-read the current code and follow this instruction
verbatim, as if it were addressed to you together with the code above and
the `witness_columns`/`canary_columns` list from `pit-check`'s output:

```
The code above may leak data from the future. Requirement: features for
seed_time must be computed ONLY from information that already exists at
seed_time. Note that different columns of the same row can become available
at different times: some values only appear after the row's own time.
The check currently found leakage in these columns: {witness_columns +
canary_columns from the last pit-check output}.
Check your code against this requirement and return the corrected version of
get_features (same signature), in a single ```python ...``` block, with no
explanation.
```

Write the corrected function back into the same file (`get_features.py`),
replacing the previous version. The signature and function name stay the
same, imports stay `pandas`/`numpy` only. The structure of the instruction
above is fixed by the protocol this skill was validated under; don't
paraphrase the requirement or add guesses about specific columns beyond
what the check actually found -- comparability with the recorded numbers
depends on keeping the wording unchanged. Substituting the run's actual
`witness_columns`/`canary_columns` for a fixed dataset's example is exactly
what makes this generalize to any database, not a departure from protocol.

### 3. Re-check

Run the same command (same `--db-dir`, `--availability-map`,
`--entity-table`/`--entity-column`, `--dev-seeds`/`--held-out-seeds` as at
detection):

```bash
pit-check --code get_features.py --db-dir ... --availability-map ... \
  --entity-table ... --entity-column ... \
  --dev-seeds ... --held-out-seeds ...
```

The script tracks iteration count itself in `pit_state.json` next to the
code, and re-checks every prediction time on every call, including the
held-out ones the model never saw at detection. A candidate that fails to
execute still spends an iteration.

### 4. Loop and stopping rule

Repeat steps 2 and 3 until you get `CLEAN`, but no more than five repair
iterations. After the fifth failure the script sets status `failed` and
prints `FAILED`; at that point stop repairing and report instead. Don't
hand-edit or delete `pit_state.json` to reset the counter.

## Final report

End your work with a short report in this form (values taken from
`pit-check`'s last JSON output and from `pit_state.json`):

```
pit-repair: verdict=<CLEAN|FAILED>, clean_at=<iteration or null>,
iterations_used=<n>, witness_final=[...], canary_final=[...],
state=<path to pit_state.json>
```

## Reading disputed cases

- Witness is proof; canary is a strong signal. On programs already known to
  be clean, canary can still fire as a false positive (measured at roughly
  1 in 8 clean programs on the Olist corpus) -- a comparable rate is
  plausible on other databases if the availability map is rich. A lone
  canary hit on a program where witness stays silent is formally "not
  CLEAN" under the protocol, but the column and mechanism are worth naming
  in the report so a human can tell a false positive from a real leak.
- If the model's reading of the code disagrees with the availability map
  (on Olist: a mutable order-status field, delayed payment availability, a
  review-response timestamp read where the map declares the review-creation
  timestamp) -- repair against the map you passed to `--availability-map`:
  it is this run's protocol reference, and disputing it belongs in the
  report, not in the code.
- A known holdout on Olist: a delivery-recency feature on one program stays
  stuck on canary after all five repair iterations. The same class of
  outcome (canary that doesn't clear within the iteration budget) is
  possible on any database and isn't a defect of the skill.

Details, numbers, and where the protocol comes from (on Olist):
`references/background.md`. Availability-map format and how to write one
for a new database: `references/availability-map.md`.
