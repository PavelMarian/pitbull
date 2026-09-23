# Availability map format

The JSON file you pass to `pit-check --availability-map`. The checker never
infers availability from column names or schema -- everything it knows
comes from this file. A full worked example for Olist:
`example-availability-map.olist.json`.

```json
{
  "time_cols": {"<table>": "<column>"},
  "self_availability_cols": {"<table>": ["<column>", "..."]},
  "gatekeeper_cols": {"<table>": ["<column>", "..."]},
  "key_hints": ["_id", "id"]
}
```

- **`time_cols`** (required, at least one entry) -- for every table whose
  rows "live" in time, the column that gives that time (usually the row's
  creation date). A lookup table with no time channel is simply omitted --
  it passes through `truncate`/`perturb_canary` untouched.
- **`self_availability_cols`** -- columns whose OWN availability equals
  their value, not the time of the row they sit in (the classic case is a
  delivery date: the order row itself exists from the moment of purchase,
  but the delivery-date value only appears once delivery happens). These
  columns aren't protected by truncating on row time; canary perturbs them
  in any row where the value isn't yet available at `seed_time`.
- **`gatekeeper_cols`** -- columns whose value ≤ `seed_time` controls
  whether OTHER columns with availability > `seed_time` are visible. Never
  perturb these -- doing so breaks the `D'|t = D|t` property. Empty by
  default: if you don't name one explicitly, the checker assumes your
  schema has none, rather than silently discovering them.
- **`key_hints`** -- substrings in a column name that mark it as an
  identifier/key, never perturbed by canary (default `_id`, `id`; add your
  schema's characteristic suffixes if it has any).

## Writing a map for a new database

1. List the tables that accumulate over time (not lookup tables) -- for
   each, find the column that says "this row was created at such-and-such
   time."
2. Separately find columns whose value fills in NOT at row-creation time
   but later (statuses, delivery/response/confirmation dates) -- these are
   `self_availability_cols`, even if they formally sit in a table you
   already listed under `time_cols`.
3. Check whether any column is both (a) known at `seed_time` and (b)
   determines whether ANOTHER column with availability later than
   `seed_time` is visible -- if so, perturbing it would break
   `perturb_canary`'s own invariant. If you find one, list it under
   `gatekeeper_cols` and leave it unperturbed.
4. Don't guess from column names (`_at`/`_date` doesn't mean "available
   immediately"): check the data source's own documentation for when a
   value actually becomes known, not when the row was physically created.

The discipline for ambiguous cases (a lone canary hit while witness stays
silent, disagreement with the model's own reading of the code) doesn't
depend on the database in general -- see `SKILL.md`, "Reading disputed
cases".
