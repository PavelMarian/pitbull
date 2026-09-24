"""checker_core: self-contained point-in-time oracle for pit-repair.

Standalone implementation of the PITBULL differential-execution protocol,
with every dataset-specific default removed. Every channel list here is a
required argument (or an explicit default of "none"), not a module-level
constant tied to one database. No import, path, or environment variable in
this file points outside the skill's own `scripts/` directory -- nothing
here depends on any other part of this repository being present.

Property: a divergence between the full-database and truncated-database
outputs is PROOF that the program read a row timestamped later than the
prediction time. Not a suspicion. The converse does not hold: agreement
does not prove correctness (the program may have read the future without
it changing the output) -- that one-sidedness is intentional, it is what
makes every LEAK verdict ironclad.
"""
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutTimeout
from pathlib import Path

import numpy as np
import pandas as pd


class Timeout(Exception):
    pass


def call_with_timeout(fn, *args, timeout):
    # signal.alarm only works on the main thread; run this in a dedicated
    # single-worker executor instead so it's safe from any caller thread.
    with _ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args)
        try:
            return fut.result(timeout=timeout)
        except _FutTimeout:
            raise Timeout()


def load_db(db_dir, parse_dates=None):
    """Load every *.csv in db_dir as a table named by its filename stem.

    parse_dates: optional {table: [columns]} to parse as datetimes -- pass
    the union of the availability map's time_cols/self_availability_cols
    values per table. Tables/columns not listed load as-is (object dtype).
    """
    parse_dates = parse_dates or {}
    db = {}
    for f in sorted(Path(db_dir).glob("*.csv")):
        name = f.stem
        cols = [c for c in parse_dates.get(name, []) if c]
        db[name] = pd.read_csv(f, parse_dates=cols) if cols else pd.read_csv(f)
    if not db:
        raise ValueError(f"no .csv tables found in {db_dir}")
    return db


def sample_entities(db, table, column, n, seed=0):
    """Deterministic sample of up to n unique ids from db[table][column]."""
    if table not in db:
        raise ValueError(f"entity table {table!r} not in db (have: {sorted(db)})")
    if column not in db[table].columns:
        raise ValueError(f"entity column {column!r} not in table {table!r}")
    ids = np.sort(db[table][column].dropna().unique())
    if len(ids) == 0:
        raise ValueError(f"no non-null values in {table}.{column} to sample entities from")
    if len(ids) <= n:
        return ids
    return np.random.RandomState(seed).choice(ids, size=n, replace=False)


def truncate(db: dict, seed_time, time_cols: dict) -> dict:
    """Keep only rows with time <= seed_time. A table absent from time_cols
    (or whose declared column isn't present) passes through untouched --
    that's how a lookup table with no time channel is declared."""
    seed_time = pd.Timestamp(seed_time)
    out = {}
    for name, df in db.items():
        tc = time_cols.get(name)
        out[name] = df if (tc is None or tc not in df.columns) else df[df[tc] <= seed_time]
    return out


def frames_equal(a: pd.DataFrame, b: pd.DataFrame, atol=1e-9, nan_equal=True) -> bool:
    # A program that deterministically returns nothing on both calls (e.g.
    # forgot a return) must compare equal to itself, not register as a
    # spurious LEAK -- `a is b` correctly equates two Nones, while a lone
    # None against a real frame is still a legitimate divergence.
    if a is None or b is None:
        return a is b
    if a.shape != b.shape:
        return False
    if list(a.columns) != list(b.columns):
        return False
    a = a.sort_index(axis=0)
    b = b.reindex(a.index)
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype="float64"), y.to_numpy(dtype="float64")
            both_nan = np.isnan(xv) & np.isnan(yv)
            if not nan_equal and both_nan.any():
                return False
            diff = np.abs(np.where(both_nan, 0.0, xv - yv))
            if not np.all(np.where(both_nan, True, diff <= atol)):
                return False
        else:
            xs, ys = x.astype(object).where(x.notna(), None), y.astype(object).where(y.notna(), None)
            if not xs.equals(ys):
                return False
    return True


# ---------------------------------------------------------------- canary ----
#
# Canary perturbation: a rerun on a copy where everything truncation would
# otherwise hide is disturbed instead of removed. Three channel categories,
# all declared by the caller via the availability map -- none are inferred:
#
#   1. Ordinary columns in FUTURE rows (time_cols) -- truncation drops the
#      whole row, so canary perturbs everything in it except keys and its
#      own time column: numbers shift, dates move past the data horizon,
#      categories/strings get a sentinel.
#   2. Self-availability channels -- columns whose OWN availability equals
#      their value (a delivery date is known on the delivery date), not the
#      row's time: truncating by row time doesn't protect them, so they are
#      perturbed in ANY row where the value is missing or > seed_time.
#   3. Gatekeeper columns -- a value <= t that gates the availability of
#      OTHER columns whose own availability is > t; these must never be
#      perturbed, or D'|t != D|t breaks. Empty by default; a caller who
#      finds one for their schema should list it explicitly rather than
#      silently accept the blind spot.

DEFAULT_KEY_HINTS = ("_id", "id")
DEFAULT_NUM_SHIFT = 7919.0                 # large prime: any aggregation shows it
DEFAULT_CAT_SENTINEL = "__canary_sentinel__"
DEFAULT_DATE_MARGIN_DAYS = 3650            # 10 years past the horizon, unmistakable


def _data_horizon(db, time_cols, self_avail_cols=None):
    """Latest timestamp actually present in the data (by time_cols and, if
    given, self-availability channels) -- the anchor for the date shift."""
    self_avail_cols = self_avail_cols or {}
    stamps = []
    for name, df in db.items():
        tc = time_cols.get(name)
        if tc is not None and tc in df.columns:
            m = df[tc].max()
            if pd.notna(m):
                stamps.append(pd.Timestamp(m))
        for c in self_avail_cols.get(name, []):
            if c in df.columns:
                m = df[c].max()
                if pd.notna(m):
                    stamps.append(pd.Timestamp(m))
    if not stamps:
        raise ValueError("could not determine the data horizon: no non-empty time column")
    return max(stamps)


def _check_sentinel_safe(db, sentinel, protected_by_table, key_hints):
    """Refuse loudly, not silently, if the sentinel already occurs in the
    data as a real value -- otherwise canary stops being distinguishable
    from a genuine row."""
    for name, df in db.items():
        protected = protected_by_table.get(name, set())
        for c in df.columns:
            if c in protected or any(h in c.lower() for h in key_hints):
                continue
            is_textlike = pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
            if is_textlike and (df[c] == sentinel).any():
                raise ValueError(
                    f"sentinel {sentinel!r} already occurs as a real value in "
                    f"{name}.{c} -- pick a different cat_sentinel"
                )


def perturb_canary(db: dict, seed_time, time_cols: dict,
                    self_avail_cols=None,
                    gatekeeper_cols=None,
                    key_hints=DEFAULT_KEY_HINTS,
                    num_shift=DEFAULT_NUM_SHIFT,
                    cat_sentinel=DEFAULT_CAT_SENTINEL,
                    date_margin_days=DEFAULT_DATE_MARGIN_DAYS) -> dict:
    """See the category breakdown above. All channel lists are supplied by
    the caller (typically loaded from the availability map); none are
    inferred from column names beyond the generic key_hints skip-list."""
    self_avail_cols = self_avail_cols or {}
    gatekeeper_cols = gatekeeper_cols or {}
    seed_time = pd.Timestamp(seed_time)
    horizon = _data_horizon(db, time_cols, self_avail_cols)
    date_target = horizon + pd.Timedelta(days=date_margin_days)

    protected_by_table = {
        name: {time_cols.get(name)} | set(gatekeeper_cols.get(name, []))
        for name in db
    }
    _check_sentinel_safe(db, cat_sentinel, protected_by_table, key_hints)

    out = {}
    for name, df in db.items():
        tc = time_cols.get(name)
        df2 = df.copy()
        protected = protected_by_table[name]
        self_avail = set(self_avail_cols.get(name, []))

        future_mask = (df2[tc] > seed_time) if (tc is not None and tc in df2.columns) \
            else pd.Series(False, index=df2.index)

        for c in df2.columns:
            if c in protected:
                continue
            if c in self_avail:
                not_avail = df2[c].isna() | (df2[c] > seed_time)
                if not_avail.any():
                    df2.loc[not_avail, c] = date_target
                continue
            if any(h in c.lower() for h in key_hints):
                continue
            if not future_mask.any():
                continue
            if pd.api.types.is_datetime64_any_dtype(df2[c]):
                df2.loc[future_mask, c] = date_target
            elif pd.api.types.is_numeric_dtype(df2[c]):
                df2.loc[future_mask, c] = df2.loc[future_mask, c] + num_shift
            elif isinstance(df2[c].dtype, pd.CategoricalDtype):
                if cat_sentinel not in df2[c].cat.categories:
                    df2[c] = df2[c].cat.add_categories([cat_sentinel])
                df2.loc[future_mask, c] = cat_sentinel
            else:
                df2.loc[future_mask, c] = cat_sentinel
        out[name] = df2
    return out
