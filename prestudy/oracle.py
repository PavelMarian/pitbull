"""
Оракул корректности по времени — дифференциальное исполнение (§4 шаг 1 runbook).

Свойство: расхождение выходов на полной и усечённой базе есть ДОКАЗАТЕЛЬСТВО того,
что программа прочитала строку с временем > момента предсказания. Не подозрение.

Обратное неверно: совпадение не доказывает корректность (программа могла прочитать
будущее и не использовать его). Это осознанная односторонность, она нам и нужна:
все вердикты «нарушение» — железные.
"""
import numpy as np
import pandas as pd

# какие таблицы Olist по какой колонке живут во времени
TIME_COLS = {
    "orders": "order_purchase_timestamp",
    "order_items": "ts",
    "reviews": "review_creation_date",
    "payments": "ts",
    # sellers / products / customers — справочники без времени
}


def truncate(db: dict, seed_time, time_cols=TIME_COLS) -> dict:
    """Оставить только строки с time <= seed_time. Справочники не трогаем."""
    seed_time = pd.Timestamp(seed_time)
    out = {}
    for name, df in db.items():
        tc = time_cols.get(name)
        if tc is None or tc not in df.columns:
            out[name] = df
        else:
            out[name] = df[df[tc] <= seed_time]
    return out


def frames_equal(a: pd.DataFrame, b: pd.DataFrame, atol=1e-9, nan_equal=True) -> bool:
    # ВАЖНО (найдено R3 PRESTUDY2_runbook.md, отрицательный контроль): было
    # `return False` безусловно при любом None -- значит функция, которая
    # детерминированно ничего не возвращает (например, забыла return), давала
    # False on None even in identical calls и всегда получала LEAK, хотя полный
    # и усечённый вызов физически не могли разойтись. a is b корректно уравнивает
    # два None между собой, оставляя одиночный None как расхождение (законный LEAK/ошибка).
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


def is_pit_correct(program, db, entity, seed_time, time_cols=TIME_COLS, return_detail=False):
    """program(db, entity, seed_time) -> DataFrame признаков, индекс = сущности."""
    full = program(db, entity, seed_time)
    trunc = program(truncate(db, seed_time, time_cols), entity, seed_time)
    ok = frames_equal(full, trunc)
    if not return_detail:
        return ok
    detail = {}
    if not ok and full is not None and trunc is not None and full.shape == trunc.shape \
            and list(full.columns) == list(trunc.columns):
        bad = []
        for c in full.columns:
            if not frames_equal(full[[c]], trunc[[c]]):
                bad.append(c)
        detail["differing_columns"] = bad
        detail["n_differing"] = len(bad)
        detail["n_total"] = full.shape[1]
    return ok, detail


# ---------------------------------------------------------------- canary (harness-v1) ----
#
# Типовое возмущение по типам данных (proposals/fse2027_proposal.md, шапка + §11):
# canary — прогон на копии, где возмущено всё, что усечение делает недоступным.
# Три категории каналов:
#
#   1. Обычные колонки в БУДУЩИХ строках (time_cols) — усечение убирает всю строку,
#      canary должен возмутить в ней всё, кроме ключей и собственной колонки времени:
#      числа — сдвиг, даты — сдвиг за горизонт данных, категории/строки — sentinel.
#   2. Self-availability каналы — колонки, чья СОБСТВЕННАЯ доступность равна их
#      значению (дата доставки известна в дату доставки), а не времени строки, в
#      которой они лежат: усечение по времени строки их не защищает, поэтому они
#      возмущаются в ЛЮБОЙ строке, когда их значение отсутствует или > t. Правило
#      из плана: возмущать любым значением > t — множество недоступного не
#      меняется, D'|t = D|t сохраняется.
#   3. Gatekeeper-колонки — значение <= t управляет доступностью ДРУГИХ колонок с
#      доступностью > t; возмущать нельзя — ломает D'|t = D|t. Список пуст для
#      Olist осознанно (не выявлены при разметке TIME_COLS/SELF_AVAILABILITY_COLS,
#      не по недосмотру); если найдутся — публикуются как известное слепое пятно.

SELF_AVAILABILITY_COLS = {
    "orders": ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date"],
    "reviews": ["review_answer_timestamp"],
}

GATEKEEPER_COLS: dict = {}

# Подстроки, по которым канал считается ключом/идентификатором и не возмущается
# никогда (перенесено из pilot/pilot_bc_density_canary.py, где раньше жило).
KEY_HINTS = ("_id", "id", "zip", "prefix", "order_item")

CANARY_NUM_SHIFT = 7919.0            # большое простое: сдвиг заметен любой агрегацией
CANARY_CAT_SENTINEL = "__canary_sentinel__"
CANARY_DATE_MARGIN_DAYS = 3650       # 10 лет за горизонтом — не спутать с реальной датой


def _data_horizon(db, time_cols=TIME_COLS, self_avail_cols=None):
    """Максимальная дата, реально встреченная в данных (по time_cols и, если
    заданы, по self-availability каналам) — точка отсчёта для сдвига дат."""
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
        raise ValueError("не удалось определить горизонт данных: нет ни одной непустой временной колонки")
    return max(stamps)


def _check_sentinel_safe(db, sentinel, protected_by_table, key_hints=KEY_HINTS):
    """Отказ громко, а не молча, если sentinel уже встречается в данных как
    настоящее значение — иначе canary перестаёт быть отличимым от реального."""
    for name, df in db.items():
        protected = protected_by_table.get(name, set())
        for c in df.columns:
            if c in protected or any(h in c.lower() for h in key_hints):
                continue
            is_textlike = pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
            if is_textlike and (df[c] == sentinel).any():
                raise ValueError(
                    f"CANARY_CAT_SENTINEL {sentinel!r} уже встречается как настоящее "
                    f"значение в {name}.{c} — нужен другой sentinel"
                )


def perturb_canary(db: dict, seed_time, time_cols=TIME_COLS,
                    self_avail_cols=SELF_AVAILABILITY_COLS,
                    gatekeeper_cols=GATEKEEPER_COLS,
                    key_hints=KEY_HINTS,
                    num_shift=CANARY_NUM_SHIFT,
                    cat_sentinel=CANARY_CAT_SENTINEL,
                    date_margin_days=CANARY_DATE_MARGIN_DAYS) -> dict:
    """Типовое canary-возмущение (harness-v1). См. блок комментариев выше за
    разбором трёх категорий каналов."""
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
