"""
P3 §3 — «лестница моделей» на нейтральном промпте, без агентной обвязки RelAgent.

Односнимковая генерация: один вызов модели -> один Python-код -> оракул
дифференциального исполнения (oracle.py, уже провалидирован). Экспериментатор
(этот скрипт) НЕ трогает сгенерированный код -- синтаксическая ошибка или падение
исполнения это отдельная категория результата ("did_not_run"), не повод чинить.

Задача C (леgкая для ошибки, см. P3_spec.md §4): спрос на товар. Признаки можно
строить и по собственной истории товара, и по соседу через соединение
(товар -> позиция заказа -> продавец -> вся история продавца) -- путь для утечки
через JOIN. Побочная временная ось: отзыв пишется в среднем на 10+ дней позже
заказа (см. PRESTUDY_RESULTS.md) -- та же ловушка, на которой ошиблась и наша
собственная эталонная реализация.

Нейтральность промпта (P3_spec.md §1): даём схему таблиц, обучающую таблицу
(entity, seed_time, label) и формулировку задачи. НЕ упоминаем "утечка",
"point-in-time", необходимость временного фильтра, примеры с/без фильтра.
"""
import hashlib
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutTimeout
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import litellm

sys.path.insert(0, str(Path(__file__).parent))
from oracle import is_pit_correct, TIME_COLS

_default_data = Path(__file__).parent / "p1_repro"
if not _default_data.exists():
    _default_data = Path(__file__).parent.parent / "PITFALL_olist_data"
D = os.environ.get("P3_OLIST", str(_default_data)) + "/"
OUT_DIR = Path(__file__).parent / "p3_out" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURE = 0.7
N_REPEATS = 80  # 20 -> 40 -> 80. Доля did_not_run у кодерного яруса около 60%,
                # поэтому эффективная выборка вдвое меньше числа генераций.
                # Настройки вызова НЕ меняются между раундами: иначе генерации
                # разных раундов нельзя объединять в одну долю.
TEST_SEEDS = ["2018-01-01", "2018-04-01", "2018-07-01"]
ALL_SEEDS = ["2017-04-01", "2017-07-01", "2017-10-01", "2018-01-01", "2018-04-01", "2018-07-01"]
N_SAMPLE_PRODUCTS = 15
PROGRAM_TIMEOUT_SECONDS = 30
MAX_CONCURRENT = int(os.environ.get("P3_BASELINE_CONCURRENT", "8"))

MODELS = {
    # "нижний (free)" пропущен -- nvidia/nemotron-3.5-lightning:free не отвечает
    # за разумное время (>3 мин на вызов, судя по всему перегружен/rate-limited).
    "средний": "deepseek/deepseek-v4-flash-0731",
    "кодерный": "bytedance-seed/seed-2.0-code",
}

# ---------------------------------------------------------------- данные ----

orders = pd.read_csv(D + "olist_orders_dataset.csv",
                      parse_dates=["order_purchase_timestamp", "order_approved_at",
                                   "order_delivered_carrier_date", "order_delivered_customer_date",
                                   "order_estimated_delivery_date"])
items = pd.read_csv(D + "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
reviews = pd.read_csv(D + "olist_order_reviews_dataset.csv",
                       parse_dates=["review_creation_date", "review_answer_timestamp"])
payments = pd.read_csv(D + "olist_order_payments_dataset.csv")

orders = orders.dropna(subset=["order_purchase_timestamp"])
ots = orders.set_index("order_id").order_purchase_timestamp
items = items.copy()
items["ts"] = items.order_id.map(ots)
items = items.dropna(subset=["ts"])
payments = payments.copy()
payments["ts"] = payments.order_id.map(ots)
payments = payments.dropna(subset=["ts"])

DB = {"orders": orders, "order_items": items, "reviews": reviews, "payments": payments}
TMAX = items.ts.max()

# события товара (для построения ground-truth сущностей/меток -- инфраструктура
# эксперимента, НЕ показывается модели)
_ev = items.merge(orders[["order_id", "order_status"]], on="order_id")


def labels(seed, horizon=90, active=180, min_orders=2):
    seed = pd.Timestamp(seed)
    act = _ev[(_ev.ts > seed - pd.Timedelta(days=active)) & (_ev.ts <= seed)]
    cnt = act.groupby("product_id").order_id.nunique()
    prods = np.sort(cnt[cnt >= min_orders].index.values)
    fut = _ev[(_ev.ts > seed) & (_ev.ts <= seed + pd.Timedelta(days=horizon))]
    y = pd.Series(prods, index=prods).isin(fut.product_id.unique()).astype(int)
    return seed, prods, y


# ------------------------------------------------------------- промпт -------

def _schema_block():
    def col_info(df):
        return ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)

    return f"""
Таблица `orders` ({len(orders)} строк, ключ order_id):
  {col_info(orders)}

Таблица `order_items` ({len(items)} строк, ключ order_id+order_item_id, внешний ключ product_id, seller_id):
  {col_info(items)}
  (столбец ts = момент покупки заказа, к которому относится позиция)

Таблица `reviews` ({len(reviews)} строк, внешний ключ order_id):
  {col_info(reviews)}

Таблица `payments` ({len(payments)} строк, внешний ключ order_id):
  {col_info(payments)}
""".strip()


def _train_table_sample():
    rows = []
    for s in ALL_SEEDS:
        seed, prods, y = labels(s)
        if len(prods) < 50:
            continue
        stable_seed = int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")
        idx = np.random.RandomState(stable_seed).choice(len(prods), size=min(6, len(prods)), replace=False)
        for i in idx:
            rows.append({"product_id": prods[i], "seed_time": str(seed.date()), "label": int(y.iloc[i])})
    df = pd.DataFrame(rows)
    return df.to_string(index=False)


NEUTRAL_PROMPT = f"""
Датасет: интернет-магазин Olist (Бразилия), реляционная база из четырёх таблиц.

{_schema_block()}

Обучающая таблица для задачи: столбцы `product_id` (сущность), `seed_time`
(момент предсказания) и `label` (целевая переменная). Пример строк:

{_train_table_sample()}

Задача: предсказать `label` -- закажут ли товар product_id в ближайшие 90 дней
после seed_time. Тебе нужно написать функцию, которая ДЛЯ ДАННОГО МОМЕНТА
seed_time и списка product_id возвращает матрицу признаков (не сами предсказания,
не финальную модель -- только признаки, которые потом пойдут в классификатор).

Требования к коду:
- Язык: Python, доступны pandas (как pd) и numpy (как np), больше ничего не импортируй.
- Функция должна называться `get_features` и иметь сигнатуру:
  def get_features(db: dict, entity_ids, seed_time) -> pd.DataFrame
  где db = {{"orders": DataFrame, "order_items": DataFrame, "reviews": DataFrame, "payments": DataFrame}}
  (ровно те четыре таблицы со схемой выше), entity_ids -- список/массив product_id,
  seed_time -- pandas.Timestamp.
- Результат: DataFrame, индексированный по entity_ids (все значения из entity_ids,
  в любом порядке допустимо, но реиндексируй под entity_ids перед возвратом),
  с произвольным числом числовых столбцов-признаков.
- Используй любые из четырёх таблиц и любые объединения между ними, если считаешь
  это полезным для качества предсказания.
- Верни ТОЛЬКО код функции (и вспомогательных функций при необходимости) в одном
  блоке ```python ... ```, без пояснений до или после блока.
""".strip()

# ------------------------------------------------------------- харнесс ------


class _Timeout(Exception):
    pass


def _extract_code(text: str) -> str | None:
    # Берём ПОСЛЕДНИЙ полный блок -- некоторые модели показывают промежуточные
    # обрывки кода в рассуждениях до финального ответа.
    blocks = re.findall(r"```python\s*(.*?)```", text, re.S)
    if not blocks:
        blocks = re.findall(r"```\s*(.*?)```", text, re.S)
    if blocks:
        return blocks[-1].strip()
    # Фолбэк для харнесса (R2 PRESTUDY2_runbook.md): некоторые модели пишут
    # полный корректный код без ```-обрамления, хотя инструкция явно его
    # требует -- это не повод терять валидный ответ. Не меняет содержание кода.
    m = re.search(r"def get_features\(.*", text, re.S)
    return m.group(0).strip() if m else None


def _load_program(code: str):
    ns = {"pd": pd, "np": np, "__builtins__": __builtins__}
    exec(code, ns)  # noqa: S102 -- испытуемый код, не чиним и не защищаем от него дальше песочницы таймаута
    fn = ns.get("get_features")
    if fn is None or not callable(fn):
        raise ValueError("get_features not defined or not callable")
    return fn


def _call_with_timeout(fn, *args, timeout=PROGRAM_TIMEOUT_SECONDS):
    # signal.alarm работает только в главном потоке процесса -- а run_one()
    # теперь запускается из воркеров ThreadPoolExecutor. Поэтому таймаут через
    # отдельный однопоточный executor вместо SIGALRM.
    with _ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args)
        try:
            return fut.result(timeout=timeout)
        except _FutTimeout:
            raise _Timeout()


def check_generation(code: str) -> dict:
    try:
        fn = _load_program(code)
    except Exception as e:
        return {"status": "did_not_run", "stage": "load", "error": f"{type(e).__name__}: {e}"[:500]}

    seed_results = []
    for s in TEST_SEEDS:
        seed, prods, y = labels(s)
        if len(prods) < N_SAMPLE_PRODUCTS:
            continue
        sample = np.random.RandomState(0).choice(prods, size=N_SAMPLE_PRODUCTS, replace=False)

        def prog(db, entity, seed_time, _fn=fn):
            return _call_with_timeout(_fn, db, entity, seed_time)

        try:
            ok, detail = is_pit_correct(prog, DB, sample, seed, TIME_COLS, return_detail=True)
            seed_results.append({"seed": s, "status": "ran", "clean": ok, "detail": detail})
        except _Timeout:
            seed_results.append({"seed": s, "status": "timeout"})
        except Exception as e:
            seed_results.append({"seed": s, "status": "error", "error": f"{type(e).__name__}: {e}"[:500]})

    ran = [r for r in seed_results if r["status"] == "ran"]
    if not ran:
        return {"status": "did_not_run", "stage": "execute", "seed_results": seed_results}
    n_leak = sum(1 for r in ran if not r["clean"])
    verdict = "LEAK" if n_leak > 0 else "CLEAN"
    return {"status": "ok", "verdict": verdict, "n_seeds_ran": len(ran),
            "n_seeds_leak": n_leak, "seed_results": seed_results}


def run_one(tier: str, model_id: str, idx: int, prompt: str = None) -> dict:
    t0 = time.time()
    prompt = prompt if prompt is not None else NEUTRAL_PROMPT
    try:
        resp = litellm.completion(
            model=f"openrouter/{model_id}",
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=14000,
            timeout=300,
            # без ограничения "мышления" некоторые модели съедают весь max_tokens
            # на reasoning и не успевают выдать ответ (finish_reason=length,
            # content=""); это калибровка бюджета, не подсказка по содержанию.
            extra_body={"reasoning": {"max_tokens": 1500}},
        )
        raw_text = resp.choices[0].message.content or ""
        finish_reason = resp.choices[0].finish_reason
        usage = resp.usage.model_dump() if resp.usage else None
    except Exception as e:
        return {"tier": tier, "model": model_id, "idx": idx, "status": "api_error",
                "error": f"{type(e).__name__}: {e}"[:500], "wall_seconds": time.time() - t0}

    code = _extract_code(raw_text)
    result = {"tier": tier, "model": model_id, "idx": idx, "raw_response": raw_text,
              "finish_reason": finish_reason, "usage": usage, "wall_seconds": time.time() - t0}
    if code is None:
        stage = "truncated_reasoning" if finish_reason == "length" and not raw_text else "extract_code"
        result.update({"status": "did_not_run", "stage": stage})
        return result

    result["code"] = code
    try:
        check = check_generation(code)
    except Exception as e:
        check = {"status": "harness_error", "error": f"{type(e).__name__}: {e}"[:500],
                  "traceback": traceback.format_exc()[:1000]}
    result.update(check)
    return result


def main():
    out_path = OUT_DIR / "results.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["tier"], r["idx"]))
            except Exception:
                pass

    tasks = [(tier, model_id, idx) for tier, model_id in MODELS.items()
             for idx in range(1, N_REPEATS + 1) if (tier, idx) not in done]

    with out_path.open("a") as out, _ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(run_one, tier, model_id, idx): (tier, idx)
                   for tier, model_id, idx in tasks}
        for fut in as_completed(futures):
            tier, idx = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"tier": tier, "idx": idx, "status": "driver_error", "error": str(e)[:300]}
            out.write(json.dumps(res, default=str) + "\n")
            out.flush()
            print(f"[{tier:12s} {idx:02d}] status={res.get('status')} "
                  f"verdict={res.get('verdict')} leak_seeds={res.get('n_seeds_leak')} "
                  f"wall_s={res.get('wall_seconds', 0):.1f}", flush=True)


if __name__ == "__main__":
    main()
