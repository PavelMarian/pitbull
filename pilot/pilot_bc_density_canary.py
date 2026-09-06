"""
Пилоты B и C (FSE RQ3 + двухуровневый вердикт), на Olist.

B — плотность нарушения: для каждой протекающей программы из корпуса G0 доля
моментов сетки, на которых дифференциальная проверка расходится.
Сетка: месячные моменты 2017-01-01 .. 2018-08-01 (все, где хватает сущностей).

C — канареечный прогон: третий запуск на копии базы, где возмущено всё, что
усечение делает недоступным, типовым образом по типам данных (harness-v1,
oracle.perturb_canary): числа и даты будущих строк — сдвиг, категории/строки —
sentinel, self-availability каналы (дата доставки и т.п.) — по своему значению
независимо от времени строки; ключи и timestamp-колонки, определяющие admission
строки, нетронуты.
  уровень 1 (witness): full vs truncated — точный вердикт;
  уровень 2 (canary):  full vs canary   — «программа прочла будущее», даже если
                        усечение выхода не изменило.
Негативный контроль (ОБЯЗАТЕЛЬНО ПЕРВЫМ): корректная программа — оба уровня молчат
на всей сетке. Кейс совпадения: наивный max (max review_score истории продавца) —
witness должен часто молчать (max=5 уже в прошлом), канарейка — стрелять.

Порядок закреплён до прогона; параметры не подгоняются по результатам.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "prestudy_code"))

from oracle import truncate, frames_equal, TIME_COLS, perturb_canary  # noqa: E402
import p3_baseline_run as H  # noqa: E402  (грузит Olist, даёт labels/DB/_load_program/_call_with_timeout)

GRID = [f"{y}-{m:02d}-01" for y in (2017, 2018) for m in range(1, 13)
        if not (y == 2018 and m > 8) and not (y == 2017 and m < 1)]
N_ENT = 15
PER_PROGRAM_BUDGET_S = 360
MAX_LEAK_PROGRAMS = 20   # по 10 на модель, детерминированно по порядку в корпусе
N_CLEAN_CONTROL = 3      # негативный контроль из корпуса

# Типовое возмущение по типам данных (harness-v1) теперь живёт в oracle.py как
# perturb_canary — общий модуль, а не локальная копия пилота: тег фиксирует
# версию оттуда. См. oracle.py за разбором трёх категорий каналов
# (обычные / self-availability / gatekeeper) и proposals/fse2027_proposal.md,
# шапка + §11, за определением.


def run_three(prog, seed):
    """-> dict(status, witness: bool расхождение full/trunc, canary: bool full/canary,
    differing_w, differing_c)."""
    seed_t = pd.Timestamp(seed)
    _, prods, _ = H.labels(seed)
    if len(prods) < N_ENT:
        return {"status": "skip_few_entities"}
    ents = np.random.RandomState(0).choice(prods, size=N_ENT, replace=False)
    try:
        full = H._call_with_timeout(prog, H.DB, ents, seed_t, timeout=20)
        trunc = H._call_with_timeout(prog, truncate(H.DB, seed_t), ents, seed_t, timeout=20)
        can = H._call_with_timeout(prog, perturb_canary(H.DB, seed_t), ents, seed_t, timeout=20)
    except H._Timeout:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"[:200]}

    def diff_cols(a, b):
        if a is None or b is None or a.shape != b.shape or list(a.columns) != list(b.columns):
            return ["__shape__"]
        return [c for c in a.columns if not frames_equal(a[[c]], b[[c]])]

    dw, dc = diff_cols(full, trunc), diff_cols(full, can)
    return {"status": "ran", "witness": bool(dw), "canary": bool(dc),
            "differing_w": dw[:8], "differing_c": dc[:8]}


# ------------------------------------------------- испытуемые программы -----

def corpus_programs():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]
    ok = [r for r in rows if r.get("status") == "ok"]
    leak = [r for r in ok if r.get("verdict") == "LEAK"]
    clean = [r for r in ok if r.get("verdict") == "CLEAN"]
    sel_leak, per_model = [], {}
    for r in leak:
        m = r["model"]
        if per_model.get(m, 0) < MAX_LEAK_PROGRAMS // 2:
            sel_leak.append(r)
            per_model[m] = per_model.get(m, 0) + 1
    return sel_leak, clean[:N_CLEAN_CONTROL]


def correct_reference(db, entity_ids, seed_time):
    """Корректная вручную программа (карта известна): счётчики истории товара и
    средний review_score по данным, доступным на момент t."""
    t = seed_time
    it = db["order_items"]
    it = it[it["ts"] <= t]
    rv = db["reviews"]
    rv = rv[rv["review_creation_date"] <= t]
    per = it.groupby("product_id").agg(n_items=("order_id", "count"),
                                       price_mean=("price", "mean"))
    r = it.merge(rv[["order_id", "review_score"]], on="order_id")
    rs = r.groupby("product_id").review_score.mean().rename("avg_review")
    out = per.join(rs)
    return out.reindex(entity_ids)


def naive_max(db, entity_ids, seed_time):
    """Наивный max: лучшая оценка отзывов товара за ВСЮ историю (без фильтра по
    времени отзывов) поверх корректной истории заказов. Классический кейс
    совпадения: если max=5 достигнут в прошлом, усечение выхода не меняет."""
    t = seed_time
    it = db["order_items"]
    it = it[it["ts"] <= t]
    r = it.merge(db["reviews"][["order_id", "review_score"]], on="order_id")
    mx = r.groupby("product_id").review_score.max().rename("max_review")
    n = it.groupby("product_id").order_id.count().rename("n_items")
    return pd.concat([n, mx], axis=1).reindex(entity_ids)


def main():
    t_start = time.time()
    results = []

    # --- негативный контроль ПЕРВЫМ: корректная программа по всей сетке ---
    print("== негативный контроль (correct_reference)", flush=True)
    for s in GRID:
        r = run_three(correct_reference, s)
        r.update({"program": "correct_reference", "kind": "control", "seed": s})
        results.append(r)
        if r["status"] == "ran" and (r["witness"] or r["canary"]):
            print(f"  !! КОНТРОЛЬ НЕ ПРОШЁЛ на {s}: {r}", flush=True)
    ctrl = [r for r in results if r["status"] == "ran"]
    fired = [r for r in ctrl if r["witness"] or r["canary"]]
    print(f"контроль: {len(ctrl)} моментов, срабатываний {len(fired)}", flush=True)
    if fired:
        print("СТОП: контроль грязный, измерение не имеет смысла.", flush=True)
        json.dump(results, open(os.path.join(HERE, "pilot_bc_results.json"), "w"),
                  indent=1, default=str)
        return

    # --- кейс совпадения: naive_max ---
    print("== кейс совпадения (naive_max)", flush=True)
    for s in GRID:
        r = run_three(naive_max, s)
        r.update({"program": "naive_max", "kind": "coincidence", "seed": s})
        results.append(r)
    sub = [r for r in results if r["program"] == "naive_max" and r["status"] == "ran"]
    w = sum(r["witness"] for r in sub); c = sum(r["canary"] for r in sub)
    print(f"naive_max: моментов {len(sub)}, witness {w}, canary {c}", flush=True)

    # --- корпус: LEAK-программы (+ CLEAN-контроли корпуса) ---
    sel_leak, sel_clean = corpus_programs()
    print(f"== корпус: {len(sel_leak)} LEAK + {len(sel_clean)} CLEAN", flush=True)
    for kind, rows in (("corpus_leak", sel_leak), ("corpus_clean", sel_clean)):
        for rec in rows:
            pid = f"{rec['model'].split('/')[-1]}#{rec['idx']}"
            try:
                fn = H._load_program(rec["code"])
            except Exception as e:
                results.append({"program": pid, "kind": kind, "status": "load_error",
                                "error": str(e)[:200]})
                continue
            t0 = time.time()
            n_ran = n_w = n_c = 0
            for s in GRID:
                if time.time() - t0 > PER_PROGRAM_BUDGET_S:
                    results.append({"program": pid, "kind": kind, "seed": s,
                                    "status": "budget_exceeded"})
                    break
                r = run_three(fn, s)
                r.update({"program": pid, "kind": kind, "seed": s})
                results.append(r)
                if r["status"] == "ran":
                    n_ran += 1; n_w += r["witness"]; n_c += r["canary"]
            print(f"  {kind} {pid}: ran {n_ran}, witness {n_w}, canary {n_c} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    json.dump(results, open(os.path.join(HERE, "pilot_bc_results.json"), "w"),
              indent=1, default=str)
    print(f"total {time.time()-t_start:.0f}s -> pilot_bc_results.json", flush=True)


if __name__ == "__main__":
    main()
