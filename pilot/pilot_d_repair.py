"""
Пилот D (FSE RQ2 / ядро ICLR): чинят ли модели свою утечку по witness-фидбэку.

Дизайн (закреплён до прогона):
  - Испытуемые: первые 6 LEAK-программ каждой из двух моделей корпуса G0 (12 шт.),
    отбор детерминированный, по порядку в results.jsonl.
  - Негативный контроль: перед починкой каждая программа перепроверяется локально;
    если вердикт не воспроизвёлся (не LEAK) — программа исключается и это рапортуется.
  - Два плеча, парные (каждая программа в обоих):
      F1 — концепция словами: признаки должны считаться только по данным, доступным
           на момент seed_time; отзыв/доставка появляются позже покупки. Без фактов
           о том, ГДЕ ошибка.
      F3 — исполняемое свидетельство: код прогнали на полной и на усечённой базе,
           выходы разошлись; список расходящихся колонок и моментов.
  - До k=3 итераций: починка -> дифференциальная проверка -> если LEAK, следующая
    итерация (в F3 — со свежим witness'ом).
  - Модель-ремонтник = модель-автор. temperature=0.2 (фиксировано), один прогон.
  - Прокси-метрика «починка выбрасыванием»: сколько колонок в выходе до/после и
    остались ли в выходе колонки, которые расходились (semantic repair) или их
    просто удалили (deletion repair).
"""
import json, os, re, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_prestudy_code = os.path.join(HERE, "prestudy_code")
if not os.path.isdir(_prestudy_code):
    _prestudy_code = os.path.join(ROOT, "prestudy")
sys.path.insert(0, _prestudy_code)

import litellm  # noqa: E402
from oracle import truncate  # noqa: E402
import p3_baseline_run as H  # noqa: E402

N_PER_MODEL = int(os.environ.get("PILOT_N_PER_MODEL", "6"))
MAX_ITERS = int(os.environ.get("PILOT_MAX_ITERS", "3"))
TEMPERATURE = 0.2
ARMS = tuple(os.environ.get("PILOT_ARMS", "F1,F3").split(","))
REPAIR_MODEL = os.environ.get("PILOT_REPAIR_MODEL")
_tag = "" if ARMS == ("F1", "F3") else "_" + "_".join(ARMS)
OUT = os.environ.get("PILOT_OUT", os.path.join(HERE, f"pilot_d_results{_tag}.json"))

F0_TEXT = """Проверь свой код на ошибки и верни финальную версию функции get_features
(та же сигнатура), в одном блоке ```python ...```, без пояснений."""

F1_TEXT = """В коде выше может быть утечка данных из будущего. Требование: признаки
для момента seed_time должны вычисляться ТОЛЬКО по информации, которая уже существует
на момент seed_time. Учитывай, что события в базе появляются в разное время: например,
отзыв на заказ пишется позже самой покупки, а даты доставки заполняются после неё.
Проверь свой код на это требование и верни исправленную версию функции get_features
(та же сигнатура), в одном блоке ```python ...```, без пояснений."""


def f3_text(seed_results):
    lines = []
    for sr in seed_results:
        if sr.get("status") == "ran" and not sr.get("clean", True):
            det = sr.get("detail") or {}
            cols = det.get("differing_columns") or ["<форма выхода различается>"]
            lines.append(f"  seed_time={sr['seed']}: расходятся колонки {cols}")
    body = "\n".join(lines) if lines else "  (расхождение без покомпонентной детализации)"
    return f"""Твой код проверили дифференциальным исполнением: запустили ДВАЖДЫ —
на полной базе и на копии, из которой удалены все строки с временем позже seed_time
(orders по order_purchase_timestamp, reviews по review_creation_date, order_items и
payments по времени родительского заказа). Правильная программа обязана дать
одинаковый результат. Твоя дала разный:
{body}
Разный результат означает, что перечисленные признаки читают данные из будущего
относительно seed_time. Исправь функцию get_features (та же сигнатура) и верни её
в одном блоке ```python ...```, без пояснений."""


def repair_call(model_id, code, feedback):
    prompt = (H.NEUTRAL_PROMPT
              + "\n\nВот текущая версия кода:\n```python\n" + code + "\n```\n\n"
              + feedback)
    resp = litellm.completion(
        model=f"openrouter/{REPAIR_MODEL or model_id}",
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE, max_tokens=14000, timeout=300,
        extra_body={"reasoning": {"max_tokens": 1500}},
    )
    return H._extract_code(resp.choices[0].message.content or "")


def out_columns(code):
    """Колонки выхода программы на одном моменте (для прокси deletion vs semantic)."""
    try:
        fn = H._load_program(code)
        seed = pd.Timestamp(H.TEST_SEEDS[0])
        _, prods, _ = H.labels(H.TEST_SEEDS[0])
        ents = np.random.RandomState(0).choice(prods, size=10, replace=False)
        df = H._call_with_timeout(fn, H.DB, ents, seed, timeout=20)
        return list(df.columns) if df is not None else None
    except Exception:
        return None


def run_arm(rec, arm):
    model_id, code = rec["model"], rec["code"]
    orig_cols = out_columns(code)
    diverging = sorted({c for sr in rec["seed_results"]
                        for c in ((sr.get("detail") or {}).get("differing_columns") or [])})
    hist, seed_results = [], rec["seed_results"]
    cur = code
    for it in range(1, MAX_ITERS + 1):
        fb = (F0_TEXT if arm == "F0"
              else F1_TEXT if arm == "F1"
              else f3_text(seed_results))
        try:
            new_code = repair_call(model_id, cur, fb)
        except Exception as e:
            hist.append({"iter": it, "status": "api_error", "error": str(e)[:300]})
            break
        if not new_code:
            hist.append({"iter": it, "status": "no_code"})
            break
        check = H.check_generation(new_code)
        hist.append({"iter": it, "status": check.get("status"),
                     "verdict": check.get("verdict"),
                     "n_seeds_leak": check.get("n_seeds_leak")})
        cur = new_code
        if check.get("status") == "ok":
            seed_results = check["seed_results"]
            if check["verdict"] == "CLEAN":
                break
        # did_not_run: даём модели ту же обратную связь ещё раз (witness старый)
    final_clean = bool(hist) and hist[-1].get("verdict") == "CLEAN"
    new_cols = out_columns(cur) if final_clean else None
    kept = (None if not (final_clean and orig_cols and new_cols and diverging)
            else sum(1 for c in diverging if c in new_cols))
    return {"arm": arm, "history": hist, "clean": final_clean,
            "iters_used": len(hist),
            "orig_n_cols": len(orig_cols) if orig_cols else None,
            "new_n_cols": len(new_cols) if new_cols else None,
            "diverging_cols": diverging,
            "diverging_kept_in_output": kept}


def main():
    results_in = os.environ.get(
        "PILOT_RESULTS_IN",
        os.path.join(ROOT, "prestudy", "p3_out", "baseline", "results.jsonl"),
    )
    rows = [json.loads(l) for l in open(results_in)]
    leak = [r for r in rows if r.get("status") == "ok" and r.get("verdict") == "LEAK"]
    sel, per_model = [], {}
    for r in leak:
        if per_model.get(r["model"], 0) < N_PER_MODEL:
            sel.append(r)
            per_model[r["model"]] = per_model.get(r["model"], 0) + 1
    print(f"выбрано {len(sel)} программ: {per_model}", flush=True)

    results = []
    for rec in sel:
        pid = f"{rec['model'].split('/')[-1]}#{rec['idx']}"
        # негативный контроль: вердикт воспроизводится?
        chk = H.check_generation(rec["code"])
        if chk.get("verdict") != "LEAK":
            print(f"  {pid}: вердикт НЕ воспроизвёлся ({chk.get('verdict') or chk.get('status')}) — исключена", flush=True)
            results.append({"program": pid, "excluded": True,
                            "reason": chk.get("verdict") or chk.get("status")})
            continue
        rec = {**rec, "seed_results": chk["seed_results"]}
        for arm in ARMS:
            t0 = time.time()
            r = run_arm(rec, arm)
            r.update({"program": pid, "model": rec["model"],
                      "repair_model": REPAIR_MODEL or rec["model"],
                      "wall": round(time.time() - t0, 1)})
            results.append(r)
            print(f"  {pid} {arm}: clean={r['clean']} iters={r['iters_used']} "
                  f"cols {r['orig_n_cols']}->{r['new_n_cols']} kept={r['diverging_kept_in_output']} "
                  f"({r['wall']}s)", flush=True)
        json.dump(results, open(OUT, "w"), indent=1, default=str)

    done = [r for r in results if not r.get("excluded")]
    for arm in ARMS:
        a = [r for r in done if r["arm"] == arm]
        print(f"{arm}: CLEAN@{MAX_ITERS} = {sum(r['clean'] for r in a)}/{len(a)}", flush=True)
    json.dump(results, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
