"""Fetch-only stage for the isolated A2 draft repair run.

This process talks to the model provider and stores responses, but never executes
model-produced code.  Validation is deliberately delegated to
``a2_draft_validate.py`` in a separate, network-disabled process.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import litellm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "prestudy"))
import p3_baseline_run as H  # noqa: E402

MODEL = os.environ.get("A2_DRAFT_MODEL", "z-ai/glm-5.3-flash")
N_PER_MODEL = int(os.environ.get("A2_DRAFT_N_PER_MODEL", "1"))
OUT = Path(os.environ.get("A2_DRAFT_FETCH_OUT", HERE / "a2_draft_glm53_fetch.json"))

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
            cols = (sr.get("detail") or {}).get("differing_columns") or ["<форма выхода различается>"]
            lines.append(f"  seed_time={sr['seed']}: расходятся колонки {cols}")
    body = "\n".join(lines) or "  (расхождение без покомпонентной детализации)"
    return f"""Твой код проверили дифференциальным исполнением: запустили ДВАЖДЫ —
на полной базе и на копии, из которой удалены все строки с временем позже seed_time
(orders по order_purchase_timestamp, reviews по review_creation_date, order_items и
payments по времени родительского заказа). Правильная программа обязана дать
одинаковый результат. Твоя дала разный:
{body}
Разный результат означает, что перечисленные признаки читают данные из будущего
относительно seed_time. Исправь функцию get_features (та же сигнатура) и верни её
в одном блоке ```python ...```, без пояснений."""


def select_programs():
    source = ROOT / "prestudy" / "p3_out" / "baseline" / "results.jsonl"
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    leaks = [r for r in rows if r.get("status") == "ok" and r.get("verdict") == "LEAK"
             and r.get("finish_reason") != "length"]
    selected, per_model = [], {}
    for rec in leaks:
        if per_model.get(rec["model"], 0) < N_PER_MODEL:
            selected.append(rec)
            per_model[rec["model"]] = per_model.get(rec["model"], 0) + 1
    return selected


def main():
    results = []
    for rec in select_programs():
        pid = f"{rec['model'].split('/')[-1]}#{rec['idx']}"
        for arm, feedback in (("F1", F1_TEXT), ("F3", f3_text(rec["seed_results"]))):
            prompt = (H.NEUTRAL_PROMPT + "\n\nВот текущая версия кода:\n```python\n"
                      + rec["code"] + "\n```\n\n" + feedback)
            item = {
                "program": pid,
                "author_model": rec["model"],
                "repair_model": MODEL,
                "arm": arm,
                "source_code_sha256": hashlib.sha256(rec["code"].encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            started = time.time()
            try:
                response = litellm.completion(
                    model=f"openrouter/{MODEL}",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=14000,
                    timeout=300,
                    extra_body={"reasoning": {"max_tokens": 1500}},
                )
                raw = response.choices[0].message.content or ""
                item.update({
                    "status": "ok",
                    "finish_reason": response.choices[0].finish_reason,
                    "raw_response": raw,
                    "code": H._extract_code(raw),
                    "usage": response.usage.model_dump() if response.usage else None,
                })
            except Exception as exc:
                item.update({"status": "api_error", "error": f"{type(exc).__name__}: {exc}"[:500]})
            item["wall_seconds"] = round(time.time() - started, 1)
            results.append(item)
            OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            print(pid, arm, item["status"], f"{item['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
