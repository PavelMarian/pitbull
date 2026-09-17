"""Network-only F1 repair fetch stage for the full A2 run; never executes responses."""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "prestudy"))
import p3_baseline_run as H  # noqa: E402

MANIFEST = Path(os.environ.get("A2_FULL_MANIFEST", HERE / "a2_full_manifest.json"))
RESPONSES = Path(os.environ.get("A2_FULL_RESPONSES", HERE / "a2_full_responses"))
MODEL = os.environ.get("A2_FULL_MODEL", "z-ai/glm-5.3-flash")
ITERATION = int(os.environ["A2_FULL_ITERATION"])
CONCURRENCY = int(os.environ.get("A2_FULL_CONCURRENCY", "4"))

F1_TEXT = """В коде выше может быть утечка данных из будущего. Требование: признаки
для момента seed_time должны вычисляться ТОЛЬКО по информации, которая уже существует
на момент seed_time. Учитывай, что события в базе появляются в разное время: например,
отзыв на заказ пишется позже самой покупки, а даты доставки заполняются после неё.
Проверь свой код на это требование и верни исправленную версию функции get_features
(та же сигнатура), в одном блоке ```python ...```, без пояснений."""


def fetch(program, out_path):
    prompt = (H.NEUTRAL_PROMPT + "\n\nВот текущая версия кода:\n```python\n"
              + program["current_code"] + "\n```\n\n" + F1_TEXT)
    item = {"program_uid": program["program_uid"], "program": program["program"],
            "iteration": ITERATION, "repair_model": MODEL,
            "source_code_sha256": hashlib.sha256(program["current_code"].encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    started = time.time()
    try:
        response = litellm.completion(
            model=f"openrouter/{MODEL}", messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=14000, timeout=300,
            extra_body={"reasoning": {"max_tokens": 1500}},
        )
        raw = response.choices[0].message.content or ""
        item.update({"status": "ok", "finish_reason": response.choices[0].finish_reason,
                     "raw_response": raw, "code": H._extract_code(raw),
                     "usage": response.usage.model_dump() if response.usage else None})
    except Exception as exc:
        item.update({"status": "api_error", "error": f"{type(exc).__name__}: {exc}"[:500]})
    item["wall_seconds"] = round(time.time() - started, 1)
    out_path.write_text(json.dumps(item, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return item


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out_dir = RESPONSES / f"iter_{ITERATION:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [p for p in manifest["programs"] if p["status"] != "clean"
               and not any(h.get("iteration") == ITERATION for h in p.get("history", []))]
    todo = [(p, out_dir / f"{p['program_uid']}.json") for p in pending
            if not (out_dir / f"{p['program_uid']}.json").exists()]
    print("iteration", ITERATION, "pending", len(pending), "to_fetch", len(todo), flush=True)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(fetch, p, path): p for p, path in todo}
        for future in as_completed(futures):
            item = future.result()
            print(item["program"], item["program_uid"], item["status"],
                  f"{item['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
