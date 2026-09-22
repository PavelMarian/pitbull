"""Fetch A3 detector judgments. Responses are text/JSON and are never executed."""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
import litellm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = Path(os.environ.get("A3_MANIFEST", HERE / "a3_manifest.json"))
OUT = Path(os.environ.get("A3_RESPONSES", HERE / "a3_responses"))
MODEL = os.environ.get("A3_MODEL", "z-ai/glm-5.3-flash")
CONCURRENCY = int(os.environ.get("A3_CONCURRENCY", "6"))

F0 = """Review the feature code for correctness. Decide whether it contains a bug that makes the features invalid for the stated prediction setting. Return only JSON: {\"verdict\": \"LEAK\" or \"CLEAN\", \"rationale\": \"one short sentence\"}."""
F1 = """Check specifically for temporal data leakage: at prediction time, a feature may use only values already available by that time, including after joins and for fields whose availability is later than their row timestamp. Return only JSON: {\"verdict\": \"LEAK\" or \"CLEAN\", \"rationale\": \"one short sentence\"}."""


def prompt(item, arm):
    return f"Prediction setting and data semantics:\n{item['context']}\n\nCode:\n```\n{item['code']}\n```\n\n{F0 if arm == 'F0' else F1}"


def fetch(item, arm, path):
    text = prompt(item, arm)
    result = {"item_id": item["item_id"], "pair_id": item["pair_id"],
              "category": item["category"], "arm": arm, "model": MODEL,
              "code_sha256": hashlib.sha256(item["code"].encode()).hexdigest(),
              "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
    started = time.time()
    try:
        response = litellm.completion(
            model=f"openrouter/{MODEL}", messages=[{"role": "user", "content": text}],
            temperature=0, max_tokens=800, timeout=300,
            extra_body={"reasoning": {"max_tokens": 1200}},
        )
        result.update({"status": "ok", "finish_reason": response.choices[0].finish_reason,
                       "raw_response": response.choices[0].message.content or "",
                       "usage": response.usage.model_dump() if response.usage else None})
    except Exception as exc:
        result.update({"status": "api_error", "error": f"{type(exc).__name__}: {exc}"[:500]})
    result["wall_seconds"] = round(time.time() - started, 1)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return result


def main():
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    todo = []
    for item in manifest["items"]:
        for arm in manifest["arms"]:
            path = OUT / f"{item['item_id']}_{arm}.json"
            if not path.exists():
                todo.append((item, arm, path))
    print("calls to fetch", len(todo), flush=True)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(fetch, *job): job for job in todo}
        for future in as_completed(futures):
            result = future.result()
            print(result["item_id"], result["arm"], result["status"],
                  f"{result['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
