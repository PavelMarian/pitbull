"""Parse and score A3 responses under the frozen conservative policy."""
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "a3_manifest.json"
RESPONSES = HERE / "a3_responses"
OUT = HERE / "a3_results.json"


def parse_verdict(raw):
    candidates = re.findall(r"\{[^{}]*\}", raw or "", flags=re.S)
    for candidate in candidates:
        try:
            value = str(json.loads(candidate).get("verdict", "")).upper()
        except (ValueError, TypeError, AttributeError):
            continue
        if value in {"LEAK", "CLEAN"}:
            return value
    return None


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [c-h, c+h]


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for item in manifest["items"]:
        for arm in manifest["arms"]:
            path = RESPONSES / f"{item['item_id']}_{arm}.json"
            response = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "missing"}
            pred = parse_verdict(response.get("raw_response")) if response.get("status") == "ok" else None
            rows.append({**{k: item[k] for k in ("item_id", "pair_id", "category", "truth", "provenance")},
                         "arm": arm, "prediction": pred, "response_status": response.get("status"),
                         "correct": pred == item["truth"]})
    metrics = defaultdict(dict)
    for category in ("checker_pair", "handwritten_pair"):
        for arm in manifest["arms"]:
            sub = [r for r in rows if r["category"] == category and r["arm"] == arm]
            pos = [r for r in sub if r["truth"] == "LEAK"]
            neg = [r for r in sub if r["truth"] == "CLEAN"]
            tp = sum(r["prediction"] == "LEAK" for r in pos)
            # Invalid is conservatively an FP for a negative item and an FN for a positive item.
            fp = sum(r["prediction"] != "CLEAN" for r in neg)
            invalid = sum(r["prediction"] is None for r in sub)
            predicted_pos = tp + sum(r["prediction"] == "LEAK" for r in neg)
            metrics[category][arm] = {
                "n_pairs": len(pos), "tp": tp, "positive_n": len(pos),
                "recall": tp/len(pos), "recall_wilson95": wilson(tp, len(pos)),
                "fp_conservative": fp, "negative_n": len(neg),
                "false_positive_rate_conservative": fp/len(neg),
                "fpr_wilson95": wilson(fp, len(neg)), "invalid": invalid,
                "precision_observed": tp/predicted_pos if predicted_pos else None,
            }
    primary = metrics["checker_pair"][manifest["primary_arm"]]
    rule = manifest["decision_rule"]
    passed = (primary["recall"] >= rule["checker_pair_recall_min"] and
              primary["false_positive_rate_conservative"] <= rule["checker_pair_false_positive_rate_max"])
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "api_errors": 0}
    for path in RESPONSES.glob("*.json"):
        response = json.loads(path.read_text(encoding="utf-8"))
        usage["calls"] += 1
        usage["api_errors"] += response.get("status") != "ok"
        u = response.get("usage") or {}
        usage["prompt_tokens"] += u.get("prompt_tokens") or 0
        usage["completion_tokens"] += u.get("completion_tokens") or 0
        usage["cost"] += u.get("cost") or 0.0
    result = {"experiment": manifest["experiment"], "model": manifest["model"],
              "metrics": metrics, "primary_passed": passed, "usage": usage, "rows": rows}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "primary_passed": passed, "usage": usage}, indent=1))


if __name__ == "__main__":
    main()
