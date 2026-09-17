"""Score the completed A4 repetitions using the preregistered estimands."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "a4_manifest.json"
OUT = HERE / "a4_results.json"
RESPONSES = HERE / "a4_responses"
SEED = 20260915
N_BOOT = 100_000


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [centre - half, centre + half]


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = []
    for repetition in manifest["repetitions"]:
        for program in repetition["programs"]:
            ok_iterations = {
                h["iteration"] for h in program.get("history", [])
                if h.get("status") == "ok"
            }
            records.append({
                "uid": program["program_uid"],
                "repeat": repetition["repeat"],
                "mechanism": program["mechanism"],
                "clean_at": program.get("clean_at"),
                "ok_iterations": ok_iterations,
            })

    by_uid: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_uid[record["uid"]].append(record)
    uids = sorted(by_uid)
    if len(records) != 186 or len(uids) != 62:
        raise ValueError(f"Expected 186 records in 62 clusters, got {len(records)} in {len(uids)}")

    rng = np.random.default_rng(SEED)
    metrics = []
    for k in range(1, 6):
        clean = np.array([
            int(record["clean_at"] is not None and record["clean_at"] <= k)
            for record in records
        ], dtype=np.int8)
        executable = np.array([
            int(any(iteration <= k for iteration in record["ok_iterations"]))
            for record in records
        ], dtype=np.int8)

        cluster_rates = np.array([
            sum(record["clean_at"] is not None and record["clean_at"] <= k
                for record in by_uid[uid]) / 3
            for uid in uids
        ])
        boot = np.empty(N_BOOT)
        for start in range(0, N_BOOT, 10_000):
            stop = min(start + 10_000, N_BOOT)
            indices = rng.integers(0, len(uids), size=(stop - start, len(uids)))
            boot[start:stop] = cluster_rates[indices].mean(axis=1)

        majority = {
            uid: int(sum(record["clean_at"] is not None and record["clean_at"] <= k
                         for record in by_uid[uid]) >= 2)
            for uid in uids
        }
        metrics.append({
            "k": k,
            "all": {
                "clean": int(clean.sum()), "n": len(clean),
                "rate": float(clean.mean()), "wilson95": wilson(int(clean.sum()), len(clean)),
                "cluster_bootstrap95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            },
            "executable_by_k": {
                "clean": int(clean.sum()), "n": int(executable.sum()),
                "rate": float(clean.sum() / executable.sum()),
            },
            "majority_clean_programs": sum(majority.values()),
        })

    majority_1 = {
        uid: int(sum(record["clean_at"] is not None and record["clean_at"] <= 1
                     for record in by_uid[uid]) >= 2)
        for uid in uids
    }
    majority_5 = {
        uid: int(sum(record["clean_at"] is not None and record["clean_at"] <= 5
                     for record in by_uid[uid]) >= 2)
        for uid in uids
    }
    b = sum(majority_1[uid] == 1 and majority_5[uid] == 0 for uid in uids)
    c = sum(majority_1[uid] == 0 and majority_5[uid] == 1 for uid in uids)

    mechanisms = {}
    for mechanism in sorted({record["mechanism"] for record in records}):
        subset = [record for record in records if record["mechanism"] == mechanism]
        mechanisms[mechanism] = []
        for k in range(1, 6):
            count = sum(record["clean_at"] is not None and record["clean_at"] <= k
                        for record in subset)
            mechanisms[mechanism].append({
                "k": k, "clean": count, "n": len(subset), "rate": count / len(subset)
            })

    repeat_summary = []
    for repetition in manifest["repetitions"]:
        programs = repetition["programs"]
        repeat_summary.append({
            "repeat": repetition["repeat"],
            "clean_at_5": sum(p["status"] == "clean" for p in programs),
            "failed_at_5": sum(p["status"] == "failed" for p in programs),
        })

    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "total_tokens": 0, "cost": 0.0, "api_errors": 0}
    for path in RESPONSES.glob("repeat_*/iter_*/*.json"):
        response = json.loads(path.read_text(encoding="utf-8"))
        for attempt in response.get("attempts", []):
            usage["calls"] += 1
            usage["api_errors"] += int(attempt.get("status") == "api_error")
        response_usage = response.get("usage", {})
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            usage[key] += response_usage.get(key) or 0
    usage["cost"] = round(usage["cost"], 6)

    result = {
        "experiment": "A4 RQ2b F1 held-out CLEAN@k",
        "status": "complete; in_review",
        "verdict": "positive" if metrics[-1]["all"]["rate"] >= 0.8 else "negative",
        "threshold": "CLEAN@5 >= 0.8",
        "records": len(records), "program_clusters": len(uids),
        "repeat_summary": repeat_summary,
        "clean_at_k": metrics,
        "by_mechanism": mechanisms,
        "mcnemar_majority_clean_1_vs_5": {
            "clean_1_not_5": b, "not_1_clean_5": c,
            "exact_two_sided_p": exact_mcnemar(b, c),
        },
        "bootstrap": {"resamples": N_BOOT, "seed": SEED},
        "new_a4_network_usage_excluding_reused_a2": usage,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest["status"] = "complete; in_review"
    manifest["checkpoint"] = {
        "stopped_after": "all 3 repeats complete through iteration 5",
        "repeat_1": "complete: 61/62 clean",
        "repeat_2": "complete: 60/62 clean",
        "repeat_3": "complete: 61/62 clean",
        "next_step": "human review of pilot/a4_results.json and experiment report",
        "pending_program_uids": [],
    }
    manifest["results"] = str(OUT.relative_to(ROOT := HERE.parent)).replace("\\", "/")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
