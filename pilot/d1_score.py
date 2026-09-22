"""Score the D1 skill-parity run against A2 using the preregistered rule (skill-artifact-plan.md §4).

Reads pilot/d1_runs/<uid>/{run.json,pit_state.json} and pilot/a2_full_manifest.json.
Outputs pilot/d1_results.json and prints a summary.

Rule (fixed before the run):
  1. exact McNemar on 76 paired CLEAN/not-CLEAN outcomes (skill vs A2), p > 0.05;
  2. CLEAN@5 (skill) >= 0.952 (lower bound of the A4 cluster-bootstrap CI);
  3. every skill failure is either the known A2 failure / payment mechanism or explained separately.
Cost and call counts are diagnostics, not gates.
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "a2_full_manifest.json"
RUNS = Path(os.environ.get("D1_RUNS", HERE / "d1_runs"))
OUT = Path(os.environ.get("D1_OUT", HERE / "d1_results.json"))
SEED = 20260915
N_BOOT = 100_000
A4_LOWER_BOUND = 0.952
KNOWN_A2_FAILURE = "f1cade690ca333e8"
# OpenRouter list price for z-ai/glm-5.3-flash at run time, USD per 1M tokens (filled by runner env if known)
PRICE_IN = float(os.environ.get("D1_PRICE_IN_PER_M", "0"))
PRICE_OUT = float(os.environ.get("D1_PRICE_OUT_PER_M", "0"))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [centre - half, centre + half]


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    a2 = {p["program_uid"]: p for p in manifest["programs"]}
    rows = []
    for p in manifest["programs"]:
        uid = p["program_uid"]
        run_path = RUNS / uid / "run.json"
        state_path = RUNS / uid / "pit_state.json"
        run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else None
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        hist = (state or {}).get("history", [])
        # skill clean_at counts repair iterations (state iteration 0 = detection); A2 clean_at counts iterations 1..5
        clean_at = (state or {}).get("clean_at")
        detected = bool(hist) and not hist[0].get("clean", False) and hist[0].get("status") == "ok"
        final_checks = hist[-1].get("checks", []) if hist else []
        rows.append({
            "uid": uid, "program": p["program"], "mechanism": p["mechanism"],
            "a2_clean_at": p.get("clean_at"), "a2_clean": p["status"] == "clean",
            "run_status": run and run["status"], "skill_loaded": run and run.get("skill_loaded"),
            "pit_check_calls": run and run.get("pit_check_calls"), "turns": run and run.get("assistant_turns"),
            "usage": (run or {}).get("session", {}) and run["session"].get("usage"),
            "wall_seconds": run and run.get("wall_seconds"),
            "detected_at_0": detected, "iterations_used": max(len(hist) - 1, 0) if hist else None,
            "skill_status": (state or {}).get("status"), "skill_clean_at": clean_at,
            "skill_clean": (state or {}).get("status") == "clean" and (clean_at or 0) >= 1,
            "clean_at_0": (state or {}).get("status") == "clean" and clean_at == 0,
            "final_witness": sorted({c for r in final_checks for c in r.get("witness_columns", [])}),
            "final_canary": sorted({c for r in final_checks for c in r.get("canary_columns", [])}),
            "final_error": hist[-1].get("error") if hist else None,
            "missing": run is None or state is None,
        })

    n = len(rows)
    missing = [r["uid"] for r in rows if r["missing"]]
    # a program counted CLEAN for the skill if the loop ended clean at a repair iteration k>=1;
    # a spurious clean at iteration 0 (detector silent on a known-leaking original) is reported separately.
    clean_at_k = []
    for k in range(1, 6):
        c = sum(1 for r in rows if r["skill_clean"] and r["skill_clean_at"] <= k)
        clean_at_k.append({"k": k, "clean": c, "n": n, "rate": c / n, "wilson95": wilson(c, n)})
    a2_clean_at_k = [{"k": k, "clean": sum(1 for r in rows if r["a2_clean"] and r["a2_clean_at"] <= k)} for k in range(1, 6)]

    rng = np.random.default_rng(SEED)
    outcomes = np.array([int(r["skill_clean"]) for r in rows])
    boot = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 10_000):
        stop = min(start + 10_000, N_BOOT)
        idx = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = outcomes[idx].mean(axis=1)
    ci = [float(x) for x in np.quantile(boot, [0.025, 0.975])]

    b = sum(1 for r in rows if r["a2_clean"] and not r["skill_clean"])   # A2 clean, skill not
    c = sum(1 for r in rows if not r["a2_clean"] and r["skill_clean"])   # skill clean, A2 not
    p_mcnemar = exact_mcnemar(b, c)

    by_mech = {}
    for mech in sorted({r["mechanism"] for r in rows}):
        sub = [r for r in rows if r["mechanism"] == mech]
        by_mech[mech] = {"n": len(sub), "skill_clean5": sum(r["skill_clean"] for r in sub),
                         "a2_clean5": sum(r["a2_clean"] for r in sub)}

    failures = [r for r in rows if not r["skill_clean"]]
    new_failure_class = [r["uid"] for r in failures
                         if r["uid"] != KNOWN_A2_FAILURE and r["mechanism"] != "PAYMENTS_NO_FILTER"]

    usage_in = sum((r["usage"] or {}).get("input_tokens", 0) for r in rows if r["usage"])
    usage_out = sum((r["usage"] or {}).get("output_tokens", 0) for r in rows if r["usage"])
    cost = (usage_in * PRICE_IN + usage_out * PRICE_OUT) / 1e6 if PRICE_IN or PRICE_OUT else None
    turns = [r["turns"] for r in rows if r["turns"] is not None]

    rate5 = clean_at_k[-1]["rate"]
    gate = {"mcnemar_p_gt_0.05": p_mcnemar > 0.05, "clean5_ge_a4_lower_bound": rate5 >= A4_LOWER_BOUND,
            "no_new_failure_class": len(new_failure_class) == 0}
    result = {
        "experiment": "D1 skill-repair parity vs A2", "runs_dir": str(RUNS), "n": n, "missing": missing,
        "clean_at_k": clean_at_k, "a2_clean_at_k": a2_clean_at_k,
        "clean5_cluster_bootstrap95": ci, "bootstrap": {"resamples": N_BOOT, "seed": SEED},
        "mcnemar_vs_a2": {"a2_clean_skill_not": b, "skill_clean_a2_not": c, "exact_two_sided_p": p_mcnemar,
                          "discordant_uids": [r["uid"] for r in rows if r["a2_clean"] != r["skill_clean"]]},
        "by_mechanism": by_mech,
        "skill_loaded": sum(1 for r in rows if r["skill_loaded"]),
        "detected_at_0": sum(1 for r in rows if r["detected_at_0"]),
        "spurious_clean_at_0": [r["uid"] for r in rows if r["clean_at_0"]],
        "run_status": dict(Counter(r["run_status"] for r in rows)),
        "failures": [{k: r[k] for k in ("uid", "program", "mechanism", "skill_status", "iterations_used",
                                         "final_witness", "final_canary", "final_error", "run_status", "a2_clean")}
                     for r in failures],
        "new_failure_class_candidates": new_failure_class,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out, "estimated_cost_usd": cost,
                  "price_per_M": [PRICE_IN, PRICE_OUT], "sessions_with_usage": sum(1 for r in rows if r["usage"])},
        "turns": {"mean": float(np.mean(turns)) if turns else None, "max": max(turns) if turns else None},
        "pit_check_calls_total": sum(r["pit_check_calls"] or 0 for r in rows),
        "gate": gate, "gate_all": all(gate.values()),
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
