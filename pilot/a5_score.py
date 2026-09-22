"""Recompute the final A5 summary from immutable/checkpointed artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLIST = ROOT / "pilot" / "a5_olist_density.json"
EVENT = ROOT / "rel" / "out" / "sql_oracle_a5_event_test.csv"
F1 = ROOT / "rel" / "out" / "sql_oracle_a5_f1.csv"
EVENT_TASKS = ROOT / "PITFALL_ext_data" / "rel-event" / "tasks"
OUT = ROOT / "pilot" / "a5_results.json"


def detection_probability(hits: int, n: int, k: int) -> float:
    if n < k:
        return float("nan")
    misses = n - hits
    return 1.0 - (math.comb(misses, k) / math.comb(n, k) if misses >= k else 0.0)


def mean(xs: list[float]) -> float:
    vals = [x for x in xs if not math.isnan(x)]
    return sum(vals) / len(vals)


def curve(programs: list[dict], verdict: str, mode: str) -> list[dict]:
    out = []
    selected = programs
    if mode == "complete_grid":
        selected = [p for p in programs if all(c["status"] == "ran" for c in p["checks"])]
    for k in range(1, 6):
        probabilities = []
        for program in selected:
            ran = [c for c in program["checks"] if c["status"] == "ran"]
            hits = sum(bool(c.get(verdict)) for c in ran)
            n = len(program["checks"]) if mode == "scheduled_conservative" else len(ran)
            probabilities.append(detection_probability(hits, n, k))
        out.append({"k": k, "mean_program_detection": mean(probabilities)})
    return out


def program_fp(programs: list[dict], verdict: str) -> dict:
    fired = sum(any(bool(c.get(verdict)) for c in p["checks"] if c["status"] == "ran")
                for p in programs)
    n = len(programs)
    return {
        "programs_fired": fired,
        "programs": n,
        "rate": fired / n,
        "rule_of_three_upper95": 3 / n if fired == 0 else None,
        "exact_zero_upper95": 1 - 0.05 ** (1 / n) if fired == 0 else None,
    }


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def phases(rows: list[dict]) -> dict:
    return {
        phase: dict(Counter(r["verdict"] for r in rows if r["phase"] == phase))
        for phase in ("determinism", "negative_control", "main")
    }


def main() -> None:
    state = json.loads(OLIST.read_text(encoding="utf-8"))
    programs = state["programs"]
    if len(programs) != 92 or any(p["status"] != "complete" for p in programs):
        raise RuntimeError("A5 Olist grid is not complete")
    if any(len(p["checks"]) != 20 for p in programs):
        raise RuntimeError("Every A5 program must contain 20 scheduled checks")

    leak = [p for p in programs if p["kind"] == "leak"]
    clean = [p for p in programs if p["kind"] == "clean"]
    leak_full = [p for p in leak if all(c["status"] == "ran" for c in p["checks"])]
    clean_full = [p for p in clean if all(c["status"] == "ran" for c in p["checks"])]
    event = read_csv(EVENT)
    f1 = read_csv(F1)

    test_times = {}
    for task_dir in sorted(EVENT_TASKS.iterdir()):
        test_file = task_dir / "test.parquet"
        if test_file.exists():
            values = pd.read_parquet(test_file, columns=["timestamp"])["timestamp"].dropna().unique()
            test_times[task_dir.name] = sorted(str(pd.Timestamp(v)) for v in values)

    result = {
        "experiment": "A5 RQ3 density + RelBench baselines + machine time",
        "status": "complete; in_review",
        "olist": {
            "programs": {"leak": len(leak), "clean": len(clean), "total": len(programs)},
            "scheduled_checks": len(programs) * 20,
            "check_status": dict(Counter(c["status"] for p in programs for c in p["checks"])),
            "full_20_programs": {"leak": len(leak_full), "clean": len(clean_full)},
            "curves": {
                verdict: {
                    "conditional_on_execution": curve(leak, verdict, "conditional_on_execution"),
                    "scheduled_conservative": curve(leak, verdict, "scheduled_conservative"),
                    "complete_grid_only": curve(leak, verdict, "complete_grid"),
                }
                for verdict in ("witness", "canary")
            },
            "clean_program_false_positive": {
                verdict: {
                    "all_attempted_programs": program_fp(clean, verdict),
                    "complete_grid_only": program_fp(clean_full, verdict),
                }
                for verdict in ("witness", "canary")
            },
            "recorded_program_wall_seconds_lower_bound": state["results"]["wall_seconds"],
        },
        "rel_event": {
            "rows": len(event),
            "phases": phases(event),
            "evaluated_seeds": sorted({r["seed"] for r in event if r["phase"] == "main"}),
            "official_test_label_timestamps": test_times,
            "is_exact_benchmark_test_timestamp": all(
                r["seed"] in test_times.get(r["task"], []) for r in event if r["phase"] == "main"
            ),
            "seconds_total": sum(float(r["seconds"]) for r in event),
            "seconds_main": sum(float(r["seconds"]) for r in event if r["phase"] == "main"),
        },
        "rel_f1": {
            "rows": len(f1),
            "phases": phases(f1),
            "seconds_total": sum(float(r["seconds"]) for r in f1),
            "seconds_main": sum(float(r["seconds"]) for r in f1 if r["phase"] == "main"),
        },
        "notes": [
            "Olist timeout/error checks are misses in scheduled_conservative curves.",
            "Program-level FP is reported both for all attempted controls and 20/20 complete grids.",
            "Recorded Olist program time is a lower bound because resumed partial attempts were overwritten.",
            "The rel-event run used informative pre-test label moments, not the official 2012-11-29 test timestamp.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
