"""A5 program-level Olist density run with resumable per-program checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "pilot"
OUT = PILOT / "a5_olist_density.json"
A2 = PILOT / "a2_full_manifest.json"
BASELINE = ROOT / "prestudy" / "p3_out" / "baseline" / "results.jsonl"

import sys
sys.path.insert(0, str(PILOT))
import pilot_bc_density_canary as BC  # noqa: E402


GRID = BC._default_grid
MAX_SECONDS_PER_PROGRAM = 1800
N_CLEAN = 30


def uid(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def corpus() -> list[dict]:
    a2 = json.loads(A2.read_text(encoding="utf-8"))
    leaks = [
        {
            "program_uid": p["program_uid"], "program": p["program"],
            "kind": "leak", "mechanism": p["mechanism"], "code": p["source_code"],
        }
        for p in a2["programs"] if p.get("detection_origin") == "witness"
    ]
    if len(leaks) != 62:
        raise ValueError(f"Expected 62 executable witness leaks, got {len(leaks)}")

    clean = []
    seen = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") != "ok" or row.get("verdict") != "CLEAN":
            continue
        program_uid = uid(row["code"])
        if program_uid in seen:
            continue
        seen.add(program_uid)
        clean.append({
            "program_uid": program_uid,
            "program": f"{row['model'].split('/')[-1]}#{row['idx']}",
            "kind": "clean", "mechanism": "CLEAN_CONTROL", "code": row["code"],
        })
        if len(clean) == N_CLEAN:
            break
    if len(clean) != N_CLEAN:
        raise ValueError(f"Expected {N_CLEAN} clean controls, got {len(clean)}")
    return leaks + clean


def save(state: dict) -> None:
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    temp.replace(OUT)


def score(state: dict) -> dict:
    completed = [p for p in state["programs"] if p["status"] == "complete"]
    curves = {}
    for verdict in ("witness", "canary"):
        leak = [p for p in completed if p["kind"] == "leak"]
        curve = []
        for k in range(1, 6):
            probabilities = []
            for program in leak:
                checks = [c for c in program["checks"] if c["status"] == "ran"]
                n = len(checks)
                hits = sum(bool(c.get(verdict)) for c in checks)
                miss_probability = (
                    math.comb(n - hits, k) / math.comb(n, k)
                    if n >= k and n - hits >= k else 0.0
                )
                probabilities.append(1 - miss_probability)
            curve.append({"k": k, "mean_program_detection": sum(probabilities) / len(probabilities)})
        curves[verdict] = curve

    clean = [p for p in completed if p["kind"] == "clean"]
    false_positive = {}
    for verdict in ("witness", "canary"):
        fired = sum(any(c.get(verdict) for c in p["checks"] if c["status"] == "ran") for p in clean)
        n = len(clean)
        false_positive[verdict] = {
            "programs_fired": fired, "programs": n,
            "rate": fired / n,
            "rule_of_three_upper95": 3 / n if fired == 0 else None,
            "exact_zero_upper95": 1 - 0.05 ** (1 / n) if fired == 0 else None,
        }

    leak_density = {}
    for verdict in ("witness", "canary"):
        values = []
        for p in completed:
            if p["kind"] != "leak":
                continue
            checks = [c for c in p["checks"] if c["status"] == "ran"]
            values.append(sum(bool(c.get(verdict)) for c in checks) / len(checks))
        leak_density[verdict] = {
            "programs": len(values), "mean": sum(values) / len(values),
            "min": min(values), "max": max(values),
        }
    return {
        "completed_programs": len(completed),
        "total_programs": len(state["programs"]),
        "curves": curves,
        "clean_program_false_positive": false_positive,
        "leak_density": leak_density,
        "wall_seconds": sum(p.get("wall_seconds", 0) for p in completed),
    }


def main() -> None:
    if OUT.exists():
        state = json.loads(OUT.read_text(encoding="utf-8"))
        # Older checkpoint passes may have stopped a slow program at the former
        # six-minute guard.  A5 needs the full 20-point grid, so make those rows
        # resumable instead of silently treating them as complete.
        for program in state["programs"]:
            program["checks"] = [
                row for row in program["checks"] if row["status"] != "budget_exceeded"
            ]
            if len({row["seed"] for row in program["checks"]}) < len(GRID):
                program["status"] = "pending"
    else:
        state = {
            "experiment": "A5 Olist program-level density",
            "status": "in_process; not_finished",
            "grid": GRID, "leak_programs": 62, "clean_programs": N_CLEAN,
            "programs": [{**p, "status": "pending", "checks": []} for p in corpus()],
        }
        save(state)

    for index, program in enumerate(state["programs"], 1):
        if program["status"] == "complete":
            continue
        try:
            fn = BC.H._load_program(program["code"])
        except Exception as exc:
            program.update({"status": "complete", "load_error": f"{type(exc).__name__}: {exc}",
                            "wall_seconds": 0.0})
            save(state)
            continue
        started = time.time()
        checked_seeds = {row["seed"] for row in program["checks"]}
        for seed in GRID:
            if seed in checked_seeds:
                continue
            if time.time() - started > MAX_SECONDS_PER_PROGRAM:
                program["checks"].append({"seed": seed, "status": "budget_exceeded"})
                break
            row = BC.run_three(fn, seed)
            row["seed"] = seed
            program["checks"].append(row)
            save(state)
        complete_grid = len({row["seed"] for row in program["checks"]}) == len(GRID)
        program["status"] = "complete" if complete_grid else "pending"
        program["wall_seconds"] = round(time.time() - started, 3)
        save(state)
        ran = [c for c in program["checks"] if c["status"] == "ran"]
        print(f"{index:02d}/{len(state['programs'])} {program['kind']} {program['program']} "
              f"ran={len(ran)} witness={sum(c.get('witness', False) for c in ran)} "
              f"canary={sum(c.get('canary', False) for c in ran)} "
              f"seconds={program['wall_seconds']:.1f}", flush=True)

    if any(program["status"] != "complete" for program in state["programs"]):
        state["status"] = "in_process; not_finished"
        save(state)
        raise SystemExit("Some programs remain incomplete; rerun to resume them.")
    state["status"] = "complete; in_review"
    state["results"] = score(state)
    save(state)
    print(json.dumps(state["results"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
