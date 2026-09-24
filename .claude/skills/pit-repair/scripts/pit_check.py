#!/usr/bin/env python3
"""pit-repair: differential check of a feature function for reading the future.

One call = one iteration of the "check -> repair -> check" loop:

    pit_check.py --code get_features.py --db-dir DB/ --availability-map map.json \\
                  --entity-table orders --entity-column product_id

What it does:
  1. Refuses to run if an LLM provider key is present in the environment
     (OPENROUTER_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY). This process
     executes unvetted model-generated code; the network step and the
     execution step must be separate processes.
  2. Loads the code in a sandbox (AST filter + a restricted builtins set)
     and calls get_features three times per prediction time: full database /
     truncated database (checker_core.truncate) / canary-perturbed database
     (checker_core.perturb_canary).
  3. A full-vs-truncated divergence is a witness (proof the program read a
     row timestamped later than seed_time). A full-vs-canary divergence is a
     canary hit (reading a field that becomes available only after its own
     row's time, e.g. a delivery date). The program is CLEAN only if both
     levels stay silent at every tested time: development and held-out.
  4. Writes iteration history to a state file next to the code.

Exit codes: 0 = CLEAN, 1 = LEAK (needs repair), 2 = candidate didn't execute,
3 = safety-discipline refusal / argument error, 4 = iteration budget spent.

Dependencies: only checker_core.py, next to this script -- no repository
root, no external database loader. You supply the database (--db-dir, a
directory of *.csv tables) and the availability map (--availability-map, a
JSON file naming which column times each table and which columns are
self-available). See ../references/availability-map.schema.json and
../references/example-availability-map.olist.json for the format and a
worked example.
"""
import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checker_core as core  # noqa: E402

FORBIDDEN_ENV = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_N_ENTITIES = 15

ALLOWED_IMPORTS = {"pandas", "numpy"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "input", "breakpoint", "help", "globals",
                "locals", "vars", "getattr", "setattr", "delattr", "__import__"}
BANNED_METHODS = {
    "read_csv", "read_json", "read_parquet", "read_pickle", "read_excel", "read_sql",
    "read_fwf", "read_feather", "read_hdf", "read_html", "read_xml", "ExcelFile", "HDFStore",
    "to_csv", "to_json", "to_parquet", "to_pickle", "to_excel", "to_sql", "to_feather", "to_hdf",
    "load", "save", "loadtxt", "savetxt", "memmap", "dump", "dumps",
    "system", "popen", "spawn", "fork", "connect", "request", "urlopen",
}


def refuse_if_keys_in_env():
    present = [k for k in FORBIDDEN_ENV if os.environ.get(k)]
    if present:
        sys.stderr.write(
            "pit_check: refusing. LLM provider key present in environment: "
            + ", ".join(present)
            + ". This process executes generated code; run it in an environment "
              "without keys (unset ...) -- the network step and the execution step "
              "must be separate processes.\n")
        sys.exit(3)


def load_availability_map(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    time_cols = raw.get("time_cols", {})
    if not time_cols:
        sys.stderr.write("pit_check: availability map has an empty time_cols -- "
                          "nothing would ever be truncated.\n")
        sys.exit(3)
    return {
        "time_cols": time_cols,
        "self_availability_cols": raw.get("self_availability_cols", {}),
        "gatekeeper_cols": raw.get("gatekeeper_cols", {}),
        "key_hints": tuple(raw.get("key_hints", core.DEFAULT_KEY_HINTS)),
    }


def parse_dates_from_map(avail_map):
    """Union of time_cols and self_availability_cols per table, so load_db
    parses exactly the columns the map declares as time-bearing."""
    out = {}
    for name, col in avail_map["time_cols"].items():
        out.setdefault(name, []).append(col)
    for name, cols in avail_map["self_availability_cols"].items():
        out.setdefault(name, []).extend(cols)
    return out


def inspect_code(code):
    tree = ast.parse(code)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"from {node.module} import")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
                problems.append(f"call {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_METHODS:
                problems.append(f"method {node.func.attr}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"dunder attribute {node.attr}")
    return sorted(set(problems))


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import blocked: {name}")
    return __import__(name, globals, locals, fromlist, level)


def load_program(code, pd, np):
    problems = inspect_code(code)
    if problems:
        raise ValueError("unsafe code: " + ", ".join(problems))
    allowed = {
        "__import__": safe_import, "len": len, "range": range, "min": min, "max": max,
        "sum": sum, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate,
        "zip": zip, "list": list, "dict": dict, "set": set, "tuple": tuple, "float": float,
        "int": int, "str": str, "bool": bool, "any": any, "all": all, "isinstance": isinstance,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    }
    namespace = {"pd": pd, "np": np, "__builtins__": allowed}
    exec(compile(code, "<candidate>", "exec"), namespace)
    fn = namespace.get("get_features")
    if not callable(fn):
        raise ValueError("get_features missing")
    return fn


def make_checker(db, avail_map, entity_table, entity_column, n_entities, timeout):
    def entities(seed):
        return core.sample_entities(db, entity_table, entity_column, n_entities, seed=0)

    def diff_columns(left, right):
        if left is None or right is None or not hasattr(left, "shape") or not hasattr(right, "shape"):
            return ["__shape__"] if left is not right else []
        if left.shape != right.shape or list(left.columns) != list(right.columns):
            return ["__shape__"]
        return [col for col in left.columns if not core.frames_equal(left[[col]], right[[col]])]

    def check(fn, seed):
        import pandas as pd
        t = pd.Timestamp(seed)
        ids = entities(seed)
        full = core.call_with_timeout(fn, db, ids, t, timeout=timeout)
        trunc_db = core.truncate(db, t, avail_map["time_cols"])
        canary_db = core.perturb_canary(db, t, avail_map["time_cols"],
                                         self_avail_cols=avail_map["self_availability_cols"],
                                         gatekeeper_cols=avail_map["gatekeeper_cols"],
                                         key_hints=avail_map["key_hints"])
        trunc = core.call_with_timeout(fn, trunc_db, ids, t, timeout=timeout)
        canary = core.call_with_timeout(fn, canary_db, ids, t, timeout=timeout)
        witness_cols = diff_columns(full, trunc)
        canary_cols = diff_columns(full, canary)
        return {"seed": seed, "witness": bool(witness_cols), "canary": bool(canary_cols),
                "witness_columns": witness_cols, "canary_columns": canary_cols,
                "n_columns": len(full.columns) if hasattr(full, "columns") else None}

    return check


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_state(path, code, args):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "tool": "pit-repair", "program_uid": sha(code)[:16], "source_sha256": sha(code),
        "source_code": code, "max_iterations": args.max_iterations,
        "dev_seeds": args.dev_seeds, "held_out_seeds": args.held_out_seeds,
        "status": "pending", "clean_at": None, "history": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", required=True, help="file with get_features (the candidate)")
    parser.add_argument("--db-dir", required=True, help="directory of *.csv tables")
    parser.add_argument("--availability-map", required=True,
                         help="JSON: time_cols (required), self_availability_cols, "
                              "gatekeeper_cols, key_hints (all optional)")
    parser.add_argument("--entity-table", required=True, help="table to sample entity ids from")
    parser.add_argument("--entity-column", required=True, help="id column within --entity-table")
    parser.add_argument("--n-entities", type=int, default=DEFAULT_N_ENTITIES)
    parser.add_argument("--dev-seeds", required=True, help="comma-separated prediction times")
    parser.add_argument("--held-out-seeds", required=True,
                         help="comma-separated prediction times not shown in --dev-seeds' role")
    parser.add_argument("--state", default=None, help="history state file (default: pit_state.json next to --code)")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--timeout", type=int, default=30, help="seconds allowed per get_features call")
    parser.add_argument("--json", action="store_true", help="print only the final JSON summary")
    args = parser.parse_args()
    args.dev_seeds = [s for s in args.dev_seeds.split(",") if s]
    args.held_out_seeds = [s for s in args.held_out_seeds.split(",") if s]

    refuse_if_keys_in_env()
    avail_map = load_availability_map(args.availability_map)
    code_path = Path(args.code)
    if not code_path.exists():
        sys.stderr.write(f"pit_check: no such file {code_path}\n")
        sys.exit(3)
    code = code_path.read_text(encoding="utf-8")
    state_path = Path(args.state) if args.state else code_path.with_name("pit_state.json")
    state = load_state(state_path, code, args)

    if state["status"] in ("clean", "failed"):
        summary = {"verdict": state["status"].upper(), "iteration": len(state["history"]) - 1,
                   "clean_at": state["clean_at"], "program_uid": state["program_uid"],
                   "next_action": "loop already finished; new checks don't count"}
        print(json.dumps(summary, ensure_ascii=False))
        sys.exit(0 if state["status"] == "clean" else 4)

    iteration = len(state["history"])  # 0 = initial detection, 1..N = repair candidates
    db = core.load_db(args.db_dir, parse_dates=parse_dates_from_map(avail_map))
    import numpy as np
    import pandas as pd
    check = make_checker(db, avail_map, args.entity_table, args.entity_column, args.n_entities, args.timeout)

    entry = {"iteration": iteration, "candidate_sha256": sha(code), "code": code,
             "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    if iteration > 0 and sha(code) == state["history"][-1]["candidate_sha256"]:
        entry["note"] = "candidate identical to previous iteration"
    started = time.time()
    try:
        fn = load_program(code, pd, np)
        checks = []
        for split, seeds in (("dev", state["dev_seeds"]), ("held_out", state["held_out_seeds"])):
            for seed in seeds:
                row = check(fn, seed)
                row["split"] = split
                checks.append(row)
        clean = all(not r["witness"] and not r["canary"] for r in checks)
        entry.update({"status": "ok", "clean": clean, "checks": checks})
    except Exception as exc:
        entry.update({"status": "rejected_or_error", "clean": False,
                      "error": f"{type(exc).__name__}: {exc}"[:500]})
    entry["wall_seconds"] = round(time.time() - started, 1)
    state["history"].append(entry)

    if entry.get("clean"):
        state["status"] = "clean"
        state["clean_at"] = iteration
    elif iteration >= state["max_iterations"]:
        state["status"] = "failed"
    state["environment"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__}
    state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    witness = sorted({c for r in entry.get("checks", []) for c in r["witness_columns"]})
    canary = sorted({c for r in entry.get("checks", []) for c in r["canary_columns"]})
    if entry["status"] != "ok":
        verdict, code_ = "ERROR", 2
        nxt = (f"candidate didn't execute ({entry['error']}); iteration {iteration}/{state['max_iterations']} "
               "counted. " + ("Fix the cause, apply the repair step from SKILL.md, and check again."
                               if state["status"] == "pending" else "Iteration budget spent: status failed."))
    elif entry["clean"]:
        verdict, code_ = "CLEAN", 0
        nxt = ("source is clean, no repair needed." if iteration == 0
               else f"repair confirmed at iteration {iteration}; stop.")
    elif state["status"] == "failed":
        verdict, code_ = "FAILED", 4
        nxt = f"leak persists after {state['max_iterations']} repair iterations; stop and report."
    else:
        verdict, code_ = "LEAK", 1
        nxt = (f"leak found; used {iteration}/{state['max_iterations']} repair iterations. "
               "Rewrite get_features per the repair step in SKILL.md and check again.")
    summary = {"verdict": verdict, "iteration": iteration, "program_uid": state["program_uid"],
               "witness_columns": witness, "canary_columns": canary,
               "per_seed": [{k: r[k] for k in ("seed", "split", "witness", "canary")} for r in entry.get("checks", [])],
               "status": state["status"], "clean_at": state["clean_at"],
               "state_file": str(state_path), "next_action": nxt}
    if entry["status"] != "ok":
        summary["error"] = entry["error"]
    if not args.json:
        print(f"[pit-repair] iteration {iteration}: {verdict}")
        if witness:
            print("  witness (proof of reading the future):", ", ".join(witness))
        if canary:
            print("  canary (reading a late-available field):", ", ".join(canary))
        for r in entry.get("checks", []):
            print(f"  {r['split']:8s} {r['seed']}: witness={int(r['witness'])} canary={int(r['canary'])}")
        print("  next:", nxt)
    print(json.dumps(summary, ensure_ascii=False))
    sys.exit(code_)


if __name__ == "__main__":
    main()
