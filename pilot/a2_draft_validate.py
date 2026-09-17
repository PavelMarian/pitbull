"""Offline validation stage for model responses fetched by a2_draft_fetch.py."""
import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "prestudy"))
import p3_baseline_run as H  # noqa: E402
from oracle import frames_equal, perturb_canary, truncate  # noqa: E402

SOURCE = Path(os.environ.get("A2_DRAFT_FETCH_OUT", HERE / "a2_draft_glm53_fetch.json"))
OUT = Path(os.environ.get("A2_DRAFT_VALIDATE_OUT", HERE / "a2_draft_glm53_validation.json"))
DEV_SEEDS = ["2018-01-01", "2018-04-01", "2018-07-01"]
HELD_OUT_SEEDS = ["2017-10-01", "2018-02-01", "2018-06-01"]

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


def load_program(code):
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
    exec(compile(code, "<model-response>", "exec"), namespace)
    fn = namespace.get("get_features")
    if not callable(fn):
        raise ValueError("get_features missing")
    return fn


def entities(seed):
    _, products, _ = H.labels(seed)
    return np.random.RandomState(0).choice(products, size=15, replace=False)


def diff_columns(left, right):
    if left is None or right is None or not hasattr(left, "shape") or not hasattr(right, "shape"):
        return ["__shape__"] if left is not right else []
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return ["__shape__"]
    return [col for col in left.columns if not frames_equal(left[[col]], right[[col]])]


def check(fn, seed):
    t = pd.Timestamp(seed)
    ids = entities(seed)
    full = H._call_with_timeout(fn, H.DB, ids, t, timeout=30)
    trunc = H._call_with_timeout(fn, truncate(H.DB, t), ids, t, timeout=30)
    canary = H._call_with_timeout(fn, perturb_canary(H.DB, t), ids, t, timeout=30)
    witness_cols = diff_columns(full, trunc)
    canary_cols = diff_columns(full, canary)
    return {"seed": seed, "witness": bool(witness_cols), "canary": bool(canary_cols),
            "witness_columns": witness_cols, "canary_columns": canary_cols,
            "n_columns": len(full.columns) if hasattr(full, "columns") else None}


def main():
    fetched = json.loads(SOURCE.read_text(encoding="utf-8"))
    corpus_path = ROOT / "prestudy" / "p3_out" / "baseline" / "results.jsonl"
    corpus = [json.loads(line) for line in corpus_path.open(encoding="utf-8")]
    by_program = {
        f"{rec['model'].split('/')[-1]}#{rec['idx']}": rec
        for rec in corpus if rec.get("status") == "ok" and rec.get("code")
    }
    validated = []
    for item in fetched:
        result = {k: item.get(k) for k in ("program", "author_model", "repair_model", "arm",
                                             "source_code_sha256", "prompt_sha256")}
        code = item.get("code")
        started = time.time()
        if item.get("status") != "ok" or not code:
            result.update({"status": "no_code", "error": item.get("error")})
        else:
            try:
                fn = load_program(code)
                checks = []
                for split, seeds in (("dev", DEV_SEEDS), ("held_out", HELD_OUT_SEEDS)):
                    for seed in seeds:
                        row = check(fn, seed)
                        row["split"] = split
                        checks.append(row)
                clean = all(not row["witness"] and not row["canary"] for row in checks)
                source_rec = by_program.get(item["program"])
                original_columns = None
                new_columns = None
                diverging = []
                if source_rec:
                    original_fn = H._load_program(source_rec["code"])
                    ids = entities(DEV_SEEDS[0])
                    original_out = H._call_with_timeout(
                        original_fn, H.DB, ids, pd.Timestamp(DEV_SEEDS[0]), timeout=30
                    )
                    new_out = H._call_with_timeout(
                        fn, H.DB, ids, pd.Timestamp(DEV_SEEDS[0]), timeout=30
                    )
                    original_columns = list(original_out.columns) if hasattr(original_out, "columns") else None
                    new_columns = list(new_out.columns) if hasattr(new_out, "columns") else None
                    diverging = sorted({
                        col for seed_result in source_rec.get("seed_results", [])
                        for col in ((seed_result.get("detail") or {}).get("differing_columns") or [])
                    })
                kept = None if new_columns is None else [col for col in diverging if col in new_columns]
                result.update({"status": "ok", "clean": clean, "checks": checks,
                               "original_n_columns": len(original_columns) if original_columns is not None else None,
                               "new_n_columns": len(new_columns) if new_columns is not None else None,
                               "diverging_columns": diverging,
                               "diverging_kept": kept})
            except Exception as exc:
                result.update({"status": "rejected_or_error",
                               "error": f"{type(exc).__name__}: {exc}"[:500]})
        result["wall_seconds"] = round(time.time() - started, 1)
        validated.append(result)
        OUT.write_text(json.dumps(validated, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        print(result["program"], result["arm"], result["status"], result.get("clean"), flush=True)


if __name__ == "__main__":
    main()
