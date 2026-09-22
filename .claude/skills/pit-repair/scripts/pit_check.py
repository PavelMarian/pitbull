#!/usr/bin/env python3
"""pit-repair: дифференциальная проверка функции признаков на чтение будущего.

Один вызов = одна итерация цикла «проверка → починка → проверка»:

    pit_check.py --code get_features.py [--state pit_state.json]

Что делает:
  1. Отказывается работать, если в окружении есть ключ LLM-провайдера
     (OPENROUTER_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY). Скрипт
     исполняет непроверенный код, сгенерированный моделью; сетевой этап и
     этап исполнения должны быть разными процессами (дисциплина A2).
  2. Загружает код в песочнице (AST-фильтр + урезанные builtins,
     как в pilot/a2_draft_validate.py) и вызывает get_features трижды на
     каждом моменте: полная база / усечённая база (oracle.truncate) /
     база с canary-возмущением (oracle.perturb_canary).
  3. Расхождение full-vs-truncated = witness (доказательство чтения строк
     с временем > seed_time). Расхождение full-vs-canary = canary
     (чтение поздно доступных полей). Программа «чистая», только если оба
     уровня молчат на всех моментах: диагностических и held-out.
  4. Пишет историю итераций в state-файл (состав полей как в
     pilot/a2_full_manifest.json: clean, clean_at, history[...].checks).

Коды выхода: 0 = CLEAN, 1 = LEAK (нужна починка), 2 = кандидат не
исполнился, 3 = отказ по дисциплине безопасности / ошибка аргументов,
4 = бюджет итераций исчерпан (статус failed).

Зависимости: prestudy/oracle.py и prestudy/p3_baseline_run.py из репозитория
PITBULL (корень ищется от расположения этого файла, либо --pitbull-root /
PITBULL_ROOT). Данные Olist: P3_OLIST или <root>/PITFALL_olist_data.
"""
import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

FORBIDDEN_ENV = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
DEFAULT_DEV_SEEDS = ["2018-01-01", "2018-04-01", "2018-07-01"]
DEFAULT_HELD_OUT_SEEDS = ["2017-10-01", "2018-02-01", "2018-06-01"]
DEFAULT_MAX_ITERATIONS = 5
N_ENTITIES = 15

# Тот же AST-фильтр, что в pilot/a2_draft_validate.py (A2/A4).
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
            "pit_check: отказ. В окружении есть ключ провайдера: "
            + ", ".join(present)
            + ". Этот процесс исполняет сгенерированный код; запустите его в окружении "
              "без ключей (unset ...), как требует дисциплина A2.\n")
        sys.exit(3)


def find_root(explicit):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("PITBULL_ROOT"):
        candidates.append(Path(os.environ["PITBULL_ROOT"]))
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    for c in candidates:
        if (c / "prestudy" / "oracle.py").exists() and (c / "prestudy" / "p3_baseline_run.py").exists():
            return c.resolve()
    sys.stderr.write("pit_check: не найден корень PITBULL (prestudy/oracle.py); "
                     "укажите --pitbull-root или PITBULL_ROOT.\n")
    sys.exit(3)


def load_harness(root):
    # p3_baseline_run импортирует litellm ради генерации; здесь сеть не нужна,
    # и в окружении проверки клиента LLM быть не должно. Подменяем модуль пустышкой.
    sys.modules.setdefault("litellm", types.ModuleType("litellm"))
    sys.path.insert(0, str(root / "prestudy"))
    import p3_baseline_run as H  # noqa: E402
    from oracle import frames_equal, perturb_canary, truncate  # noqa: E402
    return H, frames_equal, perturb_canary, truncate


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


def make_checker(H, frames_equal, perturb_canary, truncate, timeout):
    import numpy as np
    import pandas as pd

    def entities(seed):
        _, products, _ = H.labels(seed)
        return np.random.RandomState(0).choice(products, size=N_ENTITIES, replace=False)

    def diff_columns(left, right):
        if left is None or right is None or not hasattr(left, "shape") or not hasattr(right, "shape"):
            return ["__shape__"] if left is not right else []
        if left.shape != right.shape or list(left.columns) != list(right.columns):
            return ["__shape__"]
        return [col for col in left.columns if not frames_equal(left[[col]], right[[col]])]

    def check(fn, seed):
        t = pd.Timestamp(seed)
        ids = entities(seed)
        full = H._call_with_timeout(fn, H.DB, ids, t, timeout=timeout)
        trunc = H._call_with_timeout(fn, truncate(H.DB, t), ids, t, timeout=timeout)
        canary = H._call_with_timeout(fn, perturb_canary(H.DB, t), ids, t, timeout=timeout)
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
    parser.add_argument("--code", required=True, help="файл с get_features (кандидат)")
    parser.add_argument("--state", default=None, help="state-файл истории (по умолчанию pit_state.json рядом с кодом)")
    parser.add_argument("--dev-seeds", default=",".join(DEFAULT_DEV_SEEDS))
    parser.add_argument("--held-out-seeds", default=",".join(DEFAULT_HELD_OUT_SEEDS))
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--timeout", type=int, default=30, help="секунд на один вызов get_features")
    parser.add_argument("--pitbull-root", default=None)
    parser.add_argument("--json", action="store_true", help="печатать только JSON-итог")
    args = parser.parse_args()
    args.dev_seeds = [s for s in args.dev_seeds.split(",") if s]
    args.held_out_seeds = [s for s in args.held_out_seeds.split(",") if s]

    refuse_if_keys_in_env()
    root = find_root(args.pitbull_root)
    code_path = Path(args.code)
    if not code_path.exists():
        sys.stderr.write(f"pit_check: нет файла {code_path}\n")
        sys.exit(3)
    code = code_path.read_text(encoding="utf-8")
    state_path = Path(args.state) if args.state else code_path.with_name("pit_state.json")
    state = load_state(state_path, code, args)

    if state["status"] in ("clean", "failed"):
        summary = {"verdict": state["status"].upper(), "iteration": len(state["history"]) - 1,
                   "clean_at": state["clean_at"], "program_uid": state["program_uid"],
                   "next_action": "цикл уже завершён; новые проверки не засчитываются"}
        print(json.dumps(summary, ensure_ascii=False))
        sys.exit(0 if state["status"] == "clean" else 4)

    iteration = len(state["history"])  # 0 = исходная детекция, 1..N = кандидаты починки
    H, frames_equal, perturb_canary, truncate = load_harness(root)
    import numpy as np
    import pandas as pd
    check = make_checker(H, frames_equal, perturb_canary, truncate, args.timeout)

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
        nxt = (f"кандидат не исполнился ({entry['error']}); итерация {iteration}/{state['max_iterations']} "
               f"засчитана. " + ("Устраните причину, выполните шаг починки из SKILL.md и запустите проверку снова." if state["status"] == "pending"
                                 else "Бюджет итераций исчерпан: статус failed."))
    elif entry["clean"]:
        verdict, code_ = "CLEAN", 0
        nxt = ("исходный код чист, починка не нужна." if iteration == 0
               else f"починка подтверждена на итерации {iteration}; остановитесь.")
    elif state["status"] == "failed":
        verdict, code_ = "FAILED", 4
        nxt = f"утечка сохраняется после {state['max_iterations']} итераций починки; остановитесь и отчитайтесь."
    else:
        verdict, code_ = "LEAK", 1
        nxt = (f"утечка найдена; использовано {iteration}/{state['max_iterations']} итераций починки. "
               "Перепишите get_features по инструкции починки из SKILL.md и запустите проверку снова.")
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
            print("  witness (доказательство чтения будущего):", ", ".join(witness))
        if canary:
            print("  canary (чтение поздно доступных полей):", ", ".join(canary))
        for r in entry.get("checks", []):
            print(f"  {r['split']:8s} {r['seed']}: witness={int(r['witness'])} canary={int(r['canary'])}")
        print("  next:", nxt)
    print(json.dumps(summary, ensure_ascii=False))
    sys.exit(code_)


if __name__ == "__main__":
    main()
