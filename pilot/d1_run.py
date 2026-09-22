"""D1: agent-driven repair through OpenHarness + the `pit-repair` skill.

For every program of the A2 manifest (same M=76, identified by program_uid),
create a work dir with the original leaking code, launch OpenHarness in
non-interactive mode with a task prompt that does NOT name the skill, and let
the agent (`z-ai/glm-5.3-flash` via OpenRouter) decide to load `pit-repair`,
run the differential check, repair and re-check. The skill's own state file
(`pit_state.json`) is the measurement; the stream-json transcript is kept
to verify that the skill was actually invoked.

Network stage and execution stage stay separate processes: the API key lives
only in the OpenHarness settings file (see d1_make_ohcfg.py); this runner
strips *_API_KEY from the environment it passes to OpenHarness, so the
Bash tool that executes candidates never sees a key, and pit_check.py refuses
to run if one is present.

Env: D1_OH_CONFIG_DIR (default ~/.openharness-d1), D1_RUNS (default
pilot/d1_runs), D1_UIDS (comma list to restrict), D1_CONCURRENCY (default 3),
D1_MAX_TURNS (default 60), D1_ROUND (label stored in run.json, default 1).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "prestudy"))

MANIFEST = HERE / "a2_full_manifest.json"
RUNS = Path(os.environ.get("D1_RUNS", HERE / "d1_runs"))
CFG = Path(os.environ.get("D1_OH_CONFIG_DIR", Path.home() / ".openharness-d1"))
OH = Path(os.environ.get("D1_OH_BIN", Path.home() / ".openharness-venv" / "bin" / "oh"))
PIT_PYTHON = Path(os.environ.get("PIT_PYTHON", Path.home() / ".venvs" / "pitbull-d1" / "bin" / "python"))
CONCURRENCY = int(os.environ.get("D1_CONCURRENCY", "3"))
MAX_TURNS = int(os.environ.get("D1_MAX_TURNS", "60"))
ROUND = int(os.environ.get("D1_ROUND", "1"))
RUN_TIMEOUT = int(os.environ.get("D1_RUN_TIMEOUT", "2400"))

TASK_SUFFIX = """

Текущая реализация этой функции лежит в файле `get_features.py` в рабочей
директории. Проверь её на корректность по времени (утечку данных из будущего)
и, если утечка есть, почини код в этом же файле. В конце сообщи итоговый вердикт."""


def neutral_prompt():
    import types
    sys.modules.setdefault("litellm", types.ModuleType("litellm"))
    import p3_baseline_run as H  # noqa: E402  (loads Olist once; needed for the schema block)
    return H.NEUTRAL_PROMPT


def clean_env():
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    env["OPENHARNESS_CONFIG_DIR"] = str(CFG)
    env["PIT_PYTHON"] = str(PIT_PYTHON)
    env["P3_OLIST"] = str(ROOT / "PITFALL_olist_data")
    env["PITBULL_ROOT"] = str(ROOT)
    env["PATH"] = f"{ROOT / '.claude' / 'skills' / 'pit-repair' / 'bin'}:{PIT_PYTHON.parent}:{env.get('PATH', '')}"
    return env


def session_usage(cwd):
    """OpenHarness stores per-session usage under <cfg>/data/sessions/<name>/latest.json."""
    best = None
    for path in (CFG / "data" / "sessions").glob("*/latest.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("cwd") == str(cwd):
            if best is None or data.get("created_at", "") > best.get("created_at", ""):
                best = data
    if best is None:
        return None
    return {"usage": best.get("usage"), "message_count": best.get("message_count"),
            "session_id": best.get("session_id"), "model": best.get("model")}


def run_program(program, prompt, env):
    uid = program["program_uid"]
    work = RUNS / uid
    if (work / "run.json").exists():
        return uid, "skip(exists)"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "get_features.py").write_text(program["source_code"], encoding="utf-8")
    (work / "prompt.txt").write_text(prompt, encoding="utf-8")
    started = time.time()
    cmd = [str(OH), "-p", prompt, "--output-format", "stream-json", "--max-turns", str(MAX_TURNS),
           "--permission-mode", "full_auto"]
    try:
        proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=RUN_TIMEOUT)
        status, rc = "ok", proc.returncode
        (work / "oh_stream.jsonl").write_text(proc.stdout, encoding="utf-8")
        (work / "oh_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        status, rc = "timeout", None
        (work / "oh_stream.jsonl").write_text(exc.stdout or "", encoding="utf-8")
        (work / "oh_stderr.txt").write_text(exc.stderr or "", encoding="utf-8")
    events = []
    for line in (work / "oh_stream.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    tools = [e for e in events if e.get("type") == "tool_started"]
    skill_loaded = any(t["tool_name"] == "skill" and "pit" in json.dumps(t.get("tool_input", {})).lower()
                       for t in tools)
    checks = sum(1 for t in tools if "pit-check" in json.dumps(t.get("tool_input", {}))
                 or "pit_check" in json.dumps(t.get("tool_input", {})))
    final_text = next((e.get("text") for e in reversed(events) if e.get("type") == "assistant_complete"), None)
    errors = [e.get("message") for e in events if e.get("type") == "error"]
    state = None
    if (work / "pit_state.json").exists():
        state = json.loads((work / "pit_state.json").read_text(encoding="utf-8"))
    run = {"program_uid": uid, "program": program["program"], "mechanism": program["mechanism"],
           "round": ROUND, "status": status, "returncode": rc, "wall_seconds": round(time.time() - started, 1),
           "assistant_turns": sum(1 for e in events if e.get("type") == "assistant_complete"),
           "tool_calls": len(tools), "skill_loaded": skill_loaded, "pit_check_calls": checks,
           "errors": errors[:5], "final_text": final_text,
           "pit_status": state and state.get("status"), "clean_at": state and state.get("clean_at"),
           "iterations": state and len(state.get("history", [])),
           "session": session_usage(work)}
    (work / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    return uid, f"{status} pit={run['pit_status']} clean_at={run['clean_at']} turns={run['assistant_turns']} skill={skill_loaded}"


def main():
    if not (CFG / "settings.json").exists():
        sys.exit(f"no OpenHarness config at {CFG}; run pilot/d1_make_ohcfg.py first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    programs = manifest["programs"]
    if os.environ.get("D1_UIDS"):
        wanted = set(os.environ["D1_UIDS"].split(","))
        programs = [p for p in programs if p["program_uid"] in wanted]
    prompt = neutral_prompt() + TASK_SUFFIX
    env = clean_env()
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"programs={len(programs)} concurrency={CONCURRENCY} max_turns={MAX_TURNS} runs={RUNS}", flush=True)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(run_program, p, prompt, env): p for p in programs}
        for future in as_completed(futures):
            uid, msg = future.result()
            print(uid, futures[future]["program"], msg, flush=True)


if __name__ == "__main__":
    main()
