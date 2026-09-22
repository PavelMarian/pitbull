"""Offline, checkpointed validation stage for one full-A2 repair iteration."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "prestudy"))
import a2_draft_validate as V  # noqa: E402

MANIFEST = Path(os.environ.get("A2_FULL_MANIFEST", HERE / "a2_full_manifest.json"))
RESPONSES = Path(os.environ.get("A2_FULL_RESPONSES", HERE / "a2_full_responses"))
ITERATION = int(os.environ["A2_FULL_ITERATION"])


def save(manifest):
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def main():
    if os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("validation refuses to run while OPENROUTER_API_KEY is present")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    response_dir = RESPONSES / f"iter_{ITERATION:02d}"
    for program in manifest["programs"]:
        if program["status"] == "clean" or any(h.get("iteration") == ITERATION for h in program["history"]):
            continue
        response_path = response_dir / f"{program['program_uid']}.json"
        if not response_path.exists():
            continue
        response = json.loads(response_path.read_text(encoding="utf-8"))
        history = {"iteration": ITERATION, "response": str(response_path.relative_to(ROOT))}
        code = response.get("code")
        if response.get("status") != "ok" or not code:
            history.update({"status": "no_code_or_api_error", "error": response.get("error")})
        else:
            try:
                fn = V.load_program(code)
                checks = []
                for split, seeds in (("dev", manifest["dev_seeds"]),
                                     ("held_out", manifest["held_out_seeds"])):
                    for seed in seeds:
                        row = V.check(fn, seed)
                        row["split"] = split
                        checks.append(row)
                clean = all(not row["witness"] and not row["canary"] for row in checks)
                history.update({"status": "ok", "clean": clean,
                                "candidate_sha256": hashlib.sha256(code.encode()).hexdigest(),
                                "checks": checks})
                program["current_code"] = code
                if clean:
                    program["status"] = "clean"
                    program["clean_at"] = ITERATION
            except Exception as exc:
                history.update({"status": "rejected_or_error",
                                "error": f"{type(exc).__name__}: {exc}"[:500]})
        program["history"].append(history)
        if ITERATION >= manifest["max_iterations"] and program["status"] != "clean":
            program["status"] = "failed"
        save(manifest)
        print(program["program"], program["program_uid"], history["status"],
              history.get("clean"), flush=True)
    counts = {k: sum(p["status"] == k for p in manifest["programs"])
              for k in ("pending", "clean", "failed")}
    print(counts, flush=True)


if __name__ == "__main__":
    main()
