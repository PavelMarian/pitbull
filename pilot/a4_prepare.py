"""Freeze A4/RQ2b: A1 witness corpus, three nested F1 repair repetitions."""
import copy
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
A2 = HERE / "a2_full_manifest.json"
OUT = HERE / "a4_manifest.json"


def fresh(program):
    return {
        "program_uid": program["program_uid"], "source_row": program["source_row"],
        "program": program["program"], "author_model": program["author_model"],
        "source_idx": program["source_idx"], "mechanism": program["mechanism"],
        "source_code": program["source_code"], "current_code": program["source_code"],
        "status": "pending", "history": [],
    }


def main():
    a2 = json.loads(A2.read_text(encoding="utf-8"))
    corpus = [p for p in a2["programs"] if p.get("detection_origin") == "witness"]
    if len(corpus) != 62:
        raise ValueError(f"expected 62 executable A1 leaks, got {len(corpus)}")
    repetitions = []
    repetitions.append({"repeat": 1, "source": "A2 reuse: identical F1 protocol",
                        "programs": copy.deepcopy(corpus)})
    for repeat in (2, 3):
        repetitions.append({"repeat": repeat, "source": "fresh independent repair run",
                            "programs": [fresh(p) for p in corpus]})
    manifest = {
        "experiment": "A4 RQ2b F1 held-out CLEAN@k",
        "frozen_before_new_calls": True,
        "model": "z-ai/glm-5.3-flash", "temperature": 0.2,
        "max_iterations": 5, "n_repetitions": 3,
        "dev_seeds": a2["dev_seeds"], "held_out_seeds": a2["held_out_seeds"],
        "corpus_rule": "A1 executable witness leaks only; generation artifact and A2 canary-only additions excluded",
        "no_code_policy": "one immediate identical retry; if still absent, continue at next k; all-denominator counts failure",
        "estimands": ["CLEAN@k among all program-repetitions",
                       "CLEAN@k among program-repetitions with an executable candidate by k"],
        "primary_ci": "cluster bootstrap by program uid, 100000 resamples, seed 20260915",
        "secondary": "Wilson over program-repetitions; exact McNemar CLEAN@1 vs CLEAN@5 on per-program majority",
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__},
        "repetitions": repetitions,
    }
    OUT.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("programs", len(corpus), "repetitions", len(repetitions),
          "mechanisms", {m: sum(p["mechanism"] == m for p in corpus)
                         for m in sorted({p["mechanism"] for p in corpus})})


if __name__ == "__main__":
    main()
