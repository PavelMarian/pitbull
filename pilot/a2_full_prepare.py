"""Build the stable A2 manifest after the pinned clean-corpus inventory."""
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORPUS = ROOT / "prestudy" / "p3_out" / "baseline" / "results.jsonl"
LABELS = ROOT / "prestudy" / "p3_out" / "baseline" / "mechanism_labels.csv"
INVENTORIES = [HERE / "a2_full_clean_inventory_pinned.json",
               HERE / "a2_full_clean_inventory_pinned_part2.json",
               HERE / "a2_full_clean_inventory_pinned_part3.json",
               HERE / "a2_full_clean_inventory_pinned_part4.json"]
OUT = HERE / "a2_full_manifest.json"
DEV_SEEDS = ["2018-01-01", "2018-04-01", "2018-07-01"]
HELD_OUT_SEEDS = ["2017-10-01", "2018-02-01", "2018-06-01"]


def uid(code):
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def main():
    rows = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    leaks = [(i, rec) for i, rec in enumerate(rows)
             if rec.get("status") == "ok" and rec.get("verdict") == "LEAK"]
    labels = list(csv.DictReader(LABELS.open(encoding="utf-8-sig")))
    if len(leaks) != len(labels):
        raise ValueError(f"LEAK/label mismatch: {len(leaks)} != {len(labels)}")

    entries = []
    excluded = []
    for (row_index, rec), label in zip(leaks, labels):
        if rec["model"] != label["model"] or str(rec["idx"]) != str(label["idx"]):
            raise ValueError(f"label order mismatch at source row {row_index}")
        mechanism = label["mechanism"]
        base = {"program_uid": uid(rec["code"]), "source_row": row_index,
                "program": f"{rec['model'].split('/')[-1]}#{rec['idx']}",
                "author_model": rec["model"], "source_idx": rec["idx"],
                "mechanism": mechanism, "detection_origin": "witness",
                "source_code": rec["code"], "current_code": rec["code"],
                "status": "pending", "history": []}
        if mechanism == "GENERATION_ARTIFACT":
            excluded.append({**base, "reason": "not an executable leakage mechanism"})
        else:
            entries.append(base)

    inventory = []
    for path in INVENTORIES:
        inventory.extend(json.loads(path.read_text(encoding="utf-8")))
    grouped = defaultdict(list)
    for result in inventory:
        if result.get("kind") == "corpus_clean" and result.get("program_uid"):
            grouped[result["program_uid"]].append(result)
    inventory_losses = []
    for program_uid, results in grouped.items():
        ran = [r for r in results if r.get("status") == "ran"]
        source_row = results[0]["source_row"]
        rec = rows[source_row]
        if any(r.get("witness") or r.get("canary") for r in ran):
            entries.append({"program_uid": program_uid, "source_row": source_row,
                            "program": results[0]["program"], "author_model": rec["model"],
                            "source_idx": rec["idx"], "mechanism": "SELF_AVAILABILITY_CANARY",
                            "detection_origin": "canary_only", "source_code": rec["code"],
                            "current_code": rec["code"], "status": "pending", "history": [],
                            "inventory_checks": ran})
        elif len(ran) < len(DEV_SEEDS):
            inventory_losses.append({"program_uid": program_uid, "source_row": source_row,
                                     "program": results[0]["program"],
                                     "ran": len(ran), "statuses": [r.get("status") for r in results]})

    if len({e["program_uid"] for e in entries}) != len(entries):
        raise ValueError("duplicate program_uid in M")
    controls = {}
    for name in ("correct_reference", "naive_max", "delivery_date_feature"):
        by_seed = {r.get("seed"): r for r in inventory
                   if r.get("program") == name and r.get("status") == "ran"}
        subset = list(by_seed.values())
        controls[name] = {"ran": len(subset),
                          "witness": sum(bool(r.get("witness")) for r in subset),
                          "canary": sum(bool(r.get("canary")) for r in subset)}
    manifest = {
        "experiment": "A2 RQ0 Olist full run",
        "map_level": "manual per-table + declared self-availability channels",
        "repair_arm": "F1", "max_iterations": 5,
        "dev_seeds": DEV_SEEDS, "held_out_seeds": HELD_OUT_SEEDS,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__},
        "controls": controls, "M": len(entries), "programs": entries,
        "excluded": excluded, "inventory_losses": inventory_losses,
    }
    OUT.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("M", len(entries), "witness", sum(e["detection_origin"] == "witness" for e in entries),
          "canary_only", sum(e["detection_origin"] == "canary_only" for e in entries),
          "losses", len(inventory_losses))


if __name__ == "__main__":
    main()
