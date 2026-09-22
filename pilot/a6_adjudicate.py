"""Validate A6 annotations and calculate independent-annotation agreement."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pilot" / "a6"
LABELS = {"row_time", "column_time", "timeless", "unknown", "exclude"}
REQUIRED = {
    "availability_class", "boundary", "mutable", "checkability",
    "confidence", "rationale",
}


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["item_id"]: row for row in csv.DictReader(handle)}


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    lc, rc = Counter(left), Counter(right)
    expected = sum(lc[label] * rc[label] for label in LABELS) / len(left) ** 2
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)


def main() -> None:
    a1 = read(DATA / "annotator_1.csv")
    a2 = read(DATA / "annotator_2.csv")
    if set(a1) != set(a2):
        raise SystemExit("The annotator sheets contain different item IDs.")

    incomplete: dict[str, list[str]] = {"annotator_1": [], "annotator_2": []}
    for name, sheet in (("annotator_1", a1), ("annotator_2", a2)):
        for item_id, row in sheet.items():
            if not REQUIRED.issubset({k for k, value in row.items() if value.strip()}):
                incomplete[name].append(item_id)
            label = row["availability_class"].strip()
            if label and label not in LABELS:
                raise SystemExit(f"Invalid availability_class for {name}/{item_id}: {label}")
            if label in {"row_time", "column_time"} and not row["availability_column"].strip():
                raise SystemExit(f"Missing availability_column for {name}/{item_id}")

    complete_ids = [
        item_id for item_id in sorted(a1)
        if item_id not in incomplete["annotator_1"]
        and item_id not in incomplete["annotator_2"]
    ]
    disagreements = []
    exact = 0
    for item_id in complete_ids:
        left, right = a1[item_id], a2[item_id]
        signature_fields = ["availability_class", "availability_column", "boundary"]
        same = all(left[field].strip() == right[field].strip() for field in signature_fields)
        exact += int(same)
        if not same:
            disagreements.append({
                "item_id": item_id,
                "dataset": left["dataset"], "table": left["table"],
                "column": left["column"],
                "annotator_1": " | ".join(left[f].strip() for f in signature_fields),
                "annotator_2": " | ".join(right[f].strip() for f in signature_fields),
                "adjudicated_class": "", "adjudicated_column": "",
                "adjudicated_boundary": "", "adjudication_rationale": "",
            })

    with (DATA / "disagreements.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "item_id", "dataset", "table", "column", "annotator_1", "annotator_2",
            "adjudicated_class", "adjudicated_column", "adjudicated_boundary",
            "adjudication_rationale",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(disagreements)

    labels1 = [a1[item_id]["availability_class"].strip() for item_id in complete_ids]
    labels2 = [a2[item_id]["availability_class"].strip() for item_id in complete_ids]
    result = {
        "status": "ready_for_adjudication" if len(complete_ids) == len(a1) else "not_ready",
        "items_total": len(a1),
        "items_complete_by_both": len(complete_ids),
        "incomplete": {name: len(ids) for name, ids in incomplete.items()},
        "cohen_kappa_availability_class": cohen_kappa(labels1, labels2),
        "exact_map_agreement": exact / len(complete_ids) if complete_ids else None,
        "disagreements": len(disagreements),
    }
    (DATA / "agreement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
