"""Prepare the two independent per-column annotation sheets for A6.

The schema is pinned to RelBench 2.1.2.  Candidate labels are deliberately
written to a separate file: they are useful for provenance and later error
analysis, but must not be shown to either human annotator before submission.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pilot" / "a6"


EVENT = {
    "user_friends": [("user", "key"), ("friend", "key")],
    "event_interest": [
        ("user", "key"), ("event", "key"), ("timestamp", "datetime"),
        ("invited", "int"), ("interested", "int"),
        ("not_interested", "int"),
    ],
    "event_attendees": [
        ("event", "key"), ("user_id", "key"),
        ("start_time", "datetime"), ("status", "str"),
    ],
    "users": [
        ("user_id", "key"), ("joinedAt", "datetime"),
        ("birthyear", "float"), ("timezone", "float"),
        ("locale", "str"), ("gender", "str"), ("location", "str"),
    ],
    "events": [
        ("event_id", "key"), ("user_id", "key"),
        ("start_time", "datetime"), ("lat", "float"), ("lng", "float"),
        *[(f"c_{i}", "int") for i in range(1, 101)],
        ("c_other", "int"), ("city", "str"), ("state", "str"),
        ("zip", "str"), ("country", "str"),
    ],
}

STACK = {
    "badges": [
        ("Id", "key"), ("UserId", "key"), ("Date", "datetime"),
        ("Class", "int"), ("Name", "str"), ("TagBased", "bool"),
    ],
    "votes": [
        ("Id", "key"), ("UserId", "key"), ("PostId", "key"),
        ("CreationDate", "datetime"), ("VoteTypeId", "int"),
    ],
    "users": [
        ("Id", "key"), ("CreationDate", "datetime"),
        ("AccountId", "float"), ("DisplayName", "str"),
        ("Location", "str"), ("WebsiteUrl", "str"), ("AboutMe", "str"),
    ],
    "comments": [
        ("Id", "key"), ("PostId", "key"), ("UserId", "key"),
        ("CreationDate", "datetime"), ("ContentLicense", "str"),
        ("UserDisplayName", "str"), ("Text", "str"),
    ],
    "posts": [
        ("Id", "key"), ("OwnerUserId", "key"), ("ParentId", "key"),
        ("CreationDate", "datetime"), ("PostTypeId", "int"),
        ("OwnerDisplayName", "str"), ("Title", "str"), ("Tags", "str"),
        ("ContentLicense", "str"), ("Body", "str"),
    ],
    "postLinks": [
        ("Id", "key"), ("RelatedPostId", "key"), ("PostId", "key"),
        ("CreationDate", "datetime"), ("LinkTypeId", "int"),
    ],
    "postHistory": [
        ("Id", "key"), ("PostId", "key"), ("UserId", "key"),
        ("CreationDate", "datetime"), ("PostHistoryTypeId", "int"),
        ("UserDisplayName", "str"), ("ContentLicense", "str"),
        ("RevisionGUID", "str"), ("Text", "str"), ("Comment", "str"),
    ],
}

TIME_COLS = {
    "rel-event": {
        "user_friends": "", "event_interest": "timestamp",
        "event_attendees": "start_time", "users": "joinedAt",
        "events": "start_time",
    },
    "rel-stack": {
        "badges": "Date", "votes": "CreationDate",
        "users": "CreationDate", "comments": "CreationDate",
        "posts": "CreationDate", "postLinks": "CreationDate",
        "postHistory": "CreationDate",
    },
}

FORM_FIELDS = [
    "item_id", "dataset", "table", "column", "data_type", "role",
    "table_time_col", "availability_class", "availability_column",
    "boundary", "mutable", "checkability", "confidence", "rationale",
]


def role(dataset: str, table: str, column: str, dtype: str) -> str:
    if column == TIME_COLS[dataset][table]:
        return "time"
    if dtype == "key":
        return "key"
    return "value"


def candidate(dataset: str, table: str, column: str) -> tuple[str, str, str, str]:
    """Return class, availability column, checkability, rationale."""
    time_col = TIME_COLS[dataset][table]
    if not time_col:
        return (
            "unknown", "", "uncheckable",
            "The source table has no row or edge timestamp.",
        )
    if dataset == "rel-stack" and table == "users" and column in {
        "DisplayName", "Location", "WebsiteUrl", "AboutMe"
    }:
        return (
            "unknown", "", "uncheckable",
            "Mutable profile snapshot; RelBench keeps no edit history for the field.",
        )
    if dataset == "rel-stack" and table == "posts" and column in {
        "OwnerDisplayName", "Title", "Tags", "ContentLicense", "Body"
    }:
        return (
            "unknown", "", "uncheckable",
            "Posts.csv is a current snapshot and this field may reflect later edits.",
        )
    if dataset == "rel-event" and table in {"users", "event_attendees"}:
        return (
            "row_time", time_col, "proxy",
            "RelBench supplies only the table time; field-level observation time is absent.",
        )
    return (
        "row_time", time_col, "direct",
        "The value belongs to the event record represented by the table time.",
    )


def build_items() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for dataset, schema in (("rel-stack", STACK), ("rel-event", EVENT)):
        number = 0
        for table, columns in schema.items():
            for column, dtype in columns:
                number += 1
                items.append({
                    "item_id": f"{dataset}:{number:03d}",
                    "dataset": dataset,
                    "table": table,
                    "column": column,
                    "data_type": dtype,
                    "role": role(dataset, table, column, dtype),
                    "table_time_col": TIME_COLS[dataset][table],
                })
    return items


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_blank_sheet_once(path: Path, rows: list[dict[str, str]]) -> None:
    """Never erase annotations when the preparation command is re-run."""
    if not path.exists():
        write_csv(path, FORM_FIELDS, rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = build_items()
    write_csv(OUT / "annotation_items.csv", FORM_FIELDS[:7], items)

    blank_rows = [{**item, **{field: "" for field in FORM_FIELDS[7:]}}
                  for item in items]
    write_blank_sheet_once(OUT / "annotator_1.csv", blank_rows)
    write_blank_sheet_once(OUT / "annotator_2.csv", blank_rows)

    candidate_fields = FORM_FIELDS[:7] + [
        "availability_class", "availability_column", "boundary",
        "checkability", "candidate_rationale",
    ]
    candidate_rows = []
    for item in items:
        klass, avail_col, checkability, rationale = candidate(
            item["dataset"], item["table"], item["column"]
        )
        candidate_rows.append({
            **item,
            "availability_class": klass,
            "availability_column": avail_col,
            "boundary": "inclusive" if klass == "row_time" else "unknown",
            "checkability": checkability,
            "candidate_rationale": rationale,
        })
    write_csv(OUT / "bootstrap_map.csv", candidate_fields, candidate_rows)

    manifest_path = OUT / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {}
    )
    manifest.update({
        "experiment": "A6",
        "status": "in_process; not_finished",
        "relbench_version": "2.1.2",
        "scope": {"rel-stack": 50, "rel-event": 129, "total": len(items)},
        "human_annotations_required": 2,
        "human_annotations_complete": 0,
        "independence_rule": "Do not expose bootstrap_map.csv before both submissions.",
        "next_step": "Two eligible coauthors fill annotator_1.csv and annotator_2.csv independently.",
    })
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
