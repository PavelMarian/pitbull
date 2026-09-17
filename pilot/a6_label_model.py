"""Create a complete, conservative model-assisted A6 annotation.

This is an operational draft and must not be counted as either of the two
independent human annotations required by the experiment protocol.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pilot" / "a6"

FIELDS = [
    "item_id", "dataset", "table", "column", "data_type", "role",
    "table_time_col", "availability_class", "availability_column",
    "boundary", "mutable", "checkability", "confidence", "rationale",
    "operational_action",
]

STACK_MUTABLE = {
    "users": {"DisplayName", "Location", "WebsiteUrl", "AboutMe"},
    "comments": {"ContentLicense", "UserDisplayName", "Text"},
    "posts": {
        "OwnerUserId", "OwnerDisplayName", "Title", "Tags",
        "ContentLicense", "Body",
    },
}

EVENT_PROFILE = {"birthyear", "timezone", "locale", "gender", "location"}


def annotate(row: dict[str, str]) -> dict[str, str]:
    dataset, table, column = row["dataset"], row["table"], row["column"]
    time_col = row["table_time_col"]

    if not time_col:
        values = (
            "unknown", "", "unknown", "unknown", "uncheckable", "high",
            "The table has no timestamp for creation or observation of this edge.",
            "exclude",
        )
    elif dataset == "rel-stack" and column in STACK_MUTABLE.get(table, set()):
        values = (
            "unknown", "", "unknown", "yes", "uncheckable", "high",
            "The retained value is a current mutable snapshot, but its field-level edit time was not retained.",
            "exclude",
        )
    elif dataset == "rel-event" and table == "users" and column in EVENT_PROFILE:
        values = (
            "unknown", "", "unknown", "yes", "uncheckable", "medium",
            "The profile value may change and the dataset contains no field-level observation or edit time.",
            "exclude",
        )
    elif dataset == "rel-event" and table == "events" and column not in {
        "event_id", "user_id", "start_time"
    }:
        values = (
            "row_time", time_col, "inclusive", "yes", "proxy", "low",
            "Event start is a conservative proxy: metadata should be known by then, but edit history is absent.",
            "use_with_proxy_flag",
        )
    elif dataset == "rel-event" and table == "event_attendees":
        mutability = "yes" if column == "status" else "no"
        values = (
            "row_time", time_col, "inclusive", mutability, "proxy", "low",
            "Only event start is retained; it is used as a conservative proxy for RSVP/attendance-edge availability.",
            "use_with_proxy_flag",
        )
    elif dataset == "rel-event" and table == "users":
        values = (
            "row_time", time_col, "inclusive", "no", "direct", "high",
            "User identity and join time are established by the user creation event.",
            "use",
        )
    elif dataset == "rel-event" and table == "event_interest":
        values = (
            "row_time", time_col, "inclusive", "no", "direct", "high",
            "The interaction record carries its own observation timestamp.",
            "use",
        )
    elif dataset == "rel-stack" and table == "users" and column == "AccountId":
        values = (
            "row_time", time_col, "inclusive", "unknown", "proxy", "medium",
            "Account linkage is treated as available at user creation, though later account merges are not timestamped.",
            "use_with_proxy_flag",
        )
    elif dataset == "rel-stack" and table == "posts" and column in {
        "ParentId", "PostTypeId"
    }:
        values = (
            "row_time", time_col, "inclusive", "no", "proxy", "medium",
            "The structural post field is normally fixed at creation; no separate field timestamp is retained.",
            "use_with_proxy_flag",
        )
    else:
        values = (
            "row_time", time_col, "inclusive", "no", "direct", "high",
            "The value is part of the event record represented by the table time.",
            "use",
        )

    keys = [
        "availability_class", "availability_column", "boundary", "mutable",
        "checkability", "confidence", "rationale", "operational_action",
    ]
    return {**row, **dict(zip(keys, values))}


def main() -> None:
    with (DATA / "annotation_items.csv").open(newline="", encoding="utf-8") as handle:
        rows = [annotate(row) for row in csv.DictReader(handle)]

    with (DATA / "model_annotation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    map_fields = [
        "dataset", "table", "column", "availability_class",
        "availability_column", "boundary", "checkability", "confidence",
        "operational_action",
    ]
    with (DATA / "provisional_operational_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=map_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "complete_model_assisted_draft",
        "items": len(rows),
        "by_dataset": dict(Counter(row["dataset"] for row in rows)),
        "availability_class": dict(Counter(row["availability_class"] for row in rows)),
        "checkability": dict(Counter(row["checkability"] for row in rows)),
        "operational_action": dict(Counter(row["operational_action"] for row in rows)),
        "human_vote": False,
    }
    (DATA / "model_annotation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_assisted_annotation"] = {
        "complete": True,
        "items": len(rows),
        "human_vote": False,
        "summary": "model_annotation_summary.json",
        "operational_map": "provisional_operational_map.csv",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
