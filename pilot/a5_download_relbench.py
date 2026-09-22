"""Download the two public RelBench subsets required by A5 from the Hub."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "PITFALL_ext_data"
REPO = "stanford-star/relbench-v1"


def main() -> None:
    info = HfApi().dataset_info(REPO)
    revision = info.sha
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["rel-event/**", "rel-f1/**"],
        local_dir=DEST,
    )
    files = [
        {"path": str(path.relative_to(DEST)).replace("\\", "/"),
         "bytes": path.stat().st_size}
        for path in sorted(DEST.rglob("*")) if path.is_file() and ".cache" not in path.parts
    ]
    record = {
        "source": f"https://huggingface.co/datasets/{REPO}",
        "revision": revision,
        "subsets": ["rel-event", "rel-f1"],
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "inventory": files,
    }
    (DEST / "a5_download_manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: record[key] for key in record if key != "inventory"}, indent=2))


if __name__ == "__main__":
    main()
