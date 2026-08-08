#!/usr/bin/env python3
"""Convert RealBiomFall's safe, built-in-only coarse labels to JSON.

The upstream annotations are Python pickle files. Never call pickle.load() on
downloaded data: this restricted unpickler rejects every global/class lookup,
and the script validates the resulting shape before using it.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


class BuiltinsOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"global object forbidden: {module}.{name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-links", type=Path)
    args = parser.parse_args()

    label_path = args.dataset / "labels-100" / "labels_temporal_coarse.pkl"
    video_dir = args.dataset / "video_clips-trimmed_cropped_padded_resized-100"
    with label_path.open("rb") as source:
        labels = BuiltinsOnlyUnpickler(source).load()
    if not isinstance(labels, dict) or len(labels) != 100:
        raise SystemExit(f"expected 100 label records, got {type(labels).__name__}/{len(labels)}")

    manifest: list[dict] = []
    for filename, record in sorted(labels.items()):
        if not isinstance(filename, str) or not isinstance(record, dict):
            raise SystemExit("unexpected label record type")
        actions = record.get("actions")
        if not isinstance(actions, list):
            raise SystemExit(f"missing actions: {filename}")
        fall_spans = [
            (float(start), float(end)) for action, start, end in actions
            if int(action) == 14
        ]
        if not fall_spans:
            raise SystemExit(f"no fall action (class 14): {filename}")
        path = video_dir / filename
        if not path.is_file():
            raise SystemExit(f"video missing: {path}")
        manifest.append({
            "path": path.as_posix(),
            "label": 1,
            "onset_sec": min(span[0] for span in fall_spans),
            "duration_sec": float(record["duration"]),
            "upstream_subset": str(record["subset"]),
        })

    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.test_links is not None:
        destination = args.test_links / "external" / "Fall"
        destination.mkdir(parents=True, exist_ok=True)
        for row in manifest:
            if row["upstream_subset"] != "testing":
                continue
            link = destination / Path(row["path"]).name
            if not link.exists():
                link.symlink_to(Path(row["path"]).resolve())
    subsets = {name: sum(row["upstream_subset"] == name for row in manifest)
               for name in sorted({row["upstream_subset"] for row in manifest})}
    print(json.dumps({"clips": len(manifest), "subsets": subsets}))


if __name__ == "__main__":
    main()
