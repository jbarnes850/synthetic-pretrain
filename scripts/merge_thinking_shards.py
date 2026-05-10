#!/usr/bin/env python3
"""Merge and audit interleaved-thinking shard outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import jsonl_iter, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-val", type=int, required=True)
    parser.add_argument("--require-preserved", action="store_true")
    parser.add_argument("--trim-to-expected", action="store_true")
    parser.add_argument("--seed", type=int, default=5337)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    input_counts: dict[str, int] = {}
    for input_path in args.input_jsonl:
        path = Path(input_path)
        shard_rows = list(jsonl_iter(path))
        input_counts[str(path)] = len(shard_rows)
        rows.extend(shard_rows)

    ids = [row["id"] for row in rows]
    duplicate_ids = [row_id for row_id, count in Counter(ids).items() if count > 1]
    split_counts = Counter(row.get("split") for row in rows)
    preserved_count = sum(bool(row.get("original_text_preserved")) for row in rows)
    repaired_count = sum(bool(row.get("preservation_repaired")) for row in rows)
    placement_repaired_count = sum(bool(row.get("placement_repaired")) for row in rows)

    errors: list[str] = []
    if duplicate_ids:
        errors.append(f"duplicate_ids:{len(duplicate_ids)}")
    if args.require_preserved and preserved_count != len(rows):
        errors.append(f"not_preserved:{len(rows) - preserved_count}")
    if errors:
        raise RuntimeError("; ".join(errors))

    if args.trim_to_expected:
        def stable_key(row: dict[str, Any]) -> str:
            payload = f"{args.seed}:{row.get('id', '')}:{row.get('source_sha', '')}"
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        train_rows = sorted((row for row in rows if row.get("split") == "train"), key=stable_key)
        val_rows = sorted((row for row in rows if row.get("split") == "val"), key=stable_key)
        if len(train_rows) < args.expected_train:
            raise RuntimeError(f"train:{len(train_rows)}<{args.expected_train}")
        if len(val_rows) < args.expected_val:
            raise RuntimeError(f"val:{len(val_rows)}<{args.expected_val}")
        rows = train_rows[: args.expected_train] + val_rows[: args.expected_val]
        rows.sort(key=stable_key)

    split_counts = Counter(row.get("split") for row in rows)
    preserved_count = sum(bool(row.get("original_text_preserved")) for row in rows)
    repaired_count = sum(bool(row.get("preservation_repaired")) for row in rows)
    placement_repaired_count = sum(bool(row.get("placement_repaired")) for row in rows)
    if len(rows) != args.expected_total:
        raise RuntimeError(f"total:{len(rows)}!={args.expected_total}")
    if split_counts.get("train", 0) != args.expected_train:
        raise RuntimeError(f"train:{split_counts.get('train', 0)}!={args.expected_train}")
    if split_counts.get("val", 0) != args.expected_val:
        raise RuntimeError(f"val:{split_counts.get('val', 0)}!={args.expected_val}")
    if args.require_preserved and preserved_count != len(rows):
        raise RuntimeError(f"not_preserved:{len(rows) - preserved_count}")

    output_path = Path(args.output_jsonl)
    n = write_jsonl(output_path, rows)
    meta = {
        "output_jsonl": str(output_path),
        "input_counts": input_counts,
        "trim_to_expected": args.trim_to_expected,
        "rows": n,
        "split_counts": dict(split_counts),
        "duplicate_ids": 0,
        "original_text_preserved_rate": preserved_count / max(1, n),
        "preservation_repaired_rate": repaired_count / max(1, n),
        "placement_repaired_rate": placement_repaired_count / max(1, n),
        "status": "complete",
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
