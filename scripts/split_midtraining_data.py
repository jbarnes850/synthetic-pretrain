#!/usr/bin/env python3
"""Create RAM-style SFT/RL/heldout splits from interleaved thinking chunks.

The RAM thinking-midtraining recipe uses one augmented corpus, then separates
the SFT half from the RL half. This script writes physically separate JSONL
files so existing trainers can stay simple and the split boundary is auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from common import jsonl_iter, write_jsonl


def stable_key(row: dict[str, Any]) -> str:
    return str(row.get("source_sha") or row.get("id") or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "missing"))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def with_split(rows: list[dict[str, Any]], split: str, midtraining_split: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["split"] = split
        item["midtraining_split"] = midtraining_split
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--sft-jsonl", required=True)
    parser.add_argument("--rl-jsonl", required=True)
    parser.add_argument("--heldout-jsonl", default="")
    parser.add_argument("--seed", type=int, default=7337)
    parser.add_argument("--sft-count", type=int, default=32768)
    parser.add_argument("--rl-count", type=int, default=28672)
    parser.add_argument("--heldout-count", type=int, default=4096)
    parser.add_argument("--strict-source-disjoint", action="store_true", default=True)
    args = parser.parse_args()

    rows = list(jsonl_iter(args.input_jsonl))
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_jsonl}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(stable_key(row), []).append(row)
    group_items = list(groups.items())
    rng = random.Random(args.seed)
    rng.shuffle(group_items)

    buckets: dict[str, list[dict[str, Any]]] = {"sft": [], "rl": [], "heldout": []}
    targets = {"sft": args.sft_count, "rl": args.rl_count, "heldout": args.heldout_count}
    order = ["sft", "rl", "heldout"]
    cursor = 0
    for _, group in group_items:
        while cursor < len(order) and len(buckets[order[cursor]]) >= targets[order[cursor]]:
            cursor += 1
        if cursor >= len(order):
            break
        bucket = order[cursor]
        if len(buckets[bucket]) + len(group) <= targets[bucket]:
            buckets[bucket].extend(group)
        elif len(group) == 1:
            buckets[bucket].extend(group)
        if len(buckets[bucket]) >= targets[bucket]:
            cursor += 1

    missing = {name: targets[name] - len(buckets[name]) for name in order if len(buckets[name]) < targets[name]}
    if missing:
        raise RuntimeError(f"Insufficient rows for requested split counts: {missing}; input_rows={len(rows)}")

    sft_rows = with_split(buckets["sft"], "train", "sft")
    rl_rows = with_split(buckets["rl"], "train", "rl")
    heldout_rows = with_split(buckets["heldout"], "val", "heldout")

    sft_path = Path(args.sft_jsonl)
    rl_path = Path(args.rl_jsonl)
    heldout_path = Path(args.heldout_jsonl) if args.heldout_jsonl else None
    write_jsonl(sft_path, sft_rows)
    write_jsonl(rl_path, rl_rows)
    if heldout_path:
        write_jsonl(heldout_path, heldout_rows)

    source_sets = {name: {stable_key(row) for row in bucket_rows} for name, bucket_rows in buckets.items()}
    overlaps = {
        "sft_rl": len(source_sets["sft"] & source_sets["rl"]),
        "sft_heldout": len(source_sets["sft"] & source_sets["heldout"]),
        "rl_heldout": len(source_sets["rl"] & source_sets["heldout"]),
    }
    if args.strict_source_disjoint and any(overlaps.values()):
        raise RuntimeError(f"Source split overlap detected: {overlaps}")

    meta = {
        "input_jsonl": args.input_jsonl,
        "sft_jsonl": str(sft_path),
        "rl_jsonl": str(rl_path),
        "heldout_jsonl": str(heldout_path) if heldout_path else None,
        "seed": args.seed,
        "targets": targets,
        "bucket_counts": {name: len(bucket_rows) for name, bucket_rows in buckets.items()},
        "written_counts": {
            "sft_jsonl": len(sft_rows),
            "rl_jsonl": len(rl_rows),
            "heldout_jsonl": len(heldout_rows) if heldout_path else 0,
        },
        "source_overlaps": overlaps,
        "sft_split_counts": count_by(sft_rows, "split"),
        "rl_split_counts": count_by(rl_rows, "split"),
    }
    meta_path = sft_path.with_suffix(".split_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
