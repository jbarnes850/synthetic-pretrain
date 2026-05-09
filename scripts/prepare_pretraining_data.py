#!/usr/bin/env python3
"""Materialize FineWeb-Edu prefix/suffix examples for SIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from common import latest_snapshot, load_config, resolve_hf_path, set_seed, write_jsonl
from transformers import AutoTokenizer

REQUIRED_KEYS = {
    "id",
    "prefix_ids",
    "original_suffix_ids",
    "split",
    "source_sha",
}


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def stable_id(namespace: str, source_id: str, source_text: str) -> str:
    h = hashlib.sha256()
    h.update(namespace.encode("utf-8"))
    h.update(source_id.encode("utf-8", errors="ignore"))
    h.update(source_text[:512].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:24]


def split_for_id(ex_id: str, val_fraction: float) -> str:
    threshold = int(val_fraction * 10000)
    return "val" if (int(ex_id[:8], 16) % 10000) < threshold else "train"


def parquet_files(repo_cache: str, subset: str, limit: int) -> list[Path]:
    snapshot = latest_snapshot(repo_cache)
    subset_dir = snapshot / subset
    files = sorted(subset_dir.glob("*.parquet"))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {subset_dir}")
    return files


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def validate_contract(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No rows materialized")
    for idx, row in enumerate(rows[:128]):
        if set(row) != REQUIRED_KEYS:
            raise RuntimeError(f"Row {idx} schema mismatch: {sorted(row)}")
        for key in ["prefix_ids", "original_suffix_ids"]:
            ids = row[key]
            if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
                raise RuntimeError(f"Row {idx} has invalid {key}")
        if not row["prefix_ids"] or not row["original_suffix_ids"]:
            raise RuntimeError(f"Row {idx} has empty prefix or suffix")
        if row["split"] not in {"train", "val"}:
            raise RuntimeError(f"Row {idx} has invalid split={row['split']!r}")
        if not isinstance(row["source_sha"], str) or len(row["source_sha"]) != 64:
            raise RuntimeError(f"Row {idx} has invalid source_sha")


def materialize_raw(cfg: dict[str, Any], tokenizer) -> tuple[list[dict[str, Any]], list[Path]]:
    data_cfg = cfg["data"]
    files = parquet_files(
        data_cfg["fineweb_edu_repo_cache"],
        data_cfg.get("fineweb_edu_subset", "sample/10BT"),
        int(data_cfg.get("fineweb_max_parquet_files") or 0),
    )
    max_examples = int(data_cfg.get("raw_max_examples") or data_cfg["max_examples"])
    val_fraction = float(data_cfg["val_fraction"])
    prefix_tokens = int(data_cfg["prefix_tokens"])
    suffix_tokens = int(data_cfg["suffix_tokens"])
    min_source_tokens = int(data_cfg["min_source_tokens"])

    rows: list[dict[str, Any]] = []
    for file_path in files:
        table = pq.read_table(file_path)
        for row in table.to_pylist():
            source_text = str(row.get("text") or "")
            if not source_text:
                continue
            source_ids = encode(tokenizer, source_text)
            if len(source_ids) < min_source_tokens:
                continue
            prefix_ids = source_ids[:prefix_tokens]
            original_suffix_ids = source_ids[prefix_tokens : prefix_tokens + suffix_tokens]
            if len(original_suffix_ids) < suffix_tokens // 2:
                continue
            raw_id = str(row.get("id") or row.get("url") or "")
            ex_id = stable_id("fineweb_edu", raw_id, source_text)
            rows.append(
                {
                    "id": ex_id,
                    "prefix_ids": prefix_ids,
                    "original_suffix_ids": original_suffix_ids,
                    "split": split_for_id(ex_id, val_fraction),
                    "source_sha": text_sha(source_text),
                }
            )
            if len(rows) >= max_examples:
                break
        if len(rows) >= max_examples:
            break
    validate_contract(rows)
    return rows, files


def write_meta(cfg: dict[str, Any], tokenizer_path: Path, rows: list[dict[str, Any]], files: list[Path]) -> None:
    counts = {"train": 0, "val": 0}
    for row in rows:
        counts[row["split"]] += 1
    meta = {
        "examples": len(rows),
        "counts": counts,
        "tokenizer": str(tokenizer_path),
        "fineweb_edu_files": [str(path) for path in files],
        "prefix_tokens": int(cfg["data"]["prefix_tokens"]),
        "suffix_tokens": int(cfg["data"]["suffix_tokens"]),
    }
    meta_path = Path(cfg["data"]["processed_dir"]) / "materialize_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))

    output_path = Path(cfg["data"].get("online_dpo_examples_jsonl") or cfg["data"]["raw_examples_jsonl"])
    if output_path.exists() and not args.force:
        print(f"prepare_pretraining_data: using existing {output_path}")
        if args.validate:
            validate_contract([json.loads(line) for line in output_path.read_text().splitlines() if line])
        return

    tokenizer_path = resolve_hf_path(cfg["data"]["tokenizer_repo_cache"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows, files = materialize_raw(cfg, tokenizer)
    n = write_jsonl(output_path, rows)
    raw_path = Path(cfg["data"].get("raw_examples_jsonl", output_path))
    examples_path = Path(cfg["data"].get("examples_jsonl", output_path))
    if raw_path != output_path:
        write_jsonl(raw_path, rows)
    if examples_path != output_path and examples_path != raw_path:
        write_jsonl(examples_path, rows)
    write_meta(cfg, tokenizer_path, rows, files)
    print(json.dumps({"output_jsonl": str(output_path), "rows": n}, indent=2))


if __name__ == "__main__":
    main()
