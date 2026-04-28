#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from common import latest_snapshot, load_config, set_seed, write_jsonl
from transformers import AutoTokenizer

REQUIRED_KEYS = {
    "id",
    "prefix_ids",
    "original_suffix_ids",
    "finephrase_suffix_ids",
    "split",
    "source_sha",
    "finephrase_sha",
}


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def rollout_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("text") or "")
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"] or "")
        if "rollout_results" in value:
            return rollout_text(value["rollout_results"])
    return ""


def stable_id(namespace: str, source_id: str, source_text: str, synthetic_text: str = "") -> str:
    h = hashlib.sha256()
    h.update(namespace.encode("utf-8"))
    h.update(source_id.encode("utf-8", errors="ignore"))
    h.update(source_text[:512].encode("utf-8", errors="ignore"))
    h.update(synthetic_text[:512].encode("utf-8", errors="ignore"))
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


def validate_contract(rows: list[dict[str, Any]], *, paired: bool) -> None:
    if not rows:
        raise RuntimeError("No rows materialized")
    for idx, row in enumerate(rows[:128]):
        if set(row) != REQUIRED_KEYS:
            raise RuntimeError(f"Row {idx} schema mismatch: {sorted(row)}")
        for key in ["prefix_ids", "original_suffix_ids", "finephrase_suffix_ids"]:
            ids = row[key]
            if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
                raise RuntimeError(f"Row {idx} has invalid {key}")
        if not row["prefix_ids"] or not row["original_suffix_ids"]:
            raise RuntimeError(f"Row {idx} has empty prefix or original suffix")
        if paired and not row["finephrase_suffix_ids"]:
            raise RuntimeError(f"Row {idx} has empty FinePhrase suffix")
        if row["split"] not in {"train", "val"}:
            raise RuntimeError(f"Row {idx} has invalid split={row['split']!r}")
        if not isinstance(row["source_sha"], str) or len(row["source_sha"]) != 64:
            raise RuntimeError(f"Row {idx} has invalid source_sha")
        if paired and (not isinstance(row["finephrase_sha"], str) or len(row["finephrase_sha"]) != 64):
            raise RuntimeError(f"Row {idx} has invalid finephrase_sha")


def materialize_paired(cfg: dict[str, Any], tokenizer) -> tuple[list[dict[str, Any]], list[Path]]:
    data_cfg = cfg["data"]
    files = parquet_files(
        data_cfg["finephrase_repo_cache"],
        data_cfg.get("finephrase_subset", "math"),
        int(data_cfg.get("max_parquet_files") or 0),
    )
    max_examples = int(data_cfg["max_examples"])
    val_fraction = float(data_cfg["val_fraction"])
    prefix_tokens = int(data_cfg["prefix_tokens"])
    suffix_tokens = int(data_cfg["suffix_tokens"])
    min_source_tokens = int(data_cfg["min_source_tokens"])

    rows: list[dict[str, Any]] = []
    for file_path in files:
        table = pq.read_table(file_path)
        for row in table.to_pylist():
            source_text = str(row.get("text") or "")
            synthetic_text = rollout_text(row.get("rollout_results"))
            if not source_text or not synthetic_text:
                continue
            source_ids = encode(tokenizer, source_text)
            synthetic_ids = encode(tokenizer, synthetic_text)
            if len(source_ids) < min_source_tokens or len(synthetic_ids) < max(16, suffix_tokens // 2):
                continue
            prefix_ids = source_ids[:prefix_tokens]
            original_suffix_ids = source_ids[prefix_tokens : prefix_tokens + suffix_tokens]
            finephrase_suffix_ids = synthetic_ids[:suffix_tokens]
            if len(original_suffix_ids) < suffix_tokens // 2 or len(finephrase_suffix_ids) < suffix_tokens // 2:
                continue
            raw_id = str(row.get("id") or row.get("url") or "")
            ex_id = stable_id("finephrase_math", raw_id, source_text, synthetic_text)
            rows.append(
                {
                    "id": ex_id,
                    "prefix_ids": prefix_ids,
                    "original_suffix_ids": original_suffix_ids,
                    "finephrase_suffix_ids": finephrase_suffix_ids,
                    "split": split_for_id(ex_id, val_fraction),
                    "source_sha": text_sha(source_text),
                    "finephrase_sha": text_sha(synthetic_text),
                }
            )
            if len(rows) >= max_examples:
                break
        if len(rows) >= max_examples:
            break
    validate_contract(rows, paired=True)
    return rows, files


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
            ex_id = stable_id("fineweb_edu_raw", raw_id, source_text)
            rows.append(
                {
                    "id": ex_id,
                    "prefix_ids": prefix_ids,
                    "original_suffix_ids": original_suffix_ids,
                    "finephrase_suffix_ids": [],
                    "split": split_for_id(ex_id, val_fraction),
                    "source_sha": text_sha(source_text),
                    "finephrase_sha": "",
                }
            )
            if len(rows) >= max_examples:
                break
        if len(rows) >= max_examples:
            break
    validate_contract(rows, paired=False)
    return rows, files


def write_meta(cfg: dict[str, Any], tokenizer_path: Path, paired_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], paired_files: list[Path], raw_files: list[Path]) -> None:
    counts = {
        "paired": {"train": 0, "val": 0},
        "raw": {"train": 0, "val": 0},
    }
    for row in paired_rows:
        counts["paired"][row["split"]] += 1
    for row in raw_rows:
        counts["raw"][row["split"]] += 1
    meta = {
        "paired_examples": len(paired_rows),
        "raw_examples": len(raw_rows),
        "counts": counts,
        "tokenizer": str(tokenizer_path),
        "finephrase_files": [str(p) for p in paired_files],
        "fineweb_edu_files": [str(p) for p in raw_files],
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

    paired_path = Path(cfg["data"]["examples_jsonl"])
    raw_path = Path(cfg["data"]["raw_examples_jsonl"])
    if paired_path.exists() and raw_path.exists() and not args.force:
        print(f"prepare_pretraining_data: using existing {paired_path} and {raw_path}")
        if args.validate:
            validate_contract([json.loads(line) for line in paired_path.read_text().splitlines() if line], paired=True)
            validate_contract([json.loads(line) for line in raw_path.read_text().splitlines() if line], paired=False)
        return

    tokenizer_path = latest_snapshot(cfg["data"]["tokenizer_repo_cache"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    paired_rows, paired_files = materialize_paired(cfg, tokenizer)
    raw_rows, raw_files = materialize_raw(cfg, tokenizer)
    write_jsonl(paired_path, paired_rows)
    write_jsonl(raw_path, raw_rows)
    write_meta(cfg, tokenizer_path, paired_rows, raw_rows, paired_files, raw_files)


if __name__ == "__main__":
    main()
