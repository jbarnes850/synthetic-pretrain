#!/usr/bin/env python3
"""Fetch paper-shaped reasoning eval datasets from Hugging Face Dataset Viewer."""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASETS = {
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "url": "https://hf.co/datasets/openai/gsm8k",
    },
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500",
        "config": "default",
        "split": "test",
        "url": "https://hf.co/datasets/HuggingFaceH4/MATH-500",
    },
    "gpqa_diamond": {
        "dataset": "hendrydong/gpqa_diamond_mc",
        "config": "default",
        "split": "test",
        "url": "https://hf.co/datasets/hendrydong/gpqa_diamond_mc",
    },
    "olympiadbench": {
        "dataset": "zwhe99/simplerl-OlympiadBench",
        "config": "default",
        "split": "test",
        "url": "https://hf.co/datasets/zwhe99/simplerl-OlympiadBench",
    },
}


def dataset_viewer(path: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"https://datasets-server.huggingface.co/{path}?{query}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_rows(dataset: str, config: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    size = dataset_viewer("size", {"dataset": dataset})
    total = 0
    for item in size.get("size", {}).get("splits", []):
        if item.get("config") == config and item.get("split") == split:
            total = int(item["num_rows"])
            break
    if total <= 0:
        raise RuntimeError(f"Could not resolve row count for {dataset}/{config}/{split}")
    if limit is not None:
        total = min(total, limit)

    rows: list[dict[str, Any]] = []
    for offset in range(0, total, 100):
        length = min(100, total - offset)
        payload = dataset_viewer(
            "rows",
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length},
        )
        rows.extend(item["row"] for item in payload["rows"])
    return rows


def normalize_row(bench: str, idx: int, row: dict[str, Any]) -> dict[str, Any] | None:
    if bench == "gsm8k":
        return {
            "benchmark": bench,
            "id": f"gsm8k/test/{idx}",
            "question": row["question"],
            "answer": row["answer"],
            "answer_type": "numeric",
            "source_url": DATASETS[bench]["url"],
        }
    if bench == "math500":
        return {
            "benchmark": bench,
            "id": row.get("unique_id") or f"math500/test/{idx}",
            "question": row["problem"],
            "answer": row["answer"],
            "solution": row.get("solution"),
            "subject": row.get("subject"),
            "level": row.get("level"),
            "answer_type": "math",
            "source_url": DATASETS[bench]["url"],
        }
    if bench == "gpqa_diamond":
        return {
            "benchmark": bench,
            "id": f"gpqa_diamond/test/{idx}",
            "question": row["problem"],
            "answer": row["solution"],
            "domain": row.get("domain"),
            "answer_type": "multiple_choice_letter",
            "source_url": DATASETS[bench]["url"],
        }
    if bench == "olympiadbench":
        answers = row.get("final_answer") or []
        if row.get("error") or row.get("is_multiple_answer") or not answers:
            return None
        return {
            "benchmark": bench,
            "id": f"olympiadbench/test/{row.get('id', idx)}",
            "question": row["question"],
            "answer": answers[0],
            "solution": (row.get("solution") or [None])[0],
            "subfield": row.get("subfield"),
            "answer_type": row.get("answer_type") or "math",
            "source_url": DATASETS[bench]["url"],
        }
    raise ValueError(f"Unknown benchmark: {bench}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/eval/reasoning_benchmarks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--olympiad-limit", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "Hugging Face Dataset Viewer API",
        "datasets": {},
    }
    for bench, spec in DATASETS.items():
        limit = args.olympiad_limit if bench == "olympiadbench" else args.limit
        raw_rows = fetch_rows(spec["dataset"], spec["config"], spec["split"], limit)
        rows = [normalize_row(bench, idx, row) for idx, row in enumerate(raw_rows)]
        rows = [row for row in rows if row is not None]
        out_path = out_dir / f"{bench}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["datasets"][bench] = {
            **spec,
            "rows": len(rows),
            "path": str(out_path),
        }
        print(json.dumps({"benchmark": bench, "rows": len(rows), "path": str(out_path)}), flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
