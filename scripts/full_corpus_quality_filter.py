#!/usr/bin/env python3
"""Run full-corpus LLM QA and write a filtered interleaved-thinking corpus."""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import jsonl_iter, write_jsonl
from eval_corpus_quality_gate import (
    DISALLOWED_CATEGORIES,
    critic_prompt,
    extract_thoughts,
    parse_critic_response,
    row_flags,
    shorten,
)

KEEP_CATEGORIES = {"clean", "source_qa_legitimate", "legitimate_domain_user"}


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row_id = str(obj.get("id") or "")
            if row_id:
                records[row_id] = obj
    return records


def append_jsonl(path: Path, obj: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(obj, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def make_record(row: dict[str, Any], judgment: dict[str, str], endpoint: str, attempt: str, row_index: int) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "row_index": row_index,
        "split": row.get("split"),
        "source_sha": row.get("source_sha", ""),
        "attempt": attempt,
        "endpoint": endpoint,
        "flags": row.get("_flags", {}),
        "thought_preview": shorten(" ".join(extract_thoughts(row.get("augmented_text", ""))[:2]), 600),
        **judgment,
    }


def judge_row(row: dict[str, Any], endpoint: str, model: str, args: argparse.Namespace) -> dict[str, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": critic_prompt(row, args.max_chars)}],
        "temperature": args.critic_temperature,
        "top_p": args.critic_top_p,
        "max_tokens": args.critic_max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if args.response_format_json:
        payload["response_format"] = {"type": "json_object"}
    import requests

    with requests.post(
        endpoint.rstrip("/") + "/v1/chat/completions",
        json=payload,
        timeout=(20.0, args.critic_timeout),
    ) as resp:
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content") or ""
    parsed = parse_critic_response(content)
    parsed["raw_critic"] = content[:2000]
    return parsed


def run_endpoint_pool(
    *,
    rows: list[tuple[int, dict[str, Any]]],
    endpoint: str,
    model: str,
    args: argparse.Namespace,
    output_path: Path,
    output_lock: threading.Lock,
    progress: dict[str, int],
    progress_lock: threading.Lock,
    attempt: str,
) -> None:
    if not rows:
        return
    with ThreadPoolExecutor(max_workers=args.workers_per_endpoint) as ex:
        futures = {ex.submit(judge_row, row, endpoint, model, args): (row_index, row) for row_index, row in rows}
        for fut in as_completed(futures):
            row_index, row = futures[fut]
            try:
                judgment = fut.result()
            except Exception as exc:
                judgment = {
                    "category": "uncertain",
                    "severity": "warn",
                    "reason": f"critic_error: {exc}",
                    "evidence": "",
                    "raw_critic": f"ERROR: {exc}",
                }
            record = make_record(row, judgment, endpoint, attempt, row_index)
            append_jsonl(output_path, record, output_lock)
            with progress_lock:
                progress["done"] += 1
                category = str(record.get("category") or "uncertain")
                progress[f"cat:{category}"] = progress.get(f"cat:{category}", 0) + 1


def endpoint_for(row_index: int, endpoints: list[str]) -> str:
    return endpoints[row_index % len(endpoints)]


def run_full_pass(
    rows: list[dict[str, Any]],
    endpoints: list[str],
    model: str,
    args: argparse.Namespace,
    output_path: Path,
    existing: dict[str, dict[str, Any]],
    attempt: str,
) -> dict[str, dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_lock = threading.Lock()
    progress_lock = threading.Lock()
    progress: dict[str, int] = {"done": 0, "total": 0}

    rows_by_endpoint: dict[str, list[tuple[int, dict[str, Any]]]] = {endpoint: [] for endpoint in endpoints}
    for row_index, row in enumerate(rows):
        row_id = str(row.get("id") or "")
        if row_id in existing:
            continue
        endpoint = endpoint_for(row_index, endpoints)
        rows_by_endpoint[endpoint].append((row_index, row))
        progress["total"] += 1

    if progress["total"] == 0:
        print(json.dumps({"attempt": attempt, "status": "resume_complete", "records": len(existing)}), flush=True)
        return existing

    stop = threading.Event()

    def reporter() -> None:
        start = time.time()
        while not stop.wait(args.progress_interval):
            with progress_lock:
                done = progress["done"]
                total = progress["total"]
                cats = {key[4:]: value for key, value in progress.items() if key.startswith("cat:")}
            elapsed = max(1e-6, time.time() - start)
            rate = done / elapsed
            remaining = max(0, total - done)
            eta = remaining / rate if rate > 0 else None
            print(
                json.dumps(
                    {
                        "attempt": attempt,
                        "done": done,
                        "total": total,
                        "rate_rows_per_min": rate * 60.0,
                        "eta_minutes": eta / 60.0 if eta is not None else None,
                        "categories": cats,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    reporter_thread = threading.Thread(target=reporter, daemon=True)
    reporter_thread.start()
    with ThreadPoolExecutor(max_workers=len(endpoints)) as ex:
        endpoint_futures = [
            ex.submit(
                run_endpoint_pool,
                rows=endpoint_rows,
                endpoint=endpoint,
                model=model,
                args=args,
                output_path=output_path,
                output_lock=output_lock,
                progress=progress,
                progress_lock=progress_lock,
                attempt=attempt,
            )
            for endpoint, endpoint_rows in rows_by_endpoint.items()
        ]
        for fut in as_completed(endpoint_futures):
            fut.result()
    stop.set()
    reporter_thread.join(timeout=1.0)
    return load_jsonl_by_id(output_path)


def stable_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: str(row.get("source_sha") or row.get("id") or ""))


def write_filtered_splits(filtered_rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not args.write_splits:
        return {}
    rows = stable_order(filtered_rows)
    total = len(rows)
    heldout = min(args.max_heldout_count, max(0, total - 2))
    if total >= args.min_high_confidence_rows:
        heldout = min(args.max_heldout_count, max(args.min_heldout_count, total // 16))
    sft = min(args.max_sft_count, max(0, (total - heldout) // 2))
    rl = total - heldout - sft
    sft_rows = [dict(row, split="train", midtraining_split="sft") for row in rows[:sft]]
    rl_rows = [dict(row, split="train", midtraining_split="rl") for row in rows[sft : sft + rl]]
    heldout_rows = [dict(row, split="val", midtraining_split="heldout") for row in rows[sft + rl :]]
    write_jsonl(out_dir / "interleaved_thinking_sft.filtered.jsonl", sft_rows)
    write_jsonl(out_dir / "interleaved_thinking_rl.filtered.jsonl", rl_rows)
    write_jsonl(out_dir / "interleaved_thinking_heldout.filtered.jsonl", heldout_rows)
    return {
        "sft": len(sft_rows),
        "rl": len(rl_rows),
        "heldout": len(heldout_rows),
        "sft_jsonl": str(out_dir / "interleaved_thinking_sft.filtered.jsonl"),
        "rl_jsonl": str(out_dir / "interleaved_thinking_rl.filtered.jsonl"),
        "heldout_jsonl": str(out_dir / "interleaved_thinking_heldout.filtered.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="data/processed/interleaved_thinking_full.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--critic-endpoint", action="append", required=True)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--workers-per-endpoint", type=int, default=32)
    parser.add_argument("--critic-temperature", type=float, default=0.0)
    parser.add_argument("--critic-top-p", type=float, default=1.0)
    parser.add_argument("--critic-max-tokens", type=int, default=220)
    parser.add_argument("--critic-timeout", type=float, default=120.0)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--response-format-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--min-high-confidence-rows", type=int, default=20000)
    parser.add_argument("--write-splits", action="store_true")
    parser.add_argument("--max-sft-count", type=int, default=16384)
    parser.add_argument("--max-heldout-count", type=int, default=2048)
    parser.add_argument("--min-heldout-count", type=int, default=1024)
    args = parser.parse_args()

    endpoints = list(dict.fromkeys(args.critic_endpoint))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(jsonl_iter(args.input_jsonl))
    for row in rows:
        row["_flags"] = row_flags(row)

    print(
        json.dumps(
            {
                "stage": "full_corpus_quality_filter_start",
                "rows": len(rows),
                "endpoints": endpoints,
                "workers_per_endpoint": args.workers_per_endpoint,
                "total_workers": len(endpoints) * args.workers_per_endpoint,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    first_path = out_dir / "critic_records.first_pass.jsonl"
    first_records = run_full_pass(
        rows,
        endpoints,
        args.critic_model,
        args,
        first_path,
        load_jsonl_by_id(first_path),
        "first_pass",
    )

    uncertain_ids = {row_id for row_id, record in first_records.items() if record.get("category") == "uncertain"}
    uncertain_rows = [row for row in rows if row.get("id") in uncertain_ids]
    rejudge_path = out_dir / "critic_records.uncertain_rejudge.jsonl"
    rejudge_records = run_full_pass(
        uncertain_rows,
        endpoints,
        args.critic_model,
        args,
        rejudge_path,
        load_jsonl_by_id(rejudge_path),
        "uncertain_rejudge",
    )

    final_records: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("id") or "")
        record = dict(first_records[row_id])
        if record.get("category") == "uncertain" and row_id in rejudge_records:
            retry = dict(rejudge_records[row_id])
            retry["first_category"] = record.get("category")
            retry["first_reason"] = record.get("reason", "")
            retry["rejudged"] = True
            record = retry
        else:
            record["rejudged"] = False
        final_records.append(record)
    write_jsonl(out_dir / "critic_records.final.jsonl", final_records)

    final_by_id = {str(record.get("id")): record for record in final_records}
    keep_ids = {
        row_id
        for row_id, record in final_by_id.items()
        if record.get("category") in KEEP_CATEGORIES and record.get("severity") != "fail"
    }
    filtered_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows if row.get("id") in keep_ids]
    discarded_records = [record for record in final_records if str(record.get("id")) not in keep_ids]
    write_jsonl(out_dir / "interleaved_thinking_full.filtered.jsonl", filtered_rows)
    write_jsonl(out_dir / "critic_records.discarded.jsonl", discarded_records)
    split_meta = write_filtered_splits(filtered_rows, out_dir, args)

    category_counts = Counter(record.get("category") for record in final_records)
    severity_counts = Counter(record.get("severity") for record in final_records)
    deterministic_flag_counts = Counter()
    for row in rows:
        for name, value in row["_flags"].items():
            if value is True:
                deterministic_flag_counts[name] += 1
    summary = {
        "status": "pass" if len(filtered_rows) >= args.min_high_confidence_rows else "fail",
        "input_jsonl": args.input_jsonl,
        "output_dir": str(out_dir),
        "rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "discarded_rows": len(rows) - len(filtered_rows),
        "keep_categories": sorted(KEEP_CATEGORIES),
        "discard_categories": sorted(DISALLOWED_CATEGORIES | {"uncertain"}),
        "category_counts": dict(category_counts),
        "severity_counts": dict(severity_counts),
        "uncertain_first_pass": len(uncertain_ids),
        "persistent_uncertain": category_counts.get("uncertain", 0),
        "deterministic_flag_counts": dict(deterministic_flag_counts),
        "config": vars(args),
        "split_meta": split_meta,
        "go_no_go": {
            "passed": len(filtered_rows) >= args.min_high_confidence_rows,
            "min_high_confidence_rows": args.min_high_confidence_rows,
            "filtered_rows": len(filtered_rows),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(out_dir / "summary.json"), **summary["go_no_go"]}, indent=2), flush=True)
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
