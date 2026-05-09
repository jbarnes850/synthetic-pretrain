#!/usr/bin/env python3
"""Add strong-teacher suffix rewrites for the SIP rewrite-pool ablation."""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import jsonl_iter, load_config, resolve_hf_path, write_jsonl
from transformers import AutoTokenizer

try:
    import requests
except Exception:  # pragma: no cover - dependency guard
    requests = None


PROMPT = """Rewrite the continuation so it is higher-quality pretraining text while staying faithful to the prefix.

Requirements:
- Preserve the topic, facts, style, and local meaning implied by the prefix.
- Do not answer as an assistant.
- Do not add analysis, markdown, JSON, or explanations.
- Return only the rewritten continuation.

Prefix:
{prefix}

Original continuation:
{suffix}
"""


def normalize_text(text: str) -> str:
    text = re.sub(r"^```(?:text)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    return text


def call_teacher(
    prompt: str,
    endpoint: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    if requests is None:
        raise RuntimeError("requests is required for teacher rewriting")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return normalize_text(resp.json()["choices"][0]["message"].get("content", ""))
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"teacher rewrite failed: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--teacher-endpoint", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-temperature", type=float, default=0.6)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-max-tokens", type=int, default=256)
    parser.add_argument("--teacher-retries", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_hf_path(cfg["data"]["tokenizer_repo_cache"]),
        local_files_only=True,
        trust_remote_code=True,
    )
    suffix_tokens = int(cfg["data"]["suffix_tokens"])
    input_rows = list(jsonl_iter(args.input_jsonl))
    rows = input_rows
    output_path = Path(args.output_jsonl)
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        existing = {row["id"]: row for row in jsonl_iter(output_path)}
        rows = [row for row in rows if row["id"] not in existing]
    elif output_path.exists():
        output_path.unlink()

    def build(row: dict[str, Any]) -> dict[str, Any]:
        prefix = tokenizer.decode(row["prefix_ids"], skip_special_tokens=True)
        suffix = tokenizer.decode(row["original_suffix_ids"], skip_special_tokens=True)
        rewrite = call_teacher(
            PROMPT.format(prefix=prefix, suffix=suffix),
            endpoint=args.teacher_endpoint,
            model=args.teacher_model,
            temperature=args.teacher_temperature,
            top_p=args.teacher_top_p,
            max_tokens=args.teacher_max_tokens,
            timeout=120.0,
            retries=args.teacher_retries,
        )
        rewrite_ids = tokenizer.encode(rewrite, add_special_tokens=False)[:suffix_tokens]
        if len(rewrite_ids) < max(16, suffix_tokens // 4):
            raise RuntimeError(f"rewrite too short for row={row['id']}: {len(rewrite_ids)} tokens")
        out = dict(row)
        out["rewrite_suffix_ids"] = rewrite_ids
        out["rewrite_text"] = tokenizer.decode(rewrite_ids, skip_special_tokens=True)
        out["rewrite_model"] = args.teacher_model
        return out

    built_by_id: dict[str, dict[str, Any]] = dict(existing)
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(build, row): row["id"] for row in rows}
        for idx, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            built_by_id[row["id"]] = row
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if idx % 25 == 0 or idx == len(futures):
                print(json.dumps({"rewritten": idx, "remaining": len(futures) - idx}), flush=True)

    final_rows = [built_by_id[row["id"]] for row in input_rows if row["id"] in built_by_id]
    write_jsonl(output_path, final_rows)
    meta = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "rows": len(final_rows),
        "teacher_model": args.teacher_model,
        "suffix_tokens": suffix_tokens,
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
