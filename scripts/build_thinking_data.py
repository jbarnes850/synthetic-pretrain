#!/usr/bin/env python3
"""Build interleaved-thinking mid-training data from raw pretraining chunks."""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import jsonl_iter, load_config, resolve_hf_path, set_seed, write_jsonl
from transformers import AutoTokenizer

try:
    import requests
except Exception:  # pragma: no cover - dependency guard
    requests = None


PROMPT = """You are augmenting pretraining text with interleaved reasoning traces for thinking mid-training.

Given a contiguous text chunk scraped from a web page, return the same text in the same order, but insert brief missing intermediate contexts and reasoning/actions at semantically appropriate positions.

This should imitate how an intelligent learner actively reconstructs context, reasons, self-verifies, and learns reusable patterns while reading or producing the text.

Requirements:
- Preserve the original text content and order. Do not summarize, rewrite, omit, or answer as an assistant.
- Insert {min_thoughts}-{max_thoughts} total short interleaved spans across the entire chunk, not per sentence or per paragraph.
- Prefer only the highest-leverage positions: moments where a reader would need context, causal inference, self-verification, or an abstraction to predict what comes next.
- Use only these paired XML action tags: <global_context>...</global_context>, <world_model>...</world_model>, <recall_knowledge>...</recall_knowledge>, <simulate>...</simulate>, <reusable_lessons>...</reusable_lessons>, <verification>...</verification>.
- Every inserted action tag must have a matching closing tag before the original text resumes. Never use bare opening tags as labels.
- Each inserted span should be specific to the local text, not generic commentary.
- Do not mention a user or prompt.
- Do not include a separate analysis, thinking process, markdown fence, JSON object, or explanation.
- Return only the augmented text.

Miniature format example:
Original sentence one. <global_context>This note reconstructs why sentence two should follow.</global_context> Original sentence two. <verification>This checks the local claim before the next detail.</verification> Original sentence three.

Text chunk:
{chunk}
"""

ACTION_TAGS = (
    "global_context",
    "think",
    "world_model",
    "recall_knowledge",
    "simulate",
    "simulation",
    "reusable_lessons",
    "verification",
    "task",
    "set_goal",
    "tool_use",
    "python",
    "web_search",
)


def parse_augmented_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"teacher returned non-string content: {type(text).__name__}")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            value = obj.get("augmented_text", "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass
    match = re.search(r'"augmented_text"\s*:\s*"(?P<value>.*)"\s*}', text, flags=re.DOTALL)
    if match:
        try:
            return json.loads('"' + match.group("value") + '"').strip()
        except Exception:
            return match.group("value").replace(r"\/", "/").strip()
    return text.strip()


def normalize_think_tags(text: str) -> str:
    normalized = text
    for tag in ACTION_TAGS:
        normalized = re.sub(fr"<\s*{tag}\s*>", "<think>", normalized, flags=re.IGNORECASE)
        normalized = re.sub(fr"<\s*/\s*{tag}\s*>", "</think>", normalized, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def strip_think_tags(text: str) -> str:
    return re.sub(r"</?think>", "", text).strip()


def remove_think_spans(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def canonical_original_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def prune_think_spans(text: str, max_thoughts: int) -> str:
    spans_seen = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal spans_seen
        spans_seen += 1
        return match.group(0) if spans_seen <= max_thoughts else ""

    return re.sub(r"<think>.*?</think>", repl, text, flags=re.IGNORECASE | re.DOTALL)


def word_set(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9'’-]{2,}", text)}


def validate_augmented(raw_text: str, augmented_text: str, min_thoughts: int, max_thoughts: int) -> dict[str, Any]:
    think_count = len(re.findall(r"<think>", augmented_text, flags=re.IGNORECASE))
    close_count = len(re.findall(r"</think>", augmented_text, flags=re.IGNORECASE))
    stripped_text = remove_think_spans(augmented_text)
    stripped_words = word_set(stripped_text)
    raw_words = word_set(raw_text)
    coverage = len(raw_words & stripped_words) / max(1, len(raw_words))
    raw_canonical = canonical_original_text(raw_text)
    stripped_canonical = canonical_original_text(stripped_text)
    preserved_exactly = raw_canonical == stripped_canonical
    errors = []
    if think_count < min_thoughts:
        errors.append(f"too_few_thoughts:{think_count}")
    if think_count > max_thoughts:
        errors.append(f"too_many_thoughts:{think_count}")
    if think_count != close_count:
        errors.append(f"unbalanced_tags:{think_count}!={close_count}")
    if coverage < 0.60:
        errors.append(f"low_raw_word_coverage:{coverage:.3f}")
    if not preserved_exactly:
        errors.append("original_text_not_preserved")
    return {
        "think_count": think_count,
        "raw_word_coverage": coverage,
        "original_text_preserved": preserved_exactly,
        "errors": errors,
    }


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
        raise RuntimeError("requests is required for teacher generation")
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
            message = resp.json()["choices"][0]["message"]
            return parse_augmented_text(message.get("content"))
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"teacher call failed: {last_err}")


def choose_rows(
    rows: list[dict[str, Any]],
    train_count: int,
    val_count: int,
    seed: int,
    candidate_multiplier: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    train = [row for row in rows if row.get("split") == "train"]
    val = [row for row in rows if row.get("split") == "val"]
    rng.shuffle(train)
    rng.shuffle(val)
    if len(train) < train_count or len(val) < val_count:
        raise RuntimeError(
            f"Not enough rows: requested {train_count=} {val_count=}, got train={len(train)} val={len(val)}"
        )
    train_candidates = max(train_count, int(train_count * candidate_multiplier))
    val_candidates = max(val_count, int(val_count * candidate_multiplier))
    return train[:train_candidates] + val[:val_candidates]


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_meta(path: str | Path, meta: dict[str, Any]) -> None:
    Path(path).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--train-count", type=int, default=8192)
    parser.add_argument("--val-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5337)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--max-augmented-tokens", type=int, default=768)
    parser.add_argument("--tokenizer-cache", default="")
    parser.add_argument("--min-thoughts", type=int, default=3)
    parser.add_argument("--max-thoughts", type=int, default=6)
    parser.add_argument("--teacher-endpoint", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-temperature", type=float, default=0.6)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-max-tokens", type=int, default=1536)
    parser.add_argument("--teacher-retries", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=float, default=1.0)
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    tokenizer_cache = args.tokenizer_cache or cfg["data"]["tokenizer_repo_cache"]
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_hf_path(tokenizer_cache),
        local_files_only=True,
        trust_remote_code=True,
    )
    rows = choose_rows(
        list(jsonl_iter(args.input_jsonl)),
        args.train_count,
        args.val_count,
        args.seed,
        args.candidate_multiplier,
    )

    def prepare_item(idx_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, row = idx_row
        raw_ids = list(row["prefix_ids"]) + list(row["original_suffix_ids"])
        raw_ids = raw_ids[: args.chunk_tokens]
        raw_text = tokenizer.decode(raw_ids, skip_special_tokens=True)
        prompt = PROMPT.format(chunk=raw_text, min_thoughts=args.min_thoughts, max_thoughts=args.max_thoughts)
        return {"idx": idx, "row": row, "raw_ids": raw_ids, "raw_text": raw_text, "prompt": prompt}

    def finish_item(item: dict[str, Any], raw_augmented: str) -> dict[str, Any]:
        idx = item["idx"]
        row = item["row"]
        raw_text = item["raw_text"]
        augmented_text = prune_think_spans(normalize_think_tags(raw_augmented), args.max_thoughts)
        validation = validate_augmented(raw_text, augmented_text, args.min_thoughts, args.max_thoughts)
        augmented_ids = tokenizer.encode(augmented_text, add_special_tokens=False)[: args.max_augmented_tokens]
        decoded_augmented = tokenizer.decode(augmented_ids, skip_special_tokens=False)
        decoded_validation = validate_augmented(raw_text, decoded_augmented, args.min_thoughts, args.max_thoughts)
        errors = validation["errors"] + [f"decoded_{err}" for err in decoded_validation["errors"]]
        if errors or len(augmented_ids) < 32:
            preview = raw_augmented[:600].replace("\n", " ")
            raise RuntimeError(f"invalid augmentation for row={idx} id={row['id']}: {errors} preview={preview!r}")
        return {
            "id": row["id"],
            "split": row["split"],
            "source_sha": row.get("source_sha", ""),
            "raw_chunk_ids": item["raw_ids"],
            "interleaved_thinking_ids": augmented_ids,
            "raw_text": raw_text,
            "augmented_text": decoded_augmented,
            "teacher_model": args.teacher_model,
            "teacher_backend": "http",
            "prompt_kind": "interleaved_thinking",
            "chunk_tokens": args.chunk_tokens,
            "max_augmented_tokens": args.max_augmented_tokens,
            "think_count": decoded_validation["think_count"],
            "raw_word_coverage": decoded_validation["raw_word_coverage"],
            "original_text_preserved": decoded_validation["original_text_preserved"],
        }

    def build_one(idx_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        item = prepare_item(idx_row)
        idx = item["idx"]
        row = item["row"]
        last_error = ""
        for attempt in range(args.teacher_retries + 1):
            raw_augmented = call_teacher(
                item["prompt"],
                endpoint=args.teacher_endpoint,
                model=args.teacher_model,
                temperature=args.teacher_temperature,
                top_p=args.teacher_top_p,
                max_tokens=args.teacher_max_tokens,
                timeout=120.0,
                retries=0,
            )
            try:
                return finish_item(item, raw_augmented)
            except RuntimeError as exc:
                last_error = str(exc)
            if attempt < args.teacher_retries:
                time.sleep(0.75 * (attempt + 1))
        raise RuntimeError(f"invalid augmentation for row={idx} id={row['id']}: {last_error}")

    output_path = Path(args.output_jsonl)
    meta_path = output_path.with_suffix(".meta.json")
    skipped_path = output_path.with_suffix(".skipped.jsonl")
    if output_path.exists() and not args.resume:
        output_path.unlink()
    if meta_path.exists() and not args.resume:
        meta_path.unlink()
    if skipped_path.exists() and not args.resume:
        skipped_path.unlink()

    skipped_existing = {row["id"] for row in jsonl_iter(skipped_path)} if args.resume and skipped_path.exists() else set()
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        existing = {row["id"]: row for row in jsonl_iter(output_path)}
        rows = [row for row in rows if row["id"] not in existing and row["id"] not in skipped_existing]
        print(
            json.dumps(
                {
                    "resume_existing": len(existing),
                    "resume_skipped": len(skipped_existing),
                    "remaining": len(rows),
                }
            ),
            flush=True,
        )

    def record_skipped(row: dict[str, Any], error: str) -> None:
        append_jsonl(
            skipped_path,
            {
                "id": row.get("id"),
                "split": row.get("split"),
                "source_sha": row.get("source_sha", ""),
                "error": error[:500],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    out: list[dict[str, Any]] = []
    skipped_invalid = 0
    started_at = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(build_one, item): item for item in enumerate(rows)}
        for done, fut in enumerate(as_completed(futures), start=1):
            idx, src_row = futures[fut]
            try:
                row = fut.result()
            except RuntimeError as exc:
                if not args.skip_invalid:
                    raise
                skipped_invalid += 1
                record_skipped(src_row, str(exc))
                print(
                    json.dumps({"skipped_invalid_row": src_row.get("id"), "split": src_row.get("split"), "error": str(exc)[:500]}),
                    flush=True,
                )
                continue
            out.append(row)
            append_jsonl(output_path, row)
            if done % args.checkpoint_every == 0 or done == len(futures):
                elapsed = max(1e-9, time.time() - started_at)
                rate = done / elapsed
                progress = {
                    "output_jsonl": args.output_jsonl,
                    "status": "running",
                    "built_this_run": len(out),
                    "seen_this_run": done,
                    "remaining_this_run": len(futures) - done,
                    "resume_existing": len(existing),
                    "skipped_invalid": skipped_invalid,
                    "rows_per_hour": rate * 3600,
                    "eta_hours_remaining": (len(futures) - done) / max(1e-9, rate) / 3600,
                }
                write_meta(output_path, progress)
                print(json.dumps(progress), flush=True)

    final_rows = list(existing.values()) + out if existing else out
    n = write_jsonl(args.output_jsonl, final_rows) if existing else len(final_rows)
    meta = {
        "output_jsonl": args.output_jsonl,
        "rows": n,
        "train_count": sum(1 for row in final_rows if row["split"] == "train"),
        "val_count": sum(1 for row in final_rows if row["split"] == "val"),
        "chunk_tokens": args.chunk_tokens,
        "max_augmented_tokens": args.max_augmented_tokens,
        "teacher_endpoint": args.teacher_endpoint,
        "teacher_model": args.teacher_model,
        "teacher_backend": "http",
        "prompt_kind": "interleaved_thinking",
        "avg_think_count": sum(row["think_count"] for row in final_rows) / max(1, len(final_rows)),
        "avg_raw_word_coverage": sum(row["raw_word_coverage"] for row in final_rows) / max(1, len(final_rows)),
        "original_text_preserved_rate": sum(float(row["original_text_preserved"]) for row in final_rows)
        / max(1, len(final_rows)),
        "skipped_invalid": skipped_invalid,
        "status": "complete",
    }
    write_meta(args.output_jsonl, meta)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
