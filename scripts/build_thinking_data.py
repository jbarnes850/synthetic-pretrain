#!/usr/bin/env python3
"""Build paper-aligned interleaved thinking SFT data from raw pretraining chunks."""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import jsonl_iter, latest_snapshot, load_config, set_seed, write_jsonl
from transformers import AutoTokenizer

try:
    import requests
except Exception:  # pragma: no cover - optional for local MLX generation
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
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def strip_think_tags(text: str) -> str:
    return re.sub(r"</?think>", "", text).strip()


def prune_think_spans(text: str, max_thoughts: int) -> str:
    spans_seen = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal spans_seen
        spans_seen += 1
        if spans_seen <= max_thoughts:
            return match.group(0)
        return ""

    return re.sub(r"<think>.*?</think>", repl, text, flags=re.IGNORECASE | re.DOTALL)


def word_set(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9'’-]{2,}", text)}


def validate_augmented(raw_text: str, augmented_text: str, min_thoughts: int, max_thoughts: int) -> dict[str, Any]:
    think_count = len(re.findall(r"<think>", augmented_text, flags=re.IGNORECASE))
    close_count = len(re.findall(r"</think>", augmented_text, flags=re.IGNORECASE))
    stripped_words = word_set(strip_think_tags(augmented_text))
    raw_words = word_set(raw_text)
    coverage = len(raw_words & stripped_words) / max(1, len(raw_words))
    errors = []
    if think_count < min_thoughts:
        errors.append(f"too_few_thoughts:{think_count}")
    if think_count > max_thoughts:
        errors.append(f"too_many_thoughts:{think_count}")
    if think_count != close_count:
        errors.append(f"unbalanced_tags:{think_count}!={close_count}")
    if coverage < 0.60:
        errors.append(f"low_raw_word_coverage:{coverage:.3f}")
    return {
        "think_count": think_count,
        "raw_word_coverage": coverage,
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
        raise RuntimeError("requests is required for --teacher-backend=http")
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


def call_mlx_teacher(
    prompt: str,
    model: Any,
    tokenizer: Any,
    sampler: Any,
    max_tokens: int,
) -> str:
    from mlx_lm import generate

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return parse_augmented_text(
        generate(model, tokenizer, prompt=prompt_ids, max_tokens=max_tokens, sampler=sampler, verbose=False)
    )


def call_mlx_teacher_batch(
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    sampler: Any,
    max_tokens: int,
) -> list[str]:
    from mlx_lm import batch_generate

    prompt_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    response = batch_generate(model, tokenizer, prompt_ids, max_tokens=max_tokens, sampler=sampler, verbose=False)
    return [parse_augmented_text(text) for text in response.texts]


def choose_rows(
    rows: list[dict[str, Any]],
    train_count: int,
    val_count: int,
    seed: int,
    candidate_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    train = [row for row in rows if row.get("split") == "train"]
    val = [row for row in rows if row.get("split") == "val"]
    rng.shuffle(train)
    rng.shuffle(val)
    train_candidates = max(train_count, int(train_count * candidate_multiplier))
    val_candidates = max(val_count, int(val_count * candidate_multiplier))
    selected = train[:train_candidates] + val[:val_candidates]
    if len(train) < train_count or len(val) < val_count:
        raise RuntimeError(
            f"Not enough rows: requested {train_count=} {val_count=}, got train={len(train)} val={len(val)}"
        )
    return selected


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
    parser.add_argument("--teacher-endpoint", default="")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--teacher-backend", choices=("http", "mlx"), default="http")
    parser.add_argument("--mlx-model-path", default="")
    parser.add_argument("--teacher-temperature", type=float, default=0.6)
    parser.add_argument("--teacher-top-p", type=float, default=0.95)
    parser.add_argument("--teacher-max-tokens", type=int, default=1536)
    parser.add_argument("--teacher-retries", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--mlx-batch-size", type=int, default=1)
    parser.add_argument("--candidate-multiplier", type=float, default=1.0)
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    if args.teacher_backend == "http" and not args.teacher_endpoint:
        raise RuntimeError("--teacher-endpoint is required for --teacher-backend=http")
    if args.teacher_backend == "mlx" and not args.mlx_model_path:
        raise RuntimeError("--mlx-model-path is required for --teacher-backend=mlx")

    set_seed(args.seed)
    cfg = load_config(args.config)
    tokenizer_cache = args.tokenizer_cache or cfg["data"]["tokenizer_repo_cache"]
    tokenizer = AutoTokenizer.from_pretrained(
        latest_snapshot(tokenizer_cache),
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

    mlx_model = None
    mlx_tokenizer = None
    mlx_sampler = None
    if args.teacher_backend == "mlx":
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        mlx_model, mlx_tokenizer = load(args.mlx_model_path)
        mlx_sampler = make_sampler(temp=args.teacher_temperature, top_p=args.teacher_top_p)
        args.max_workers = 1

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
        raw_ids = item["raw_ids"]
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
            "raw_chunk_ids": raw_ids,
            "interleaved_thinking_ids": augmented_ids,
            "raw_text": raw_text,
            "augmented_text": decoded_augmented,
            "teacher_model": args.teacher_model or args.mlx_model_path,
            "teacher_backend": args.teacher_backend,
            "prompt_kind": "paper_interleaved_thinking_v1",
            "chunk_tokens": args.chunk_tokens,
            "max_augmented_tokens": args.max_augmented_tokens,
            "think_count": decoded_validation["think_count"],
            "raw_word_coverage": decoded_validation["raw_word_coverage"],
        }

    def build_one(idx_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        item = prepare_item(idx_row)
        idx = item["idx"]
        row = item["row"]
        last_error = ""
        for attempt in range(args.teacher_retries + 1):
            if args.teacher_backend == "mlx":
                raw_augmented = call_mlx_teacher(
                    item["prompt"],
                    model=mlx_model,
                    tokenizer=mlx_tokenizer,
                    sampler=mlx_sampler,
                    max_tokens=args.teacher_max_tokens,
                )
            else:
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

    def build_mlx_batch(idx_rows: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
        remaining = [prepare_item(item) for item in idx_rows]
        finished: list[dict[str, Any]] = []
        last_errors: dict[str, str] = {}
        for attempt in range(args.teacher_retries + 1):
            outputs = call_mlx_teacher_batch(
                [item["prompt"] for item in remaining],
                model=mlx_model,
                tokenizer=mlx_tokenizer,
                sampler=mlx_sampler,
                max_tokens=args.teacher_max_tokens,
            )
            retry_items = []
            for item, raw_augmented in zip(remaining, outputs, strict=True):
                row = item["row"]
                try:
                    finished.append(finish_item(item, raw_augmented))
                except RuntimeError as exc:
                    last_errors[row["id"]] = str(exc)
                    retry_items.append(item)
            if not retry_items:
                return finished
            remaining = retry_items
            if attempt < args.teacher_retries:
                time.sleep(0.75 * (attempt + 1))
        first = remaining[0]
        row = first["row"]
        raise RuntimeError(last_errors.get(row["id"], f"invalid augmentation for row={first['idx']} id={row['id']}"))

    output_path = Path(args.output_jsonl)
    if output_path.exists() and not args.resume:
        output_path.unlink()
    meta_path = output_path.with_suffix(".meta.json")
    if meta_path.exists() and not args.resume:
        meta_path.unlink()
    skipped_path = output_path.with_suffix(".skipped.jsonl")
    if skipped_path.exists() and not args.resume:
        skipped_path.unlink()
    skipped_existing: set[str] = set()
    if args.resume and skipped_path.exists():
        skipped_existing = {row["id"] for row in jsonl_iter(skipped_path)}
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

    out: list[dict[str, Any] | None] = []
    existing_train = sum(1 for row in existing.values() if row["split"] == "train")
    existing_val = sum(1 for row in existing.values() if row["split"] == "val")
    need_train = max(0, args.train_count - existing_train)
    need_val = max(0, args.val_count - existing_val)
    built_train = 0
    built_val = 0
    skipped_invalid = 0
    started_at = time.time()
    if args.teacher_backend == "mlx":
        enumerated_rows = [
            item
            for item in enumerate(rows)
            if (item[1].get("split") == "train" and need_train > 0) or (item[1].get("split") == "val" and need_val > 0)
        ]
        done = 0
        next_checkpoint = args.checkpoint_every
        for start in range(0, len(enumerated_rows), args.mlx_batch_size):
            if built_train >= need_train and built_val >= need_val:
                break
            batch = []
            for item in enumerated_rows[start : start + args.mlx_batch_size]:
                split = item[1].get("split")
                if split == "train" and built_train >= need_train:
                    continue
                if split == "val" and built_val >= need_val:
                    continue
                batch.append(item)
            if not batch:
                continue
            try:
                batch_rows = build_mlx_batch(batch)
            except RuntimeError as exc:
                if not args.skip_invalid:
                    raise
                batch_rows = []
                batch_error = str(exc)[:500]
                for item in batch:
                    try:
                        batch_rows.append(build_one(item))
                    except RuntimeError as item_exc:
                        skipped_invalid += 1
                        record_skipped(item[1], str(item_exc))
                        print(
                            json.dumps(
                                {
                                    "skipped_invalid_row": item[1].get("id"),
                                    "split": item[1].get("split"),
                                    "batch_error": batch_error,
                                    "row_error": str(item_exc)[:500],
                                }
                            ),
                            flush=True,
                        )
            for row in batch_rows:
                if row["split"] == "train" and built_train >= need_train:
                    continue
                if row["split"] == "val" and built_val >= need_val:
                    continue
                out.append(row)
                append_jsonl(output_path, row)
                if row["split"] == "train":
                    built_train += 1
                elif row["split"] == "val":
                    built_val += 1
            done += len(batch_rows)
            if done >= next_checkpoint or done == len(enumerated_rows):
                elapsed = max(1e-9, time.time() - started_at)
                rate = done / elapsed
                target_remaining = max(0, need_train - built_train) + max(0, need_val - built_val)
                progress = {
                    "output_jsonl": args.output_jsonl,
                    "status": "running",
                    "built_this_run": done,
                    "remaining_this_run": target_remaining,
                    "resume_existing": len(existing),
                    "existing_train": existing_train,
                    "existing_val": existing_val,
                    "built_train_this_run": built_train,
                    "built_val_this_run": built_val,
                    "skipped_invalid": skipped_invalid,
                    "resume_skipped": len(skipped_existing),
                    "mlx_batch_size": args.mlx_batch_size,
                    "rows_per_hour": rate * 3600,
                    "eta_hours_remaining": target_remaining / max(1e-9, rate) / 3600,
                }
                write_meta(output_path, progress)
                print(
                    json.dumps(
                        {
                            "built": done,
                            "total": len(enumerated_rows),
                            "resume_existing": len(existing),
                            "target_remaining": target_remaining,
                            "skipped_invalid": skipped_invalid,
                            "mlx_batch_size": args.mlx_batch_size,
                            "rows_per_hour": rate * 3600,
                            "eta_hours_remaining": target_remaining / max(1e-9, rate) / 3600,
                        }
                    ),
                    flush=True,
                )
                while next_checkpoint <= done:
                    next_checkpoint += args.checkpoint_every
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(build_one, item): item[0] for item in enumerate(rows)}
            done = 0
            for fut in as_completed(futures):
                row = fut.result()
                out.append(row)
                append_jsonl(output_path, row)
                done += 1
            if done % args.checkpoint_every == 0:
                elapsed = max(1e-9, time.time() - started_at)
                rate = done / elapsed
                progress = {
                    "output_jsonl": args.output_jsonl,
                    "status": "running",
                    "built_this_run": done,
                    "remaining_this_run": len(rows) - done,
                    "resume_existing": len(existing),
                    "rows_per_hour": rate * 3600,
                    "eta_hours_remaining": (len(rows) - done) / max(1e-9, rate) / 3600,
                }
                write_meta(output_path, progress)
                print(
                        json.dumps(
                            {
                                "built": done,
                                "total": len(rows),
                                "resume_existing": len(existing),
                                "rows_per_hour": rate * 3600,
                                "eta_hours_remaining": (len(rows) - done) / max(1e-9, rate) / 3600,
                            }
                        ),
                        flush=True,
                    )

    if existing:
        final_rows = list(existing.values()) + out
        n = write_jsonl(args.output_jsonl, final_rows)
    else:
        final_rows = out
        n = len(final_rows)
    meta = {
        "output_jsonl": args.output_jsonl,
        "rows": n,
        "train_count": sum(1 for row in final_rows if row["split"] == "train"),
        "val_count": sum(1 for row in final_rows if row["split"] == "val"),
        "chunk_tokens": args.chunk_tokens,
        "max_augmented_tokens": args.max_augmented_tokens,
        "teacher_endpoint": args.teacher_endpoint,
        "teacher_model": args.teacher_model or args.mlx_model_path,
        "teacher_backend": args.teacher_backend,
        "prompt_kind": "paper_interleaved_thinking_v1",
        "avg_think_count": sum(row["think_count"] for row in final_rows) / max(1, len(final_rows)),
        "avg_raw_word_coverage": sum(row["raw_word_coverage"] for row in final_rows) / max(1, len(final_rows)),
        "status": "complete",
    }
    write_meta(args.output_jsonl, meta)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
