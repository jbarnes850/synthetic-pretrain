#!/usr/bin/env python3
"""Pre-training data integrity gate for interleaved thinking data.

This is the dataset-level check we wanted before spending training compute:
given held-out prefix/suffix examples, verify that inserted teacher thoughts
make suffix prediction better than blank or generic controls.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import torch
from common import jsonl_iter, load_config, now_run_id, resolve_hf_path, safe_mean, set_seed
from eval_reward_gate import parse_thought_only, word_count
from transformers import AutoModelForCausalLM, AutoTokenizer

AUTODATA_URL = "https://facebookresearch.github.io/RAM/blogs/autodata/"

GENERIC_THOUGHTS = [
    "Consider the local context and continue with a coherent relevant passage.",
    "Use the preceding text to infer what information should naturally come next.",
    "Maintain topic continuity and predict the next part of the source text.",
    "Focus on semantic consistency with the prefix before writing the continuation.",
]

DEFAULT_CONDITIONS = [
    "blank_thought",
    "generic_thought",
    "teacher_thought",
    "student_generated_thought",
    "swapped_teacher_thought",
]

JUDGE_PROMPT = """# Data Quality Judge

You are evaluating a held-out data-quality gate for thinking mid-training.
This adapts the Autodata acceptance loop: generate candidate data, evaluate it
with a judge, and accept the data recipe only when it creates a useful gap over
controls. Here, the controls are blank or generic thoughts, and the proposed
data signal is an inserted thought conditioned on the same prefix.
Corpus-style issues such as assistant-role leakage are checked by a separate
corpus-quality gate before this stage; do not penalize thought style unless it
affects the predicted continuation.

## Your Goal
Decide whether the model continuation is semantically useful for predicting the
ground-truth continuation. Reward the continuation, not the beauty of the
thought. Penalize generic, off-topic, contradictory, copied, or non-continuation
answers.

## Input
Prefix:
{prefix}

Inserted thought:
{thought}

Ground-truth continuation:
{reference}

Model predicted continuation:
{candidate}

## Acceptance Criteria
Return score 1 only when the predicted continuation is coherent, locally
relevant, and substantially matches or helps predict the ground-truth
continuation. Return score 0 when it is generic, unrelated, too short to judge,
mostly repeats the prefix, contradicts the reference, or fails to continue the
document.

Return only valid JSON:
{{"score": 0 or 1, "failure_mode": "none|generic|off_topic|too_short|repeat_prefix|contradiction|non_continuation|other", "reason": "brief reason"}}
"""


def load_tokenizer(cfg: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_hf_path(cfg["data"]["tokenizer_repo_cache"]),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_pretrained_model(cfg: dict[str, Any], device: torch.device):
    attn_impl = os.environ.get("SPARK_ATTN_IMPL", "sdpa")
    if attn_impl == "sdpa":
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception as exc:
            print(f"warning: could not configure SDP backends: {exc}", flush=True)
    dtype = torch.bfloat16 if cfg["runtime"].get("dtype") == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        resolve_hf_path(cfg["train"]["init_from_pretrained"]),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    model.to(device)
    model.eval()
    return model


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def clean_thought(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def remove_thought_spans(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def source_bucket(row: dict[str, Any]) -> str:
    for key in ("source", "source_name", "dataset", "corpus", "source_dataset"):
        value = row.get(key)
        if value:
            return str(value)
    source_id = str(row.get("source_id") or "")
    for known in ("dclm", "finemath", "fine_math", "fineweb"):
        if known in source_id.lower():
            return known
    return "unknown"


def build_gate_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    raw_prefix_tokens: int,
    suffix_tokens: int,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        augmented_text = row.get("augmented_text") or ""
        matches = list(re.finditer(r"<think>.*?</think>", augmented_text, flags=re.IGNORECASE | re.DOTALL))
        for match in matches:
            raw_before = remove_thought_spans(augmented_text[: match.start()]).strip()
            raw_after = remove_thought_spans(augmented_text[match.end() :]).strip()
            before_ids = encode(tokenizer, raw_before)
            after_ids = encode(tokenizer, raw_after)
            if len(before_ids) < raw_prefix_tokens // 2 or len(after_ids) < suffix_tokens // 2:
                continue
            prefix_ids = before_ids[-raw_prefix_tokens:]
            suffix_ids = after_ids[:suffix_tokens]
            prefix_text = decode(tokenizer, prefix_ids)
            out.append(
                {
                    "source_id": row["id"],
                    "source_bucket": source_bucket(row),
                    "prefix_ids": prefix_ids,
                    "reference_ids": suffix_ids,
                    "prefix_text": prefix_text,
                    "prompt_text": prefix_text.rstrip() + "\n<think>",
                    "reference_text": decode(tokenizer, suffix_ids),
                    "teacher_thought": match.group(0),
                }
            )
            break
    return out


def pad_left(tokenizer, rows: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(row) for row in rows)
    input_rows, attn_rows = [], []
    for row in rows:
        pad = max_len - len(row)
        input_rows.append([pad_id] * pad + row)
        attn_rows.append([0] * pad + [1] * len(row))
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(attn_rows, dtype=torch.long, device=device),
        max_len,
    )


@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    prompts: list[list[int]],
    samples_per_prompt: int,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[list[list[int]]]:
    flat_prompts = [prompt for prompt in prompts for _ in range(samples_per_prompt)]
    flat_outputs: list[list[int]] = []
    for start in range(0, len(flat_prompts), batch_size):
        batch = flat_prompts[start : start + batch_size]
        input_ids, attn, max_len = pad_left(tokenizer, batch, device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        for seq in out.tolist():
            flat_outputs.append(seq[max_len:])
    return [flat_outputs[idx : idx + samples_per_prompt] for idx in range(0, len(flat_outputs), samples_per_prompt)]


def deranged_indices(n: int, seed: int) -> list[int]:
    if n < 2:
        raise ValueError("swapped_teacher_thought requires at least two prefixes")
    indices = list(range(n))
    rng = random.Random(seed)
    for _ in range(100):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if all(src != dst for src, dst in zip(indices, shuffled)):
            return shuffled
    return indices[1:] + indices[:1]


def build_static_thoughts(
    eval_rows: list[dict[str, Any]],
    condition: str,
    samples_per_prefix: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if condition == "blank_thought":
        return [
            [
                {
                    "thought_text": "",
                    "thought_source": "blank",
                    "thought_source_prefix_index": prefix_index,
                    "thought_words": 0,
                }
                for _ in range(samples_per_prefix)
            ]
            for prefix_index in range(len(eval_rows))
        ]
    if condition == "generic_thought":
        out = []
        for prefix_index in range(len(eval_rows)):
            group = []
            for sample_index in range(samples_per_prefix):
                text = GENERIC_THOUGHTS[(prefix_index + sample_index) % len(GENERIC_THOUGHTS)]
                group.append(
                    {
                        "thought_text": text,
                        "thought_source": "generic",
                        "thought_source_prefix_index": prefix_index,
                        "thought_words": word_count(text),
                    }
                )
            out.append(group)
        return out
    if condition == "teacher_thought":
        return [
            [
                {
                    "thought_text": clean_thought(row.get("teacher_thought")),
                    "thought_source": "teacher",
                    "thought_source_prefix_index": prefix_index,
                    "thought_words": word_count(clean_thought(row.get("teacher_thought"))),
                }
                for _ in range(samples_per_prefix)
            ]
            for prefix_index, row in enumerate(eval_rows)
        ]
    if condition == "swapped_teacher_thought":
        swap = deranged_indices(len(eval_rows), seed)
        out = []
        for prefix_index, source_prefix_index in enumerate(swap):
            text = clean_thought(eval_rows[source_prefix_index].get("teacher_thought"))
            out.append(
                [
                    {
                        "thought_text": text,
                        "thought_source": "swapped_teacher",
                        "thought_source_prefix_index": source_prefix_index,
                        "thought_words": word_count(text),
                    }
                    for _ in range(samples_per_prefix)
                ]
            )
        return out
    raise ValueError(f"unsupported static condition: {condition}")


def build_student_thoughts(
    model,
    tokenizer,
    eval_rows: list[dict[str, Any]],
    samples_per_prefix: int,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[list[dict[str, Any]]]:
    prompts = [encode(tokenizer, row["prompt_text"]) for row in eval_rows]
    grouped = generate_samples(model, tokenizer, prompts, samples_per_prefix, max_new_tokens, batch_size, temperature, top_p, device)
    out = []
    for prefix_index, group in enumerate(grouped):
        thought_group = []
        for thought_ids in group:
            raw = decode(tokenizer, thought_ids)
            parsed = parse_thought_only(raw)
            thought_group.append(
                {
                    "thought_text": parsed["thought_text"],
                    "thought_source": "student_generated",
                    "thought_source_prefix_index": prefix_index,
                    "thought_words": parsed["thought_words"],
                    "autonomous_closed_think": parsed["autonomous_closed_think"],
                    "raw_thought_generation": raw,
                }
            )
        out.append(thought_group)
    return out


def parse_judge_response(content: str) -> dict[str, Any]:
    try:
        obj = json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        obj = json.loads(match.group(0)) if match else {}
    score = obj.get("score")
    if score in {0, 1}:
        parsed_score = int(score)
    elif isinstance(score, str) and score.strip() in {"0", "1"}:
        parsed_score = int(score.strip())
    else:
        parsed_score = None
    return {
        "score": parsed_score,
        "failure_mode": str(obj.get("failure_mode") or ""),
        "reason": str(obj.get("reason") or ""),
    }


def judge_one(
    prompt: str,
    endpoint: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = parse_judge_response(content)
            parsed["raw_judge"] = content
            return parsed
        except Exception as exc:
            last_err = exc
    return {"score": None, "failure_mode": "judge_error", "reason": str(last_err), "raw_judge": f"ERROR: {last_err}"}


def judge_repeated(
    prompts: list[str],
    endpoint: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_workers: int,
    repeats: int,
    retries: int,
) -> list[dict[str, Any]]:
    repeated = [(idx, prompt) for idx, prompt in enumerate(prompts) for _ in range(repeats)]
    grouped: list[list[dict[str, Any]]] = [[] for _ in prompts]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(judge_one, prompt, endpoint, model_name, temperature, top_p, max_tokens, 60.0, retries): idx
            for idx, prompt in repeated
        }
        for fut in as_completed(futures):
            grouped[futures[fut]].append(fut.result())
    out = []
    for judgments in grouped:
        valid_scores = [j["score"] for j in judgments if j.get("score") in {0, 1}]
        out.append(
            {
                "score": safe_mean([float(score) for score in valid_scores]) if valid_scores else None,
                "valid_judge_repeats": len(valid_scores),
                "invalid_judge_repeats": len(judgments) - len(valid_scores),
                "judge_repeats": judgments,
            }
        )
    return out


def evaluate_condition(
    model,
    tokenizer,
    eval_rows: list[dict[str, Any]],
    condition: str,
    condition_thoughts: list[list[dict[str, Any]]],
    args,
    judge_cfg: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    suffix_prompts: list[list[int]] = []
    prompt_records: list[dict[str, Any]] = []
    for prefix_index, (row, thoughts) in enumerate(zip(eval_rows, condition_thoughts)):
        for sample_index, thought in enumerate(thoughts):
            suffix_prompt = row["prompt_text"] + thought["thought_text"].strip() + "</think>"
            suffix_prompts.append(encode(tokenizer, suffix_prompt))
            prompt_records.append(
                {
                    "condition": condition,
                    "prefix_index": prefix_index,
                    "sample_index": sample_index,
                    "source_id": row["source_id"],
                    "source_bucket": source_bucket(row),
                    "prefix_text": row["prefix_text"],
                    "prompt_text": row["prompt_text"],
                    "reference_text": row["reference_text"],
                    "teacher_thought": clean_thought(row.get("teacher_thought")),
                    **thought,
                }
            )

    grouped_suffixes = generate_samples(
        model,
        tokenizer,
        suffix_prompts,
        1,
        args.suffix_tokens,
        args.gen_batch_size,
        args.temperature,
        args.top_p,
        device,
    )
    records: list[dict[str, Any]] = []
    judge_prompts: list[str] = []
    for record, suffix_group in zip(prompt_records, grouped_suffixes):
        predicted_suffix = decode(tokenizer, suffix_group[0]).strip()
        judge_prompts.append(
            JUDGE_PROMPT.format(
                prefix=record["prefix_text"][: args.max_judge_chars],
                thought=record["thought_text"][: args.max_judge_chars],
                reference=record["reference_text"][: args.max_judge_chars],
                candidate=predicted_suffix[: args.max_judge_chars],
            )
        )
        records.append(
            {
                **record,
                "predicted_suffix": predicted_suffix,
                "predicted_suffix_words": word_count(predicted_suffix),
                "suffix_tokens": len(suffix_group[0]),
            }
        )

    judgments = judge_repeated(judge_prompts, **judge_cfg)
    for record, judgment in zip(records, judgments):
        record.update(judgment)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prefix_means(records: list[dict[str, Any]]) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in records:
        if isinstance(row.get("score"), int | float):
            grouped[int(row["prefix_index"])].append(float(row["score"]))
    return {idx: safe_mean(vals) for idx, vals in grouped.items() if vals}


def bootstrap_ci(values: list[float], rounds: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "low": None, "high": None, "rounds": 0}
    if len(values) == 1 or rounds <= 0:
        mean = safe_mean(values)
        return {"mean": mean, "low": mean, "high": mean, "rounds": 0}
    rng = random.Random(seed)
    samples = []
    for _ in range(rounds):
        samples.append(safe_mean([values[rng.randrange(len(values))] for _ in values]))
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return {"mean": safe_mean(values), "low": low, "high": high, "rounds": rounds}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in records if isinstance(row.get("score"), int | float)]
    scores = [float(row["score"]) for row in valid]
    by_prefix = prefix_means(records)
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in valid:
        by_source[str(row.get("source_bucket") or "unknown")].append(float(row["score"]))
    return {
        "records": len(records),
        "valid": len(valid),
        "invalid": len(records) - len(valid),
        "invalid_rate": (len(records) - len(valid)) / max(1, len(records)),
        "reward_mean": safe_mean(scores),
        "reward_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "prefix_mean_reward": safe_mean(list(by_prefix.values())),
        "prefixes": len(by_prefix),
        "avg_predicted_suffix_words": safe_mean([row["predicted_suffix_words"] for row in records]),
        "avg_thought_words": safe_mean([row["thought_words"] for row in records]),
        "by_source": {
            source: {
                "records": len(values),
                "reward_mean": safe_mean(values),
            }
            for source, values in sorted(by_source.items())
        },
    }


def summarize_comparisons(records_by_condition: dict[str, list[dict[str, Any]]], args) -> dict[str, Any]:
    means = {condition: prefix_means(records) for condition, records in records_by_condition.items()}
    blank = means.get("blank_thought", {})
    generic = means.get("generic_thought", {})
    teacher = means.get("teacher_thought", {})
    student = means.get("student_generated_thought", {})
    swapped = means.get("swapped_teacher_thought", {})

    shared_controls = sorted(set(teacher) & set(blank) & set(generic))
    teacher_best_deltas = [teacher[idx] - max(blank[idx], generic[idx]) for idx in shared_controls]
    teacher_blank_deltas = [teacher[idx] - blank[idx] for idx in sorted(set(teacher) & set(blank))]
    teacher_generic_deltas = [teacher[idx] - generic[idx] for idx in sorted(set(teacher) & set(generic))]
    teacher_swapped_deltas = [teacher[idx] - swapped[idx] for idx in sorted(set(teacher) & set(swapped))]
    student_best_shared = sorted(set(student) & set(blank) & set(generic))
    student_best_deltas = [student[idx] - max(blank[idx], generic[idx]) for idx in student_best_shared]

    ci = bootstrap_ci(teacher_best_deltas, args.bootstrap_rounds, args.seed)
    comparison = {
        "teacher_minus_blank": {
            "shared_prefixes": len(teacher_blank_deltas),
            "mean_delta": safe_mean(teacher_blank_deltas),
            "teacher_better_prefix_rate": safe_mean([float(delta > 0) for delta in teacher_blank_deltas]),
        },
        "teacher_minus_generic": {
            "shared_prefixes": len(teacher_generic_deltas),
            "mean_delta": safe_mean(teacher_generic_deltas),
            "teacher_better_prefix_rate": safe_mean([float(delta > 0) for delta in teacher_generic_deltas]),
        },
        "teacher_minus_best_control": {
            "shared_prefixes": len(teacher_best_deltas),
            "mean_delta": safe_mean(teacher_best_deltas),
            "bootstrap_ci_95": ci,
            "teacher_better_prefix_rate": safe_mean([float(delta > 0) for delta in teacher_best_deltas]),
            "control_better_prefix_rate": safe_mean([float(delta < 0) for delta in teacher_best_deltas]),
            "tie_prefix_rate": safe_mean([float(abs(delta) < 1e-9) for delta in teacher_best_deltas]),
        },
        "teacher_minus_swapped": {
            "shared_prefixes": len(teacher_swapped_deltas),
            "mean_delta": safe_mean(teacher_swapped_deltas),
            "teacher_better_prefix_rate": safe_mean([float(delta > 0) for delta in teacher_swapped_deltas]),
        },
        "student_generated_minus_best_control": {
            "shared_prefixes": len(student_best_deltas),
            "mean_delta": safe_mean(student_best_deltas),
            "student_better_prefix_rate": safe_mean([float(delta > 0) for delta in student_best_deltas]),
        },
    }
    max_invalid_rate = max((summarize_records(records)["invalid_rate"] for records in records_by_condition.values()), default=1.0)
    lower_ci = ci["low"]
    comparison["go_no_go"] = {
        "passed": (
            len(teacher_best_deltas) >= args.min_prefixes
            and max_invalid_rate <= args.max_invalid_rate
            and safe_mean(teacher_best_deltas) >= args.min_teacher_control_delta
            and comparison["teacher_minus_best_control"]["teacher_better_prefix_rate"] >= args.min_teacher_control_win_rate
            and (lower_ci is not None and lower_ci > 0.0)
            and (not teacher_swapped_deltas or safe_mean(teacher_swapped_deltas) >= args.min_teacher_swapped_delta)
        ),
        "min_prefixes_ok": len(teacher_best_deltas) >= args.min_prefixes,
        "judge_validity_ok": max_invalid_rate <= args.max_invalid_rate,
        "teacher_margin_ok": safe_mean(teacher_best_deltas) >= args.min_teacher_control_delta,
        "teacher_win_rate_ok": comparison["teacher_minus_best_control"]["teacher_better_prefix_rate"]
        >= args.min_teacher_control_win_rate,
        "teacher_ci_ok": lower_ci is not None and lower_ci > 0.0,
        "swapped_control_ok": (not teacher_swapped_deltas or safe_mean(teacher_swapped_deltas) >= args.min_teacher_swapped_delta),
        "max_invalid_rate": max_invalid_rate,
    }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/thinking_sft_base.yaml")
    parser.add_argument("--input-jsonl", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--judge-endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--judge-model", default="qwen36-35b-a3b")
    parser.add_argument("--seed", type=int, default=6337)
    parser.add_argument("--num-prefixes", type=int, default=128)
    parser.add_argument("--samples-per-prefix", type=int, default=4)
    parser.add_argument("--raw-prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--thought-max-new-tokens", type=int, default=48)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--judge-temperature", type=float, default=0.6)
    parser.add_argument("--judge-top-p", type=float, default=0.95)
    parser.add_argument("--judge-max-tokens", type=int, default=160)
    parser.add_argument("--judge-max-workers", type=int, default=32)
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--judge-retries", type=int, default=1)
    parser.add_argument("--max-judge-chars", type=int, default=2000)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    parser.add_argument("--min-prefixes", type=int, default=64)
    parser.add_argument("--max-invalid-rate", type=float, default=0.05)
    parser.add_argument("--min-teacher-control-delta", type=float, default=0.03)
    parser.add_argument("--min-teacher-control-win-rate", type=float, default=0.55)
    parser.add_argument("--min-teacher-swapped-delta", type=float, default=0.0)
    args = parser.parse_args()

    unknown = [condition for condition in args.conditions if condition not in DEFAULT_CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}; available={DEFAULT_CONDITIONS}")
    for required in ("blank_thought", "generic_thought", "teacher_thought"):
        if required not in args.conditions:
            raise ValueError(f"{required} is required for the data integrity gate")

    set_seed(args.seed)
    cfg = load_config(args.config)
    input_jsonl = args.input_jsonl or cfg["data"].get("heldout_examples_jsonl") or cfg["data"]["interleaved_thinking_examples_jsonl"]
    rows = [row for row in jsonl_iter(input_jsonl) if row.get("split") in {"val", "heldout"}]
    if not rows:
        rows = list(jsonl_iter(input_jsonl))
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    tokenizer = load_tokenizer(cfg)
    eval_rows = build_gate_rows(rows, tokenizer, args.raw_prefix_tokens, args.suffix_tokens)[: args.num_prefixes]
    if len(eval_rows) < args.min_prefixes:
        raise RuntimeError(f"Only {len(eval_rows)} valid eval rows found; required at least {args.min_prefixes}")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"].get("device") == "cuda" else "cpu")
    model = load_pretrained_model(cfg, device)
    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/data_integrity_gate") / now_run_id("data-integrity")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "eval_rows.jsonl", eval_rows)

    judge_cfg = {
        "endpoint": args.judge_endpoint,
        "model_name": args.judge_model,
        "temperature": args.judge_temperature,
        "top_p": args.judge_top_p,
        "max_tokens": args.judge_max_tokens,
        "max_workers": args.judge_max_workers,
        "repeats": args.judge_repeats,
        "retries": args.judge_retries,
    }
    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "config": vars(args),
        "input_jsonl": input_jsonl,
        "num_eval_prefixes": len(eval_rows),
        "paper_alignment": {
            "autodata_url": AUTODATA_URL,
            "adapted_acceptance_loop": "generate candidate data -> evaluate with judge -> accept only if useful gap over controls",
            "weak_strong_mapping": "blank/generic controls are weak baselines; matched teacher thoughts are the proposed strong data signal",
            "gate_question": "Do teacher/generated thoughts improve suffix prediction over blank/generic controls on held-out prefixes?",
            "no_training": "This script performs no weight updates.",
        },
        "conditions": {},
    }

    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in args.conditions:
        print(f"evaluating condition={condition}", flush=True)
        if condition == "student_generated_thought":
            thoughts = build_student_thoughts(
                model,
                tokenizer,
                eval_rows,
                args.samples_per_prefix,
                args.thought_max_new_tokens,
                args.gen_batch_size,
                args.temperature,
                args.top_p,
                device,
            )
        else:
            thoughts = build_static_thoughts(eval_rows, condition, args.samples_per_prefix, args.seed)
        records = evaluate_condition(model, tokenizer, eval_rows, condition, thoughts, args, judge_cfg, device)
        records_by_condition[condition] = records
        records_path = out_dir / f"{condition}.jsonl"
        write_jsonl(records_path, records)
        summary["conditions"][condition] = summarize_records(records) | {"records_path": str(records_path)}
        (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    summary["comparisons"] = summarize_comparisons(records_by_condition, args)
    summary["status"] = "pass" if summary["comparisons"]["go_no_go"]["passed"] else "fail"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(out_dir / "summary.json"), "status": summary["status"]}, indent=2), flush=True)
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
