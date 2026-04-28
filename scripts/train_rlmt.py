#!/usr/bin/env python3
"""Small custom RLMT loop for thinking mid-training.

Implements the paper-aligned object:
prefix -> generated thought -> predicted suffix -> judge(predicted suffix, true suffix)
followed by a group-relative policy update over K samples per prefix.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from common import jsonl_iter, load_config, read_available_mem_gib, safe_mean, set_seed
from eval_reward_gate import (
    JUDGE_PROMPT,
    build_eval_rows,
    parse_thought_only,
    word_count,
)
from eval_thinking import ARM_SPECS, decode, encode, judge_pointwise, load_model, load_tokenizer
from transformers import get_cosine_schedule_with_warmup

RLMT_ARMS = {
    "think_base": {
        "config": ARM_SPECS["think_base"]["config"],
        "checkpoint": ARM_SPECS["think_base"]["checkpoint"],
        "output_dir": "outputs/rlmt_base",
    },
    "think_phase3": {
        "config": ARM_SPECS["think_phase3"]["config"],
        "checkpoint": ARM_SPECS["think_phase3"]["checkpoint"],
        "output_dir": "outputs/rlmt_self_improved",
    },
}


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
    grouped: list[list[list[int]]] = []
    for idx in range(0, len(flat_outputs), samples_per_prompt):
        grouped.append(flat_outputs[idx : idx + samples_per_prompt])
    return grouped


def pad_training_batch(
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(row["input_ids"]) for row in records)
    input_rows, label_rows, attn_rows = [], [], []
    for row in records:
        pad = max_len - len(row["input_ids"])
        input_rows.append(row["input_ids"] + [pad_id] * pad)
        label_rows.append(row["labels"] + [-100] * pad)
        attn_rows.append([1] * len(row["input_ids"]) + [0] * pad)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(label_rows, dtype=torch.long, device=device),
        torch.tensor(attn_rows, dtype=torch.long, device=device),
    )


def sequence_logps(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    safe_targets = targets.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    counts = mask.sum(dim=1).clamp_min(1)
    return token_logps.sum(dim=1) / counts, counts


def artifact_flag(text: str) -> bool:
    lowered = text.lower()
    if "�" in text or lowered.count("<think>") > 2 or lowered.count("</think>") > 2:
        return True
    if "as an ai" in lowered or "i cannot" in lowered or "i'm unable" in lowered:
        return True
    if re.search(r"(.)\1{24,}", text):
        return True
    visible = re.sub(r"\s+", "", text)
    if len(visible) >= 40:
        alpha_ratio = sum(ch.isalpha() for ch in visible) / max(1, len(visible))
        if alpha_ratio < 0.35:
            return True
    return False


def group_advantages(records: list[dict[str, Any]], eps: float) -> tuple[list[float], list[dict[str, Any]]]:
    by_prefix: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_prefix[int(row["prefix_index"])].append(row)
    advantages = [0.0] * len(records)
    stats = []
    index_by_id = {id(row): idx for idx, row in enumerate(records)}
    for prefix_index, group in sorted(by_prefix.items()):
        rewards = [float(row["score"]) for row in group]
        mean = safe_mean(rewards)
        variance = safe_mean([(reward - mean) ** 2 for reward in rewards])
        std = math.sqrt(variance)
        for row, reward in zip(group, rewards):
            advantages[index_by_id[id(row)]] = 0.0 if std < eps else (reward - mean) / (std + eps)
        stats.append(
            {
                "prefix_index": prefix_index,
                "mean": mean,
                "std": std,
                "range": max(rewards) - min(rewards) if rewards else 0.0,
                "valid": len(rewards),
            }
        )
    return advantages, stats


def summarize_step(
    records: list[dict[str, Any]],
    group_stats: list[dict[str, Any]],
    elapsed_s: float,
    tokens_scored: int,
) -> dict[str, Any]:
    valid = [row for row in records if row["score"] in {0, 1}]
    scores = [float(row["score"]) for row in valid]
    near_zero = [g for g in group_stats if g["valid"] >= 2 and g["std"] < 1e-5]
    mixed = [g for g in group_stats if g["range"] > 0]
    all_zero = [g for g in group_stats if g["valid"] > 0 and g["mean"] <= 0.0]
    all_one = [g for g in group_stats if g["valid"] > 0 and g["mean"] >= 1.0]
    total_new = [row["thought_tokens"] + row["suffix_tokens"] for row in records]
    return {
        "reward_mean": safe_mean(scores),
        "reward_std": math.sqrt(safe_mean([(score - safe_mean(scores)) ** 2 for score in scores])) if scores else 0.0,
        "invalid_judge_rate": (len(records) - len(valid)) / max(1, len(records)),
        "near_zero_group_rate": len(near_zero) / max(1, len(group_stats)),
        "mixed_group_rate": len(mixed) / max(1, len(group_stats)),
        "all_zero_group_rate": len(all_zero) / max(1, len(group_stats)),
        "all_one_group_rate": len(all_one) / max(1, len(group_stats)),
        "avg_thought_words": safe_mean([row["thought_words"] for row in records]),
        "avg_suffix_words": safe_mean([row["predicted_suffix_words"] for row in records]),
        "avg_total_new_tokens": safe_mean(total_new),
        "artifact_rate": safe_mean([float(row["artifact_flag"]) for row in records]),
        "samples_per_sec": len(records) / max(1e-6, elapsed_s),
        "tok_per_sec": tokens_scored / max(1e-6, elapsed_s),
    }


def resolve_rows(args, tokenizer) -> list[dict[str, Any]]:
    rows = [row for row in jsonl_iter(args.data_path) if row["split"] == args.rl_split]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.row_offset:
        rows = rows[args.row_offset :]
    eval_rows = build_eval_rows(
        rows,
        tokenizer,
        args.raw_prefix_tokens,
        args.suffix_tokens,
        "raw",
        "two_stage_external_boundary",
    )
    if len(eval_rows) < args.prefixes_per_step:
        raise RuntimeError(f"Only {len(eval_rows)} usable RLMT rows found")
    return eval_rows


def build_rollout_records(
    tokenizer,
    eval_rows: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    grouped_thoughts: list[list[list[int]]],
    grouped_suffixes: list[list[list[int]]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    boundary_ids = encode(tokenizer, "</think>")
    records: list[dict[str, Any]] = []
    judge_rows: list[dict[str, str]] = []
    for row, thought_outputs, suffix_outputs in zip(batch_rows, grouped_thoughts, grouped_suffixes):
        prefix_index = eval_rows.index(row)
        prompt_ids = encode(tokenizer, row["prompt_text"])
        for sample_index, (thought_gen_ids, suffix_group) in enumerate(zip(thought_outputs, suffix_outputs)):
            raw_thought = decode(tokenizer, thought_gen_ids)
            parsed = parse_thought_only(raw_thought)
            thought_ids = encode(tokenizer, parsed["thought_text"].strip())
            suffix_ids = suffix_group[0]
            predicted_suffix = decode(tokenizer, suffix_ids).strip()
            input_ids = prompt_ids + thought_ids + boundary_ids + suffix_ids
            labels = [-100] * len(prompt_ids) + thought_ids + [-100] * len(boundary_ids) + suffix_ids
            record = {
                "prefix_index": prefix_index,
                "sample_index": sample_index,
                "source_id": row["source_id"],
                "prompt_text": row["prompt_text"],
                "reference_text": row["reference_text"],
                "thought_text": parsed["thought_text"],
                "raw_thought_generation": raw_thought,
                "predicted_suffix": predicted_suffix,
                "raw_generation": parsed["thought_text"] + "</think>" + predicted_suffix,
                "thought_words": parsed["thought_words"],
                "predicted_suffix_words": word_count(predicted_suffix),
                "thought_tokens": len(thought_ids),
                "suffix_tokens": len(suffix_ids),
                "artifact_flag": artifact_flag(parsed["thought_text"] + "\n" + predicted_suffix),
                "input_ids": input_ids,
                "labels": labels,
            }
            judge_rows.append(
                {
                    "judge_prompt": JUDGE_PROMPT.format(
                        prefix=row["prefix_text"][:2000],
                        reference=row["reference_text"][:2000],
                        candidate=predicted_suffix[:2000],
                    )
                }
            )
            records.append(record)
    return records, judge_rows


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            clean = {k: v for k, v in record.items() if k not in {"input_ids", "labels"}}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(RLMT_ARMS), required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-path", default="data/processed/interleaved_thinking.jsonl")
    parser.add_argument("--judge-endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--judge-model", default="qwen-judge")
    parser.add_argument("--seed", type=int, default=4337)
    parser.add_argument("--rl-split", choices=["train", "val"], default="train")
    parser.add_argument("--row-offset", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--samples-per-prefix", type=int, default=16)
    parser.add_argument("--prefixes-per-step", type=int, default=2)
    parser.add_argument("--raw-prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--thought-max-new-tokens", type=int, default=48)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--judge-temperature", type=float, default=0.7)
    parser.add_argument("--judge-top-p", type=float, default=0.6)
    parser.add_argument("--judge-max-tokens", type=int, default=64)
    parser.add_argument("--judge-max-workers", type=int, default=32)
    parser.add_argument("--min-available-mem-gib", type=float, default=4.0)
    parser.add_argument("--max-invalid-rate", type=float, default=0.05)
    parser.add_argument("--max-artifact-rate", type=float, default=0.10)
    parser.add_argument("--near-zero-worsen-pp", type=float, default=0.20)
    parser.add_argument("--max-length-growth", type=float, default=0.20)
    parser.add_argument(
        "--enforce-stop-conditions",
        action="store_true",
        help="Stop automatically on guardrail alerts. By default alerts are logged only for manual monitoring.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    spec = RLMT_ARMS[args.arm]
    cfg = load_config(args.config or spec["config"])
    checkpoint = Path(args.checkpoint or spec["checkpoint"])
    output_dir = Path(args.output_dir or spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"].get("device") == "cuda" else "cpu")
    tokenizer = load_tokenizer(cfg)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_rows = resolve_rows(args, tokenizer)
    model = load_model(cfg, checkpoint, device)
    model.train()
    ref_model = load_model(cfg, checkpoint, device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.steps)
    judge_cfg = {
        "endpoint": args.judge_endpoint,
        "model_name": args.judge_model,
        "temperature": args.judge_temperature,
        "top_p": args.judge_top_p,
        "max_tokens": args.judge_max_tokens,
        "max_workers": args.judge_max_workers,
    }

    cursor = 0
    baseline_near_zero = None
    baseline_total_tokens = None
    all_logs = []
    stopped_reason = None
    reward_collapse_streak = 0
    train_start = time.time()

    for step in range(1, args.steps + 1):
        step_start = time.time()
        if cursor + args.prefixes_per_step > len(eval_rows):
            random.Random(args.seed + step).shuffle(eval_rows)
            cursor = 0
        batch_rows = eval_rows[cursor : cursor + args.prefixes_per_step]
        cursor += args.prefixes_per_step

        thought_prompts = [encode(tokenizer, row["prompt_text"]) for row in batch_rows]
        grouped_thoughts = generate_samples(
            model,
            tokenizer,
            thought_prompts,
            args.samples_per_prefix,
            args.thought_max_new_tokens,
            args.gen_batch_size,
            args.temperature,
            args.top_p,
            device,
        )
        suffix_prompts: list[list[int]] = []
        for row, thought_outputs in zip(batch_rows, grouped_thoughts):
            for thought_ids in thought_outputs:
                parsed = parse_thought_only(decode(tokenizer, thought_ids))
                suffix_prompts.append(encode(tokenizer, row["prompt_text"] + parsed["thought_text"].strip() + "</think>"))
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
        grouped_suffixes_by_prefix = [
            grouped_suffixes[idx : idx + args.samples_per_prefix]
            for idx in range(0, len(grouped_suffixes), args.samples_per_prefix)
        ]
        records, judge_rows = build_rollout_records(tokenizer, eval_rows, batch_rows, grouped_thoughts, grouped_suffixes_by_prefix)
        judgments = judge_pointwise(judge_rows, **judge_cfg)
        for record, judgment in zip(records, judgments):
            record["score"] = judgment["score"]
            record["raw_judge"] = judgment["raw_judge"]

        valid_records = [row for row in records if row["score"] in {0, 1}]
        if not valid_records:
            stopped_reason = "all_judges_invalid"
            break
        advantages, group_stats = group_advantages(valid_records, args.advantage_eps)
        for record, advantage in zip(valid_records, advantages):
            record["advantage"] = advantage

        input_ids, labels, attn = pad_training_batch(tokenizer, valid_records, device)
        with torch.no_grad():
            old_logps, _ = sequence_logps(model, input_ids, labels, attn)
            ref_logps, _ = sequence_logps(ref_model, input_ids, labels, attn)
        new_logps, counts = sequence_logps(model, input_ids, labels, attn)
        adv = torch.tensor(advantages, dtype=torch.float32, device=device)
        ratio = torch.exp((new_logps - old_logps).clamp(-20, 20))
        clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
        policy_loss = -torch.minimum(ratio * adv, clipped_ratio * adv).mean()
        log_ratio_ref = (ref_logps - new_logps).clamp(-20, 20)
        kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
        loss = policy_loss + args.kl_coef * kl.mean()
        if not torch.isfinite(loss):
            stopped_reason = "non_finite_loss"
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - step_start
        tokens_scored = int(counts.sum().detach().cpu())
        step_summary = summarize_step(valid_records, group_stats, elapsed, tokens_scored)
        mem = read_available_mem_gib()
        if baseline_near_zero is None:
            baseline_near_zero = step_summary["near_zero_group_rate"]
            baseline_total_tokens = step_summary["avg_total_new_tokens"]
        length_growth = (
            step_summary["avg_total_new_tokens"] / max(1e-6, float(baseline_total_tokens)) - 1.0
            if baseline_total_tokens
            else 0.0
        )
        log_entry = {
            "step": step,
            "train_loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "kl": float(kl.mean().detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "learning_rate": scheduler.get_last_lr()[0],
            "available_mem_gib": mem,
            "response_length_growth": length_growth,
            "tokens_scored": tokens_scored,
            **step_summary,
        }
        stop_alerts = []
        if mem >= 0 and mem < args.min_available_mem_gib:
            stop_alerts.append("low_memory")
        if log_entry["invalid_judge_rate"] > args.max_invalid_rate:
            stop_alerts.append("judge_invalid_rate")
        if log_entry["artifact_rate"] > args.max_artifact_rate:
            stop_alerts.append("artifact_rate")
        if length_growth > args.max_length_growth:
            stop_alerts.append("response_length_growth")
        if log_entry["near_zero_group_rate"] > baseline_near_zero + args.near_zero_worsen_pp:
            stop_alerts.append("near_zero_group_worsened")
        if log_entry["all_zero_group_rate"] >= 1.0 or log_entry["all_one_group_rate"] >= 1.0:
            reward_collapse_streak += 1
        else:
            reward_collapse_streak = 0
        if reward_collapse_streak >= 2:
            stop_alerts.append("reward_collapsed")
        log_entry["stop_alerts"] = stop_alerts
        log_entry["stop_conditions_enforced"] = args.enforce_stop_conditions
        print(json.dumps(log_entry), flush=True)
        all_logs.append(log_entry)
        save_jsonl(output_dir / f"step_{step:04d}_samples.jsonl", records)

        if args.enforce_stop_conditions and stop_alerts:
            stopped_reason = stop_alerts[0]
            break

        if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
            torch.save(model.state_dict(), output_dir / f"step_{step}.pt")

    final_step = all_logs[-1]["step"] if all_logs else 0
    torch.save(model.state_dict(), output_dir / "final.pt")
    metrics = {
        "arm": args.arm,
        "output_dir": str(output_dir),
        "source_checkpoint": str(checkpoint),
        "completed_steps": final_step,
        "requested_steps": args.steps,
        "stopped_reason": stopped_reason,
        "wall_time_s": time.time() - train_start,
        "final": all_logs[-1] if all_logs else {},
        "history": all_logs,
        "paper_alignment": {
            "objective": "prefix -> generated thought -> predicted suffix -> binary judge reward on predicted suffix",
            "optimizer": "small GRPO/DrGRPO-style clipped group-relative policy update with frozen SFT reference KL",
            "boundary": "two_stage_external_boundary; </think> is externally inserted and not rewarded directly",
            "data_split": "Uses the configured RL split/offset as the D_RL approximation; prefixes are built from augmented chunks and references are visible/raw suffix text after the thought boundary.",
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"final_checkpoint": str(output_dir / "final.pt"), **metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
