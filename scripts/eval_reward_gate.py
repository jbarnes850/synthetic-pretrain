#!/usr/bin/env python3
"""Small RLMT reward-signal gate for thinking SFT arms.

This is a read-only pre-RL check. It samples multiple candidate
"thought + predicted suffix" completions per held-out prefix, asks the judge
whether each predicted suffix matches the held-out continuation, and reports
whether the binary reward has enough within-prefix variance for RLMT.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from common import load_config, load_split_rows, now_run_id, safe_mean, set_seed
from eval_thinking import ARM_SPECS, decode, encode, judge_pointwise, load_model, load_tokenizer

DEFAULT_THINK_ARMS = ["think_base", "think_self_improved"]

JUDGE_PROMPT = """You are judging a reward for thinking mid-training.

The model saw a prefix from a pretraining document, generated intermediate thinking, and then generated a predicted continuation.

Prefix:
{prefix}

Ground-truth continuation:
{reference}

Model predicted continuation:
{candidate}

Return only valid JSON: {{"score": 1}} if the model predicted continuation is coherent, locally relevant, and semantically useful for predicting the ground-truth continuation. Return {{"score": 0}} otherwise.
"""


@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    prompts: list[list[int]],
    samples_per_prefix: int,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[list[list[int]]]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    flat_prompts = [prompt for prompt in prompts for _ in range(samples_per_prefix)]
    flat_outputs: list[list[int]] = []
    for start in range(0, len(flat_prompts), batch_size):
        print(f"  generate batch {start // batch_size + 1}/{max(1, (len(flat_prompts) + batch_size - 1) // batch_size)}", flush=True)
        batch = flat_prompts[start : start + batch_size]
        max_len = max(len(ids) for ids in batch)
        input_rows, attn_rows = [], []
        for ids in batch:
            pad = max_len - len(ids)
            input_rows.append([pad_id] * pad + ids)
            attn_rows.append([0] * pad + [1] * len(ids))
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attn = torch.tensor(attn_rows, dtype=torch.long, device=device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
        )
        for seq in out.tolist():
            flat_outputs.append(seq[max_len:])
    grouped: list[list[list[int]]] = []
    for idx in range(0, len(flat_outputs), samples_per_prefix):
        grouped.append(flat_outputs[idx : idx + samples_per_prefix])
    return grouped


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def parse_generation(raw_text: str) -> dict[str, Any]:
    if "</think>" in raw_text:
        before, after = raw_text.split("</think>", 1)
        thought_text = before.replace("<think>", "").strip()
        candidate = after.strip()
        return {
            "closed_think": True,
            "thought_words": word_count(thought_text),
            "predicted_suffix": candidate or raw_text.strip(),
        }
    stripped = re.sub(r"^<think>", "", raw_text.strip(), flags=re.IGNORECASE).strip()
    return {
        "closed_think": False,
        "thought_words": 0,
        "predicted_suffix": stripped or raw_text.strip(),
    }


def parse_thought_only(raw_text: str) -> dict[str, Any]:
    if "</think>" in raw_text:
        thought_text = raw_text.split("</think>", 1)[0].replace("<think>", "").strip()
        autonomous_closed = True
    else:
        thought_text = re.sub(r"^<think>", "", raw_text.strip(), flags=re.IGNORECASE).strip()
        autonomous_closed = False
    return {
        "thought_text": thought_text,
        "thought_words": word_count(thought_text),
        "autonomous_closed_think": autonomous_closed,
    }


def strip_thought_spans(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def build_eval_rows(
    rows: list[dict[str, Any]],
    tokenizer,
    raw_prefix_tokens: int,
    suffix_tokens: int,
    reference_mode: str,
    interface_mode: str,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if interface_mode.startswith("teacher_boundary") or interface_mode == "two_stage_external_boundary":
            matches = list(re.finditer(r"<think>.*?</think>", row["augmented_text"], flags=re.DOTALL))
            chosen = None
            for match in matches:
                before = row["augmented_text"][: match.start()]
                after = row["augmented_text"][match.end() :]
                before_ids = encode(tokenizer, before)
                suffix_source = strip_thought_spans(after) if interface_mode == "two_stage_external_boundary" else after
                after_ids = encode(tokenizer, suffix_source)
                if len(before_ids) >= raw_prefix_tokens // 2 and len(after_ids) >= suffix_tokens // 2:
                    chosen = (match, before, after_ids)
                    break
            if chosen is None:
                continue
            match, before, after_ids = chosen
            if interface_mode in {"teacher_boundary_open_tag", "two_stage_external_boundary"}:
                prompt_text = before.rstrip() + " <think>"
            elif interface_mode == "teacher_boundary_explicit_format":
                prompt_text = (
                    before.rstrip()
                    + "\n\nContinue the document in this exact format:\n"
                    + "<think>one brief local thought about what comes next</think>\n"
                    + "then continue the original document text.\n"
                    + "<think>"
                )
            elif interface_mode == "teacher_boundary_no_cue":
                prompt_text = before.rstrip()
            else:
                raise ValueError(f"unsupported interface_mode: {interface_mode}")
            prefix_ids = encode(tokenizer, prompt_text)
            suffix_ids = after_ids[:suffix_tokens]
            reference_text = decode(tokenizer, suffix_ids)
            teacher_thought = match.group(0)
        elif reference_mode == "augmented":
            ids = encode(tokenizer, row["augmented_text"])
            prefix_ids = ids[:raw_prefix_tokens]
            suffix_ids = ids[raw_prefix_tokens : raw_prefix_tokens + suffix_tokens]
            prompt_text = decode(tokenizer, prefix_ids).rstrip() + "\n<think>"
            reference_text = decode(tokenizer, suffix_ids)
            teacher_thought = None
        else:
            ids = list(row["raw_chunk_ids"])
            prefix_ids = ids[:raw_prefix_tokens]
            suffix_ids = ids[raw_prefix_tokens : raw_prefix_tokens + suffix_tokens]
            prompt_text = decode(tokenizer, prefix_ids).rstrip() + "\n<think>"
            reference_text = decode(tokenizer, suffix_ids)
            teacher_thought = None
        if len(prefix_ids) < raw_prefix_tokens // 2 or len(suffix_ids) < suffix_tokens // 2:
            continue
        out.append(
            {
                "source_id": row["id"],
                "prefix_ids": prefix_ids,
                "reference_ids": suffix_ids,
                "prefix_text": decode(tokenizer, prefix_ids),
                "prompt_text": prompt_text,
                "reference_text": reference_text,
                "teacher_thought": teacher_thought,
            }
        )
    return out


def summarize_arm(records: list[dict[str, Any]], samples_per_prefix: int) -> dict[str, Any]:
    valid = [row for row in records if row["score"] in {0, 1}]
    invalid = len(records) - len(valid)
    scores = [float(row["score"]) for row in valid]
    by_prefix: dict[int, list[float]] = defaultdict(list)
    for row in valid:
        by_prefix[int(row["prefix_index"])].append(float(row["score"]))
    group_stats = []
    for prefix_index, group_scores in sorted(by_prefix.items()):
        if len(group_scores) < 2:
            std = 0.0
        else:
            std = statistics.pstdev(group_scores)
        group_stats.append(
            {
                "prefix_index": prefix_index,
                "mean": safe_mean(group_scores),
                "std": std,
                "range": max(group_scores) - min(group_scores) if group_scores else 0.0,
                "valid": len(group_scores),
            }
        )
    near_zero_groups = [g for g in group_stats if g["valid"] >= 2 and g["std"] < 1e-5]
    mixed_groups = [g for g in group_stats if g["range"] > 0]
    total_tokens = [row["total_new_tokens"] for row in records]
    suffix_words = [row["predicted_suffix_words"] for row in records]
    return {
        "num_records": len(records),
        "valid": len(valid),
        "invalid": invalid,
        "invalid_rate": invalid / max(1, len(records)),
        "reward_mean": safe_mean(scores),
        "reward_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "reward_min": min(scores) if scores else None,
        "reward_max": max(scores) if scores else None,
        "num_groups": len(group_stats),
        "samples_per_prefix": samples_per_prefix,
        "mean_group_reward": safe_mean([g["mean"] for g in group_stats]),
        "mean_group_std": safe_mean([g["std"] for g in group_stats]),
        "mean_group_range": safe_mean([g["range"] for g in group_stats]),
        "near_zero_group_rate": len(near_zero_groups) / max(1, len(group_stats)),
        "mixed_group_rate": len(mixed_groups) / max(1, len(group_stats)),
        "any_success_group_rate": safe_mean([float(g["mean"] > 0.0) for g in group_stats]),
        "all_success_group_rate": safe_mean([float(g["mean"] >= 1.0) for g in group_stats]),
        "avg_total_new_tokens": safe_mean(total_tokens),
        "avg_predicted_suffix_words": safe_mean(suffix_words),
        "closed_think_rate": safe_mean([float(row["closed_think"]) for row in records]),
        "autonomous_closed_think_rate": safe_mean(
            [float(row.get("autonomous_closed_think", row["closed_think"])) for row in records]
        ),
        "avg_thought_words_when_closed": safe_mean(
            [row["thought_words"] for row in records if row["closed_think"]]
        ),
    }


def compare_arms(records_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    means_by_arm: dict[str, dict[int, float]] = {}
    for arm, records in records_by_arm.items():
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in records:
            if row["score"] in {0, 1}:
                grouped[int(row["prefix_index"])].append(float(row["score"]))
        means_by_arm[arm] = {idx: safe_mean(vals) for idx, vals in grouped.items() if vals}
    base = means_by_arm.get("think_base", {})
    self_improved = means_by_arm.get("think_self_improved", {})
    shared = sorted(set(base) & set(self_improved))
    deltas = [self_improved[idx] - base[idx] for idx in shared]
    pairwise: dict[str, Any] = {}
    for left in sorted(means_by_arm):
        for right in sorted(means_by_arm):
            if left >= right:
                continue
            shared_pair = sorted(set(means_by_arm[left]) & set(means_by_arm[right]))
            pair_deltas = [means_by_arm[right][idx] - means_by_arm[left][idx] for idx in shared_pair]
            pairwise[f"{right}_minus_{left}"] = {
                "shared_prefixes": len(shared_pair),
                "mean_reward_delta": safe_mean(pair_deltas),
                "right_better_prefix_rate": safe_mean([float(delta > 0) for delta in pair_deltas]),
                "left_better_prefix_rate": safe_mean([float(delta < 0) for delta in pair_deltas]),
                "tie_prefix_rate": safe_mean([float(abs(delta) < 1e-9) for delta in pair_deltas]),
            }
    return {
        "shared_prefixes": len(shared),
        "mean_self_improved_minus_base_reward": safe_mean(deltas),
        "self_improved_better_prefix_rate": safe_mean([float(delta > 0) for delta in deltas]),
        "base_better_prefix_rate": safe_mean([float(delta < 0) for delta in deltas]),
        "tie_prefix_rate": safe_mean([float(abs(delta) < 1e-9) for delta in deltas]),
        "pairwise": pairwise,
    }


def evaluate_two_stage_arm(
    arm_name: str,
    model,
    tokenizer,
    eval_rows: list[dict[str, Any]],
    args,
    judge_cfg: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    thought_prompts = [encode(tokenizer, row["prompt_text"]) for row in eval_rows]
    print(f"sampling thoughts {arm_name}", flush=True)
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
    suffix_prompt_records: list[dict[str, Any]] = []
    suffix_prompts: list[list[int]] = []
    for prefix_index, (row, thought_outputs) in enumerate(zip(eval_rows, grouped_thoughts)):
        for sample_index, thought_ids in enumerate(thought_outputs):
            raw_thought = decode(tokenizer, thought_ids)
            parsed = parse_thought_only(raw_thought)
            suffix_prompt = row["prompt_text"] + parsed["thought_text"].strip() + "</think>"
            suffix_prompts.append(encode(tokenizer, suffix_prompt))
            suffix_prompt_records.append(
                {
                    "arm": arm_name,
                    "prefix_index": prefix_index,
                    "sample_index": sample_index,
                    "source_id": row["source_id"],
                    "prefix_text": row["prefix_text"],
                    "prompt_text": row["prompt_text"],
                    "reference_text": row["reference_text"],
                    "teacher_thought": row["teacher_thought"],
                    "raw_thought_generation": raw_thought,
                    "thought_text": parsed["thought_text"],
                    "thought_words": parsed["thought_words"],
                    "autonomous_closed_think": parsed["autonomous_closed_think"],
                }
            )
    print(f"sampling suffixes {arm_name}", flush=True)
    flat_suffix_outputs = generate_samples(
        model,
        tokenizer,
        suffix_prompts,
        1,
        args.max_new_tokens,
        args.gen_batch_size,
        args.temperature,
        args.top_p,
        device,
    )
    judge_rows = []
    records: list[dict[str, Any]] = []
    for record, suffix_group in zip(suffix_prompt_records, flat_suffix_outputs):
        suffix_ids = suffix_group[0]
        predicted_suffix = decode(tokenizer, suffix_ids).strip()
        judge_rows.append(
            {
                "judge_prompt": JUDGE_PROMPT.format(
                    prefix=record["prefix_text"][:2000],
                    reference=record["reference_text"][:2000],
                    candidate=predicted_suffix[:2000],
                )
            }
        )
        records.append(
            {
                **record,
                "raw_generation": record["raw_thought_generation"] + "</think>" + predicted_suffix,
                "predicted_suffix": predicted_suffix,
                "closed_think": True,
                "total_new_tokens": len(encode(tokenizer, record["thought_text"])) + len(suffix_ids),
                "predicted_suffix_words": word_count(predicted_suffix),
                "external_boundary": True,
            }
        )
    print(f"judging {arm_name}", flush=True)
    judgments = judge_pointwise(judge_rows, **judge_cfg)
    for record, judgment in zip(records, judgments):
        record["score"] = judgment["score"]
        record["raw_judge"] = judgment["raw_judge"]
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--judge-endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--judge-model", default="qwen36-35b-a3b")
    parser.add_argument("--seed", type=int, default=4337)
    parser.add_argument("--num-prefixes", type=int, default=32)
    parser.add_argument("--samples-per-prefix", type=int, default=4)
    parser.add_argument("--raw-prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reference-mode", choices=["raw", "augmented"], default="raw")
    parser.add_argument(
        "--interface-mode",
        choices=[
            "raw_open_tag",
            "teacher_boundary_open_tag",
            "teacher_boundary_explicit_format",
            "teacher_boundary_no_cue",
            "two_stage_external_boundary",
        ],
        default="two_stage_external_boundary",
    )
    parser.add_argument("--thought-max-new-tokens", type=int, default=48)
    parser.add_argument("--judge-temperature", type=float, default=0.7)
    parser.add_argument("--judge-top-p", type=float, default=0.6)
    parser.add_argument("--judge-max-tokens", type=int, default=64)
    parser.add_argument("--judge-max-workers", type=int, default=32)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=DEFAULT_THINK_ARMS,
        help="Thinking/RLMT arm names from eval_thinking.ARM_SPECS.",
    )
    args = parser.parse_args()

    unknown_arms = [arm for arm in args.arms if arm not in ARM_SPECS]
    if unknown_arms:
        raise ValueError(f"Unknown arms: {unknown_arms}; available={sorted(ARM_SPECS)}")

    set_seed(args.seed)
    cfg = load_config(ARM_SPECS[args.arms[0]]["config"])
    tokenizer = load_tokenizer(cfg)
    rows = load_split_rows(
        cfg["data"]["interleaved_thinking_examples_jsonl"],
        "val",
        cfg["data"].get("heldout_examples_jsonl"),
    )
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    eval_rows = build_eval_rows(
        rows,
        tokenizer,
        args.raw_prefix_tokens,
        args.suffix_tokens,
        args.reference_mode,
        args.interface_mode,
    )[: args.num_prefixes]
    if not eval_rows:
        raise RuntimeError("No valid held-out rows found for RLMT reward gate")

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/reward_gate") / now_run_id("reward-gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"].get("device") == "cuda" else "cpu")
    judge_cfg = {
        "endpoint": args.judge_endpoint,
        "model_name": args.judge_model,
        "temperature": args.judge_temperature,
        "top_p": args.judge_top_p,
        "max_tokens": args.judge_max_tokens,
        "max_workers": args.judge_max_workers,
    }

    summary: dict[str, Any] = {
        "output_dir": str(out_dir),
        "config": vars(args),
        "paper_alignment": {
            "source": "Tan et al. Section 2 RLMT: prefix -> generated thinking + predicted suffix; LLM judge compares predicted suffix to held-out suffix and returns binary reward.",
            "scaled_gate": "Small held-out validation batch with multiple stochastic samples per prefix; no policy update is performed.",
            "reward_variance_gate": "Before GRPO/DrGRPO-style training, groups with reward std < 1e-5 should not dominate.",
            "two_stage_note": "two_stage_external_boundary samples tau_hat first, externally inserts the thought/suffix boundary, then samples s_hat; this preserves the paper's reward object while avoiding XML-boundary brittleness.",
        },
        "num_eval_prefixes": len(eval_rows),
        "arms": {},
    }
    records_by_arm: dict[str, list[dict[str, Any]]] = {}

    prompts = [encode(tokenizer, row["prompt_text"]) for row in eval_rows]
    for arm_name in args.arms:
        spec = ARM_SPECS[arm_name]
        print(f"loading {arm_name}", flush=True)
        arm_cfg = load_config(spec["config"])
        model = load_model(arm_cfg, Path(spec["checkpoint"]), device)
        if args.interface_mode == "two_stage_external_boundary":
            records = evaluate_two_stage_arm(arm_name, model, tokenizer, eval_rows, args, judge_cfg, device)
        else:
            print(f"sampling {arm_name}", flush=True)
            grouped_outputs = generate_samples(
                model,
                tokenizer,
                prompts,
                args.samples_per_prefix,
                args.max_new_tokens,
                args.gen_batch_size,
                args.temperature,
                args.top_p,
                device,
            )
            judge_rows = []
            records = []
            for prefix_index, (row, outputs) in enumerate(zip(eval_rows, grouped_outputs)):
                for sample_index, gen_ids in enumerate(outputs):
                    raw_generation = decode(tokenizer, gen_ids)
                    parsed = parse_generation(raw_generation)
                    predicted_suffix = parsed["predicted_suffix"]
                    judge_rows.append(
                        {
                            "judge_prompt": JUDGE_PROMPT.format(
                                prefix=row["prefix_text"][:2000],
                                reference=row["reference_text"][:2000],
                                candidate=predicted_suffix[:2000],
                            )
                        }
                    )
                    records.append(
                        {
                            "arm": arm_name,
                            "prefix_index": prefix_index,
                            "sample_index": sample_index,
                            "source_id": row["source_id"],
                            "prefix_text": row["prefix_text"],
                            "prompt_text": row["prompt_text"],
                            "reference_text": row["reference_text"],
                            "teacher_thought": row["teacher_thought"],
                            "raw_generation": raw_generation,
                            "predicted_suffix": predicted_suffix,
                            "closed_think": parsed["closed_think"],
                            "thought_words": parsed["thought_words"],
                            "total_new_tokens": len(gen_ids),
                            "predicted_suffix_words": word_count(predicted_suffix),
                        }
                    )
            print(f"judging {arm_name}", flush=True)
            judgments = judge_pointwise(judge_rows, **judge_cfg)
            for record, judgment in zip(records, judgments):
                record["score"] = judgment["score"]
                record["raw_judge"] = judgment["raw_judge"]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        records_by_arm[arm_name] = records
        arm_out = out_dir / f"{arm_name}.jsonl"
        with arm_out.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary["arms"][arm_name] = summarize_arm(records, args.samples_per_prefix) | {
            "examples_path": str(arm_out)
        }
        (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    summary["arm_comparison"] = compare_arms(records_by_arm)
    near_zero_rates = [
        arm["near_zero_group_rate"]
        for arm in summary["arms"].values()
        if isinstance(arm.get("near_zero_group_rate"), float)
    ]
    summary["go_no_go"] = {
        "reward_validity_ok": all(arm["invalid_rate"] <= 0.05 for arm in summary["arms"].values()),
        "variance_ok": all(rate <= 0.50 for rate in near_zero_rates),
    }
    if "think_base" in summary["arms"] and "think_self_improved" in summary["arms"]:
        summary["go_no_go"]["self_improved_more_reward_separable"] = (
            summary["arms"]["think_self_improved"]["mean_group_std"] > summary["arms"]["think_base"]["mean_group_std"]
            or summary["arms"]["think_self_improved"]["mixed_group_rate"] > summary["arms"]["think_base"]["mixed_group_rate"]
        )
        summary["go_no_go"]["self_improved_higher_mean_reward"] = (
            summary["arm_comparison"]["mean_self_improved_minus_base_reward"] > 0
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
