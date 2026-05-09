#!/usr/bin/env python3
"""Causal thought-use probe for thinking mid-training.

The probe intervenes on the thought text while keeping the paper-aligned reward
object fixed: judge only the predicted suffix against the true suffix.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from common import load_config, load_split_rows, now_run_id, safe_mean, set_seed
from eval_reward_gate import JUDGE_PROMPT, build_eval_rows, parse_thought_only, word_count
from eval_thinking import ARM_SPECS, decode, encode, judge_pointwise, load_model, load_tokenizer

DEFAULT_ARMS = ["think_base", "think_self_improved", "think_base_rlmt", "think_self_improved_rlmt"]
DEFAULT_CONDITIONS = ["normal_model_thought", "blank_thought", "generic_thought", "same_arm_swapped_thought"]
GENERIC_THOUGHTS = [
    "Consider the local context and continue with a coherent relevant passage.",
    "Use the preceding text to infer what information should naturally come next.",
    "Maintain topic continuity and predict the next part of the source text.",
    "Focus on semantic consistency with the prefix before writing the continuation.",
]


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


def deranged_indices(n: int, seed: int) -> list[int]:
    if n < 2:
        raise ValueError("same_arm_swapped_thought requires at least two prefixes")
    indices = list(range(n))
    rng = random.Random(seed)
    for _ in range(100):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if all(src != dst for src, dst in zip(indices, shuffled)):
            return shuffled
    return indices[1:] + indices[:1]


def build_condition_thoughts(
    condition: str,
    normal_thoughts: list[list[dict[str, Any]]],
    eval_rows: list[dict[str, Any]],
    samples_per_prefix: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if condition == "normal_model_thought":
        return normal_thoughts
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
    if condition == "same_arm_swapped_thought":
        swap = deranged_indices(len(eval_rows), seed)
        out = []
        for prefix_index, source_prefix_index in enumerate(swap):
            group = []
            for sample_index in range(samples_per_prefix):
                source = normal_thoughts[source_prefix_index][sample_index % len(normal_thoughts[source_prefix_index])]
                group.append(
                    {
                        "thought_text": source["thought_text"],
                        "thought_source": "same_arm_swapped",
                        "thought_source_prefix_index": source_prefix_index,
                        "thought_words": source["thought_words"],
                    }
                )
            out.append(group)
        return out
    if condition == "teacher_thought":
        out = []
        for prefix_index, row in enumerate(eval_rows):
            raw = row.get("teacher_thought") or ""
            text = raw.replace("<think>", "").replace("</think>", "").strip()
            group = [
                {
                    "thought_text": text,
                    "thought_source": "teacher",
                    "thought_source_prefix_index": prefix_index,
                    "thought_words": word_count(text),
                }
                for _ in range(samples_per_prefix)
            ]
            out.append(group)
        return out
    raise ValueError(f"unsupported condition: {condition}")


def summarize_records(records: list[dict[str, Any]], samples_per_prefix: int) -> dict[str, Any]:
    valid = [row for row in records if row["score"] in {0, 1}]
    scores = [float(row["score"]) for row in valid]
    by_prefix: dict[int, list[float]] = defaultdict(list)
    for row in valid:
        by_prefix[int(row["prefix_index"])].append(float(row["score"]))
    prefix_means = [safe_mean(vals) for vals in by_prefix.values()]
    return {
        "records": len(records),
        "valid": len(valid),
        "invalid": len(records) - len(valid),
        "invalid_rate": (len(records) - len(valid)) / max(1, len(records)),
        "reward_mean": safe_mean(scores),
        "reward_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "prefix_mean_reward": safe_mean(prefix_means),
        "any_success_prefix_rate": safe_mean([float(max(vals) > 0.0) for vals in by_prefix.values()]),
        "all_success_prefix_rate": safe_mean([float(min(vals) >= 1.0) for vals in by_prefix.values()]),
        "samples_per_prefix": samples_per_prefix,
        "avg_predicted_suffix_words": safe_mean([row["predicted_suffix_words"] for row in records]),
        "avg_thought_words": safe_mean([row["thought_words"] for row in records]),
    }


def summarize_deltas(records_by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    means_by_condition: dict[str, dict[int, float]] = {}
    for condition, records in records_by_condition.items():
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in records:
            if row["score"] in {0, 1}:
                grouped[int(row["prefix_index"])].append(float(row["score"]))
        means_by_condition[condition] = {idx: safe_mean(vals) for idx, vals in grouped.items()}
    normal = means_by_condition.get("normal_model_thought", {})
    out = {}
    for condition, means in means_by_condition.items():
        if condition == "normal_model_thought":
            continue
        shared = sorted(set(normal) & set(means))
        deltas = [normal[idx] - means[idx] for idx in shared]
        out[f"normal_minus_{condition}"] = {
            "shared_prefixes": len(shared),
            "mean_delta": safe_mean(deltas),
            "normal_better_prefix_rate": safe_mean([float(delta > 0) for delta in deltas]),
            "condition_better_prefix_rate": safe_mean([float(delta < 0) for delta in deltas]),
            "tie_prefix_rate": safe_mean([float(abs(delta) < 1e-9) for delta in deltas]),
        }
    return out


def evaluate_condition(
    arm_name: str,
    condition: str,
    model,
    tokenizer,
    eval_rows: list[dict[str, Any]],
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
                    "arm": arm_name,
                    "condition": condition,
                    "prefix_index": prefix_index,
                    "sample_index": sample_index,
                    "source_id": row["source_id"],
                    "prefix_text": row["prefix_text"],
                    "prompt_text": row["prompt_text"],
                    "reference_text": row["reference_text"],
                    "teacher_thought": row["teacher_thought"],
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
    judge_rows = []
    records = []
    for record, suffix_group in zip(prompt_records, grouped_suffixes):
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
                "predicted_suffix": predicted_suffix,
                "predicted_suffix_words": word_count(predicted_suffix),
                "suffix_tokens": len(suffix_ids),
            }
        )
    judgments = judge_pointwise(judge_rows, **judge_cfg)
    for record, judgment in zip(records, judgments):
        record["score"] = judgment["score"]
        record["raw_judge"] = judgment["raw_judge"]
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--thought-max-new-tokens", type=int, default=48)
    parser.add_argument("--judge-temperature", type=float, default=0.7)
    parser.add_argument("--judge-top-p", type=float, default=0.6)
    parser.add_argument("--judge-max-tokens", type=int, default=64)
    parser.add_argument("--judge-max-workers", type=int, default=32)
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    args = parser.parse_args()

    unknown_arms = [arm for arm in args.arms if arm not in ARM_SPECS]
    if unknown_arms:
        raise ValueError(f"Unknown arms: {unknown_arms}; available={sorted(ARM_SPECS)}")
    unknown_conditions = [condition for condition in args.conditions if condition not in DEFAULT_CONDITIONS + ["teacher_thought"]]
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")
    if "normal_model_thought" not in args.conditions:
        raise ValueError("normal_model_thought is required as the causal baseline")

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
        "raw",
        "two_stage_external_boundary",
    )[: args.num_prefixes]
    if len(eval_rows) < args.num_prefixes:
        raise RuntimeError(f"Only {len(eval_rows)} valid eval rows found")

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/thought_use_probe") / now_run_id("thought-use")
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
        "num_eval_prefixes": len(eval_rows),
        "paper_alignment": {
            "reward_object": "prefix -> intervened thought -> predicted suffix -> judge(predicted suffix, true suffix)",
            "claim_boundary": "thought interventions are causal behavioral probes of suffix reward, not circuit-level explanations",
        },
        "arms": {},
    }
    (out_dir / "eval_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_rows),
        encoding="utf-8",
    )

    for arm_name in args.arms:
        print(f"loading {arm_name}", flush=True)
        spec = ARM_SPECS[arm_name]
        arm_cfg = load_config(spec["config"])
        model = load_model(arm_cfg, Path(spec["checkpoint"]), device)
        thought_prompts = [encode(tokenizer, row["prompt_text"]) for row in eval_rows]
        print(f"sampling normal thoughts {arm_name}", flush=True)
        grouped_thought_ids = generate_samples(
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
        normal_thoughts: list[list[dict[str, Any]]] = []
        for prefix_index, thoughts in enumerate(grouped_thought_ids):
            group = []
            for thought_ids in thoughts:
                parsed = parse_thought_only(decode(tokenizer, thought_ids))
                group.append(
                    {
                        "thought_text": parsed["thought_text"],
                        "thought_source": "model",
                        "thought_source_prefix_index": prefix_index,
                        "thought_words": parsed["thought_words"],
                        "autonomous_closed_think": parsed["autonomous_closed_think"],
                    }
                )
            normal_thoughts.append(group)

        records_by_condition: dict[str, list[dict[str, Any]]] = {}
        arm_summary: dict[str, Any] = {"conditions": {}}
        for condition in args.conditions:
            print(f"evaluating {arm_name} {condition}", flush=True)
            condition_thoughts = build_condition_thoughts(
                condition,
                normal_thoughts,
                eval_rows,
                args.samples_per_prefix,
                args.seed + len(records_by_condition),
            )
            records = evaluate_condition(
                arm_name,
                condition,
                model,
                tokenizer,
                eval_rows,
                condition_thoughts,
                args,
                judge_cfg,
                device,
            )
            records_by_condition[condition] = records
            path = out_dir / f"{arm_name}__{condition}.jsonl"
            write_jsonl(path, records)
            arm_summary["conditions"][condition] = summarize_records(records, args.samples_per_prefix) | {
                "records_path": str(path)
            }
            summary["arms"][arm_name] = arm_summary
            (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        arm_summary["causal_deltas"] = summarize_deltas(records_by_condition)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        summary["arms"][arm_name] = arm_summary
        (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    summary["status"] = "complete"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(out_dir / "summary.json"), "status": "complete"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
