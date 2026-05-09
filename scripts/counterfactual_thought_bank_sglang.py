#!/usr/bin/env python3
"""Counterfactual thought-bank probe using SGLang-served policy models.

This is a read-only extension of the thought-use probe:

prefix -> intervened thought -> predicted suffix -> judge(predicted suffix, reference suffix)

The policy arm is expected to be served through an OpenAI-compatible SGLang
endpoint. The judge endpoint can be a separate SGLang server.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

try:
    from common import load_config, load_split_rows, now_run_id, safe_mean, set_seed
except ImportError as exc:  # pragma: no cover - runtime environment guard
    raise SystemExit(f"Could not import repo common helpers: {exc}") from exc

from eval_reward_gate import JUDGE_PROMPT, build_eval_rows, parse_thought_only, word_count
from eval_thinking import ARM_SPECS, judge_pointwise, load_tokenizer

DEFAULT_CONDITIONS = [
    "normal_model_thought",
    "blank_thought",
    "generic_thought",
    "same_arm_swapped_thought",
]

GENERIC_THOUGHTS = [
    "Consider the local context and continue with a coherent relevant passage.",
    "Use the preceding text to infer what information should naturally come next.",
    "Maintain topic continuity and predict the next part of the source text.",
    "Focus on semantic consistency with the prefix before writing the continuation.",
]


def completion_text(
    endpoint: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    timeout: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if stop:
        payload["stop"] = stop
    started = time.monotonic()
    resp = requests.post(endpoint.rstrip("/") + "/v1/completions", json=payload, timeout=timeout)
    elapsed = time.monotonic() - started
    resp.raise_for_status()
    body = resp.json()
    choice = body["choices"][0]
    return {
        "text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "matched_stop": choice.get("matched_stop"),
        "usage": body.get("usage"),
        "latency_s": elapsed,
    }


def complete_many(
    endpoint: str,
    model_name: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    max_workers: int,
    timeout: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any] | None] = [None] * len(prompts)

    def one(prompt: str) -> dict[str, Any]:
        try:
            return completion_text(endpoint, model_name, prompt, max_tokens, temperature, top_p, stop, timeout)
        except Exception as exc:
            return {
                "text": "",
                "finish_reason": "error",
                "matched_stop": None,
                "usage": None,
                "latency_s": None,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(one, prompt): idx for idx, prompt in enumerate(prompts)}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return [row or {"text": "", "finish_reason": "missing", "error": "missing"} for row in out]


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
            out.append(
                [
                    {
                        "thought_text": text,
                        "thought_source": "teacher",
                        "thought_source_prefix_index": prefix_index,
                        "thought_words": word_count(text),
                    }
                    for _ in range(samples_per_prefix)
                ]
            )
        return out
    raise ValueError(f"unsupported condition: {condition}")


def summarize_records(records: list[dict[str, Any]], samples_per_prefix: int) -> dict[str, Any]:
    valid = [row for row in records if row.get("score") in {0, 1}]
    scores = [float(row["score"]) for row in valid]
    by_prefix: dict[int, list[float]] = defaultdict(list)
    for row in valid:
        by_prefix[int(row["prefix_index"])].append(float(row["score"]))
    prefix_means = [safe_mean(vals) for vals in by_prefix.values()]
    generation_errors = [row for row in records if row.get("generation_error")]
    return {
        "records": len(records),
        "valid": len(valid),
        "invalid": len(records) - len(valid),
        "invalid_rate": (len(records) - len(valid)) / max(1, len(records)),
        "generation_errors": len(generation_errors),
        "reward_mean": safe_mean(scores),
        "reward_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "prefix_mean_reward": safe_mean(prefix_means),
        "any_success_prefix_rate": safe_mean([float(max(vals) > 0.0) for vals in by_prefix.values()]),
        "all_success_prefix_rate": safe_mean([float(min(vals) >= 1.0) for vals in by_prefix.values()]),
        "samples_per_prefix": samples_per_prefix,
        "avg_predicted_suffix_words": safe_mean([row["predicted_suffix_words"] for row in records]),
        "avg_thought_words": safe_mean([row["thought_words"] for row in records]),
        "avg_generation_latency_s": safe_mean(
            [row["generation_latency_s"] for row in records if isinstance(row.get("generation_latency_s"), float)]
        ),
    }


def summarize_deltas(records_by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    means_by_condition: dict[str, dict[int, float]] = {}
    for condition, records in records_by_condition.items():
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in records:
            if row.get("score") in {0, 1}:
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_summary(out_dir: Path, args: argparse.Namespace, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = out_dir / "summary.json"
    partial_path = out_dir / "summary.partial.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if partial_path.exists():
        return json.loads(partial_path.read_text(encoding="utf-8"))
    return {
        "output_dir": str(out_dir),
        "config": vars(args),
        "num_eval_prefixes": len(eval_rows),
        "data_source": args.data_source_label,
        "serving": {
            "policy_backend": "sglang_openai_completions",
            "judge_backend": "sglang_openai_chat_completions",
        },
        "paper_alignment": {
            "reward_object": "prefix -> intervened thought -> predicted suffix -> judge(predicted suffix, true suffix)",
            "claim_boundary": "thought interventions are causal behavioral probes of suffix reward, not circuit-level explanations",
        },
        "arms": {},
    }


def evaluate_condition(
    arm_name: str,
    condition: str,
    eval_rows: list[dict[str, Any]],
    condition_thoughts: list[list[dict[str, Any]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    suffix_prompts: list[str] = []
    prompt_records: list[dict[str, Any]] = []
    for prefix_index, (row, thoughts) in enumerate(zip(eval_rows, condition_thoughts)):
        for sample_index, thought in enumerate(thoughts):
            suffix_prompt = row["prompt_text"] + thought["thought_text"].strip() + "</think>"
            suffix_prompts.append(suffix_prompt)
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
    suffix_outputs = complete_many(
        args.model_endpoint,
        args.model_name,
        suffix_prompts,
        args.suffix_tokens,
        args.temperature,
        args.top_p,
        None,
        args.generation_workers,
        args.request_timeout,
    )
    judge_rows = []
    records = []
    for record, output in zip(prompt_records, suffix_outputs):
        predicted_suffix = output["text"].strip()
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
                "suffix_tokens": (output.get("usage") or {}).get("completion_tokens"),
                "generation_finish_reason": output.get("finish_reason"),
                "generation_matched_stop": output.get("matched_stop"),
                "generation_latency_s": output.get("latency_s"),
                "generation_error": output.get("error"),
            }
        )
    judgments = judge_pointwise(
        judge_rows,
        endpoint=args.judge_endpoint,
        model_name=args.judge_model,
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        max_tokens=args.judge_max_tokens,
        max_workers=args.judge_workers,
    )
    for record, judgment in zip(records, judgments):
        record["score"] = judgment["score"]
        record["raw_judge"] = judgment["raw_judge"]
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(ARM_SPECS))
    parser.add_argument("--model-endpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--judge-endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--judge-model", default="qwen36-35b-a3b")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=4337)
    parser.add_argument("--num-prefixes", type=int, default=32)
    parser.add_argument("--samples-per-prefix", type=int, default=4)
    parser.add_argument("--raw-prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--thought-max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--generation-workers", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument("--judge-temperature", type=float, default=0.7)
    parser.add_argument("--judge-top-p", type=float, default=0.6)
    parser.add_argument("--judge-max-tokens", type=int, default=64)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument(
        "--data-source-label",
        default="local interleaved-thinking corpus",
    )
    parser.add_argument("--mark-complete", action="store_true")
    args = parser.parse_args()

    unknown_conditions = [condition for condition in args.conditions if condition not in DEFAULT_CONDITIONS + ["teacher_thought"]]
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")
    if "normal_model_thought" not in args.conditions:
        raise ValueError("normal_model_thought is required as the causal baseline")

    set_seed(args.seed)
    cfg = load_config(ARM_SPECS[args.arm]["config"])
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

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/counterfactual_thought_bank") / now_run_id("thought-bank")
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_rows_path = out_dir / "eval_rows.jsonl"
    if not eval_rows_path.exists():
        write_jsonl(eval_rows_path, eval_rows)

    summary = load_existing_summary(out_dir, args, eval_rows)
    thought_prompts = [row["prompt_text"] for row in eval_rows]
    flat_thought_prompts = [prompt for prompt in thought_prompts for _ in range(args.samples_per_prefix)]
    thought_outputs = complete_many(
        args.model_endpoint,
        args.model_name,
        flat_thought_prompts,
        args.thought_max_new_tokens,
        args.temperature,
        args.top_p,
        ["</think>"],
        args.generation_workers,
        args.request_timeout,
    )
    normal_thoughts: list[list[dict[str, Any]]] = []
    cursor = 0
    for prefix_index in range(len(eval_rows)):
        group = []
        for _ in range(args.samples_per_prefix):
            output = thought_outputs[cursor]
            cursor += 1
            parsed = parse_thought_only(output["text"])
            group.append(
                {
                    "thought_text": parsed["thought_text"],
                    "thought_source": "model",
                    "thought_source_prefix_index": prefix_index,
                    "thought_words": parsed["thought_words"],
                    "autonomous_closed_think": output.get("matched_stop") == "</think>",
                    "thought_finish_reason": output.get("finish_reason"),
                    "thought_latency_s": output.get("latency_s"),
                    "thought_generation_error": output.get("error"),
                }
            )
        normal_thoughts.append(group)

    records_by_condition: dict[str, list[dict[str, Any]]] = {}
    arm_summary: dict[str, Any] = {
        "model_endpoint": args.model_endpoint,
        "model_name": args.model_name,
        "conditions": {},
    }
    for condition in args.conditions:
        condition_thoughts = build_condition_thoughts(
            condition,
            normal_thoughts,
            eval_rows,
            args.samples_per_prefix,
            args.seed + len(records_by_condition),
        )
        records = evaluate_condition(args.arm, condition, eval_rows, condition_thoughts, args)
        records_by_condition[condition] = records
        records_path = out_dir / f"{args.arm}__{condition}.jsonl"
        write_jsonl(records_path, records)
        arm_summary["conditions"][condition] = summarize_records(records, args.samples_per_prefix) | {
            "records_path": str(records_path)
        }
        summary["arms"][args.arm] = arm_summary
        (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    arm_summary["causal_deltas"] = summarize_deltas(records_by_condition)
    means = {
        condition: arm_summary["conditions"][condition]["reward_mean"]
        for condition in arm_summary["conditions"]
    }
    controls = {k: v for k, v in means.items() if k != "normal_model_thought"}
    arm_summary["model_thought_comparison"] = {
        "normal_reward_mean": means.get("normal_model_thought"),
        "best_control": max(controls, key=controls.get) if controls else None,
        "best_control_reward_mean": max(controls.values()) if controls else None,
        "normal_beats_best_control": (
            means.get("normal_model_thought", -1.0) > max(controls.values()) if controls else None
        ),
    }
    summary["arms"][args.arm] = arm_summary
    summary["status"] = "complete" if args.mark_complete else "partial"
    (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.mark_complete:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(out_dir / ("summary.json" if args.mark_complete else "summary.partial.json")), "arm": args.arm}, indent=2), flush=True)


if __name__ == "__main__":
    main()
