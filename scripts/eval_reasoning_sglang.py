#!/usr/bin/env python3
"""SGLang/OpenAI-compatible reasoning eval for one served model arm."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import jsonl_iter, now_run_id, set_seed

PROMPT_HEADER = """Answer the following problem. Work out the reasoning, then put the final answer in \\boxed{}.

"""

GSM8K_FEW_SHOT = """Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Solution: Natalia sold 48/2 = 24 clips in May. Altogether she sold 48 + 24 = 72 clips. Therefore the final answer is \\boxed{72}.

Problem: Weng earns $12 an hour for babysitting. Yesterday, she babysat for 50 minutes. How much did she earn?
Solution: Weng worked 50/60 = 5/6 hours. She earned 12 * 5/6 = 10 dollars. Therefore the final answer is \\boxed{10}.

"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(jsonl_iter(path))


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\(?:boxed|mathrm|text|left|right)\s*", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def extract_number(text: str) -> str | None:
    text = text.replace(",", "")
    boxed = extract_boxed(text)
    if boxed:
        text = boxed
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    out = matches[-1]
    return out[:-2] if out.endswith(".0") else out


def extract_letter(text: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed:
        match = re.search(r"\b([ABCD])\b", boxed.upper())
        if match:
            return match.group(1)
    matches = re.findall(r"(?:answer is|final answer is|therefore|thus|^|\s)([ABCD])(?:[\).,\s]|$)", text.upper())
    return matches[-1] if matches else None


def gold_answer(row: dict[str, Any]) -> str:
    if row["benchmark"] == "gsm8k":
        match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", row["answer"])
        if match:
            return match.group(1).replace(",", "")
    if row["benchmark"] == "gpqa_diamond":
        letter = extract_letter(row["answer"])
        if letter:
            return letter
    return str(row["answer"]).strip()


def prediction(row: dict[str, Any], completion: str) -> str | None:
    if row["benchmark"] == "gpqa_diamond":
        return extract_letter(completion)
    if row["benchmark"] == "gsm8k":
        return extract_number(completion)
    boxed = extract_boxed(completion)
    if boxed:
        return boxed
    return extract_number(completion)


def correct(row: dict[str, Any], pred: str | None) -> bool:
    if pred is None:
        return False
    return normalize_text(pred) == normalize_text(gold_answer(row))


def build_prompt(row: dict[str, Any]) -> str:
    if row["benchmark"] == "gsm8k":
        return GSM8K_FEW_SHOT + f"Problem: {row['question']}\nSolution:"
    if row["benchmark"] == "gpqa_diamond":
        return (
            "Answer the multiple-choice science question. Work out the reasoning, then put only the option "
            "letter in \\boxed{}.\n\n"
            f"Question:\n{row['question']}\n\nSolution:"
        )
    return PROMPT_HEADER + f"Problem:\n{row['question']}\n\nSolution:"


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.96
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return [center - half, center + half]


def summarize(rows: list[dict[str, Any]], num_samples: int) -> dict[str, Any]:
    by_problem: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_problem[row["id"]].append(bool(row["correct"]))
    sample_correct = sum(int(row["correct"]) for row in rows)
    sample_total = len(rows)
    pass_any = sum(int(any(vals)) for vals in by_problem.values())
    problem_total = len(by_problem)
    return {
        "num_problems": problem_total,
        "num_samples_per_problem": num_samples,
        "num_samples": sample_total,
        "mean_at_k": sample_correct / max(1, sample_total),
        "mean_at_k_correct": sample_correct,
        "mean_at_k_ci95": wilson(sample_correct, sample_total),
        "pass_at_k_any": pass_any / max(1, problem_total),
        "pass_at_k_any_correct": pass_any,
        "pass_at_k_any_ci95": wilson(pass_any, problem_total),
    }


def completion_request(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout: int,
) -> tuple[str, int]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=request_timeout) as resp:
        body = json.loads(resp.read().decode())
    text = (body.get("choices") or [{}])[0].get("text") or ""
    usage = body.get("usage") or {}
    return text, int(usage.get("completion_tokens") or max(1, len(text) // 4))


def wait_for_server(endpoint: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=10).read()
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
            time.sleep(2)
    raise RuntimeError(f"server not ready at {endpoint}: {last_error}")


def evaluate_arm(args, rows_by_benchmark: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, Any]:
    arm_dir = output_dir / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for benchmark, rows in rows_by_benchmark.items():
        jobs = []
        for row in rows:
            prompt = build_prompt(row)
            for sample_idx in range(args.num_samples):
                jobs.append((row, sample_idx, prompt))

        out_rows: list[dict[str, Any]] = []
        run_start = time.perf_counter()
        cumulative_tokens = 0
        for batch_start in range(0, len(jobs), args.concurrency):
            batch_t0 = time.perf_counter()
            batch = jobs[batch_start : batch_start + args.concurrency]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = [
                    pool.submit(
                        completion_request,
                        args.endpoint,
                        args.model,
                        prompt,
                        args.max_new_tokens,
                        args.temperature,
                        args.top_p,
                        args.request_timeout,
                    )
                    for _, _, prompt in batch
                ]
                for (row, sample_idx, _), future in zip(batch, futures, strict=True):
                    completion, tokens = future.result()
                    pred = prediction(row, completion)
                    cumulative_tokens += tokens
                    out_rows.append(
                        {
                            "arm": args.arm,
                            "benchmark": benchmark,
                            "id": row["id"],
                            "sample_idx": sample_idx,
                            "gold": gold_answer(row),
                            "prediction": pred,
                            "correct": correct(row, pred),
                            "completion": completion,
                            "completion_tokens": tokens,
                        }
                    )
            batch_elapsed = time.perf_counter() - batch_t0
            elapsed = time.perf_counter() - run_start
            print(
                json.dumps(
                    {
                        "event": "generation_batch",
                        "arm": args.arm,
                        "benchmark": benchmark,
                        "batch_index": batch_start // args.concurrency + 1,
                        "num_batches": math.ceil(len(jobs) / args.concurrency),
                        "batch_size": len(batch),
                        "samples_done": len(out_rows),
                        "samples_total": len(jobs),
                        "elapsed_s": elapsed,
                        "batch_elapsed_s": batch_elapsed,
                        "generated_tokens": sum(row["completion_tokens"] for row in out_rows[-len(batch) :]),
                        "tok_per_sec": sum(row["completion_tokens"] for row in out_rows[-len(batch) :])
                        / max(batch_elapsed, 1e-9),
                        "cumulative_generated_tokens": cumulative_tokens,
                        "cumulative_tok_per_sec": cumulative_tokens / max(elapsed, 1e-9),
                    }
                ),
                flush=True,
            )
        out_path = arm_dir / f"{benchmark}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in out_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary[benchmark] = summarize(out_rows, args.num_samples) | {"examples_path": str(out_path)}
        print(json.dumps({"arm": args.arm, "benchmark": benchmark, **summary[benchmark]}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", default="data/eval/reasoning_benchmarks")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math500", "gpqa_diamond", "olympiadbench"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=4337)
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=int, default=180)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    bench_dir = Path(args.bench_dir)
    rows_by_benchmark = {}
    for benchmark in args.benchmarks:
        rows = load_jsonl(bench_dir / f"{benchmark}.jsonl")
        rng.shuffle(rows)
        if args.max_examples is not None:
            rows = rows[: args.max_examples]
        rows_by_benchmark[benchmark] = rows

    wait_for_server(args.endpoint, args.server_timeout)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/reasoning_eval_sglang") / now_run_id("reasoning-eval-sglang")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "output_dir": str(output_dir),
        "config": vars(args),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paper_alignment": {
            "metric": "Reports sample-level mean@k and problem-level pass@k_any; Table 10 reports mean@16/64 style downstream reasoning metrics.",
            "benchmarks": "GSM8K, MATH-500, GPQA-Diamond, and OlympiadBench text-only are HF-hosted approximations of the paper's downstream suite.",
            "serving": "SGLang/OpenAI-compatible completions endpoint serving materialized checkpoints.",
        },
        "arms": {args.arm: evaluate_arm(args, rows_by_benchmark, output_dir)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
