#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import jsonl_iter, latest_snapshot, load_config, write_jsonl


def repetition_penalty(ids: list[int], n: int = 4) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))


def length_score(ids: list[int], target_len: int) -> float:
    if not ids:
        return -1.0
    return -abs(len(ids) - target_len) / max(1, target_len)


def diversity_score(ids: list[int]) -> float:
    if not ids:
        return 0.0
    return len(set(ids)) / len(ids)


def heuristic_score(ids: list[int], target_len: int) -> float:
    return (
        0.55 * length_score(ids, target_len)
        + 0.35 * diversity_score(ids)
        - 0.75 * repetition_penalty(ids)
        + 0.10 * math.log1p(len(ids))
    )


def score_payload(original: float, finephrase: float, rollout: float = 0.0) -> dict[str, float]:
    return {"original": float(original), "finephrase": float(finephrase), "rollout": float(rollout)}


def selected_row(
    row: dict[str, Any],
    chosen: str,
    scores: dict[str, float],
    judge_model: str,
    prompt_version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suffix_key = f"{chosen}_suffix_ids"
    out = {
        "id": row["id"],
        "split": row["split"],
        "prefix_ids": row["prefix_ids"],
        "chosen": chosen,
        "chosen_suffix_ids": row[suffix_key],
        "scores": scores,
        "judge_model": judge_model,
        "prompt_version": prompt_version,
    }
    if extra:
        out.update(extra)
    return out


def fixed_choice(row: dict[str, Any], choice: str, prompt_version: str) -> dict[str, Any]:
    scores = score_payload(1.0 if choice == "original" else 0.0, 1.0 if choice == "finephrase" else 0.0)
    return selected_row(row, choice, scores, f"fixed_{choice}", prompt_version)


def load_tokenizer(cfg: dict[str, Any]):
    from transformers import AutoTokenizer

    tokenizer_path = latest_snapshot(cfg["data"]["tokenizer_repo_cache"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def parse_judge_response(content: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise ValueError(f"Judge response did not contain JSON: {content[:200]!r}")
    parsed = json.loads(match.group(0))
    winner = str(parsed.get("winner", "")).upper()
    if winner not in {"A", "B"}:
        raise ValueError(f"Judge response missing winner A/B: {parsed}")
    return parsed


def build_prompt(prompt_template: str, prefix: str, a: str, b: str) -> str:
    return prompt_template.format(prefix=prefix, candidate_a=a, candidate_b=b)


def openai_pairwise_score(
    prompt: str,
    endpoint: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict[str, Any]:
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    response = requests.post(endpoint.rstrip("/") + "/v1/chat/completions", json=payload, timeout=60)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return parse_judge_response(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run-prompts", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    input_path = Path(cfg["data"]["examples_jsonl"])
    output_path = Path(cfg["data"]["selected_jsonl"])
    if output_path.exists() and not args.force and not args.dry_run_prompts:
        print(f"select_targets: using existing {output_path}")
        return

    mode = cfg["selection"].get("mode", "openai")
    target_len = int(cfg["data"]["suffix_tokens"])
    prompt_template = Path(cfg["selection"].get("prompt_path", "prompts/judge_quality.txt")).read_text(encoding="utf-8")
    prompt_version = str(cfg["selection"].get("prompt_version", "quality_v1"))
    judge_endpoint = os.environ.get("JUDGE_ENDPOINT") or cfg["selection"].get("judge_endpoint")
    judge_model = os.environ.get("JUDGE_MODEL") or cfg["selection"].get("judge_model") or "qwen-judge"
    judge_repeats = int(cfg["selection"].get("judge_repeats", 1))
    judge_temperature = float(cfg["selection"].get("judge_temperature", 0.0))
    judge_top_p = float(cfg["selection"].get("judge_top_p", 1.0))
    judge_max_tokens = int(cfg["selection"].get("judge_max_tokens", 64))
    judge_retries = int(cfg["selection"].get("judge_retries", 0))
    tokenizer = load_tokenizer(cfg) if mode == "openai" or args.dry_run_prompts else None
    rows = []
    counts = Counter()
    wins_over_raw = 0
    total = 0

    for idx, row in enumerate(jsonl_iter(input_path)):
        if args.limit > 0 and idx >= args.limit:
            break
        if mode == "fixed":
            selected = fixed_choice(row, cfg["selection"].get("fixed_choice", "original"), prompt_version)
        elif mode == "openai":
            assert tokenizer is not None
            original_text = tokenizer.decode(row["original_suffix_ids"], skip_special_tokens=True)
            finephrase_text = tokenizer.decode(row["finephrase_suffix_ids"], skip_special_tokens=True)
            prefix_text = tokenizer.decode(row["prefix_ids"], skip_special_tokens=True)
            swap = int(str(row["id"])[-2:], 16) % 2 == 1
            candidate_a_name = "finephrase" if swap else "original"
            candidate_b_name = "original" if swap else "finephrase"
            candidate_a = finephrase_text if swap else original_text
            candidate_b = original_text if swap else finephrase_text
            prompt = build_prompt(prompt_template, prefix_text, candidate_a, candidate_b)
            if args.dry_run_prompts:
                parse_judge_response('{"winner":"A","scores":{"A":1.0,"B":0.0},"reason":"parser dry run"}')
                rows.append({"id": row["id"], "prompt_chars": len(prompt), "candidate_a": candidate_a_name, "candidate_b": candidate_b_name})
                continue
            if not judge_endpoint or not judge_model:
                raise ValueError("selection.mode=openai requires selection.judge_endpoint and selection.judge_model")
            win_counts = Counter()
            reasons = []
            for _ in range(max(1, judge_repeats)):
                strict_prompt = prompt
                last_error: Exception | None = None
                parsed = None
                for attempt in range(judge_retries + 1):
                    try:
                        parsed = openai_pairwise_score(
                            strict_prompt,
                            judge_endpoint,
                            judge_model,
                            0.0 if attempt else judge_temperature,
                            judge_top_p,
                            judge_max_tokens,
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        strict_prompt = (
                            prompt
                            + '\n\nYour previous response did not satisfy the schema. '
                            + 'Output exactly {"winner":"A"} or {"winner":"B"} and nothing else.'
                        )
                if parsed is None:
                    raise RuntimeError(f"Judge failed to return parseable JSON after retries: {last_error}") from last_error
                winner = parsed["winner"].upper()
                win_counts[winner] += 1
                if parsed.get("reason"):
                    reasons.append(str(parsed["reason"]))
            chosen_letter = "A" if win_counts["A"] >= win_counts["B"] else "B"
            chosen = candidate_a_name if chosen_letter == "A" else candidate_b_name
            original_score = (
                win_counts["A"] / max(1, judge_repeats)
                if candidate_a_name == "original"
                else win_counts["B"] / max(1, judge_repeats)
            )
            finephrase_score = (
                win_counts["A"] / max(1, judge_repeats)
                if candidate_a_name == "finephrase"
                else win_counts["B"] / max(1, judge_repeats)
            )
            scores = score_payload(
                original_score,
                finephrase_score,
            )
            selected = selected_row(
                row,
                chosen,
                scores,
                judge_model,
                prompt_version,
                {
                    "position_swap": swap,
                    "judge_repeats": judge_repeats,
                    "judge_temperature": judge_temperature,
                    "judge_top_p": judge_top_p,
                    "judge_max_tokens": judge_max_tokens,
                    "judge_retries": judge_retries,
                    "judge_reason": reasons[0] if reasons else "",
                    "judge_win_counts": dict(win_counts),
                },
            )
        elif mode == "heuristic":
            if not bool(cfg["selection"].get("allow_heuristic", False)):
                raise ValueError("selection.mode=heuristic requires selection.allow_heuristic=true and is not valid for research RF-NLL")
            scores = score_payload(
                heuristic_score(row["original_suffix_ids"], target_len),
                heuristic_score(row["finephrase_suffix_ids"], target_len),
            )
            chosen = max((("original", scores["original"]), ("finephrase", scores["finephrase"])), key=lambda kv: kv[1])[0]
            selected = selected_row(row, chosen, scores, "heuristic_not_research", "heuristic_v1")
        else:
            raise ValueError(f"Unknown selection.mode={mode!r}")
        rows.append(selected)
        counts[selected["chosen"]] += 1
        total += 1
        if selected["chosen"] != "original":
            wins_over_raw += 1

    if not rows:
        raise RuntimeError(f"No rows selected from {input_path}")
    if args.dry_run_prompts:
        print(json.dumps({"dry_run_prompts": len(rows), "examples": rows[:3]}, indent=2))
        return
    write_jsonl(output_path, rows)
    stats = {
        "selected": len(rows),
        "counts": dict(counts),
        "judge_win_rate_vs_raw": wins_over_raw / max(1, total),
        "mode": mode,
        "judge_model": judge_model,
        "prompt_version": prompt_version,
    }
    stats_path = output_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
