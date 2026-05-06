#!/usr/bin/env python3
"""Compare thought selectors on an existing oracle@N thought bank.

Selectors choose one sampled thought per prefix. The selection policy may see
the prefix and candidate thoughts, but never the reference continuation,
generated suffix, or judge score. Scores are used only after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

ARMS = ["think_base", "think_phase3", "think_base_rlmt", "think_phase3_rlmt"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def trim(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def stable_seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text or "")


def weird_unicode_count(text: str) -> int:
    total = 0
    for ch in text or "":
        if ord(ch) <= 127:
            continue
        category = unicodedata.category(ch)
        if category.startswith("M"):
            total += 1
        elif "CJK" in unicodedata.name(ch, "") or "HANGUL" in unicodedata.name(ch, ""):
            total += 1
        elif category.startswith("S"):
            total += 1
    return total


def format_score(record: dict[str, Any]) -> float:
    """Score format quality without using prefix, suffix, reference, or reward."""
    text = record.get("thought_text") or ""
    words = int(record.get("thought_words") or len(text.split()))
    score = 0.0
    if record.get("autonomous_closed_think"):
        score += 2.0
    if record.get("thought_finish_reason") == "length":
        score -= 1.25
    if 8 <= words <= 28:
        score += 1.5
    elif 29 <= words <= 36:
        score += 0.5
    elif words < 5:
        score -= 1.5
    elif words > 45:
        score -= 1.0
    if has_non_ascii(text):
        score -= 1.0
    score -= min(3.0, 0.5 * weird_unicode_count(text))
    if any(tok in text.lower() for tok in ["</think>", "<think>", "http://", "www."]):
        score -= 0.5
    if "\n" in text:
        score -= 0.25
    return score


def choose_format_heuristic(candidates: list[dict[str, Any]]) -> int:
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            format_score(item[1]),
            -abs(int(item[1].get("thought_words") or 0) - 18),
            -int(item[1].get("sample_index") or 0),
        ),
        reverse=True,
    )
    return ranked[0][0]


def llm_prompt(prefix: str, candidates: list[dict[str, Any]]) -> str:
    lines = [
        "You are selecting one candidate thought for a small language model.",
        "The model will condition on the selected thought, then continue the prefix.",
        "Choose the thought most likely to help predict the next natural continuation.",
        "Use only the prefix and candidate thoughts. Do not assume access to a hidden answer.",
        "Prefer thoughts that are concrete, local, coherent, and predictive.",
        "Penalize generic, malformed, overlong, non-ASCII-corrupted, or off-topic thoughts.",
        "Candidate numbers are zero-based. Choose one integer from 0 to 15.",
        'Return only valid JSON like {"choice": 3, "reason": "short reason"}.',
        "",
        "Prefix:",
        trim(prefix, 1600),
        "",
        "Candidate thoughts:",
    ]
    for idx, cand in enumerate(candidates):
        text = trim(cand.get("thought_text") or "", 360)
        words = cand.get("thought_words")
        lines.append(f"{idx}. ({words} words) {text}")
    return "\n".join(lines)


def parse_choice(text: str, n: int) -> tuple[int | None, str | None]:
    try:
        body = json.loads(text)
        choice = int(body["choice"])
        reason = str(body.get("reason", ""))
        if 0 <= choice < n:
            return choice, reason
    except Exception:
        pass
    match = re.search(r'"?choice"?\s*[:=]\s*(\d+)', text)
    if match:
        choice = int(match.group(1))
        if 0 <= choice < n:
            return choice, None
    match = re.search(r"\b(?:candidate|choice)\s+(\d+)\b", text, re.I)
    if match:
        choice = int(match.group(1))
        if 0 <= choice < n:
            return choice, None
    return None, None


def call_llm_selector(
    endpoint: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    resp = requests.post(endpoint.rstrip("/") + "/v1/chat/completions", json=payload, timeout=timeout)
    elapsed = time.monotonic() - started
    resp.raise_for_status()
    body = resp.json()
    raw = body["choices"][0]["message"]["content"]
    return {"raw": raw, "latency_s": elapsed, "usage": body.get("usage")}


def load_bank(input_dir: Path, arm: str) -> dict[int, list[dict[str, Any]]]:
    records = read_jsonl(input_dir / f"{arm}__normal_model_thought.jsonl")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["prefix_index"])].append(record)
    return {idx: sorted(rows, key=lambda r: int(r["sample_index"])) for idx, rows in grouped.items()}


def selected_metrics(groups: dict[int, list[dict[str, Any]]], selected: dict[int, int]) -> dict[str, Any]:
    scores = []
    oracle = []
    selected_when_oracle = []
    for prefix_index, rows in groups.items():
        choice = selected[prefix_index]
        score = float(rows[choice]["score"])
        oracle_score = float(any(int(row["score"]) == 1 for row in rows))
        scores.append(score)
        oracle.append(oracle_score)
        if oracle_score:
            selected_when_oracle.append(score)
    return {
        "selected_reward": safe_mean(scores),
        "oracle_at_16": safe_mean(oracle),
        "selected_given_oracle_available": safe_mean(selected_when_oracle),
        "regret_to_oracle": safe_mean(oracle) - safe_mean(scores),
        "selected_successes": int(sum(scores)),
        "oracle_success_prefixes": int(sum(oracle)),
        "prefixes": len(groups),
    }


def summarize_arm(
    arm: str,
    groups: dict[int, list[dict[str, Any]]],
    llm_choices: dict[int, dict[str, Any]] | None,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    random_selected = {}
    for prefix_index, rows in groups.items():
        rng = random.Random(stable_seed(seed, arm, prefix_index))
        random_selected[prefix_index] = rng.randrange(len(rows))
    format_selected = {prefix_index: choose_format_heuristic(rows) for prefix_index, rows in groups.items()}
    oracle_selected = {}
    for prefix_index, rows in groups.items():
        winners = [i for i, row in enumerate(rows) if int(row["score"]) == 1]
        oracle_selected[prefix_index] = winners[0] if winners else 0

    flat_scores = [float(row["score"]) for rows in groups.values() for row in rows]
    summary = {
        "random_expected_reward": safe_mean(flat_scores),
        "random_seeded": selected_metrics(groups, random_selected),
        "format_heuristic": selected_metrics(groups, format_selected),
        "oracle_at_16": selected_metrics(groups, oracle_selected),
    }
    if llm_choices is not None:
        llm_selected = {
            prefix_index: int(choice["choice"])
            for prefix_index, choice in llm_choices.items()
            if choice.get("choice") is not None
        }
        if len(llm_selected) != len(groups):
            missing = sorted(set(groups) - set(llm_selected))
            for prefix_index in missing:
                llm_selected[prefix_index] = format_selected[prefix_index]
        summary["llm_prefix_selector"] = selected_metrics(groups, llm_selected)
        summary["llm_parse_failures"] = sum(1 for choice in llm_choices.values() if choice.get("parse_failed"))
        summary["llm_mean_latency_s"] = safe_mean(
            [float(choice["latency_s"]) for choice in llm_choices.values() if isinstance(choice.get("latency_s"), float)]
        )

    decisions = []
    for prefix_index, rows in sorted(groups.items()):
        oracle_choices = [i for i, row in enumerate(rows) if int(row["score"]) == 1]
        record = {
            "arm": arm,
            "prefix_index": prefix_index,
            "source_id": rows[0].get("source_id"),
            "random_choice": random_selected[prefix_index],
            "format_choice": format_selected[prefix_index],
            "oracle_first_choice": oracle_selected[prefix_index],
            "oracle_choices": oracle_choices,
            "random_score": rows[random_selected[prefix_index]]["score"],
            "format_score": rows[format_selected[prefix_index]]["score"],
            "oracle_score": int(bool(oracle_choices)),
            "format_heuristic_score": format_score(rows[format_selected[prefix_index]]),
            "format_choice_thought": rows[format_selected[prefix_index]].get("thought_text"),
        }
        if llm_choices is not None:
            choice = llm_choices[prefix_index]
            llm_idx = int(choice["choice"]) if choice.get("choice") is not None else format_selected[prefix_index]
            record.update(
                {
                    "llm_choice": llm_idx,
                    "llm_score": rows[llm_idx]["score"],
                    "llm_reason": choice.get("reason"),
                    "llm_raw": choice.get("raw"),
                    "llm_parse_failed": choice.get("parse_failed", False),
                    "llm_choice_thought": rows[llm_idx].get("thought_text"),
                }
            )
        decisions.append(record)
    return summary, decisions


def run_llm_selectors(
    endpoint: str,
    model: str,
    arm_groups: dict[str, dict[int, list[dict[str, Any]]]],
    max_workers: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, dict[int, dict[str, Any]]]:
    tasks = []
    for arm, groups in arm_groups.items():
        for prefix_index, rows in groups.items():
            tasks.append((arm, prefix_index, llm_prompt(rows[0]["prefix_text"], rows)))

    out: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in arm_groups}

    def one(task: tuple[str, int, str]) -> tuple[str, int, dict[str, Any]]:
        arm, prefix_index, prompt = task
        try:
            result = call_llm_selector(endpoint, model, prompt, temperature, max_tokens, timeout)
            choice, reason = parse_choice(result["raw"], 16)
            return arm, prefix_index, {
                **result,
                "choice": choice,
                "reason": reason,
                "parse_failed": choice is None,
            }
        except Exception as exc:
            return arm, prefix_index, {
                "raw": "",
                "choice": None,
                "reason": None,
                "parse_failed": True,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(one, task): task for task in tasks}
        for fut in as_completed(futures):
            arm, prefix_index, result = fut.result()
            out[arm][prefix_index] = result
    return out


def write_markdown(path: Path, summary: dict[str, Any], decisions: list[dict[str, Any]]) -> None:
    lines = ["# Thought Selector Comparison\n\n"]
    lines.append("| arm | random expected | random seeded | format heuristic | prefix-only LLM | oracle@16 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for arm, arm_summary in summary["arms"].items():
        llm = arm_summary.get("llm_prefix_selector", {})
        lines.append(
            f"| {arm} | {arm_summary['random_expected_reward']:.4f} | "
            f"{arm_summary['random_seeded']['selected_reward']:.4f} | "
            f"{arm_summary['format_heuristic']['selected_reward']:.4f} | "
            f"{llm.get('selected_reward', float('nan')):.4f} | "
            f"{arm_summary['oracle_at_16']['selected_reward']:.4f} |\n"
        )
    lines.append("\n## Notes\n\n")
    lines.append("- Selectors never receive the reference continuation, generated suffix, or judge score.\n")
    lines.append("- `random expected` is the exact mean reward over all 16 sampled thoughts per prefix.\n")
    lines.append("- `oracle@16` is an upper bound that selects a successful candidate if one exists.\n")

    interesting = []
    for record in decisions:
        if record.get("oracle_score") and not record.get("llm_score") and not record.get("format_score"):
            interesting.append(("oracle_available_selectors_miss", record))
        elif record.get("llm_score") and not record.get("format_score"):
            interesting.append(("llm_finds_format_misses", record))
        elif record.get("format_score") and not record.get("llm_score"):
            interesting.append(("format_finds_llm_misses", record))
    lines.append("\n## Example Decisions\n\n")
    for label, record in interesting[:12]:
        lines.append(f"### {record['arm']} prefix {record['prefix_index']} - {label}\n\n")
        lines.append(f"oracle choices: {record['oracle_choices']}\n\n")
        lines.append(f"format choice {record['format_choice']} score {record['format_score']}: ")
        lines.append(f"{trim(record.get('format_choice_thought') or '', 260)}\n\n")
        if "llm_choice" in record:
            lines.append(f"llm choice {record['llm_choice']} score {record['llm_score']}: ")
            lines.append(f"{trim(record.get('llm_choice_thought') or '', 260)}\n\n")
            lines.append(f"llm reason: {trim(record.get('llm_reason') or record.get('llm_raw') or '', 240)}\n\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--seed", type=int, default=4337)
    parser.add_argument("--selector-endpoint", default=None)
    parser.add_argument("--selector-model", default="qwen-judge")
    parser.add_argument("--selector-workers", type=int, default=16)
    parser.add_argument("--selector-temperature", type=float, default=0.0)
    parser.add_argument("--selector-max-tokens", type=int, default=96)
    parser.add_argument("--selector-timeout", type=float, default=120.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arm_groups = {arm: load_bank(input_dir, arm) for arm in args.arms}
    llm_choices = None
    if args.selector_endpoint:
        llm_choices = run_llm_selectors(
            args.selector_endpoint,
            args.selector_model,
            arm_groups,
            args.selector_workers,
            args.selector_temperature,
            args.selector_max_tokens,
            args.selector_timeout,
        )

    all_decisions: list[dict[str, Any]] = []
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "selection_boundary": "selectors see prefix_text and candidate thought_text only; scores/references/generated suffixes are used after selection",
        "arms": {},
    }
    for arm, groups in arm_groups.items():
        arm_summary, arm_decisions = summarize_arm(
            arm,
            groups,
            llm_choices[arm] if llm_choices is not None else None,
            args.seed,
        )
        summary["arms"][arm] = arm_summary
        all_decisions.extend(arm_decisions)

    summary["macro"] = {}
    methods = ["random_seeded", "format_heuristic", "llm_prefix_selector", "oracle_at_16"]
    for method in methods:
        vals = [
            arm_summary[method]["selected_reward"]
            for arm_summary in summary["arms"].values()
            if method in arm_summary
        ]
        if vals:
            summary["macro"][method] = {
                "mean_selected_reward": safe_mean(vals),
                "std_selected_reward": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
    summary["macro"]["random_expected_reward"] = safe_mean(
        [arm_summary["random_expected_reward"] for arm_summary in summary["arms"].values()]
    )

    (output_dir / "selector_comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "selector_decisions.jsonl", all_decisions)
    write_markdown(output_dir / "selector_audit.md", summary, all_decisions)
    print(json.dumps({"summary_path": str(output_dir / "selector_comparison.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
