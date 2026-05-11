#!/usr/bin/env python3
"""Corpus-quality gate for interleaved thinking data.

This gate checks whether accepted interleaved thoughts are suitable SFT targets.
It is intentionally separate from eval_data_integrity_gate.py, which tests
whether thoughts improve suffix prediction. Here the question is whether the
thoughts have the right RAM-style form: author/agent-side latent reasoning, not
assistant-role prompt leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from common import jsonl_iter, now_run_id, safe_mean, write_jsonl

CRITIC_PROMPT = """# Interleaved Thought Corpus Critic

You are auditing one accepted interleaved-thinking training row before any
model training. This is a corpus hygiene review, not a suffix-reward judgment.

The intended data recipe is RAM-style thinking mid-training:
- preserve the original text exactly;
- insert local thoughts that reconstruct implicit reasoning needed to predict
  the surrounding text;
- thoughts should be from the perspective of the author/agent(s) in the source
  text, not from an assistant analyzing a user request;
- Q&A/forum rows may legitimately mention a user, reader, caller, patient,
  customer, API user, or software user when that role is part of the source
  domain. Do not penalize legitimate domain usage.

Classify the row into exactly one category:
- clean: no meaningful role/prompt/refusal contamination.
- legitimate_domain_user: "user" language is appropriate because the source
  domain has a real user role, e.g. software docs, forum participants,
  customers, patients, product users.
- source_qa_legitimate: the source itself is a Q&A/forum/help exchange and the
  thought correctly models the asker/responder roles without becoming an AI
  assistant.
- assistant_role_contamination: the thought frames itself as an AI assistant or
  says the user asked it to perform the augmentation/answering task.
- prompt_leak: the thought leaks the generation prompt, mentions augmenting the
  text, the annotator task, or "as an AI" style metadata unrelated to the
  source.
- refusal_or_policy_artifact: the thought contains refusal/policy/safety boilerplate
  from the teacher instead of source-grounded reasoning.
- uncertain: evidence is mixed or insufficient.

Return only valid JSON:
{{
  "category": "<one category>",
  "severity": "pass|warn|fail",
  "reason": "<brief reason>",
  "evidence": "<short quoted evidence or empty string>"
}}

Raw source text:
{raw_text}

Inserted thoughts:
{thoughts}
"""


PATTERNS = {
    "assistant_role": re.compile(
        r"\b(the user is asking me|user asks me|as an ai|ai assistant|chatgpt|"
        r"i need to augment|i am asked to|my task is to augment)\b",
        re.IGNORECASE,
    ),
    "user_language": re.compile(r"\b(the user|user asks|user is|user wants|user needs|user must)\b", re.IGNORECASE),
    "refusal_policy": re.compile(
        r"\b(i cannot|i can't|cannot generate|can't generate|unable to comply|policy|safety guidelines)\b",
        re.IGNORECASE,
    ),
    "prompt_leak": re.compile(
        r"\b(augment(?:ed)? text|insert thoughts|generation prompt|scraped from a web page|"
        r"the annotator|teacher model|i need to add missing contexts)\b",
        re.IGNORECASE,
    ),
    "markdown_fence": re.compile(r"```"),
}


DISALLOWED_CATEGORIES = {
    "assistant_role_contamination",
    "prompt_leak",
    "refusal_or_policy_artifact",
}


def extract_thoughts(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"<think>(.*?)</think>", text or "", flags=re.DOTALL | re.IGNORECASE)]


def shorten(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def row_flags(row: dict[str, Any]) -> dict[str, Any]:
    text = row.get("augmented_text", "")
    thoughts = extract_thoughts(text)
    all_thoughts = "\n".join(thoughts)
    interior = "\n".join(thoughts[1:])
    malformed = len(thoughts) != int(row.get("think_count") or len(thoughts))
    flags = {
        "assistant_role": bool(PATTERNS["assistant_role"].search(all_thoughts)),
        "user_language_any": bool(PATTERNS["user_language"].search(all_thoughts)),
        "user_language_interior": bool(PATTERNS["user_language"].search(interior)),
        "refusal_policy": bool(PATTERNS["refusal_policy"].search(all_thoughts)),
        "prompt_leak": bool(PATTERNS["prompt_leak"].search(all_thoughts)),
        "markdown_fence": bool(PATTERNS["markdown_fence"].search(all_thoughts)),
        "malformed_think_count": malformed,
        "original_text_not_preserved": not bool(row.get("original_text_preserved")),
        "low_raw_word_coverage": float(row.get("raw_word_coverage") or 0.0) < 0.995,
    }
    flags["needs_critic"] = any(
        flags[name]
        for name in (
            "assistant_role",
            "user_language_interior",
            "refusal_policy",
            "prompt_leak",
            "markdown_fence",
            "malformed_think_count",
            "original_text_not_preserved",
            "low_raw_word_coverage",
        )
    )
    return flags


def stable_sample(rows: list[dict[str, Any]], count: int, seed: int, salt: str) -> list[dict[str, Any]]:
    if count <= 0:
        return []

    def key(row: dict[str, Any]) -> str:
        payload = f"{seed}:{salt}:{row.get('id', '')}:{row.get('source_sha', '')}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return sorted(rows, key=key)[: min(count, len(rows))]


def parse_critic_response(content: str) -> dict[str, str]:
    try:
        obj = json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content or "", flags=re.DOTALL)
        obj = json.loads(match.group(0)) if match else {}
    category = str(obj.get("category") or "uncertain").strip()
    severity = str(obj.get("severity") or "warn").strip()
    if category not in {
        "clean",
        "legitimate_domain_user",
        "source_qa_legitimate",
        "assistant_role_contamination",
        "prompt_leak",
        "refusal_or_policy_artifact",
        "uncertain",
    }:
        category = "uncertain"
    if severity not in {"pass", "warn", "fail"}:
        severity = "warn"
    return {
        "category": category,
        "severity": severity,
        "reason": str(obj.get("reason") or "")[:500],
        "evidence": str(obj.get("evidence") or "")[:500],
    }


def call_critic(prompt: str, endpoint: str, model: str, temperature: float, top_p: float, max_tokens: int, timeout: float) -> dict[str, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(endpoint.rstrip("/") + "/v1/chat/completions", json=payload, timeout=(20.0, timeout))
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"].get("content") or ""
    parsed = parse_critic_response(content)
    parsed["raw_critic"] = content[:2000]
    return parsed


def critic_prompt(row: dict[str, Any], max_chars: int) -> str:
    thoughts = "\n\n".join(f"[{idx}] {shorten(thought, max_chars)}" for idx, thought in enumerate(extract_thoughts(row.get("augmented_text", "")), start=1))
    return CRITIC_PROMPT.format(raw_text=shorten(row.get("raw_text", ""), max_chars), thoughts=thoughts)


def run_critic(rows: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.critic_max_workers) as ex:
        futures = {
            ex.submit(
                call_critic,
                critic_prompt(row, args.max_chars),
                args.critic_endpoint,
                args.critic_model,
                args.critic_temperature,
                args.critic_top_p,
                args.critic_max_tokens,
                args.critic_timeout,
            ): row
            for row in rows
        }
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                judgment = fut.result()
            except Exception as exc:
                judgment = {
                    "category": "uncertain",
                    "severity": "warn",
                    "reason": f"critic_error: {exc}",
                    "evidence": "",
                    "raw_critic": f"ERROR: {exc}",
                }
            out.append(
                {
                    "id": row.get("id"),
                    "split": row.get("split"),
                    "source_sha": row.get("source_sha", ""),
                    "sample_bucket": row.get("_sample_bucket", ""),
                    "flags": row.get("_flags", {}),
                    "thought_preview": shorten(" ".join(extract_thoughts(row.get("augmented_text", ""))[:2]), 600),
                    **judgment,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="data/processed/interleaved_thinking_full.jsonl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--critic-endpoint", required=True)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--flagged-critic-rows", type=int, default=256)
    parser.add_argument("--random-critic-rows", type=int, default=64)
    parser.add_argument("--critic-max-workers", type=int, default=16)
    parser.add_argument("--critic-temperature", type=float, default=0.0)
    parser.add_argument("--critic-top-p", type=float, default=1.0)
    parser.add_argument("--critic-max-tokens", type=int, default=220)
    parser.add_argument("--critic-timeout", type=float, default=120.0)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--max-estimated-disallowed-rate", type=float, default=0.02)
    parser.add_argument("--max-uncertain-rate", type=float, default=0.10)
    parser.add_argument("--max-deterministic-prompt-leak-rate", type=float, default=0.005)
    parser.add_argument("--max-deterministic-refusal-rate", type=float, default=0.01)
    args = parser.parse_args()

    rows = list(jsonl_iter(args.input_jsonl))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    for row in rows:
        row["_flags"] = row_flags(row)

    flagged = [row for row in rows if row["_flags"]["needs_critic"]]
    cleanish = [row for row in rows if not row["_flags"]["needs_critic"]]
    for row in flagged:
        row["_sample_bucket"] = "flagged"
    for row in cleanish:
        row["_sample_bucket"] = "random"
    critic_rows = stable_sample(flagged, args.flagged_critic_rows, args.seed, "flagged") + stable_sample(
        cleanish, args.random_critic_rows, args.seed, "random"
    )

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/corpus_quality_gate") / now_run_id("corpus-quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    critic_records = run_critic(critic_rows, args)
    write_jsonl(out_dir / "critic_records.jsonl", critic_records)

    flag_counts = Counter()
    for row in rows:
        for name, value in row["_flags"].items():
            if value is True:
                flag_counts[name] += 1

    critic_by_bucket: dict[str, list[dict[str, Any]]] = {"flagged": [], "random": []}
    for record in critic_records:
        critic_by_bucket.setdefault(record.get("sample_bucket") or "", []).append(record)

    def bucket_rate(bucket: str, predicate) -> float:
        records = critic_by_bucket.get(bucket, [])
        if not records:
            return 0.0
        return safe_mean([float(predicate(record)) for record in records])

    flagged_rate = len(flagged) / max(1, len(rows))
    flagged_disallowed = bucket_rate("flagged", lambda record: record.get("category") in DISALLOWED_CATEGORIES or record.get("severity") == "fail")
    random_disallowed = bucket_rate("random", lambda record: record.get("category") in DISALLOWED_CATEGORIES or record.get("severity") == "fail")
    estimated_disallowed_rate = flagged_rate * flagged_disallowed + (1.0 - flagged_rate) * random_disallowed
    uncertain_rate = safe_mean([float(record.get("category") == "uncertain") for record in critic_records])

    deterministic_rates = {name: count / max(1, len(rows)) for name, count in sorted(flag_counts.items())}
    structural_ok = (
        deterministic_rates.get("malformed_think_count", 0.0) == 0.0
        and deterministic_rates.get("original_text_not_preserved", 0.0) == 0.0
        and deterministic_rates.get("low_raw_word_coverage", 0.0) == 0.0
    )
    go_no_go = {
        "passed": (
            structural_ok
            and estimated_disallowed_rate <= args.max_estimated_disallowed_rate
            and uncertain_rate <= args.max_uncertain_rate
            and deterministic_rates.get("prompt_leak", 0.0) <= args.max_deterministic_prompt_leak_rate
            and deterministic_rates.get("refusal_policy", 0.0) <= args.max_deterministic_refusal_rate
        ),
        "structural_ok": structural_ok,
        "estimated_disallowed_rate": estimated_disallowed_rate,
        "max_estimated_disallowed_rate": args.max_estimated_disallowed_rate,
        "uncertain_rate": uncertain_rate,
        "max_uncertain_rate": args.max_uncertain_rate,
        "prompt_leak_rate": deterministic_rates.get("prompt_leak", 0.0),
        "refusal_policy_rate": deterministic_rates.get("refusal_policy", 0.0),
    }

    summary = {
        "status": "pass" if go_no_go["passed"] else "fail",
        "input_jsonl": args.input_jsonl,
        "output_dir": str(out_dir),
        "config": vars(args),
        "rows": len(rows),
        "flagged_rows": len(flagged),
        "flagged_rate": flagged_rate,
        "deterministic_flag_counts": dict(flag_counts),
        "deterministic_flag_rates": deterministic_rates,
        "critic_rows": len(critic_records),
        "critic_category_counts": dict(Counter(record.get("category") for record in critic_records)),
        "critic_severity_counts": dict(Counter(record.get("severity") for record in critic_records)),
        "critic_bucket_counts": {bucket: len(records) for bucket, records in critic_by_bucket.items()},
        "critic_bucket_disallowed_rates": {
            "flagged": flagged_disallowed,
            "random": random_disallowed,
        },
        "go_no_go": go_no_go,
        "paper_alignment": {
            "gate_question": "Are interleaved thoughts clean RAM-style SFT targets, independent of suffix reward?",
            "separate_from_suffix_reward": True,
            "critic_scope": "role contamination, prompt leakage, refusal artifacts, preservation/form hygiene",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(out_dir / "summary.json"), "status": summary["status"]}, indent=2), flush=True)
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
