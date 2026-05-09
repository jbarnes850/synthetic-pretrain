#!/usr/bin/env python3
"""Shared judging helpers for self-improving pretraining."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


def load_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_prompt(template: str, prefix_text: str, cand_a: str, cand_b: str) -> str:
    return (
        template.replace("{prefix}", prefix_text)
        .replace("{candidate_a}", cand_a)
        .replace("{candidate_b}", cand_b)
    )


_WINNER_JSON_RE = re.compile(r'"winner"\s*:\s*"([AB])"', re.IGNORECASE)
_WINNER_PLAIN_RE = re.compile(r"\b(?:Option|option|winner|Winner)[\s:]*([AB])\b")
_WINNER_NUMERIC_RE = re.compile(r"\b(?:Option|option|winner|Winner)[\s:]*(1|2)\b")
_WINNER_FALLBACK_RE = re.compile(r"\b([AB])\b")


def parse_winner(content: str) -> str:
    match = _WINNER_JSON_RE.search(content)
    if match:
        return match.group(1).upper()
    match = _WINNER_PLAIN_RE.search(content)
    if match:
        return match.group(1).upper()
    match = _WINNER_NUMERIC_RE.search(content)
    if match:
        return "A" if match.group(1) == "1" else "B"
    match = _WINNER_FALLBACK_RE.search(content)
    if match:
        return match.group(1).upper()
    return "B"


def _judge_call(
    prompt: str,
    endpoint: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    last_err: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return parse_winner(content)
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"judge call failed after {retries + 1} attempts: {last_err}")


def judge_pairwise_batch(
    prompts: list[str],
    endpoint: str,
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.6,
    max_tokens: int = 64,
    max_workers: int = 16,
    timeout: float = 30.0,
    retries: int = 2,
) -> list[str]:
    results: list[str | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_judge_call, prompt, endpoint, model, temperature, top_p, max_tokens, timeout, retries): i
            for i, prompt in enumerate(prompts)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:
                print(f"judge batch item {i} failed: {exc}", flush=True)
                results[i] = "B"
    return [winner or "B" for winner in results]


def collate_rollout_raw(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prefix_ids": [row["prefix_ids"] for row in batch],
        "rewrite_suffix_ids": [row.get("rewrite_suffix_ids", []) for row in batch],
        "original_suffix_ids": [row["original_suffix_ids"] for row in batch],
    }
