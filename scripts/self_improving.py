#!/usr/bin/env python3
"""
Self-improving pretraining helpers: on-policy rollouts + online pairwise judge.

Implements the "RF-NLL (rollout vs rewrite)" candidate path from
Tan et al. 2026 (arxiv 2601.21343), Section 1.1.2 and Table 2. At each
training step the current policy generates 1 rollout per prefix, a
post-trained judge scores {rollout, rewrite} pairwise with position
randomization, and the winner becomes the NLL target.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import torch


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
_WINNER_FALLBACK_RE = re.compile(r"\b([AB])\b")


def parse_winner(content: str) -> str:
    m = _WINNER_JSON_RE.search(content)
    if m:
        return m.group(1).upper()
    m = _WINNER_PLAIN_RE.search(content)
    if m:
        return m.group(1).upper()
    m = _WINNER_FALLBACK_RE.search(content)
    if m:
        return m.group(1).upper()
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
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return parse_winner(content)
        except Exception as e:
            last_err = e
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
    """Concurrent judge calls. Returns list of 'A' or 'B' per prompt."""
    results: list[str | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_judge_call, p, endpoint, model, temperature, top_p, max_tokens, timeout, retries): i
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"judge batch item {i} failed: {e}", flush=True)
                results[i] = "B"
    return [w or "B" for w in results]


@torch.no_grad()
def generate_rollouts(
    model: torch.nn.Module,
    tokenizer,
    prefix_ids_list: list[list[int]],
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[list[int]]:
    """Generate exactly `max_new_tokens` from the current policy for each prefix.

    Left-pads prefixes to the longest in the batch so a single batched
    generate call produces rollouts aligned with the trailing prefix position.
    """
    was_training = model.training
    model.eval()
    try:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        max_len = max(len(p) for p in prefix_ids_list)
        padded = []
        attn = []
        for p in prefix_ids_list:
            pad_count = max_len - len(p)
            padded.append([pad_id] * pad_count + list(p))
            attn.append([0] * pad_count + [1] * len(p))
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
        )
        rollouts = []
        for i in range(out.size(0)):
            full = out[i].tolist()
            gen = full[max_len : max_len + max_new_tokens]
            if len(gen) < max_new_tokens:
                gen = gen + [pad_id] * (max_new_tokens - len(gen))
            rollouts.append(gen)
        return rollouts
    finally:
        if was_training:
            model.train()


def collate_rollout_raw(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate that preserves raw lists — the training loop builds tensors after judging."""
    return {
        "prefix_ids": [b["prefix_ids"] for b in batch],
        "rewrite_suffix_ids": [b["rewrite_suffix_ids"] for b in batch],
        "original_suffix_ids": [b["original_suffix_ids"] for b in batch],
    }


def build_rollout_vs_rewrite_batch(
    raw_batch: dict[str, Any],
    model: torch.nn.Module,
    tokenizer,
    prompt_template: str,
    judge_endpoint: str,
    judge_model: str,
    judge_temperature: float,
    judge_top_p: float,
    judge_max_tokens: int,
    judge_repeats: int,
    judge_max_workers: int,
    max_new_tokens: int,
    pad_id: int,
    device: torch.device,
    step: int,
    rollout_temperature: float = 1.0,
    rollout_top_p: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Generate rollouts, judge against rewrites, build a padded NLL batch.

    Returns (tensor_batch, stats). stats includes:
      rollout_chosen: int
      total: int
      judge_latency_s: float
      gen_latency_s: float
    """
    prefix_ids = raw_batch["prefix_ids"]
    rewrite_ids = raw_batch["rewrite_suffix_ids"]
    bsz = len(prefix_ids)

    t0 = time.time()
    rollouts = generate_rollouts(
        model, tokenizer, prefix_ids, max_new_tokens, device,
        temperature=rollout_temperature, top_p=rollout_top_p,
    )
    gen_latency_s = time.time() - t0

    prompts: list[str] = []
    swaps: list[bool] = []
    for i in range(bsz):
        prefix_text = tokenizer.decode(prefix_ids[i], skip_special_tokens=True)
        rollout_text = tokenizer.decode(rollouts[i], skip_special_tokens=True)
        rewrite_text = tokenizer.decode(rewrite_ids[i], skip_special_tokens=True)
        swap = ((step * 131 + i * 17) % 2) == 1
        swaps.append(swap)
        if swap:
            prompts.append(build_prompt(prompt_template, prefix_text, rewrite_text, rollout_text))
        else:
            prompts.append(build_prompt(prompt_template, prefix_text, rollout_text, rewrite_text))

    t1 = time.time()
    if judge_repeats <= 1:
        winners = judge_pairwise_batch(
            prompts, judge_endpoint, judge_model,
            temperature=judge_temperature, top_p=judge_top_p,
            max_tokens=judge_max_tokens, max_workers=judge_max_workers,
        )
        rollout_wins = [
            ((w == "A") and not sw) or ((w == "B") and sw)
            for w, sw in zip(winners, swaps)
        ]
    else:
        counts_a = [0] * bsz
        for _ in range(judge_repeats):
            winners = judge_pairwise_batch(
                prompts, judge_endpoint, judge_model,
                temperature=judge_temperature, top_p=judge_top_p,
                max_tokens=judge_max_tokens, max_workers=judge_max_workers,
            )
            for i, w in enumerate(winners):
                counts_a[i] += 1 if w == "A" else 0
        rollout_wins = []
        for i in range(bsz):
            a_majority = counts_a[i] > judge_repeats / 2
            rollout_wins.append(
                (a_majority and not swaps[i]) or ((not a_majority) and swaps[i])
            )
    judge_latency_s = time.time() - t1

    input_ids_rows = []
    labels_rows = []
    rollout_chosen = 0
    for i in range(bsz):
        prefix = prefix_ids[i]
        if rollout_wins[i]:
            suffix = rollouts[i]
            rollout_chosen += 1
        else:
            suffix = rewrite_ids[i]
        ids = list(prefix) + list(suffix)
        labels = [-100] * len(prefix) + list(suffix)
        input_ids_rows.append(ids)
        labels_rows.append(labels)

    max_len = max(len(ids) for ids in input_ids_rows)
    padded_input = []
    padded_labels = []
    padded_attn = []
    for ids, lbl in zip(input_ids_rows, labels_rows):
        pad = max_len - len(ids)
        padded_input.append(ids + [pad_id] * pad)
        padded_labels.append(lbl + [-100] * pad)
        padded_attn.append([1] * len(ids) + [0] * pad)

    tensor_batch = {
        "input_ids": torch.tensor(padded_input, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(padded_attn, dtype=torch.long, device=device),
    }
    stats = {
        "rollout_chosen": rollout_chosen,
        "total": bsz,
        "judge_latency_s": judge_latency_s,
        "gen_latency_s": gen_latency_s,
    }
    return tensor_batch, stats
