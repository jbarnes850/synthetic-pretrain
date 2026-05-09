#!/usr/bin/env python3
"""Read-only continuation evaluation: base vs self-improved checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import torch
from common import jsonl_iter, latest_snapshot, load_config, now_run_id, safe_mean, set_seed
from self_improving import build_prompt, load_prompt_template, parse_winner
from torch.utils.data import DataLoader
from train import SuffixDataset, pad_batch, repetition_4gram_rate
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json_objects(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            rows.append(json.loads(line))
    return rows


def summarize_training(run_dir: Path) -> dict[str, Any]:
    rows = load_json_objects(run_dir / "train.log")
    train_rows = [r for r in rows if "train_loss" in r]
    val_rows = [r for r in rows if "val_loss" in r]
    ckpt_rows = [r for r in rows if "checkpoint" in r]
    first = train_rows[0]
    final = train_rows[-1]
    pool_spread = final["pool_top_score_mean"] - final["pool_bottom_score_mean"]
    pivot_gap = final["pivot_pointwise_mean"] - final["rollout_pointwise_mean"]
    return {
        "train_records": len(train_rows),
        "val_records": val_rows,
        "checkpoint_records": ckpt_rows,
        "first_train": first,
        "final_train": final,
        "chosen_is_rollout_delta": final["chosen_is_rollout_rate"] - first["chosen_is_rollout_rate"],
        "dpo_margin_delta": final["dpo_margin_avg"] - first["dpo_margin_avg"],
        "pool_spread_final": pool_spread,
        "pivot_minus_rollout_pointwise_final": pivot_gap,
        "alerts": {
            "nonfinite_train_loss": any(not torch.isfinite(torch.tensor(r["train_loss"])) for r in train_rows),
            "low_memory": min(r.get("available_mem_gib", 999.0) for r in train_rows) < 10.0,
            "pool_spread_lt_0_3_after_50": any(
                (r["step"] > 50)
                and ("pool_top_score_mean" in r)
                and (r["pool_top_score_mean"] - r["pool_bottom_score_mean"] < 0.3)
                for r in train_rows
            ),
            "chosen_rollout_lt_0_4_after_200": any(
                (r["step"] > 200)
                and ("chosen_is_rollout_rate" in r)
                and (r["chosen_is_rollout_rate"] < 0.40)
                for r in train_rows
            ),
            "pivot_gap_gt_0_4_after_500": any(
                (r["step"] > 500)
                and ("pivot_pointwise_mean" in r)
                and (r["pivot_pointwise_mean"] - r["rollout_pointwise_mean"] > 0.4)
                for r in train_rows
            ),
        },
    }


def load_tokenizer(cfg: dict[str, Any]):
    tokenizer_path = latest_snapshot(cfg["data"]["tokenizer_repo_cache"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(cfg: dict[str, Any], checkpoint: Path | None, device: torch.device):
    attn_impl = os.environ.get("SPARK_ATTN_IMPL", "sdpa")
    if attn_impl == "sdpa":
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception as e:
            print(f"warning: could not configure SDP backends: {e}", flush=True)
    dtype = torch.bfloat16 if cfg["runtime"].get("dtype") == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg["train"]["init_from_pretrained"],
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_val_loss(model, rows: list[dict[str, Any]], tokenizer, batch_size: int, device: torch.device) -> float:
    ds = SuffixDataset(rows, "raw_ntp")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: pad_batch(b, tokenizer.pad_token_id))
    losses = []
    for batch in loader:
        batch.pop("chosen")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))
    return safe_mean(losses)


@torch.no_grad()
def generate_continuations(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
) -> list[list[int]]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    outputs: list[list[int]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prefixes = [r["prefix_ids"] for r in batch_rows]
        max_len = max(len(p) for p in prefixes)
        input_rows, attn_rows = [], []
        for prefix in prefixes:
            pad = max_len - len(prefix)
            input_rows.append([pad_id] * pad + list(prefix))
            attn_rows.append([0] * pad + [1] * len(prefix))
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn_rows, dtype=torch.long, device=device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
        for seq in out.tolist():
            outputs.append(seq[max_len : max_len + max_new_tokens])
    return outputs


def judge_one(
    prompt: str,
    endpoint: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
) -> tuple[str | None, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    r = requests.post(endpoint.rstrip("/") + "/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    winner = parse_winner(content)
    if winner not in {"A", "B"}:
        return None, content
    if not re.search(r'"winner"\s*:\s*"[AB]"', content, flags=re.IGNORECASE):
        # Keep the verdict, but mark malformed responses separately upstream.
        return winner.lower(), content
    return winner, content


def judge_pairs(
    examples: list[dict[str, Any]],
    endpoint: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(examples)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(judge_one, exm["judge_prompt"], endpoint, model, temperature, top_p, max_tokens, 60.0): i
            for i, exm in enumerate(examples)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                winner, raw = fut.result()
                results[i] = {"winner": winner, "raw_judge": raw}
            except Exception as e:
                results[i] = {"winner": None, "raw_judge": f"ERROR: {e}"}
    return [r or {"winner": None, "raw_judge": "missing"} for r in results]


def text_for(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def write_examples_markdown(path: Path, paired: list[dict[str, Any]], limit: int) -> None:
    chunks = ["# Paired Continuation Examples\n"]
    for i, row in enumerate(paired[:limit], start=1):
        chunks.append(f"\n## Example {i}: {row['winner_label']}\n")
        chunks.append("### Prefix\n")
        chunks.append(row["prefix_text"].strip() + "\n")
        chunks.append("### Base Continuation\n")
        chunks.append(row["base_text"].strip() + "\n")
        chunks.append("### Self-Improved Continuation\n")
        chunks.append(row["self_improved_text"].strip() + "\n")
    path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-pairwise", type=int, default=128)
    parser.add_argument("--num-examples-md", type=int, default=24)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3337)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "evals" / now_run_id("continuation-eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    training_summary = summarize_training(run_dir)
    (out_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["runtime"].get("device") == "cuda" else "cpu")
    tokenizer = load_tokenizer(cfg)
    raw_rows = list(jsonl_iter(cfg["data"].get("raw_examples_jsonl", cfg["data"]["examples_jsonl"])))
    val_rows = [r for r in raw_rows if r["split"] == "val"]
    if not val_rows:
        raise RuntimeError("No validation rows found")

    rng = random.Random(args.seed)
    sample_rows = list(val_rows)
    rng.shuffle(sample_rows)
    sample_rows = sample_rows[: min(args.num_pairwise, len(sample_rows))]
    max_new_tokens = int(cfg["data"]["suffix_tokens"])

    sampling_config = {
        "generation": {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,
            "system_prompt": None,
            "special_tokens": "skip_special_tokens=True for decoded text",
        },
        "judge": {
            "endpoint": cfg["selection"]["judge_endpoint"],
            "model": cfg["selection"]["judge_model"],
            "temperature": float(cfg["selection"].get("judge_temperature", 0.7)),
            "top_p": float(cfg["selection"].get("judge_top_p", 0.6)),
            "max_tokens": int(cfg["selection"].get("judge_max_tokens", 64)),
            "prompt_path": cfg["selection"].get("prompt_path", "prompts/judge_quality.txt"),
        },
        "eval_parity_note": "Base and self-improved checkpoints use identical validation rows and greedy decoding settings.",
    }
    (out_dir / "sampling_config.json").write_text(json.dumps(sampling_config, indent=2) + "\n", encoding="utf-8")

    base_model = load_model(cfg, checkpoint=None, device=device)
    base_val_loss = evaluate_val_loss(base_model, val_rows, tokenizer, args.val_batch_size, device)
    base_gens = generate_continuations(base_model, tokenizer, sample_rows, max_new_tokens, args.gen_batch_size, device)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    self_improved_model = load_model(cfg, checkpoint=run_dir / "final.pt", device=device)
    self_improved_val_loss = evaluate_val_loss(self_improved_model, val_rows, tokenizer, args.val_batch_size, device)
    self_improved_gens = generate_continuations(
        self_improved_model, tokenizer, sample_rows, max_new_tokens, args.gen_batch_size, device
    )
    del self_improved_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    prompt_template = load_prompt_template(sampling_config["judge"]["prompt_path"])
    judge_inputs = []
    swaps = []
    for i, (row, base_ids, self_improved_ids) in enumerate(zip(sample_rows, base_gens, self_improved_gens)):
        prefix = text_for(tokenizer, row["prefix_ids"])
        base = text_for(tokenizer, base_ids)
        self_improved = text_for(tokenizer, self_improved_ids)
        swap = rng.random() < 0.5
        swaps.append(swap)
        cand_a, cand_b = (self_improved, base) if swap else (base, self_improved)
        judge_inputs.append({
            "judge_prompt": build_prompt(prompt_template, prefix, cand_a, cand_b),
            "prefix_text": prefix,
            "base_text": base,
            "self_improved_text": self_improved,
            "original_suffix_text": text_for(tokenizer, row["original_suffix_ids"]),
        })

    judge_results = judge_pairs(
        judge_inputs,
        endpoint=sampling_config["judge"]["endpoint"],
        model=sampling_config["judge"]["model"],
        temperature=sampling_config["judge"]["temperature"],
        top_p=sampling_config["judge"]["top_p"],
        max_tokens=sampling_config["judge"]["max_tokens"],
        max_workers=int(cfg["selection"].get("judge_max_workers", 32)),
    )

    paired = []
    self_improved_wins = 0
    base_wins = 0
    invalid = 0
    malformed = 0
    for i, (row, judge, swap) in enumerate(zip(judge_inputs, judge_results, swaps)):
        winner = judge["winner"]
        if winner is None:
            invalid += 1
            winner_label = "invalid"
        else:
            if winner in {"a", "b"}:
                malformed += 1
                winner = winner.upper()
            self_improved_won = (winner == "A" and swap) or (winner == "B" and not swap)
            if self_improved_won:
                self_improved_wins += 1
                winner_label = "self_improved"
            else:
                base_wins += 1
                winner_label = "base"
        paired.append({
            "index": i,
            "winner_label": winner_label,
            "judge_winner": judge["winner"],
            "self_improved_was_option_a": swap,
            "prefix_text": row["prefix_text"],
            "base_text": row["base_text"],
            "self_improved_text": row["self_improved_text"],
            "original_suffix_text": row["original_suffix_text"],
            "raw_judge": judge["raw_judge"],
        })

    with (out_dir / "paired_continuations.jsonl").open("w", encoding="utf-8") as f:
        for row in paired:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_examples_markdown(out_dir / "representative_examples.md", paired, args.num_examples_md)

    valid = max(1, self_improved_wins + base_wins)
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "num_val_rows": len(val_rows),
        "num_pairwise_examples": len(sample_rows),
        "base_val_loss": base_val_loss,
        "self_improved_val_loss": self_improved_val_loss,
        "self_improved_minus_base_val_loss": self_improved_val_loss - base_val_loss,
        "self_improved_win_rate": self_improved_wins / valid,
        "base_win_rate": base_wins / valid,
        "invalid_rate": invalid / max(1, len(sample_rows)),
        "malformed_but_parsed_rate": malformed / max(1, len(sample_rows)),
        "self_improved_wins": self_improved_wins,
        "base_wins": base_wins,
        "invalid": invalid,
        "base_repetition_4gram_rate": repetition_4gram_rate(base_gens),
        "self_improved_repetition_4gram_rate": repetition_4gram_rate(self_improved_gens),
        "base_avg_chars": safe_mean([len(p["base_text"]) for p in paired]),
        "self_improved_avg_chars": safe_mean([len(p["self_improved_text"]) for p in paired]),
        "sampling_config": sampling_config,
        "training_summary_path": str(out_dir / "training_summary.json"),
        "paired_examples_path": str(out_dir / "paired_continuations.jsonl"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
