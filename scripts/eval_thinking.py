#!/usr/bin/env python3
"""Thinking mid-training evals.

This complements the plain continuation judge eval. It asks whether the
interleaved-thinking SFT arms learned the intended mechanism:
thoughts as useful scaffolding for predicting later text.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import torch
import torch.nn.functional as F
from common import load_config, load_split_rows, now_run_id, resolve_hf_path, safe_mean, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

ARM_SPECS = {
    "raw_base": {
        "config": "configs/thinking_sft_raw_control.yaml",
        "checkpoint": "outputs/thinking_sft_raw_control/final.pt",
    },
    "think_base": {
        "config": "configs/thinking_sft_base.yaml",
        "checkpoint": "outputs/thinking_sft_base/final.pt",
    },
    "think_self_improved": {
        "config": "configs/thinking_sft_self_improved.yaml",
        "checkpoint": "outputs/thinking_sft_self_improved/final.pt",
    },
    "think_base_rlmt": {
        "config": "configs/thinking_sft_base.yaml",
        "checkpoint": "outputs/rlmt_base/final.pt",
    },
    "think_self_improved_rlmt": {
        "config": "configs/thinking_sft_self_improved.yaml",
        "checkpoint": "outputs/rlmt_self_improved/final.pt",
    },
}

JUDGE_SUFFIX_PROMPT = """You are judging whether a model continuation is semantically useful for predicting the reference continuation.

Prefix:
{prefix}

Reference continuation:
{reference}

Model continuation:
{candidate}

Return only valid JSON: {{"score": 1}} if the model continuation is coherent, locally relevant, and substantially matches or helps predict the reference continuation. Return {{"score": 0}} otherwise.
"""

GSM8K_FEW_SHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Altogether she sold 48 + 24 = 72 clips. The answer is 72.

Question: Weng earns $12 an hour for babysitting. Yesterday, she babysat for 50 minutes. How much did she earn?
Answer: Weng worked 50/60 = 5/6 hours. She earned 12 * 5/6 = 10 dollars. The answer is 10.

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents gave her $15, and her grandparents gave her twice as much as her parents. How much more money does Betty need?
Answer: Betty has 100/2 = 50 dollars. Her grandparents gave her 15 * 2 = 30 dollars. Now she has 50 + 15 + 30 = 95 dollars. She needs 100 - 95 = 5 more dollars. The answer is 5.

Question: Julie is reading a 120-page book. Yesterday she read 12 pages and today she read twice as many pages as yesterday. If she wants to finish the book tomorrow, how many pages must she read tomorrow?
Answer: Today Julie read 12 * 2 = 24 pages. She has read 12 + 24 = 36 pages. She must read 120 - 36 = 84 pages tomorrow. The answer is 84.

"""


def load_tokenizer(cfg: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_hf_path(cfg["data"]["tokenizer_repo_cache"]),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device):
    attn_impl = os.environ.get("SPARK_ATTN_IMPL", "sdpa")
    if attn_impl == "sdpa":
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception as exc:
            print(f"warning: could not configure SDP backends: {exc}", flush=True)
    dtype = torch.bfloat16 if cfg["runtime"].get("dtype") == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        resolve_hf_path(cfg["train"]["init_from_pretrained"]),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def parse_think_segments(text: str) -> list[tuple[str, str]]:
    """Return [(kind, text)] where kind is text or thought_with_tags."""
    parts: list[tuple[str, str]] = []
    pos = 0
    for match in re.finditer(r"<think>.*?</think>", text, flags=re.IGNORECASE | re.DOTALL):
        if match.start() > pos:
            parts.append(("text", text[pos : match.start()]))
        parts.append(("thought", match.group(0)))
        pos = match.end()
    if pos < len(text):
        parts.append(("text", text[pos:]))
    return [(kind, value) for kind, value in parts if value]


def extract_visible_after_thought(text: str) -> tuple[str, int, bool]:
    if "</think>" in text:
        thought_text = text.split("</think>", 1)[0]
        return text.split("</think>", 1)[1].strip(), len(encode_len_words(thought_text)), True
    stripped = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return stripped or text.strip(), 0, False


def encode_len_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def suffix_nll(model, input_ids: list[int], label_mask: list[bool], device: torch.device) -> dict[str, float]:
    if len(input_ids) != len(label_mask):
        raise ValueError("input_ids and label_mask length mismatch")
    if len(input_ids) < 2:
        return {"nll": float("nan"), "tokens": 0}
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[:, :-1, :]
    targets = ids[:, 1:]
    losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    mask = torch.tensor(label_mask[1:], dtype=torch.bool, device=device)
    selected = losses[mask]
    if selected.numel() == 0:
        return {"nll": float("nan"), "tokens": 0}
    return {"nll": float(selected.mean().detach().cpu()), "tokens": int(selected.numel())}


def evaluate_token_type_nll(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_rows: int,
) -> dict[str, Any]:
    totals: dict[str, list[float]] = defaultdict(list)
    token_counts: dict[str, int] = defaultdict(int)
    for row in rows[:max_rows]:
        ids: list[int] = []
        masks: dict[str, list[bool]] = {"thought": [], "text": [], "post_thought_16": []}
        pending_post = 0
        for kind, segment in parse_think_segments(row["augmented_text"]):
            seg_ids = encode(tokenizer, segment)
            ids.extend(seg_ids)
            is_thought = kind == "thought"
            masks["thought"].extend([is_thought] * len(seg_ids))
            masks["text"].extend([not is_thought] * len(seg_ids))
            post_mask = []
            if is_thought:
                pending_post = 16
                post_mask = [False] * len(seg_ids)
            else:
                take = min(pending_post, len(seg_ids))
                post_mask = [True] * take + [False] * (len(seg_ids) - take)
                pending_post -= take
            masks["post_thought_16"].extend(post_mask)
        ids = ids[:1024]
        for key in masks:
            masks[key] = masks[key][: len(ids)]
            result = suffix_nll(model, ids, masks[key], device)
            if result["tokens"] > 0 and math.isfinite(result["nll"]):
                totals[key].append(result["nll"])
                token_counts[key] += result["tokens"]
    return {
        key: {"mean_nll": safe_mean(values), "examples": len(values), "tokens": token_counts[key]}
        for key, values in totals.items()
    }


def build_aug_prefix(
    tokenizer,
    row: dict[str, Any],
    raw_prefix_tokens: int,
    corrupt_thought: str | None = None,
) -> list[int]:
    prefix_ids: list[int] = []
    raw_seen = 0
    for kind, segment in parse_think_segments(row["augmented_text"]):
        if kind == "text":
            seg_ids = encode(tokenizer, segment)
            remaining = raw_prefix_tokens - raw_seen
            if remaining <= 0:
                break
            prefix_ids.extend(seg_ids[:remaining])
            raw_seen += min(remaining, len(seg_ids))
            if raw_seen >= raw_prefix_tokens:
                break
        else:
            if corrupt_thought is None:
                prefix_ids.extend(encode(tokenizer, segment))
            else:
                prefix_ids.extend(encode(tokenizer, f"<think>{corrupt_thought}</think>"))
    return prefix_ids[-896:]


def evaluate_thought_ablation(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_rows: int,
    raw_prefix_tokens: int,
    suffix_tokens: int,
) -> dict[str, Any]:
    corrupt_pool = [
        "This unrelated note discusses a different topic and should not help predict the next passage.",
        "The model should reason carefully, but this content is deliberately mismatched to the source.",
        "A generic observation about causes and effects is inserted here without local evidence.",
    ]
    out: dict[str, list[float]] = {"raw_prefix": [], "thought_prefix": [], "corrupt_thought_prefix": []}
    for idx, row in enumerate(rows[:max_rows]):
        raw = list(row["raw_chunk_ids"])
        prefix_raw = raw[:raw_prefix_tokens]
        suffix = raw[raw_prefix_tokens : raw_prefix_tokens + suffix_tokens]
        if len(suffix) < suffix_tokens:
            continue
        contexts = {
            "raw_prefix": prefix_raw,
            "thought_prefix": build_aug_prefix(tokenizer, row, raw_prefix_tokens),
            "corrupt_thought_prefix": build_aug_prefix(
                tokenizer, row, raw_prefix_tokens, corrupt_pool[idx % len(corrupt_pool)]
            ),
        }
        for name, context in contexts.items():
            input_ids = (context + suffix)[-1024:]
            label_mask = [False] * max(0, len(input_ids) - len(suffix)) + [True] * min(len(suffix), len(input_ids))
            result = suffix_nll(model, input_ids, label_mask, device)
            if result["tokens"] > 0 and math.isfinite(result["nll"]):
                out[name].append(result["nll"])
    return {
        key: {"mean_suffix_nll": safe_mean(values), "examples": len(values)}
        for key, values in out.items()
    }


@torch.no_grad()
def generate_from_prompts(
    model,
    tokenizer,
    prompts: list[list[int]],
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
) -> list[list[int]]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    outputs: list[list[int]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        max_len = max(len(ids) for ids in batch)
        input_rows, attn_rows = [], []
        for ids in batch:
            pad = max_len - len(ids)
            input_rows.append([pad_id] * pad + ids)
            attn_rows.append([0] * pad + [1] * len(ids))
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attn = torch.tensor(attn_rows, dtype=torch.long, device=device)
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
        for seq in gen.tolist():
            outputs.append(seq[max_len:])
    return outputs


def judge_pointwise(
    rows: list[dict[str, Any]],
    endpoint: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_workers: int,
) -> list[dict[str, Any]]:
    def one(row: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": row["judge_prompt"]}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(endpoint.rstrip("/") + "/v1/chat/completions", json=payload, timeout=60.0)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r'"score"\s*:\s*([01])', content)
            score = int(match.group(1)) if match else None
            return {"score": score, "raw_judge": content}
        except Exception as exc:
            return {"score": None, "raw_judge": f"ERROR: {exc}"}

    out: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(one, row): idx for idx, row in enumerate(rows)}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return [row or {"score": None, "raw_judge": "missing"} for row in out]


def evaluate_thought_forced(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    args,
    judge_cfg: dict[str, Any],
    out_path: Path,
    arm_name: str,
) -> dict[str, Any]:
    sample_rows = rows[: args.num_thought_forced]
    prompts: list[list[int]] = []
    prompt_texts: list[str] = []
    references: list[str] = []
    for row in sample_rows:
        prefix_ids = row["raw_chunk_ids"][: args.raw_prefix_tokens]
        prompt = decode(tokenizer, prefix_ids).rstrip() + "\n<think>"
        prompts.append(encode(tokenizer, prompt))
        prompt_texts.append(prompt)
        references.append(decode(tokenizer, row["raw_chunk_ids"][args.raw_prefix_tokens : args.raw_prefix_tokens + args.suffix_tokens]))
    generated_ids = generate_from_prompts(model, tokenizer, prompts, args.thought_forced_max_new_tokens, args.gen_batch_size, device)
    judge_rows = []
    examples = []
    for idx, (row, gen_ids, prompt_text, reference) in enumerate(zip(sample_rows, generated_ids, prompt_texts, references)):
        raw = decode(tokenizer, gen_ids)
        visible, thought_words, closed = extract_visible_after_thought("<think>" + raw)
        candidate = visible or raw
        judge_rows.append(
            {
                "judge_prompt": JUDGE_SUFFIX_PROMPT.format(
                    prefix=prompt_text.replace("<think>", "").strip()[:2000],
                    reference=reference[:2000],
                    candidate=candidate[:2000],
                )
            }
        )
        examples.append(
            {
                "index": idx,
                "source_id": row["id"],
                "prefix": prompt_text,
                "reference": reference,
                "raw_generation": raw,
                "visible_candidate": candidate,
                "closed_think": closed,
                "thought_words": thought_words,
            }
        )
    judgments = judge_pointwise(judge_rows, **judge_cfg)
    valid_scores = [j["score"] for j in judgments if j["score"] in {0, 1}]
    with out_path.open("w", encoding="utf-8") as f:
        for ex, judgment in zip(examples, judgments):
            f.write(json.dumps({"arm": arm_name, **ex, **judgment}, ensure_ascii=False) + "\n")
    return {
        "reward_rate": sum(valid_scores) / max(1, len(valid_scores)),
        "valid": len(valid_scores),
        "invalid": len(judgments) - len(valid_scores),
        "closed_think_rate": safe_mean([float(ex["closed_think"]) for ex in examples]),
        "avg_thought_words": safe_mean([ex["thought_words"] for ex in examples]),
        "avg_visible_chars": safe_mean([len(ex["visible_candidate"]) for ex in examples]),
        "examples_path": str(out_path),
    }


def find_gsm8k_parquet() -> Path | None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    roots = [
        hf_home / "hub/datasets--openai--gsm8k/blobs",
        hf_home / "hub/datasets--gsm8k/blobs",
        Path.home() / ".cache/huggingface/hub/datasets--openai--gsm8k/blobs",
        Path.home() / ".cache/huggingface/hub/datasets--gsm8k/blobs",
    ]
    candidates = []
    for root in roots:
        if root.exists():
            for path in root.glob("*"):
                try:
                    if path.read_bytes()[:4] == b"PAR1":
                        candidates.append(path)
                except Exception:
                    pass
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def load_gsm8k_examples(limit: int, seed: int) -> list[dict[str, str]]:
    path = find_gsm8k_parquet()
    if path is None:
        raise RuntimeError("No cached GSM8K parquet found")
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows = table.to_pylist()
    rows = [row for row in rows if "question" in row and "answer" in row]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


def extract_answer_number(text: str) -> str | None:
    text = text.replace(",", "")
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if not matches:
        return None
    value = matches[-1]
    if value.endswith(".0"):
        value = value[:-2]
    return value


def gold_gsm8k_answer(answer: str) -> str | None:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if match:
        return match.group(1).replace(",", "")
    return extract_answer_number(answer)


def evaluate_gsm8k(
    model,
    tokenizer,
    device: torch.device,
    args,
    out_path: Path,
    arm_name: str,
) -> dict[str, Any]:
    examples = load_gsm8k_examples(args.num_reasoning, args.seed)
    prompts = [
        encode(tokenizer, GSM8K_FEW_SHOT + f"Question: {row['question']}\nAnswer:")
        for row in examples
    ]
    gens = generate_from_prompts(model, tokenizer, prompts, args.reasoning_max_new_tokens, args.reasoning_batch_size, device)
    correct = 0
    rows_out = []
    for idx, (row, gen_ids) in enumerate(zip(examples, gens)):
        completion = decode(tokenizer, gen_ids)
        pred = extract_answer_number(completion)
        gold = gold_gsm8k_answer(row["answer"])
        ok = pred is not None and gold is not None and pred == gold
        correct += int(ok)
        rows_out.append(
            {
                "arm": arm_name,
                "index": idx,
                "question": row["question"],
                "gold_answer": gold,
                "pred_answer": pred,
                "correct": ok,
                "completion": completion,
            }
        )
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"accuracy": correct / max(1, len(rows_out)), "correct": correct, "total": len(rows_out), "examples_path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--judge-endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--judge-model", default="qwen36-35b-a3b")
    parser.add_argument("--seed", type=int, default=4337)
    parser.add_argument("--num-thought-forced", type=int, default=128)
    parser.add_argument("--num-token-nll", type=int, default=512)
    parser.add_argument("--num-ablation", type=int, default=256)
    parser.add_argument("--num-reasoning", type=int, default=32)
    parser.add_argument("--raw-prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--thought-forced-max-new-tokens", type=int, default=192)
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=256)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--reasoning-batch-size", type=int, default=4)
    parser.add_argument("--judge-temperature", type=float, default=0.7)
    parser.add_argument("--judge-top-p", type=float, default=0.6)
    parser.add_argument("--judge-max-tokens", type=int, default=64)
    parser.add_argument("--judge-max-workers", type=int, default=32)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["raw_base", "think_base", "think_self_improved"],
        help="Arm names from ARM_SPECS to evaluate.",
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
    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/thinking_eval") / now_run_id("thinking-eval")
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
        "num_val_rows": len(rows),
        "config": vars(args),
        "paper_alignment": {
            "thought_forced": "prefix -> generated thought + suffix -> judge against reference suffix",
            "token_type_nll": "loss over original and thought tokens in augmented sequence",
            "ablation": "test whether inserted thoughts help suffix prediction vs removed/corrupted thoughts",
            "reasoning_probe": "small pass@1 GSM8K-style downstream readiness probe",
        },
        "arms": {},
    }

    for arm_name in args.arms:
        spec = ARM_SPECS[arm_name]
        print(f"loading {arm_name}", flush=True)
        arm_cfg = load_config(spec["config"])
        model = load_model(arm_cfg, Path(spec["checkpoint"]), device)
        arm_out = out_dir / arm_name
        arm_out.mkdir(parents=True, exist_ok=True)
        print(f"thought_forced {arm_name}", flush=True)
        thought_forced = evaluate_thought_forced(
            model,
            tokenizer,
            rows,
            device,
            args,
            judge_cfg,
            arm_out / "thought_forced.jsonl",
            arm_name,
        )
        print(f"token_type_nll {arm_name}", flush=True)
        token_type_nll = evaluate_token_type_nll(model, tokenizer, rows, device, args.num_token_nll)
        print(f"ablation {arm_name}", flush=True)
        ablation = evaluate_thought_ablation(
            model,
            tokenizer,
            rows,
            device,
            args.num_ablation,
            args.raw_prefix_tokens,
            args.suffix_tokens,
        )
        print(f"gsm8k {arm_name}", flush=True)
        reasoning = evaluate_gsm8k(model, tokenizer, device, args, arm_out / "gsm8k_mini.jsonl", arm_name)
        summary["arms"][arm_name] = {
            "thought_forced": thought_forced,
            "token_type_nll": token_type_nll,
            "thought_ablation": ablation,
            "gsm8k_mini": reasoning,
        }
        (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
