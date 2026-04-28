#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from common import emit_metric, jsonl_iter, latest_snapshot, load_config, read_available_mem_gib, safe_mean, set_seed
from self_improving import (
    build_rollout_vs_rewrite_batch,
    collate_rollout_raw,
    load_prompt_template,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from train_dpo import (
    _suffix_logprobs_from_logits,
    build_online_dpo_batch,
    compute_dpo_loss,
    load_ref_model,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def cosine_with_floor_schedule(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Cosine schedule that decays to `min_lr_ratio * peak_lr`, not zero.

    Paper §1.2.2 continual: min_ratio=0.1. Transformers'
    get_cosine_schedule_with_warmup decays to 0.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


class SuffixDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], condition: str):
        self.rows = rows
        self.condition = condition

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        if self.condition == "raw_ntp":
            prefix = row["prefix_ids"]
            suffix = row["original_suffix_ids"]
            chosen = "original"
        elif self.condition == "finephrase_math_ntp":
            prefix = row["prefix_ids"]
            suffix = row["finephrase_suffix_ids"]
            chosen = "finephrase"
        elif self.condition == "rfnll_original_vs_finephrase":
            prefix = row["prefix_ids"]
            suffix = row["chosen_suffix_ids"]
            chosen = row.get("chosen", "selected")
        elif self.condition == "thinking_sft":
            prefix = row["prefix_ids"]
            suffix = row["thinking_suffix_ids"]
            chosen = "thinking"
        elif self.condition == "interleaved_thinking_sft":
            prefix = []
            suffix = row["interleaved_thinking_ids"]
            chosen = "interleaved_thinking"
        elif self.condition == "raw_chunk_ntp":
            prefix = []
            suffix = row["raw_chunk_ids"]
            chosen = "raw_chunk"
        elif self.condition == "rfnll_rollout_vs_rewrite":
            return {
                "prefix_ids": list(row["prefix_ids"]),
                "rewrite_suffix_ids": list(row["finephrase_suffix_ids"]),
                "original_suffix_ids": list(row["original_suffix_ids"]),
            }
        elif self.condition == "online_dpo_selfimproving":
            return {
                "prefix_ids": list(row["prefix_ids"]),
                "rewrite_suffix_ids": list(row["finephrase_suffix_ids"]),
                "original_suffix_ids": list(row["original_suffix_ids"]),
            }
        else:
            raise ValueError(f"Unknown train.condition={self.condition!r}")
        input_ids = prefix + suffix
        labels = [-100] * len(prefix) + suffix
        return {
            "input_ids": input_ids,
            "labels": labels,
            "chosen": chosen,
            "prefix_len": len(prefix),
            "suffix_len": len(suffix),
        }


def pad_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    chosen = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
        chosen.append(item["chosen"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "chosen": chosen,
    }


def load_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    condition = cfg["train"]["condition"]
    if condition == "raw_ntp":
        path = cfg["data"].get("raw_examples_jsonl", cfg["data"]["examples_jsonl"])
    elif condition == "finephrase_math_ntp":
        path = cfg["data"]["examples_jsonl"]
    elif condition == "rfnll_original_vs_finephrase":
        path = cfg["data"]["selected_jsonl"]
    elif condition == "thinking_sft":
        path = cfg["data"].get("thinking_examples_jsonl", cfg["data"]["examples_jsonl"])
    elif condition == "interleaved_thinking_sft":
        path = cfg["data"].get("interleaved_thinking_examples_jsonl", cfg["data"]["examples_jsonl"])
    elif condition == "raw_chunk_ntp":
        path = cfg["data"].get("interleaved_thinking_examples_jsonl", cfg["data"]["examples_jsonl"])
    elif condition == "rfnll_rollout_vs_rewrite":
        path = cfg["data"]["examples_jsonl"]
    elif condition == "online_dpo_selfimproving":
        path = cfg["data"]["examples_jsonl"]
    else:
        raise ValueError(f"Unknown train.condition={condition!r}")
    rows = list(jsonl_iter(path))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def build_model(cfg: dict[str, Any], tokenizer) -> torch.nn.Module:
    attn_impl = os.environ.get("SPARK_ATTN_IMPL", "sdpa")
    if attn_impl == "sdpa":
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception as e:
            print(f"warning: could not configure SDP backends: {e}", flush=True)

    # Path A: continual pretraining from a public HF checkpoint.
    # Loads native architecture + pretrained weights in one call. cfg.model
    # dimension overrides are ignored — the pretrained weights dictate shape.
    init_from_pretrained = cfg["train"].get("init_from_pretrained")
    if init_from_pretrained:
        dtype = torch.bfloat16 if cfg["runtime"].get("dtype") == "bfloat16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            init_from_pretrained,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
        # Tokenizer may have fewer tokens than model.config.vocab_size due to
        # embedding-matrix padding (common HF pattern). Only fail if tokenizer
        # has MORE tokens than the model can embed — that would produce
        # out-of-range token IDs at training time.
        if len(tokenizer) > model.config.vocab_size:
            raise ValueError(
                f"tokenizer vocab ({len(tokenizer)}) exceeds pretrained model vocab_size "
                f"({model.config.vocab_size}); IDs would be out-of-range."
            )
        return model

    # Path B: from-scratch pretraining (Phase 1) — build empty model from
    # the reference config, override dimensions from cfg.model, init randomly.
    config_path = latest_snapshot(cfg["data"]["config_repo_cache"])
    config = AutoConfig.from_pretrained(config_path, local_files_only=True, trust_remote_code=True)
    setattr(config, "_attn_implementation", attn_impl)
    model_cfg = cfg["model"]
    for key, value in model_cfg.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    config.vocab_size = len(tokenizer)
    if hasattr(config, "pad_token_id"):
        config.pad_token_id = tokenizer.pad_token_id
    if hasattr(config, "eos_token_id"):
        config.eos_token_id = tokenizer.eos_token_id
    if hasattr(config, "bos_token_id"):
        config.bos_token_id = tokenizer.bos_token_id
    return AutoModelForCausalLM.from_config(config, trust_remote_code=True)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> tuple[float, Counter]:
    model.eval()
    losses = []
    chosen_counts = Counter()
    for batch in loader:
        chosen_counts.update(batch.pop("chosen"))
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))
    model.train()
    return safe_mean(losses), chosen_counts


def repetition_4gram_rate(token_rows: list[list[int]]) -> float:
    total = 0
    repeated = 0
    for ids in token_rows:
        if len(ids) < 4:
            continue
        grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
        counts = Counter(grams)
        total += len(grams)
        repeated += sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp:
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
    is_main = rank == 0

    set_seed(int(cfg["project"]["seed"]) + rank)

    if torch.cuda.is_available() and cfg["runtime"].get("device") == "cuda":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    tokenizer_path = latest_snapshot(cfg["data"]["tokenizer_repo_cache"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_rows(cfg)
    oracle_rows = list(jsonl_iter(cfg["data"].get("raw_examples_jsonl", cfg["data"]["examples_jsonl"])))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"] or train_rows[: max(1, min(32, len(train_rows)))]
    oracle_train_rows = [r for r in oracle_rows if r["split"] == "train"]
    oracle_val_rows = [r for r in oracle_rows if r["split"] == "val"] or oracle_train_rows[: max(1, min(32, len(oracle_train_rows)))]
    condition = cfg["train"]["condition"]

    train_ds = SuffixDataset(train_rows, condition)
    val_ds = SuffixDataset(val_rows, condition)
    pad_id = tokenizer.pad_token_id

    train_sampler = None
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=int(cfg["project"]["seed"]),
        )

    if condition in ("rfnll_rollout_vs_rewrite", "online_dpo_selfimproving"):
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_rollout_raw,
            drop_last=True,
        )
        raw_val_ds = SuffixDataset(oracle_val_rows, "raw_ntp")
        val_loader = DataLoader(
            raw_val_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            collate_fn=lambda b: pad_batch(b, pad_id),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=lambda b: pad_batch(b, pad_id),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            collate_fn=lambda b: pad_batch(b, pad_id),
        )
    if not train_loader:
        raise RuntimeError("Train loader is empty")

    judge_endpoint = os.environ.get("JUDGE_ENDPOINT") or cfg.get("selection", {}).get("judge_endpoint")
    judge_model = os.environ.get("JUDGE_MODEL") or cfg.get("selection", {}).get("judge_model", "qwen-judge")
    judge_temperature = float(cfg.get("selection", {}).get("judge_temperature", 0.7))
    judge_top_p = float(cfg.get("selection", {}).get("judge_top_p", 0.6))
    judge_max_tokens = int(cfg.get("selection", {}).get("judge_max_tokens", 64))
    judge_repeats = int(cfg.get("selection", {}).get("judge_repeats", 1))
    judge_max_workers = int(cfg.get("selection", {}).get("judge_max_workers", 16))
    prompt_path = cfg.get("selection", {}).get("prompt_path", "prompts/judge_quality.txt")
    prompt_template = load_prompt_template(prompt_path) if condition == "rfnll_rollout_vs_rewrite" else ""
    rollout_max_new_tokens = int(cfg["data"]["suffix_tokens"])
    rollout_temperature = float(cfg.get("selection", {}).get("rollout_temperature", 1.0))
    rollout_top_p = float(cfg.get("selection", {}).get("rollout_top_p", 1.0))
    if condition == "rfnll_rollout_vs_rewrite" and not (judge_endpoint and judge_model):
        raise ValueError("rfnll_rollout_vs_rewrite requires JUDGE_ENDPOINT and judge_model")
    if condition == "online_dpo_selfimproving" and not (judge_endpoint and judge_model):
        raise ValueError("online_dpo_selfimproving requires JUDGE_ENDPOINT and judge_model")

    dpo_beta = float(os.environ.get("SPARK_DPO_BETA", cfg["train"].get("dpo_beta", 0.1)))
    num_rollouts = int(cfg["train"].get("num_rollouts", 16))
    min_lr_ratio = float(cfg["train"].get("min_lr_ratio", 0.0))
    # Paper §1.2.2 continual: pivot_source = original suffix. Build a loader
    # whose collate still returns raw prefix/rewrite/original; build_online_dpo_batch
    # picks the pivot per row.
    if condition == "online_dpo_selfimproving":
        prompt_template = load_prompt_template(prompt_path)

    model = build_model(cfg, tokenizer).to(device)
    if cfg["runtime"].get("dtype") == "bfloat16" and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)

    init_from_pretrained = cfg["train"].get("init_from_pretrained")
    init_ckpt = cfg["train"].get("init_from_checkpoint") or os.environ.get("INIT_FROM_CHECKPOINT")
    if init_from_pretrained:
        if is_main:
            print(f"loaded pretrained base from {init_from_pretrained}", flush=True)
        if init_ckpt:
            state = torch.load(init_ckpt, map_location=device)
            model.load_state_dict(state, strict=True)
            if is_main:
                print(f"loaded checkpoint over pretrained architecture from {init_ckpt}", flush=True)
    elif init_ckpt:
        state = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(state, strict=True)
        if is_main:
            print(f"loaded warm-start checkpoint from {init_ckpt}", flush=True)

    ref_model = None
    if condition == "online_dpo_selfimproving":
        if not (init_ckpt or init_from_pretrained):
            raise ValueError(
                "online_dpo_selfimproving requires init_from_checkpoint or init_from_pretrained"
            )
        ref_model = load_ref_model(model, device)
        if is_main:
            print("reference model cloned from warm-start policy, grad disabled", flush=True)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    max_steps = int(cfg["train"]["max_steps"])
    grad_accum = int(cfg["train"]["grad_accum_steps"])
    if min_lr_ratio > 0.0:
        scheduler = cosine_with_floor_schedule(
            optimizer,
            num_warmup_steps=int(cfg["train"]["warmup_steps"]),
            num_training_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg["train"]["warmup_steps"]),
            num_training_steps=max_steps,
        )

    output_dir = Path(cfg["train"]["output_dir"])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    save_every = int(cfg["train"].get("save_every", 0))

    start = time.time()
    mem_min = read_available_mem_gib()
    tokens_seen = 0
    step = 0
    running = []
    rollout_chosen_total = 0
    rollout_total = 0
    judge_latency_total = 0.0
    gen_latency_total = 0.0
    # DPO accumulators (only used when condition == online_dpo_selfimproving)
    dpo_total_prefixes = 0
    dpo_kept_prefixes = 0
    dpo_chosen_is_pivot = 0
    dpo_rejected_is_pivot = 0
    dpo_pivot_pointwise_sum = 0.0
    dpo_rollout_pointwise_sum = 0.0
    dpo_pool_top_score_sum = 0.0
    dpo_pool_bottom_score_sum = 0.0
    dpo_margin_sum = 0.0
    dpo_acc_sum = 0.0
    dpo_micro_steps = 0
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)
    model.train()

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)
            if condition == "rfnll_rollout_vs_rewrite":
                batch, roll_stats = build_rollout_vs_rewrite_batch(
                    batch,
                    model,
                    tokenizer,
                    prompt_template,
                    judge_endpoint,
                    judge_model,
                    judge_temperature,
                    judge_top_p,
                    judge_max_tokens,
                    judge_repeats,
                    judge_max_workers,
                    rollout_max_new_tokens,
                    pad_id,
                    device,
                    step * grad_accum + rollout_total // max(1, int(cfg["train"]["batch_size"])),
                    rollout_temperature=rollout_temperature,
                    rollout_top_p=rollout_top_p,
                )
                rollout_chosen_total += roll_stats["rollout_chosen"]
                rollout_total += roll_stats["total"]
                judge_latency_total += roll_stats["judge_latency_s"]
                gen_latency_total += roll_stats["gen_latency_s"]
                out = model(**batch)
                if not torch.isfinite(out.loss):
                    raise RuntimeError(f"non-finite training loss at step={step + 1}: {float(out.loss.detach().cpu())}")
                loss = out.loss / grad_accum
                loss.backward()
                accum_loss += float(loss.detach().cpu())
                tokens_seen += int(batch["attention_mask"].sum().detach().cpu())
            elif condition == "online_dpo_selfimproving":
                micro_step_id = step * grad_accum + dpo_micro_steps
                # Pass the unwrapped policy for rollout generation so we never
                # touch DDP's forward hooks from generate(); the policy forward
                # for DPO loss below still goes through the DDP wrapper.
                unwrapped_policy = model.module if use_ddp else model
                dpo_batch, dpo_stats = build_online_dpo_batch(
                    batch,
                    unwrapped_policy,
                    ref_model,
                    tokenizer,
                    prompt_template,
                    judge_endpoint,
                    judge_model,
                    judge_temperature,
                    judge_top_p,
                    judge_max_tokens,
                    judge_max_workers,
                    rollout_max_new_tokens,
                    num_rollouts,
                    pad_id,
                    device,
                    micro_step_id,
                    rollout_temperature=rollout_temperature,
                    rollout_top_p=rollout_top_p,
                    rank=rank,
                )
                dpo_total_prefixes += dpo_stats["total"]
                judge_latency_total += dpo_stats["judge_latency_s"]
                gen_latency_total += dpo_stats["gen_latency_s"]
                dpo_pivot_pointwise_sum += dpo_stats["pivot_pointwise_mean"] * dpo_stats["total"]
                dpo_rollout_pointwise_sum += dpo_stats["rollout_pointwise_mean"] * dpo_stats["total"]
                dpo_pool_top_score_sum += dpo_stats["pool_top_score_mean"] * dpo_stats["total"]
                dpo_pool_bottom_score_sum += dpo_stats["pool_bottom_score_mean"] * dpo_stats["total"]
                if dpo_batch is None:
                    # Unreachable with K >= 1 and {0,1} scores, but keep DDP safe.
                    dpo_micro_steps += 1
                    continue
                dpo_kept_prefixes += dpo_stats["kept"]
                dpo_chosen_is_pivot += dpo_stats["chosen_is_pivot_count"]
                dpo_rejected_is_pivot += dpo_stats["rejected_is_pivot_count"]

                # One stacked policy forward through the DDP-wrapped model.
                stacked_ids = torch.cat(
                    [dpo_batch["chosen_input_ids"], dpo_batch["rejected_input_ids"]], dim=0
                )
                stacked_attn = torch.cat(
                    [dpo_batch["chosen_attention_mask"], dpo_batch["rejected_attention_mask"]], dim=0
                )
                stacked_labels = torch.cat(
                    [dpo_batch["chosen_labels"], dpo_batch["rejected_labels"]], dim=0
                )
                out = model(input_ids=stacked_ids, attention_mask=stacked_attn)
                policy_logp = _suffix_logprobs_from_logits(out.logits, stacked_labels)
                n_kept = dpo_batch["chosen_input_ids"].size(0)
                policy_logp_chosen = policy_logp[:n_kept]
                policy_logp_rejected = policy_logp[n_kept:]
                dpo_loss, dpo_loss_stats = compute_dpo_loss(
                    policy_logp_chosen,
                    policy_logp_rejected,
                    dpo_batch["ref_logp_chosen"],
                    dpo_batch["ref_logp_rejected"],
                    beta=dpo_beta,
                )
                if not torch.isfinite(dpo_loss):
                    raise RuntimeError(f"non-finite DPO loss at step={step + 1}: {float(dpo_loss.detach().cpu())}")
                loss = dpo_loss / grad_accum
                loss.backward()
                accum_loss += float(loss.detach().cpu())
                tokens_seen += int(dpo_batch["chosen_attention_mask"].sum().detach().cpu())
                dpo_margin_sum += dpo_loss_stats["dpo_margin"]
                dpo_acc_sum += dpo_loss_stats["dpo_chosen_beats_rejected_rate"]
                dpo_micro_steps += 1
            else:
                batch.pop("chosen")
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                if not torch.isfinite(out.loss):
                    raise RuntimeError(f"non-finite training loss at step={step + 1}: {float(out.loss.detach().cpu())}")
                loss = out.loss / grad_accum
                loss.backward()
                accum_loss += float(loss.detach().cpu())
                tokens_seen += int(batch["attention_mask"].sum().detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        step += 1
        running.append(accum_loss)
        mem = read_available_mem_gib()
        mem_min = mem if mem_min < 0 else min(mem_min, mem)

        if step % int(cfg["train"]["log_every"]) == 0 or step == 1:
            elapsed = max(1e-6, time.time() - start)
            if is_main:
                log_entry = {
                    "step": step,
                    "train_loss": safe_mean(running[-int(cfg["train"]["log_every"]) :]),
                    "tokens_seen": tokens_seen,
                    "tok_per_sec": tokens_seen / elapsed,
                    "available_mem_gib": mem,
                }
                if condition == "rfnll_rollout_vs_rewrite" and rollout_total > 0:
                    log_entry["chosen_rollout_rate"] = rollout_chosen_total / rollout_total
                    log_entry["judge_latency_s_avg"] = judge_latency_total / max(1, step * grad_accum)
                    log_entry["gen_latency_s_avg"] = gen_latency_total / max(1, step * grad_accum)
                if condition == "online_dpo_selfimproving" and dpo_total_prefixes > 0:
                    log_entry["pivot_pointwise_mean"] = dpo_pivot_pointwise_sum / dpo_total_prefixes
                    log_entry["rollout_pointwise_mean"] = dpo_rollout_pointwise_sum / dpo_total_prefixes
                    log_entry["pool_top_score_mean"] = dpo_pool_top_score_sum / dpo_total_prefixes
                    log_entry["pool_bottom_score_mean"] = dpo_pool_bottom_score_sum / dpo_total_prefixes
                    if dpo_kept_prefixes > 0:
                        log_entry["chosen_is_pivot_rate"] = dpo_chosen_is_pivot / dpo_kept_prefixes
                        log_entry["rejected_is_pivot_rate"] = dpo_rejected_is_pivot / dpo_kept_prefixes
                        log_entry["chosen_is_rollout_rate"] = 1.0 - (dpo_chosen_is_pivot / dpo_kept_prefixes)
                    if dpo_micro_steps > 0:
                        log_entry["dpo_margin_avg"] = dpo_margin_sum / dpo_micro_steps
                        log_entry["dpo_chosen_beats_rejected_rate"] = dpo_acc_sum / dpo_micro_steps
                    log_entry["judge_latency_s_avg"] = judge_latency_total / max(1, step * grad_accum)
                    log_entry["gen_latency_s_avg"] = gen_latency_total / max(1, step * grad_accum)
                    log_entry["dpo_kept_prefixes"] = dpo_kept_prefixes
                    log_entry["dpo_total_prefixes"] = dpo_total_prefixes
                print(json.dumps(log_entry), flush=True)
            elif use_ddp and step % (int(cfg["train"]["log_every"]) * 10) == 0:
                print(json.dumps({
                    "rank": rank,
                    "step": step,
                    "train_loss_local": safe_mean(running[-int(cfg["train"]["log_every"]) :]),
                    "tokens_seen_local": tokens_seen,
                    "available_mem_gib": mem,
                }), flush=True)

        if is_main and (step % int(cfg["train"]["eval_every"]) == 0 or step == max_steps):
            eval_model = model.module if use_ddp else model
            val_loss, counts = evaluate(eval_model, val_loader, device)
            print(json.dumps({"step": step, "val_loss": val_loss, "chosen_counts": dict(counts)}), flush=True)

        if is_main and save_every > 0 and (step % save_every == 0 or step == max_steps):
            unwrapped = model.module if use_ddp else model
            ckpt_path = output_dir / f"step_{step}.pt"
            torch.save(unwrapped.state_dict(), ckpt_path)
            print(json.dumps({"step": step, "checkpoint": str(ckpt_path)}), flush=True)

    eval_model = model.module if use_ddp else model
    val_loss_selected, val_counts = evaluate(eval_model, val_loader, device)

    raw_cfg = dict(cfg)
    raw_cfg["train"] = dict(cfg["train"])
    raw_cfg["train"]["condition"] = "raw_ntp"
    raw_condition = "raw_chunk_ntp" if oracle_val_rows and "raw_chunk_ids" in oracle_val_rows[0] else "raw_ntp"
    raw_val = SuffixDataset(oracle_val_rows, raw_condition)
    raw_loader = DataLoader(raw_val, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, collate_fn=lambda b: pad_batch(b, pad_id))
    val_loss_raw, _ = evaluate(eval_model, raw_loader, device)

    suffixes = []
    for r in val_rows[:256]:
        if condition == "raw_ntp":
            suffixes.append(r["original_suffix_ids"])
        elif condition == "finephrase_math_ntp":
            suffixes.append(r["finephrase_suffix_ids"])
        elif condition == "rfnll_original_vs_finephrase":
            suffixes.append(r["chosen_suffix_ids"])
        elif condition == "thinking_sft":
            suffixes.append(r["thinking_suffix_ids"])
        elif condition == "interleaved_thinking_sft":
            suffixes.append(r["interleaved_thinking_ids"])
        elif condition == "raw_chunk_ntp":
            suffixes.append(r["raw_chunk_ids"])
        elif condition == "rfnll_rollout_vs_rewrite":
            suffixes.append(r["original_suffix_ids"])
        elif condition == "online_dpo_selfimproving":
            suffixes.append(r["original_suffix_ids"])
        else:
            raise ValueError(f"Unknown train.condition={condition!r}")
    rep = repetition_4gram_rate(suffixes)

    total_chosen = sum(val_counts.values())
    chosen_original = val_counts.get("original", 0) / max(1, total_chosen)
    chosen_finephrase = val_counts.get("finephrase", 0) / max(1, total_chosen)
    chosen_rollout = val_counts.get("rollout", 0) / max(1, total_chosen)
    if condition == "rfnll_rollout_vs_rewrite":
        chosen_rollout = rollout_chosen_total / max(1, rollout_total)
        chosen_finephrase = 1.0 - chosen_rollout
        chosen_original = 0.0
        judge_win_rate = chosen_rollout
        val_loss_selected = val_loss_raw
    elif condition == "online_dpo_selfimproving":
        chosen_rollout = 1.0 - (dpo_chosen_is_pivot / max(1, dpo_kept_prefixes))
        chosen_original = dpo_chosen_is_pivot / max(1, dpo_kept_prefixes)
        chosen_finephrase = 0.0
        # Paper §1.2.2 "Suffix & rewrite vs. rollouts" (p.12 Fig 8): early in
        # training the judge picks the original suffix (pivot) more often; as
        # rollouts improve the rollout pointwise score overtakes the pivot.
        # Track rollout - pivot pointwise gap as the training-dynamics signal.
        judge_win_rate = (dpo_rollout_pointwise_sum - dpo_pivot_pointwise_sum) / max(1, dpo_total_prefixes) + 0.5
        val_loss_selected = val_loss_raw
    elif condition == "rfnll_original_vs_finephrase":
        judge_win_rate = 1.0 - chosen_original
    elif condition == "finephrase_math_ntp":
        judge_win_rate = 1.0
    else:
        judge_win_rate = 0.0
    elapsed = max(1e-6, time.time() - start)
    tok_per_sec = tokens_seen / elapsed

    metrics = {
        "primary_metric": val_loss_selected,
        "val_loss_raw": val_loss_raw,
        "val_loss_selected": val_loss_selected,
        "judge_win_rate_vs_raw": judge_win_rate,
        "chosen_original_rate": chosen_original,
        "chosen_finephrase_rate": chosen_finephrase,
        "chosen_rollout_rate": chosen_rollout,
        "repetition_4gram_rate": rep,
        "tokens_seen": tokens_seen,
        "tok_per_sec": tok_per_sec,
        "available_mem_gib_min": mem_min,
        "world_size": world_size,
        "rank": rank,
    }
    if condition == "online_dpo_selfimproving":
        metrics["dpo_beta"] = dpo_beta
        metrics["num_rollouts_per_prefix"] = num_rollouts
        metrics["judge_mode"] = "full_pairwise"
        metrics["pool_size"] = num_rollouts + 1
        metrics["pairs_per_prefix"] = (num_rollouts + 1) * num_rollouts // 2
        metrics["dpo_kept_prefixes"] = dpo_kept_prefixes
        metrics["dpo_total_prefixes"] = dpo_total_prefixes
        metrics["chosen_is_pivot_rate"] = chosen_original
        metrics["chosen_is_rollout_rate"] = chosen_rollout
        metrics["dpo_rejected_is_pivot_rate"] = dpo_rejected_is_pivot / max(1, dpo_kept_prefixes)
        metrics["pivot_pointwise_mean"] = dpo_pivot_pointwise_sum / max(1, dpo_total_prefixes)
        metrics["rollout_pointwise_mean"] = dpo_rollout_pointwise_sum / max(1, dpo_total_prefixes)
        metrics["pool_top_score_mean"] = dpo_pool_top_score_sum / max(1, dpo_total_prefixes)
        metrics["pool_bottom_score_mean"] = dpo_pool_bottom_score_sum / max(1, dpo_total_prefixes)
        metrics["rollout_minus_pivot_gap"] = (dpo_rollout_pointwise_sum - dpo_pivot_pointwise_sum) / max(1, dpo_total_prefixes)
        metrics["dpo_margin_avg"] = dpo_margin_sum / max(1, dpo_micro_steps)
        metrics["dpo_chosen_beats_rejected_rate"] = dpo_acc_sum / max(1, dpo_micro_steps)
    if is_main:
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        if save_every > 0:
            unwrapped = model.module if use_ddp else model
            torch.save(unwrapped.state_dict(), output_dir / "final.pt")
        for key, value in metrics.items():
            emit_metric(key, value)

    if use_ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
