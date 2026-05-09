#!/usr/bin/env python3
"""
Online DPO for self-improving continual pretraining.

Paper: Tan et al. 2026, "Self-Improving Pretraining..." (arxiv 2601.21343).
Champion recipe (Table 1): continual pretraining + Online DPO + K=16 rollouts,
with pivot judging against the original suffix (page 10 variant).

Per training step, for each prefix:
  1. Generate K rollouts from the current policy.
  2. Judge the candidate pool full-pairwise with position randomization.
     Mainline pool = original suffix + K rollouts. Rewrite ablation pool =
     original suffix + teacher rewrite + K rollouts.
  3. Chosen = argmax score, rejected = argmin score. Skip if all equal.
  4. Forward the frozen reference on chosen+rejected to get suffix log-probs.
  5. Return a batch ready for compute_dpo_loss().

The training-loop forward through the policy still runs inside DDP so grads
flow. We pre-compute reference log-probs here so the outer loop only touches
the policy model for backward.
"""
from __future__ import annotations

import copy
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from self_improving import build_prompt, judge_pairwise_batch


def load_ref_model(policy_model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Deep-copy the policy weights into a frozen reference on the same device.

    Call this AFTER the warm-start checkpoint has been loaded into the policy
    and BEFORE wrapping in DDP. The returned module has requires_grad=False
    on every parameter and is in eval() mode.
    """
    ref = copy.deepcopy(policy_model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.to(device)
    return ref


@torch.no_grad()
def generate_k_rollouts(
    model: torch.nn.Module,
    tokenizer,
    prefix_ids_list: list[list[int]],
    K: int,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[list[list[int]]]:
    """Return rollouts[i][k] = token-id list of length max_new_tokens.

    Uses num_return_sequences=K inside a single batched generate. Left-pads
    prefixes so generated suffixes start at the same absolute position.
    """
    was_training = model.training
    model.eval()
    try:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        max_len = max(len(p) for p in prefix_ids_list)
        padded, attn = [], []
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
            num_return_sequences=K,
            pad_token_id=pad_id,
        )
        # out shape: [batch * K, max_len + max_new_tokens]
        bsz = len(prefix_ids_list)
        rollouts: list[list[list[int]]] = [[] for _ in range(bsz)]
        for i in range(out.size(0)):
            full = out[i].tolist()
            gen = full[max_len : max_len + max_new_tokens]
            if len(gen) < max_new_tokens:
                gen = gen + [pad_id] * (max_new_tokens - len(gen))
            rollouts[i // K].append(gen)
        return rollouts
    finally:
        if was_training:
            model.train()


def judge_full_pairwise_scores(
    prefix_ids_list: list[list[int]],
    candidates_ids_list: list[list[list[int]]],
    tokenizer,
    prompt_template: str,
    judge_endpoint: str,
    judge_model: str,
    judge_temperature: float,
    judge_top_p: float,
    judge_max_tokens: int,
    judge_max_workers: int,
    judge_repeats: int,
    step: int,
    rank: int = 0,
) -> tuple[list[list[float]], int]:
    """Full pairwise judging over N candidates per prefix (paper §1.2.2 p.8).

    For each prefix i, runs every unordered pair (a, b) with a < b in the
    N-element candidate pool through the pairwise judge with per-pair position
    randomization. Each pair can be judged multiple times; pointwise score of
    candidate c = mean win fraction across its N - 1 comparisons. All prefixes'
    repeated pair prompts are fired in a single concurrent batch.

    Returns (scores[i][c], total_comparisons). scores[i][c] in [0, 1].
    """
    bsz = len(prefix_ids_list)
    if bsz == 0:
        return [], 0
    N = len(candidates_ids_list[0])
    num_pairs_per_prefix = N * (N - 1) // 2

    rng = random.Random(step * 1_000_003 + rank * 2_999 + 17)
    prompts: list[str] = []
    swaps: list[bool] = []
    pair_index: list[tuple[int, int, int]] = []  # (prefix_i, cand_a_idx, cand_b_idx)

    for i in range(bsz):
        prefix_text = tokenizer.decode(prefix_ids_list[i], skip_special_tokens=True)
        cand_texts = [
            tokenizer.decode(c, skip_special_tokens=True)
            for c in candidates_ids_list[i]
        ]
        for a in range(N):
            for b in range(a + 1, N):
                swap = rng.random() < 0.5
                swaps.append(swap)
                pair_index.append((i, a, b))
                if swap:
                    # A = cand b, B = cand a
                    prompts.append(build_prompt(prompt_template, prefix_text, cand_texts[b], cand_texts[a]))
                else:
                    # A = cand a, B = cand b
                    prompts.append(build_prompt(prompt_template, prefix_text, cand_texts[a], cand_texts[b]))

    vote_rows = judge_pairwise_batch(
        prompts, judge_endpoint, judge_model,
        temperature=judge_temperature, top_p=judge_top_p,
        max_tokens=judge_max_tokens, max_workers=judge_max_workers,
        repeats=judge_repeats, return_vote_counts=True,
    )

    # Tally wins per candidate per prefix.
    wins = [[0.0 for _ in range(N)] for _ in range(bsz)]
    for idx, (i, a, b) in enumerate(pair_index):
        sw = swaps[idx]
        vote = vote_rows[idx]
        a_votes = int(vote["a_votes"])
        b_votes = int(vote["b_votes"])
        repeats = max(1, int(vote["repeats"]))
        a_fraction_in_prompt = a_votes / repeats
        # Recover which candidate the A-vote fraction refers to.
        if sw:
            # Prompt A was candidate b, prompt B was candidate a.
            a_fraction = b_votes / repeats
        else:
            # Prompt A was candidate a, prompt B was candidate b.
            a_fraction = a_fraction_in_prompt
        wins[i][a] += a_fraction
        wins[i][b] += 1.0 - a_fraction

    denom = max(1, N - 1)
    scores = [[wins[i][c] / denom for c in range(N)] for i in range(bsz)]
    return scores, bsz * num_pairs_per_prefix


def _select_chosen_rejected(
    all_scores: list[float],
    step: int,
    prefix_idx: int,
) -> tuple[int, int] | None:
    """Select (chosen_idx, rejected_idx) over a unified candidate pool.

    Paper p.4: chosen = highest-scoring, rejected = lowest-scoring.
    With N candidates and pointwise scores in k/(N-1), ties at max/min
    are common — break them deterministically via a step-seeded rotation
    of the iteration order so:
      (a) different positions get picked across steps (no positional bias
          accumulating), and
      (b) the selection is reproducible under a fixed global seed.

    Returns None if every candidate has the exact same score — no DPO signal.
    """
    N = len(all_scores)
    if N < 2:
        return None
    smax = max(all_scores)
    smin = min(all_scores)
    if smax == smin:
        return None
    seed = (step * 97 + prefix_idx * 13) % N
    order = [(i + seed) % N for i in range(N)]
    chosen = next(k for k in order if all_scores[k] == smax)
    rejected = next(k for k in order if all_scores[k] == smin)
    return chosen, rejected


def _pad_stack(
    prefix: list[int],
    suffix: list[int],
    max_len: int,
    pad_id: int,
) -> tuple[list[int], list[int], list[int]]:
    ids = list(prefix) + list(suffix)
    labels = [-100] * len(prefix) + list(suffix)
    attn = [1] * len(ids)
    pad = max_len - len(ids)
    return ids + [pad_id] * pad, labels + [-100] * pad, attn + [0] * pad


def _suffix_logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Shift-by-one causal LM sum of log-probs over labels != -100.

    logits: [B, T, V]. labels: [B, T] with -100 for prefix+pad.
    Returns [B] sum of log P_model(y_t | y_<t) over suffix positions.
    """
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    logp = F.log_softmax(shifted_logits.float(), dim=-1)
    gathered = logp.gather(-1, shifted_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (shifted_labels != -100).float()
    return (gathered * mask).sum(dim=-1)


@torch.no_grad()
def _ref_logprobs_stacked(
    ref_model: torch.nn.Module,
    chosen_input_ids: torch.Tensor,
    chosen_attn: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attn: torch.Tensor,
    rejected_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One stacked forward through the frozen reference; returns (lp_chosen, lp_rejected)."""
    stacked_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
    stacked_attn = torch.cat([chosen_attn, rejected_attn], dim=0)
    stacked_labels = torch.cat([chosen_labels, rejected_labels], dim=0)
    out = ref_model(input_ids=stacked_ids, attention_mask=stacked_attn)
    logp = _suffix_logprobs_from_logits(out.logits, stacked_labels)
    n = chosen_input_ids.size(0)
    return logp[:n], logp[n:]


def compute_dpo_loss(
    policy_logp_chosen: torch.Tensor,
    policy_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Standard DPO loss with detached reference log-probs.

    loss = -mean log sigmoid(beta * ((lp_pi_c - lp_ref_c) - (lp_pi_r - lp_ref_r)))
    """
    pi_logratios = policy_logp_chosen - policy_logp_rejected
    ref_logratios = ref_logp_chosen.detach() - ref_logp_rejected.detach()
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    with torch.no_grad():
        margin = float(logits.mean().detach().cpu())
        acc = float((logits > 0).float().mean().detach().cpu())
    return loss, {"dpo_margin": margin, "dpo_chosen_beats_rejected_rate": acc}


def build_online_dpo_batch(
    raw_batch: dict[str, Any],
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer,
    prompt_template: str,
    judge_endpoint: str,
    judge_model: str,
    judge_temperature: float,
    judge_top_p: float,
    judge_max_tokens: int,
    judge_max_workers: int,
    judge_repeats: int,
    max_new_tokens: int,
    num_rollouts: int,
    pad_id: int,
    device: torch.device,
    step: int,
    rollout_temperature: float = 1.0,
    rollout_top_p: float = 1.0,
    rank: int = 0,
    include_rewrite_candidate: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | tuple[None, dict[str, Any]]:
    """Rollouts -> pivot judge -> select chosen/rejected -> ref log-probs.

    Returns ({
        "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
        "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
        "ref_logp_chosen", "ref_logp_rejected",
    }, stats) or (None, stats) if every prefix ended in a tie.
    """
    prefix_ids = raw_batch["prefix_ids"]
    original_ids = raw_batch["original_suffix_ids"]
    rewrite_ids = raw_batch.get("rewrite_suffix_ids", [[] for _ in prefix_ids])
    bsz = len(prefix_ids)

    # 1. K rollouts from the current policy
    t0 = time.time()
    rollouts = generate_k_rollouts(
        policy_model, tokenizer, prefix_ids, num_rollouts, max_new_tokens, device,
        temperature=rollout_temperature, top_p=rollout_top_p,
    )
    gen_latency_s = time.time() - t0

    # 2. Build the unified candidate pool. Mainline uses the paper branch
    #    "suffix vs K rollouts"; the rewrite ablation uses
    #    "suffix + rewrite vs K rollouts" with the same full-pairwise judge.
    candidates_per_prefix: list[list[list[int]]] = []
    labels_per_prefix: list[list[str]] = []
    rewrite_available_count = 0
    for i in range(bsz):
        candidates = [original_ids[i]]
        labels = ["original"]
        if include_rewrite_candidate and rewrite_ids[i]:
            candidates.append(rewrite_ids[i])
            labels.append("rewrite")
            rewrite_available_count += 1
        elif include_rewrite_candidate:
            raise RuntimeError(
                "include_rewrite_candidate=True requires rewrite_suffix_ids for every row; "
                "run scripts/build_rewrite_data.py before the rewrite ablation."
            )
        candidates.extend(rollouts[i])
        labels.extend(["rollout"] * len(rollouts[i]))
        candidates_per_prefix.append(candidates)
        labels_per_prefix.append(labels)
    N = len(candidates_per_prefix[0])
    if any(len(candidates) != N for candidates in candidates_per_prefix):
        raise RuntimeError("All prefixes in a DPO batch must have the same candidate-pool size")

    t1 = time.time()
    pointwise_scores, total_pairs = judge_full_pairwise_scores(
        prefix_ids, candidates_per_prefix, tokenizer, prompt_template,
        judge_endpoint, judge_model, judge_temperature, judge_top_p,
        judge_max_tokens, judge_max_workers, judge_repeats, step, rank=rank,
    )
    judge_latency_s = time.time() - t1

    # 3. Select chosen/rejected over the full pool
    kept_prefixes: list[list[int]] = []
    kept_chosen: list[list[int]] = []
    kept_rejected: list[list[int]] = []
    chosen_is_pivot = 0
    rejected_is_pivot = 0
    chosen_is_rewrite = 0
    rejected_is_rewrite = 0
    chosen_is_rollout = 0
    rejected_is_rollout = 0
    pivot_score_sum = 0.0
    rewrite_score_sum = 0.0
    rollout_score_sum = 0.0  # averaged across K rollouts per prefix
    pool_score_max_sum = 0.0
    pool_score_min_sum = 0.0
    candidate_audit_records: list[dict[str, Any]] = []
    for i in range(bsz):
        scores_i = pointwise_scores[i]
        pivot_score_sum += scores_i[0]
        rollout_scores = [score for score, label in zip(scores_i, labels_per_prefix[i]) if label == "rollout"]
        rollout_score_sum += sum(rollout_scores) / max(1, len(rollout_scores))
        if "rewrite" in labels_per_prefix[i]:
            rewrite_score_sum += scores_i[labels_per_prefix[i].index("rewrite")]
        pool_score_max_sum += max(scores_i)
        pool_score_min_sum += min(scores_i)
        pair = _select_chosen_rejected(scores_i, step, i)
        if pair is None:
            candidate_audit_records.append(
                {
                    "step": step,
                    "rank": rank,
                    "prefix_index": i,
                    "candidate_labels": labels_per_prefix[i],
                    "candidate_scores": scores_i,
                    "chosen_index": None,
                    "rejected_index": None,
                    "chosen_label": None,
                    "rejected_label": None,
                    "tie": True,
                }
            )
            continue
        c_idx, r_idx = pair
        kept_prefixes.append(prefix_ids[i])
        kept_chosen.append(candidates_per_prefix[i][c_idx])
        kept_rejected.append(candidates_per_prefix[i][r_idx])
        chosen_label = labels_per_prefix[i][c_idx]
        rejected_label = labels_per_prefix[i][r_idx]
        if chosen_label == "original":
            chosen_is_pivot += 1
        elif chosen_label == "rewrite":
            chosen_is_rewrite += 1
        elif chosen_label == "rollout":
            chosen_is_rollout += 1
        if rejected_label == "original":
            rejected_is_pivot += 1
        elif rejected_label == "rewrite":
            rejected_is_rewrite += 1
        elif rejected_label == "rollout":
            rejected_is_rollout += 1
        candidate_audit_records.append(
            {
                "step": step,
                "rank": rank,
                "prefix_index": i,
                "candidate_labels": labels_per_prefix[i],
                "candidate_scores": scores_i,
                "chosen_index": c_idx,
                "rejected_index": r_idx,
                "chosen_label": chosen_label,
                "rejected_label": rejected_label,
                "tie": False,
            }
        )

    stats: dict[str, Any] = {
        "total": bsz,
        "kept": len(kept_prefixes),
        "chosen_is_pivot_count": chosen_is_pivot,
        "rejected_is_pivot_count": rejected_is_pivot,
        "chosen_is_rewrite_count": chosen_is_rewrite,
        "rejected_is_rewrite_count": rejected_is_rewrite,
        "chosen_is_rollout_count": chosen_is_rollout,
        "rejected_is_rollout_count": rejected_is_rollout,
        "judge_latency_s": judge_latency_s,
        "gen_latency_s": gen_latency_s,
        "num_rollouts_per_prefix": num_rollouts,
        "pool_size": N,
        "include_rewrite_candidate": include_rewrite_candidate,
        "rewrite_available_count": rewrite_available_count,
        "pairs_per_prefix": total_pairs // max(1, bsz),
        "total_judge_pairs": total_pairs,
        "judge_repeats": judge_repeats,
        "total_judge_calls": total_pairs * judge_repeats,
        "pivot_pointwise_mean": pivot_score_sum / max(1, bsz),
        "rewrite_pointwise_mean": rewrite_score_sum / max(1, rewrite_available_count),
        "rollout_pointwise_mean": rollout_score_sum / max(1, bsz),
        "pool_top_score_mean": pool_score_max_sum / max(1, bsz),
        "pool_bottom_score_mean": pool_score_min_sum / max(1, bsz),
        "original_suffix_len_mean": sum(len(o) for o in original_ids) / max(1, len(original_ids)),
        "rewrite_suffix_len_mean": sum(len(r) for r in rewrite_ids if r) / max(1, rewrite_available_count),
        "candidate_audit_records": candidate_audit_records,
    }

    if not kept_prefixes:
        return None, stats

    # 4. build chosen/rejected tensors (pad to shared max across both sides)
    chosen_lens = [len(p) + len(c) for p, c in zip(kept_prefixes, kept_chosen)]
    rejected_lens = [len(p) + len(r) for p, r in zip(kept_prefixes, kept_rejected)]
    max_len = max(max(chosen_lens), max(rejected_lens))

    c_ids, c_lbl, c_attn = [], [], []
    r_ids, r_lbl, r_attn = [], [], []
    for p, c, r in zip(kept_prefixes, kept_chosen, kept_rejected):
        ci, cl, ca = _pad_stack(p, c, max_len, pad_id)
        ri, rl, ra = _pad_stack(p, r, max_len, pad_id)
        c_ids.append(ci)
        c_lbl.append(cl)
        c_attn.append(ca)
        r_ids.append(ri)
        r_lbl.append(rl)
        r_attn.append(ra)

    chosen_input_ids = torch.tensor(c_ids, dtype=torch.long, device=device)
    chosen_labels = torch.tensor(c_lbl, dtype=torch.long, device=device)
    chosen_attn = torch.tensor(c_attn, dtype=torch.long, device=device)
    rejected_input_ids = torch.tensor(r_ids, dtype=torch.long, device=device)
    rejected_labels = torch.tensor(r_lbl, dtype=torch.long, device=device)
    rejected_attn = torch.tensor(r_attn, dtype=torch.long, device=device)

    # 5. reference log-probs (no grad, one stacked forward)
    ref_logp_chosen, ref_logp_rejected = _ref_logprobs_stacked(
        ref_model,
        chosen_input_ids, chosen_attn, chosen_labels,
        rejected_input_ids, rejected_attn, rejected_labels,
    )

    batch = {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attn,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attn,
        "rejected_labels": rejected_labels,
        "ref_logp_chosen": ref_logp_chosen.detach(),
        "ref_logp_rejected": ref_logp_rejected.detach(),
    }
    return batch, stats


def policy_suffix_logprobs(
    policy_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Same shift-by-one as _suffix_logprobs but keeps grad for the policy."""
    out = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    tgt = labels[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    gathered = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (tgt != -100).float()
    return (gathered * mask).sum(dim=-1)
