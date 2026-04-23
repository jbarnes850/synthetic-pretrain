# Training Metrics & Logs Access Schema

How to read what's happening inside a `train.py` run without attaching to the container.

## File locations (per-arm)

On spark-f7e2 (or whichever host runs the trainer), every arm writes to `~/synthetic-pretrain/outputs/<arm_name>/`:

```
outputs/<arm_name>/
├── train.log              # JSONL stream of per-step metrics (tee'd stdout)
├── resolved_config.json   # Post-include yaml snapshot (what the run actually used)
├── step_<N>.pt            # Checkpoints at every save_every steps
└── final.pt               # Present only when run completes cleanly (step == max_steps)
```

Phase 3 current arm: `outputs/phase3_dpo_qwen3_06b/`.

## Log format

`train.log` is **JSON-Lines with a non-JSON header**. Line 1 is a torch/cuda FutureWarning from the container. Every subsequent line is one JSON object. Parsers must skip non-JSON lines:

```bash
tail -n 200 train.log | grep '^{' | jq .
```

Two record types interleave, distinguished by which keys are present:

| Cadence | Trigger | Distinguishing key |
|---------|---------|--------------------|
| Train step | every `log_every` steps | `train_loss` |
| Val step   | every `eval_every` steps | `val_loss` |

## Train step schema

Emitted by the loop at step boundaries. All values numeric unless noted.

| Key | Type | Meaning |
|-----|------|---------|
| `step` | int | Optimizer step index (not tokens) |
| `train_loss` | float | DPO loss for `online_dpo_*` conditions, NTP loss for NTP arms |
| `tokens_seen` | int | Cumulative training tokens (batch × grad_accum × seq_len × step) |
| `tok_per_sec` | float | Throughput over the last logging window |
| `available_mem_gib` | float | Free host RAM as measured inside the container (GB1 = 1.073 GiB). Watch for sustained drop below 10 |

### DPO / self-improving arms only

These keys exist when `condition: online_dpo_selfimproving`. Core paper Fig 8 reproduction lives here.

| Key | Type | Meaning |
|-----|------|---------|
| `dpo_margin_avg` | float | Mean (chosen_logp - rejected_logp) - (ref_chosen_logp - ref_rejected_logp). Positive = policy prefers chosen over ref |
| `dpo_chosen_beats_rejected_rate` | float | Fraction of prefixes where policy scores chosen > rejected |
| `dpo_kept_prefixes` | int | Prefixes that survived filtering this log window |
| `dpo_total_prefixes` | int | Prefixes proposed this window. `kept / total` = acceptance rate |

### Judge pool stats (per prefix: {pivot + K=16 rollouts}, 136 pairwise calls)

| Key | Type | Meaning |
|-----|------|---------|
| `pivot_pointwise_mean` | float [0,1] | Mean pivot pointwise score = wins/(N-1). Baseline at step 1 Phase 3 = 0.742 |
| `rollout_pointwise_mean` | float [0,1] | Mean rollout pointwise score. Baseline at step 1 Phase 3 = 0.485 |
| `pool_top_score_mean` | float [0,1] | Best candidate's pointwise score per prefix, averaged. Upper-bound signal |
| `pool_bottom_score_mean` | float [0,1] | Worst candidate's pointwise score. Lower-bound signal |
| `chosen_is_pivot_rate` | float [0,1] | Fraction of training pairs where the chosen = pivot |
| `chosen_is_rollout_rate` | float [0,1] | **Paper Fig 8 signature.** Climbs as rollouts start winning. Phase 3 step 1 = 0.625, step 950 = 0.713 |
| `rejected_is_pivot_rate` | float [0,1] | Fraction of training pairs where rejected = pivot. Typically near 0 when mechanism fires |

### Latency

| Key | Type | Meaning |
|-----|------|---------|
| `judge_latency_s_avg` | float | Average wallclock per judge call over the log window |
| `gen_latency_s_avg` | float | Average wallclock per rollout generation over the log window |

## Val step schema

Emitted every `eval_every` steps (Phase 3: every 500).

| Key | Type | Meaning |
|-----|------|---------|
| `step` | int | Same optimizer index as train steps |
| `val_loss` | float | Held-out NLL on the val split (frozen across all arms from Phase 1 materialize) |
| `chosen_counts` | object | `{"original": N}` — number of val examples; stable indicator that val set loader didn't silently truncate |

## Quick-access commands

### One-line current status
```bash
ssh spark "tail -n 1 ~/synthetic-pretrain/outputs/phase3_dpo_qwen3_06b/train.log"
```

### Latest train step (skip warning line)
```bash
ssh spark "grep '^{' ~/synthetic-pretrain/outputs/phase3_dpo_qwen3_06b/train.log | tail -n 1" | jq .
```

### Latest val step
```bash
ssh spark "grep 'val_loss' ~/synthetic-pretrain/outputs/phase3_dpo_qwen3_06b/train.log | tail -n 1" | jq .
```

### Per-step trajectory (chosen_is_rollout_rate over time — paper Fig 8)
```bash
ssh spark "grep '^{' ~/synthetic-pretrain/outputs/phase3_dpo_qwen3_06b/train.log" | \
  jq -r 'select(.chosen_is_rollout_rate) | [.step, .chosen_is_rollout_rate] | @tsv'
```

### Has the run finished?
```bash
ssh spark "ls ~/synthetic-pretrain/outputs/phase3_dpo_qwen3_06b/final.pt 2>&1"
# PHASE3_FINAL_PT_PRESENT  → step == max_steps, ready to eval
# No such file             → still running or died
```

### Is the container still up?
```bash
ssh spark "docker ps --format '{{.Names}} {{.Status}}' | grep phase3-train"
```

### Is the judge reachable?
```bash
ssh spark "ssh jarrodbarnes@192.168.100.11 'curl -sf http://192.168.100.11:30000/v1/models | grep -q qwen-judge && echo JUDGE_LIVE || echo JUDGE_DEAD'"
```

### Resolved config (what this run actually used, post-include)
```bash
ssh spark "cat ~/synthetic-pretrain/outputs/phase3_dpo_qwen3_06b/resolved_config.json" | jq .
```

### Download a checkpoint for local eval
```bash
scp spark:~/synthetic-pretrain/outputs/phase3_dpo_qwen3_06b/step_500.pt ~/Downloads/
```

## Alert thresholds (bound to monitoring cron)

These are the conditions that trigger `>>> ALERT:` from the monitoring cron. Mirror them in any ad-hoc review.

| Condition | Interpretation |
|-----------|----------------|
| `final.pt` appears | Training complete → ready to evaluate |
| `phase3-train` missing from `docker ps` | Training died |
| `phase3-judge` missing on cfd0 | Judge died |
| Judge endpoint 4xx/5xx | Mechanism halts (no pairs → no DPO update) |
| Non-finite `train_loss` (NaN/Inf) | Numerical blowup — kill and restart from last checkpoint |
| `available_mem_gib < 10` on f7e2 or `< 5` on cfd0 | UMA OOM risk — GB10's shared memory between CPU/GPU makes this lethal |
| `pool_top_score_mean - pool_bottom_score_mean < 0.3` at step > 50 | Judge no longer discriminating — prompt drift or model collapse in the served judge |
| `chosen_is_rollout_rate < 0.40` at step > 200 | Rollouts regressed below Phase 3 step-1 baseline (0.625). Mechanism stalled |
| `pivot_pointwise_mean - rollout_pointwise_mean > 0.4` at step > 500 | Rollouts plateaued far below pivot. Phase 3 target: gap closes over training |

## Schema provenance

Keys are emitted by `scripts/train.py` and `scripts/online_dpo.py`. Schema is defined by where these modules call `log_metrics()`. If a key appears here but not in the log (or vice versa), the code was edited; re-derive from the emitter.
