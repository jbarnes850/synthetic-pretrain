# Spark-Native Synthetic Pretraining RF-NLL Program

## Project

Spark-native controlled analogue of arXiv 2601.21343, Section 1:
train a Qwen-style decoder-only policy model from scratch on prefix-conditioned
suffix generation. Compare standard raw next-token pretraining against
FinePhrase-Math rewrites and reward-filtered NLL target selection.

This program treats previous heuristic/parallel sweeps as infrastructure
validation only. Blog-grade claims require true FineWeb-Edu raw data,
FinePhrase-Math rewrite data, real judge-derived RF-NLL labels, and equal-token
comparisons.

## Paper-Aligned Method

The paper frames pretraining as sequence learning: split streaming pretraining
text into a prefix and a fixed-length suffix, then train the policy to generate
the highest-quality suffix candidate. For RF-NLL, choose the highest-scoring
candidate and run an NLL update on that completion.

Spark-scale candidates:

1. `original`: source suffix following the prefix.
2. `finephrase`: FinePhrase-Math generated rewrite/continuation.
3. `rollout`: current-policy rollout, enabled only after minimal coherence.

Phase 0 and Phase 1 use candidates 1 and 2 only. Rollouts are explicitly gated
because early policy rollouts are expected to be low quality.

## Editable Surface

The agent may modify only:

- `configs/*.yaml`: condition, model size, learning rate, batch size, sequence
  length, suffix length, data limits, judge settings, run budget.
- `scripts/materialize_dataset.py`: data contract/provenance/chunking bugs.
- `scripts/select_targets.py`: judge scoring and RF-NLL target selection.
- `scripts/train.py`: objective, scratch init verification, metrics/logging.
- `scripts/run_experiment.sh`, `scripts/run_validation.sh`: execution gates.
- `prompts/judge_quality.txt`: prompt wording/version.

Everything else is fixed oracle unless a validation gate proves it is broken.

## Fixed Oracle

- Tokenizer and config-shape reference: Qwen cache under `tokenizer_repo_cache`
  and `config_repo_cache`.
- Train/validation split: deterministic hash of materialized example id.
- Raw baseline source: FineWeb-Edu `sample/10BT` parquet shards.
- Rewrite/source pairs: FinePhrase-Math parquet shards.
- Evaluation metric names and parser.
- Dataset provenance fields: `source_sha` and `finephrase_sha`.
- Canonical ledger: `/home/jarrodbarnes/synthetic-pretrain/results.json` on
  cfd0.

## Data Contract

Materialized examples must contain exactly:

```json
{
  "id": "...",
  "prefix_ids": [],
  "original_suffix_ids": [],
  "finephrase_suffix_ids": [],
  "split": "train",
  "source_sha": "...",
  "finephrase_sha": "..."
}
```

Raw FineWeb-Edu examples use the same schema with an empty
`finephrase_suffix_ids` list and empty `finephrase_sha`.

Judge-selected examples must contain:

```json
{
  "id": "...",
  "scores": {"original": 0.0, "finephrase": 1.0, "rollout": 0.0},
  "chosen": "finephrase",
  "judge_model": "qwen-judge",
  "prompt_version": "quality_v1"
}
```

Training artifacts may include extra operational fields such as
`prefix_ids`, `chosen_suffix_ids`, and judge sampling metadata.

## Core Conditions

All real conditions run at equal token budget and scratch initialization.

1. `raw_ntp`: train on true FineWeb-Edu raw/source shards.
2. `finephrase_math_ntp`: train on FinePhrase-Math generated suffixes.
3. `rfnll_original_vs_finephrase`: judge original vs FinePhrase suffixes,
   choose the higher-scoring candidate, train NLL on the chosen suffix.
4. `rfnll_original_vs_finephrase_vs_rollout`: add current-policy rollout only
   after the first three conditions are stable and coherence passes.

## Metrics

Every real run must emit these exact machine-readable lines and write the same
keys to `metrics.json`:

```text
primary_metric: <float>
val_loss_raw: <float>
val_loss_selected: <float>
judge_win_rate_vs_raw: <float>
chosen_original_rate: <float>
chosen_finephrase_rate: <float>
chosen_rollout_rate: <float>
repetition_4gram_rate: <float>
tokens_seen: <int>
tok_per_sec: <float>
available_mem_gib_min: <float>
```

Primary metric is `val_loss_selected` lower-is-better. Raw FineWeb-Edu
validation loss is the anti-regression anchor.

## Keep/Discard

Keep if:

1. `val_loss_selected` improves and `val_loss_raw` regresses by <=5%.
2. Or `judge_win_rate_vs_raw` improves by >=3 absolute points and
   `val_loss_raw` regresses by <=5%.

Discard if:

- Required metrics are missing or nonfinite.
- `available_mem_gib_min` drops below 4 GiB.
- `val_loss_raw` regresses by >5%.
- `repetition_4gram_rate` materially worsens.
- Rollout-chosen rate rises while judged quality falls.
- No explicit keep rule is satisfied.

Crashes are logged as `crash`, never silently converted to discard.

## Validation Ladder

No training run may start until upstream gates pass in order:

1. `py_compile` all scripts/tools.
2. `ruff check`.
3. 128-example materialization from real FinePhrase-Math and FineWeb-Edu
   caches.
4. Tokenizer round-trip and suffix-alignment/data-contract checks.
5. Judge prompt/parser dry run for 16 examples; real judge scoring dry run when
   an endpoint is available.
6. CPU/no-CUDA trainer dry run if feasible.
7. Spark runtime CUDA/BF16/SDPA probe.
8. Single-GPU 10-step Spark smoke.
9. 1k-step `raw_ntp`.
10. 1k-step `finephrase_math_ntp`.
11. 1k-step `rfnll_original_vs_finephrase`.

Only after the first three real 1k-step runs complete may the autoresearch loop
or parallel sweeps begin.

## Spark Layout

- cfd0: `jarrodbarnes@100.113.207.120`, direct `192.168.100.11`; default
  controller/data/training node.
- f7e2: `jarrodbarnes@100.70.91.108`, direct `192.168.100.10`; preferred judge
  serving node when free.
- Runtime image: `scitrera/dgx-spark-sglang:0.5.9-t5`.
- Use CUDA 13. Do not force CUDA 12.4 or Nanotron assumptions.
- Prefer direct Spark interconnect for bulk transfer and multi-node serving.

Do not launch training while f7e2 has active GRPO/SGLang jobs and cfd0 is
simultaneously serving Qwen with <20 GiB available memory.

## Search Strategy

Mode is supervised until the first three real 1k-step conditions complete.
Afterward, use sequential hill-climbing. Use small parallel sweeps only after
baseline trio stability is proven.

Search order:

1. Judge prompt/model/repeat count.
2. RF-NLL target scoring policy.
3. FinePhrase/raw mixture ablations.
4. Model size/depth.
5. Learning rate and warmup.
6. Rollout candidate gate.

Stop-loss: pause after 5 consecutive discards or any memory floor breach.

## Commands

Non-GPU gates:

```bash
RUN_HARDWARE_PROBE=0 RUN_GPU_SMOKE=0 scripts/run_validation.sh
```

Hardware probe and 10-step smoke:

```bash
RUN_GPU_SMOKE=1 scripts/run_validation.sh
```

First real comparison:

```bash
scripts/run_experiment.sh configs/baseline_raw.yaml
scripts/run_experiment.sh configs/baseline_finephrase_math.yaml
JUDGE_ENDPOINT=http://127.0.0.1:30000 JUDGE_MODEL=qwen-judge \
  scripts/run_experiment.sh configs/rfnll_math.yaml
```

Metrics are discoverable in both `logs/<run_id>.log` and
`outputs/<condition>/train.log`.
