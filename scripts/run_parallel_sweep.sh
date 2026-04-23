#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p configs/generated logs queue/pending queue/running queue/done queue/failed locks workers

cat > configs/generated/parallel_lr1e3_depth4.yaml <<'EOF'
include: configs/rfnll_math.yaml

train:
  condition: rfnll_original_vs_finephrase
  output_dir: outputs/parallel_lr1e3_depth4
  max_steps: 200
  eval_every: 50
  batch_size: 4
  grad_accum_steps: 4
  learning_rate: 0.001
  warmup_steps: 40

model:
  num_hidden_layers: 4
EOF

cat > configs/generated/parallel_lr8e4_depth8.yaml <<'EOF'
include: configs/rfnll_math.yaml

train:
  condition: rfnll_original_vs_finephrase
  output_dir: outputs/parallel_lr8e4_depth8
  max_steps: 200
  eval_every: 50
  batch_size: 4
  grad_accum_steps: 4
  learning_rate: 0.0008
  warmup_steps: 40

model:
  num_hidden_layers: 8
EOF

PYTHONUNBUFFERED=1 python3 scripts/spark_orchestrator.py run-pair \
  --config-a configs/generated/parallel_lr1e3_depth4.yaml \
  --config-b configs/generated/parallel_lr8e4_depth8.yaml
