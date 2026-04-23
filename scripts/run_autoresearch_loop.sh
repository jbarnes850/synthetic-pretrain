#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs configs/generated

echo "autoresearch: start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "autoresearch: project=${PROJECT_DIR}"

if [ ! -d .git ]; then
  git init
  git add .
  git commit -m "initialize spark hybrid rfnll scaffold" || true
fi

scripts/run_validation.sh

BASELINE_CONFIGS=(
  "configs/baseline_raw.yaml"
  "configs/baseline_finephrase_math.yaml"
  "configs/rfnll_math.yaml"
)

for cfg in "${BASELINE_CONFIGS[@]}"; do
  echo "autoresearch: baseline ${cfg}"
  start=$(date +%s)
  status=keep
  if ! scripts/run_experiment.sh "${cfg}"; then
    status=crash
  fi
  end=$(date +%s)
  outdir=$(python3 - "${cfg}" <<'PY'
import re, sys
text=open(sys.argv[1]).read()
m=re.search(r'output_dir:\s*(\S+)', text)
print(m.group(1) if m else 'outputs/unknown')
PY
)
  metrics_file="${outdir}/metrics.json"
  metrics="$(cat "${metrics_file}" 2>/dev/null || echo '{"primary_metric": 999999}')"
  python3 tools/experiment_log.py log \
    --commit "$(git rev-parse --short HEAD 2>/dev/null || echo nogit)" \
    --metrics "${metrics}" \
    --config "{\"config\":\"${cfg}\"}" \
    --cost 0 \
    --wall-clock "$((end - start))" \
    --gpu "DGX Spark GB10" \
    --status "${status}" \
    --description "baseline ${cfg}" || true
done

iteration=0
while true; do
  iteration=$((iteration + 1))
  lr_values=(0.0004 0.0008 0.0010 0.0003)
  depth_values=(6 8 4 10)
  idx=$(( (iteration - 1) % ${#lr_values[@]} ))
  lr="${lr_values[$idx]}"
  depth="${depth_values[$idx]}"
  cfg="configs/generated/rfnll_iter_${iteration}.yaml"
  cat > "${cfg}" <<EOF
include: configs/rfnll_math.yaml

train:
  condition: rfnll_original_vs_finephrase
  output_dir: outputs/rfnll_iter_${iteration}
  max_steps: 200
  eval_every: 50
  batch_size: 4
  grad_accum_steps: 4
  learning_rate: ${lr}
  warmup_steps: 20

model:
  num_hidden_layers: ${depth}
EOF

  echo "autoresearch: iteration=${iteration} lr=${lr} depth=${depth} cfg=${cfg}"
  git add "${cfg}" && git commit -m "try rfnll lr ${lr} depth ${depth}" || true

  start=$(date +%s)
  status=keep
  if ! scripts/run_validation.sh; then
    status=crash
  elif ! scripts/run_experiment.sh "${cfg}"; then
    status=crash
  fi
  end=$(date +%s)
  metrics_file="outputs/rfnll_iter_${iteration}/metrics.json"
  metrics="$(cat "${metrics_file}" 2>/dev/null || echo '{"primary_metric": 999999}')"
  python3 tools/experiment_log.py log \
    --commit "$(git rev-parse --short HEAD 2>/dev/null || echo nogit)" \
    --metrics "${metrics}" \
    --config "{\"config\":\"${cfg}\",\"lr\":${lr},\"depth\":${depth}}" \
    --cost 0 \
    --wall-clock "$((end - start))" \
    --gpu "DGX Spark GB10" \
    --status "${status}" \
    --description "autonomous rfnll iteration ${iteration}" || true

  if [ "${status}" = "crash" ]; then
    echo "autoresearch: crash logged; continuing after cooldown"
    sleep 30
  fi
done
