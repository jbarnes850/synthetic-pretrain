#!/usr/bin/env bash
set -euo pipefail

IMAGE="${SPARK_TRAIN_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"
RUN_HARDWARE_PROBE="${RUN_HARDWARE_PROBE:-1}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p logs data/processed outputs

echo "validation: project=${PROJECT_DIR}"
echo "validation: image=${IMAGE}"
echo "validation: host memory"
free -h

echo "validation: ruff"
RUFF_TARGETS="scripts"
if command -v ruff >/dev/null 2>&1; then
  ruff check ${RUFF_TARGETS}
elif [ -x .venv/bin/ruff ]; then
  .venv/bin/ruff check ${RUFF_TARGETS}
else
  echo "validation: ruff unavailable on host; compile gate will still run"
fi

echo "validation: compile and package imports in CUDA 13 container"
docker run --rm --ipc=host --network=host \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${HOME}/.cache/huggingface:/hf:ro" \
  -v "${HOME}/models:/models:ro" \
  -e HF_HOME=/hf \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e PYTHONPATH=/workspace/scripts \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "python3 -m compileall scripts && python3 - <<'PY'
import importlib.util as u
mods = ['torch', 'transformers', 'pyarrow', 'yaml', 'numpy']
print({m: bool(u.find_spec(m)) for m in mods})
missing = [m for m in mods if not u.find_spec(m)]
raise SystemExit(1 if missing else 0)
PY"

echo "validation: CUDA/BF16/SDPA hardware probe"
if [[ "${RUN_HARDWARE_PROBE}" == "1" ]]; then
  docker run --rm --gpus all --ipc=host --network=host \
    -v "${PROJECT_DIR}:/workspace" \
    -e PYTHONPATH=/workspace/scripts \
    -w /workspace \
    "${IMAGE}" \
    python3 scripts/verify_torch.py
else
  echo "validation: hardware probe skipped because RUN_HARDWARE_PROBE=${RUN_HARDWARE_PROBE}"
fi

echo "validation: dataset contract, tokenizer round-trip, and suffix alignment"
docker run --rm --ipc=host --network=host \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${HOME}/.cache/huggingface:/hf:ro" \
  -v "${HOME}/models:/models:ro" \
  -e HF_HOME=/hf \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e PYTHONPATH=/workspace/scripts \
  -w /workspace \
  "${IMAGE}" \
  python3 scripts/prepare_pretraining_data.py --config configs/smoke.yaml --force --validate

echo "validation: GPU smoke"
if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  scripts/run_experiment.sh configs/smoke.yaml
else
  echo "validation: GPU smoke skipped because RUN_GPU_SMOKE=${RUN_GPU_SMOKE}"
fi

echo "validation: passed"
