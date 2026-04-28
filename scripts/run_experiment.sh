#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/self_improving_pretraining.yaml}"
RUN_ID="${2:-$(basename "${CONFIG}" .yaml)-$(date -u +%Y%m%d-%H%M%S)}"
IMAGE="${SPARK_TRAIN_IMAGE:-scitrera/dgx-spark-sglang:0.5.9-t5}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/${RUN_ID}.log"
OUTPUT_DIR="$(PYTHONPATH="${PROJECT_DIR}/scripts" python3 - "${CONFIG}" <<'PY'
from common import load_config
import sys
cfg = load_config(sys.argv[1])
print(cfg["train"]["output_dir"])
PY
)"
CONDITION="$(PYTHONPATH="${PROJECT_DIR}/scripts" python3 - "${CONFIG}" <<'PY'
from common import load_config
import sys
cfg = load_config(sys.argv[1])
print(cfg["train"]["condition"])
PY
)"
mkdir -p "${OUTPUT_DIR}"
WORKER_DIR="${PROJECT_DIR}/workers/${SPARK_WORKER_NAME:-local}/${RUN_ID}"
mkdir -p "${WORKER_DIR}"

echo "run_experiment: project=${PROJECT_DIR}"
echo "run_experiment: config=${CONFIG}"
echo "run_experiment: condition=${CONDITION}"
echo "run_experiment: run_id=${RUN_ID}"
echo "run_experiment: image=${IMAGE}"
echo "run_experiment: log=${RUN_LOG}"
echo "run_experiment: train_log=${OUTPUT_DIR}/train.log"
echo "run_experiment: worker_dir=${WORKER_DIR}"

docker run --rm --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${HOME}/.cache/huggingface:/hf:ro" \
  -v "${HOME}/models:/models:ro" \
  -e HF_HOME=/hf \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e PYTHONPATH=/workspace/scripts \
  -e JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-}" \
  -e JUDGE_MODEL="${JUDGE_MODEL:-}" \
  -e SPARK_SKIP_SELECT="${SPARK_SKIP_SELECT:-0}" \
  -e CONDITION="${CONDITION}" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "set -euo pipefail && python3 scripts/prepare_pretraining_data.py --config '${CONFIG}' --force && if [[ \"\${SPARK_SKIP_SELECT}\" != \"1\" && \"\${CONDITION}\" != \"online_dpo_selfimproving\" ]]; then python3 scripts/select_targets.py --config '${CONFIG}' --force; else echo 'run_experiment: skipping target selection'; fi && python3 scripts/train.py --config '${CONFIG}' 2>&1 | tee '${OUTPUT_DIR}/train.log'" \
  2>&1 | tee "${RUN_LOG}"

metrics_path="$(PYTHONPATH="${PROJECT_DIR}/scripts" python3 - "${CONFIG}" <<'PY'
from common import load_config
import sys
cfg = load_config(sys.argv[1])
print(cfg["train"]["output_dir"] + "/metrics.json")
PY
)"
cp "${metrics_path}" "${WORKER_DIR}/metrics.json"
cp "${CONFIG}" "${WORKER_DIR}/config.yaml"
printf '{"run_id":"%s","config":"%s","metrics_file":"%s","log":"%s"}\n' "${RUN_ID}" "${CONFIG}" "${metrics_path}" "${RUN_LOG}" > "${WORKER_DIR}/summary.json"
echo "run_experiment: complete log=${RUN_LOG}"
