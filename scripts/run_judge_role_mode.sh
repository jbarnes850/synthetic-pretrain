#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-eval-select}"
CONFIG="${2:-configs/judge_eval.yaml}"
RUN_ID="${3:-judge-eval-$(date -u +%Y%m%d-%H%M%S)}"
IMAGE="${SPARK_TRAIN_IMAGE:-scitrera/dgx-spark-sglang:0.5.9-t5}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUDGE_PORT="${JUDGE_PORT:-30000}"
JUDGE_MODEL_CACHE_HOST="${JUDGE_MODEL_CACHE_HOST:-${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
JUDGE_SERVED_MODEL_NAME="${JUDGE_SERVED_MODEL_NAME:-qwen-judge}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

if [[ -z "${JUDGE_MODEL_PATH}" && -d "${JUDGE_MODEL_CACHE_HOST}/snapshots" ]]; then
  JUDGE_MODEL_SNAPSHOT="$(find "${JUDGE_MODEL_CACHE_HOST}/snapshots" -mindepth 1 -maxdepth 1 -type d -print | sort | tail -1)"
  JUDGE_MODEL_PATH="${JUDGE_MODEL_SNAPSHOT/#${HOME}\/.cache\/huggingface/\/hf}"
fi

case "${ROLE}" in
  serve-judge)
    echo "judge_role: serving judge model=${JUDGE_MODEL_PATH} port=${JUDGE_PORT}"
    docker run --rm --gpus all --ipc=host --network=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v "${HOME}/.cache/huggingface:/hf:ro" \
      -v "${HOME}/models:/models:ro" \
      -e HF_HOME=/hf \
      -e JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH}" \
      -e TRANSFORMERS_OFFLINE=1 \
      -e HF_DATASETS_OFFLINE=1 \
      "${IMAGE}" \
      bash -lc "test -n \"\${JUDGE_MODEL_PATH}\" && python3 -m sglang.launch_server --model-path \"\${JUDGE_MODEL_PATH}\" --served-model-name '${JUDGE_SERVED_MODEL_NAME}' --host 0.0.0.0 --port '${JUDGE_PORT}' --trust-remote-code --mem-fraction-static 0.72"
    ;;
  eval-select)
    echo "judge_role: evaluating OpenAI-compatible judge selection config=${CONFIG} run_id=${RUN_ID}"
    docker run --rm --ipc=host --network=host \
      -v "${PROJECT_DIR}:/workspace" \
      -v "${HOME}/.cache/huggingface:/hf:ro" \
      -v "${HOME}/models:/models:ro" \
      -e HF_HOME=/hf \
      -e TRANSFORMERS_OFFLINE=1 \
      -e HF_DATASETS_OFFLINE=1 \
      -e PYTHONPATH=/workspace/scripts \
      -e JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-}" \
      -e JUDGE_MODEL="${JUDGE_MODEL:-}" \
      -w /workspace \
      "${IMAGE}" \
      bash -lc "python3 scripts/materialize_dataset.py --config '${CONFIG}' --force && python3 scripts/select_targets.py --config '${CONFIG}' --force" \
      2>&1 | tee "${LOG_DIR}/${RUN_ID}.judge_eval.log"
    ;;
  trainer)
    echo "judge_role: running trainer config=${CONFIG} run_id=${RUN_ID}"
    scripts/run_experiment.sh "${CONFIG}" "${RUN_ID}"
    ;;
  *)
    echo "Unknown role '${ROLE}'. Expected: serve-judge, eval-select, trainer" >&2
    exit 2
    ;;
esac
