#!/usr/bin/env bash
# Launch Phase 1 DDP pretraining across both DGX Sparks over 200 GbE.
# Run this on EACH node with the appropriate NODE_RANK env var.
#
# Usage (from spark-f7e2):
#   NODE_RANK=0 scripts/run_ddp_phase1.sh configs/generated/phase1_nll_250M.yaml
# Usage (from spark-cfd0, via f7e2 hop):
#   NODE_RANK=1 scripts/run_ddp_phase1.sh configs/generated/phase1_nll_250M.yaml
#
# Both invocations must be started within ~30s of each other for NCCL init.

set -euo pipefail

CONFIG="${1:-configs/generated/phase1_nll_250M.yaml}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on f7e2 or NODE_RANK=1 on cfd0}"
RUN_ID="${RUN_ID:-phase1-ddp-$(date -u +%Y%m%d-%H%M%S)}"
IMAGE="${SPARK_TRAIN_IMAGE:-synpre-flash:2509}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MASTER_ADDR="${MASTER_ADDR:-192.168.100.10}"
MASTER_PORT="${MASTER_PORT:-25000}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

OUTPUT_DIR="$(PYTHONPATH="${PROJECT_DIR}/scripts" python3 - "${CONFIG}" <<'PY'
from common import load_config
import sys
cfg = load_config(sys.argv[1])
print(cfg["train"]["output_dir"])
PY
)"
mkdir -p "${OUTPUT_DIR}"

RUN_LOG="${LOG_DIR}/${RUN_ID}.rank${NODE_RANK}.log"

echo "run_ddp_phase1: project=${PROJECT_DIR}"
echo "run_ddp_phase1: config=${CONFIG}"
echo "run_ddp_phase1: node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "run_ddp_phase1: run_id=${RUN_ID}"
echo "run_ddp_phase1: log=${RUN_LOG}"

SKIP_MATERIALIZE="${SPARK_SKIP_MATERIALIZE:-0}"

rm -rf /dev/shm/nccl-* 2>/dev/null || true

docker run --rm --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${HOME}/.cache/huggingface:/hf:ro" \
  -v "${HOME}/models:/models:ro" \
  -e HF_HOME=/hf \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e PYTHONPATH=/workspace/scripts \
  -e PYTHONUNBUFFERED=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_DEBUG=WARN \
  -e NVIDIA_IMEX_CHANNELS=0 \
  -e NCCL_NVLS_ENABLE=0 \
  -e TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -e SPARK_SKIP_MATERIALIZE="${SKIP_MATERIALIZE}" \
  -e SPARK_ATTN_IMPL="${SPARK_ATTN_IMPL:-flash_attention_2}" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "set -euo pipefail && \
    if [[ \"\${SPARK_SKIP_MATERIALIZE}\" != \"1\" ]]; then \
      python3 scripts/materialize_dataset.py --config '${CONFIG}' --force; \
    else \
      echo 'run_ddp_phase1: skipping materialize (data pre-synced)'; \
    fi && \
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=${NODE_RANK} \
      --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
      scripts/train.py --config '${CONFIG}' 2>&1 | tee '${OUTPUT_DIR}/train.rank${NODE_RANK}.log'" \
  2>&1 | tee "${RUN_LOG}"

echo "run_ddp_phase1: complete log=${RUN_LOG}"
