#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-faithful-clean-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/full_runs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

export RUN_ID
export SPARK_TRAIN_IMAGE="${SPARK_TRAIN_IMAGE:-vllm/vllm-openai:v0.20.0}"
export SPARK_PREP_IMAGE="${SPARK_PREP_IMAGE:-lmsysorg/sglang:deepseek-v4-grace-blackwell}"
export JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://192.168.100.11:30000}"
export TEACHER_ENDPOINT="${TEACHER_ENDPOINT:-${JUDGE_ENDPOINT}}"
export JUDGE_MODEL="${JUDGE_MODEL:-qwen35_35b_a3b_dflash_32768}"
export TEACHER_MODEL="${TEACHER_MODEL:-${JUDGE_MODEL}}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "launch_full_spark_run: run_id=${RUN_ID}"
echo "launch_full_spark_run: started=$(date -Is)"
echo "launch_full_spark_run: cwd=$(pwd)"
echo "launch_full_spark_run: train_image=${SPARK_TRAIN_IMAGE}"
echo "launch_full_spark_run: prep_image=${SPARK_PREP_IMAGE}"
echo "launch_full_spark_run: judge_endpoint=${JUDGE_ENDPOINT}"
echo "launch_full_spark_run: judge_model=${JUDGE_MODEL}"
echo "launch_full_spark_run: log=${LOG_FILE}"

test -f .env
grep -q '^WANDB_API_KEY=' .env
curl -fsS "${JUDGE_ENDPOINT}/v1/models" >/tmp/"${RUN_ID}".judge_models.json

python3 scripts/cache_corpus_shards.py --config configs/self_improving_pretraining.yaml

scripts/run_pipeline.sh prepare-corpus
scripts/run_pipeline.sh build-thinking
scripts/run_pipeline.sh split-thinking
scripts/run_pipeline.sh data-integrity-gate
scripts/run_pipeline.sh smoke-sip-dpo
scripts/run_pipeline.sh cpt-baseline
scripts/run_pipeline.sh sip-cpt
scripts/run_pipeline.sh sft-base
scripts/run_pipeline.sh sft-cpt
scripts/run_pipeline.sh sft-self-improved
scripts/run_pipeline.sh reward-gate-pre-rlmt
scripts/run_pipeline.sh smoke-rlmt
scripts/run_pipeline.sh rlmt-base
scripts/run_pipeline.sh rlmt-cpt
scripts/run_pipeline.sh rlmt-self-improved
scripts/run_pipeline.sh reward-gate-post-rlmt
scripts/run_pipeline.sh thinking-eval

echo "launch_full_spark_run: completed=$(date -Is)"
