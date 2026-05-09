#!/usr/bin/env bash
set -euo pipefail

# Recipe-faithful launcher. It is intentionally explicit: run one stage at a
# time when Spark capacity is available, preserving SIP -> thinking SFT -> RLMT
# -> probes/evals. It does not stop or manage existing workloads.

STAGE="${1:-help}"
RUN_ID="${RUN_ID:-run-$(date -u +%Y%m%d-%H%M%S)}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SPARK_TRAIN_IMAGE:-nvcr.io/nvidia/pytorch:26.01-py3}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://127.0.0.1:30000}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen36-35b-a3b}"
TEACHER_ENDPOINT="${TEACHER_ENDPOINT:-${JUDGE_ENDPOINT}}"
TEACHER_MODEL="${TEACHER_MODEL:-${JUDGE_MODEL}}"
MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:30001}"
MODEL_NAME="${MODEL_NAME:-policy}"
ARM="${ARM:-think_self_improved_rlmt}"

cd "${PROJECT_DIR}"
mkdir -p logs data/processed outputs

docker_base=(
  docker run --rm --ipc=host --network=host
  -v "${PROJECT_DIR}:/workspace"
  -v "${HOME}/.cache/huggingface:/hf:ro"
  -v "${HOME}/models:/models:ro"
  -e HF_HOME=/hf
  -e TRANSFORMERS_OFFLINE=1
  -e HF_DATASETS_OFFLINE=1
  -e PYTHONPATH=/workspace/scripts
  -w /workspace
  "${IMAGE}"
)

docker_gpu=(
  docker run --rm --gpus all --ipc=host --network=host
  --ulimit memlock=-1 --ulimit stack=67108864
  -v "${PROJECT_DIR}:/workspace"
  -v "${HOME}/.cache/huggingface:/hf:ro"
  -v "${HOME}/models:/models:ro"
  -e HF_HOME=/hf
  -e TRANSFORMERS_OFFLINE=1
  -e HF_DATASETS_OFFLINE=1
  -e PYTHONPATH=/workspace/scripts
  -e JUDGE_ENDPOINT="${JUDGE_ENDPOINT}"
  -e JUDGE_MODEL="${JUDGE_MODEL}"
  -w /workspace
  "${IMAGE}"
)

case "${STAGE}" in
  compat)
    "${docker_base[@]}" python3 scripts/check_models.py \
      --student-path /hf/hub/models--Qwen--Qwen3.5-0.8B-Base \
      --teacher-path /hf/hub/models--Qwen--Qwen3.6-35B-A3B
    ;;
  prepare-corpus)
    "${docker_base[@]}" python3 scripts/prepare_pretraining_data.py \
      --config configs/self_improving_pretraining.yaml --force --validate
    ;;
  sip-cpt)
    JUDGE_ENDPOINT="${JUDGE_ENDPOINT}" JUDGE_MODEL="${JUDGE_MODEL}" \
      scripts/run_experiment.sh configs/self_improving_pretraining.yaml "${RUN_ID}-sip-cpt"
    ;;
  build-rewrites)
    "${docker_base[@]}" python3 scripts/build_rewrite_data.py \
      --config configs/self_improving_pretraining_rewrite.yaml \
      --input-jsonl data/processed/pretraining_examples.jsonl \
      --output-jsonl data/processed/pretraining_examples_with_rewrites.jsonl \
      --teacher-endpoint "${TEACHER_ENDPOINT}" \
      --teacher-model "${TEACHER_MODEL}" \
      --teacher-temperature 0.6 --teacher-top-p 0.95 \
      --teacher-max-tokens 256 \
      --max-workers 16 --resume
    ;;
  sip-cpt-rewrite)
    JUDGE_ENDPOINT="${JUDGE_ENDPOINT}" JUDGE_MODEL="${JUDGE_MODEL}" \
      scripts/run_experiment.sh configs/self_improving_pretraining_rewrite.yaml "${RUN_ID}-sip-cpt-rewrite"
    ;;
  build-thinking)
    "${docker_base[@]}" bash -lc "python3 scripts/build_thinking_data.py \
      --config configs/thinking_sft_base.yaml \
      --input-jsonl data/processed/pretraining_examples.jsonl \
      --output-jsonl data/processed/interleaved_thinking_full.jsonl \
      --train-count 61440 --val-count 4096 \
      --chunk-tokens 384 --max-augmented-tokens 768 \
      --teacher-endpoint '${TEACHER_ENDPOINT}' \
      --teacher-model '${TEACHER_MODEL}' \
	      --teacher-temperature 0.6 --teacher-top-p 0.95 \
	      --teacher-max-tokens 1536 \
	      --candidate-multiplier '${THINKING_CANDIDATE_MULTIPLIER:-1.25}' \
	      --max-workers 16 --skip-invalid --resume"
    ;;
  split-thinking)
    "${docker_base[@]}" python3 scripts/split_midtraining_data.py \
      --input-jsonl data/processed/interleaved_thinking_full.jsonl \
      --sft-jsonl data/processed/interleaved_thinking_sft.jsonl \
      --rl-jsonl data/processed/interleaved_thinking_rl.jsonl \
      --heldout-jsonl data/processed/interleaved_thinking_heldout.jsonl \
      --sft-count 32768 --rl-count 28672 --heldout-count 4096
    ;;
  sft-base)
    scripts/run_experiment.sh configs/thinking_sft_base.yaml "${RUN_ID}-sft-base"
    ;;
  sft-self-improved)
    INIT_FROM_CHECKPOINT=outputs/self_improving_pretraining/final.pt \
      scripts/run_experiment.sh configs/thinking_sft_self_improved.yaml "${RUN_ID}-sft-self-improved"
    ;;
  rlmt-base)
    "${docker_gpu[@]}" python3 scripts/train_rlmt.py \
      --arm think_base \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --steps "${RLMT_STEPS:-1000}" --prefixes-per-step "${RLMT_PREFIXES_PER_STEP:-4}" \
      --samples-per-prefix 16 --enforce-stop-conditions \
      2>&1 | tee "logs/${RUN_ID}-rlmt-base.log"
    ;;
  rlmt-self-improved)
    "${docker_gpu[@]}" python3 scripts/train_rlmt.py \
      --arm think_self_improved \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --steps "${RLMT_STEPS:-1000}" --prefixes-per-step "${RLMT_PREFIXES_PER_STEP:-4}" \
      --samples-per-prefix 16 --enforce-stop-conditions \
      2>&1 | tee "logs/${RUN_ID}-rlmt-self-improved.log"
    ;;
  reward-gate-pre-rlmt)
    "${docker_gpu[@]}" python3 scripts/eval_reward_gate.py \
      --arms think_base think_self_improved \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --num-prefixes 128 --samples-per-prefix 16 \
      --output-dir "outputs/reward_gate/pre_rlmt_${RUN_ID}" \
      2>&1 | tee "logs/${RUN_ID}-reward-gate-pre-rlmt.log"
    ;;
  reward-gate-post-rlmt)
    "${docker_gpu[@]}" python3 scripts/eval_reward_gate.py \
      --arms think_base think_self_improved think_base_rlmt think_self_improved_rlmt \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --num-prefixes 128 --samples-per-prefix 16 \
      --output-dir "outputs/reward_gate/post_rlmt_${RUN_ID}" \
      2>&1 | tee "logs/${RUN_ID}-reward-gate-post-rlmt.log"
    ;;
  thinking-eval)
    "${docker_gpu[@]}" python3 scripts/eval_thinking.py \
      --arms raw_base think_base think_self_improved think_base_rlmt think_self_improved_rlmt \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --output-dir "outputs/thinking_eval/${RUN_ID}" \
      2>&1 | tee "logs/${RUN_ID}-thinking-eval.log"
    ;;
  causal-probe)
    "${docker_base[@]}" python3 scripts/counterfactual_thought_bank_sglang.py \
      --arm "${ARM}" \
      --model-endpoint "${MODEL_ENDPOINT}" --model-name "${MODEL_NAME}" \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --num-prefixes "${PROBE_PREFIXES:-128}" --samples-per-prefix 16 \
      --conditions normal_model_thought teacher_thought blank_thought generic_thought same_arm_swapped_thought \
      --mark-complete \
      --output-dir "outputs/causal_probe/${RUN_ID}/${ARM}" \
      2>&1 | tee "logs/${RUN_ID}-causal-probe-${ARM}.log"
    ;;
  selector-ablation)
    selector_args=()
    if [[ -n "${SELECTOR_ENDPOINT:-}" ]]; then
      selector_args+=(--selector-endpoint "${SELECTOR_ENDPOINT}" --selector-model "${SELECTOR_MODEL:-${JUDGE_MODEL}}")
    fi
    "${docker_base[@]}" python3 scripts/compare_thought_selectors.py \
      --input-dir "${SELECTOR_INPUT_DIR:-outputs/causal_probe/${RUN_ID}/${ARM}}" \
      --output-dir "outputs/selector_ablation/${RUN_ID}/${ARM}" \
      --arms "${ARM}" \
      "${selector_args[@]}" \
      2>&1 | tee "logs/${RUN_ID}-selector-ablation-${ARM}.log"
    ;;
  reasoning-eval)
    "${docker_base[@]}" python3 scripts/eval_reasoning_sglang.py \
      --arm "${ARM}" \
      --endpoint "${MODEL_ENDPOINT}/v1" --model "${MODEL_NAME}" \
      --output-dir "outputs/reasoning_eval/${RUN_ID}/${ARM}" \
      2>&1 | tee "logs/${RUN_ID}-reasoning-eval-${ARM}.log"
    ;;
  *)
    cat <<'EOF'
Usage: scripts/run_pipeline.sh <stage>

Stages:
  compat
  prepare-corpus
  sip-cpt
  build-rewrites
  sip-cpt-rewrite
  build-thinking
  split-thinking
  sft-base
  sft-self-improved
  reward-gate-pre-rlmt
  rlmt-base
  rlmt-self-improved
  reward-gate-post-rlmt
  thinking-eval
  causal-probe
  selector-ablation
  reasoning-eval
EOF
    ;;
esac
