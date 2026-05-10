#!/usr/bin/env bash
set -euo pipefail

# Recipe-faithful launcher. It is intentionally explicit: run one stage at a
# time when Spark capacity is available, preserving SIP -> thinking SFT -> RLMT
# -> probes/evals. It does not stop or manage existing workloads.

STAGE="${1:-help}"
RUN_ID="${RUN_ID:-run-$(date -u +%Y%m%d-%H%M%S)}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SPARK_TRAIN_IMAGE:-vllm/vllm-openai:v0.20.0}"
PREP_IMAGE="${SPARK_PREP_IMAGE:-lmsysorg/sglang:deepseek-v4-grace-blackwell}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://127.0.0.1:30000}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen36-35b-a3b}"
TEACHER_ENDPOINT="${TEACHER_ENDPOINT:-${JUDGE_ENDPOINT}}"
TEACHER_MODEL="${TEACHER_MODEL:-${JUDGE_MODEL}}"
MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:30001}"
MODEL_NAME="${MODEL_NAME:-policy}"
ARM="${ARM:-think_self_improved_rlmt}"
DATA_GATE_SUMMARY="${DATA_GATE_SUMMARY:-outputs/data_integrity_gate/${RUN_ID}/summary.json}"

cd "${PROJECT_DIR}"
mkdir -p logs data/processed outputs

require_data_integrity_gate() {
  if [[ "${ALLOW_UNGATED_TRAINING:-0}" == "1" ]]; then
    echo "warning: ALLOW_UNGATED_TRAINING=1; bypassing data integrity gate" >&2
    return 0
  fi
  python3 - "${DATA_GATE_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Data integrity gate has not passed: missing {path}")
summary = json.loads(path.read_text())
status = summary.get("status")
passed = summary.get("comparisons", {}).get("go_no_go", {}).get("passed")
if status != "pass" or passed is not True:
    raise SystemExit(f"Data integrity gate has not passed: status={status!r} passed={passed!r} path={path}")
print(f"data_integrity_gate_ok: {path}")
PY
}

docker_base=(
  docker run --rm --ipc=host --network=host
  --user "$(id -u):$(id -g)"
  -v "${PROJECT_DIR}:/workspace"
  -v "${HOME}/.cache/huggingface:/hf:ro"
  -v "${HOME}/models:/models:ro"
  -e HF_HOME=/hf
  -e TRANSFORMERS_OFFLINE=1
  -e HF_DATASETS_OFFLINE=1
  -e PYTHONPATH=/workspace/scripts
  -w /workspace
  "${PREP_IMAGE}"
)

docker_gpu=(
  docker run --rm --gpus all --ipc=host --network=host
  --entrypoint bash
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
  -e RUN_ID="${RUN_ID}"
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
    require_data_integrity_gate
    SPARK_SKIP_PREPARE="${SPARK_SKIP_PREPARE:-1}" JUDGE_ENDPOINT="${JUDGE_ENDPOINT}" JUDGE_MODEL="${JUDGE_MODEL}" \
      scripts/run_experiment.sh configs/self_improving_pretraining.yaml "${RUN_ID}-sip-cpt"
    ;;
  cpt-baseline)
    require_data_integrity_gate
    SPARK_SKIP_PREPARE="${SPARK_SKIP_PREPARE:-1}" \
      scripts/run_experiment.sh configs/standard_cpt.yaml "${RUN_ID}-cpt-baseline"
    ;;
  smoke-sip-dpo)
    require_data_integrity_gate
    JUDGE_ENDPOINT="${JUDGE_ENDPOINT}" JUDGE_MODEL="${JUDGE_MODEL}" \
      scripts/run_experiment.sh configs/smoke_sip_dpo.yaml "${RUN_ID}-smoke-sip-dpo"
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
    require_data_integrity_gate
    SPARK_SKIP_PREPARE="${SPARK_SKIP_PREPARE:-1}" JUDGE_ENDPOINT="${JUDGE_ENDPOINT}" JUDGE_MODEL="${JUDGE_MODEL}" \
      scripts/run_experiment.sh configs/self_improving_pretraining_rewrite.yaml "${RUN_ID}-sip-cpt-rewrite"
    ;;
  build-thinking)
    "${docker_base[@]}" bash -lc "python3 scripts/build_thinking_data.py \
      --config configs/thinking_sft_base.yaml \
      --input-jsonl data/processed/pretraining_examples.jsonl \
      --output-jsonl '${THINKING_OUTPUT_JSONL:-data/processed/interleaved_thinking_full.jsonl}' \
      --train-count '${THINKING_TRAIN_COUNT:-30720}' --val-count '${THINKING_VAL_COUNT:-2048}' \
      --chunk-tokens 384 --max-augmented-tokens '${THINKING_MAX_AUGMENTED_TOKENS:-6144}' \
      --teacher-endpoint '${TEACHER_ENDPOINT}' \
      --teacher-model '${TEACHER_MODEL}' \
      --teacher-temperature 0.6 --teacher-top-p 0.95 \
      --teacher-max-tokens '${THINKING_TEACHER_MAX_TOKENS:-8192}' \
      --teacher-timeout '${THINKING_TEACHER_TIMEOUT:-900}' \
      --candidate-multiplier '${THINKING_CANDIDATE_MULTIPLIER:-1.0}' \
      --num-shards '${THINKING_NUM_SHARDS:-1}' \
      --shard-index '${THINKING_SHARD_INDEX:-0}' \
      --max-workers '${THINKING_MAX_WORKERS:-24}' \
      --max-skip-rate '${THINKING_MAX_SKIP_RATE:-0.10}' \
      --skip-invalid --resume"
    ;;
  split-thinking)
    "${docker_base[@]}" python3 scripts/split_midtraining_data.py \
      --input-jsonl data/processed/interleaved_thinking_full.jsonl \
      --sft-jsonl data/processed/interleaved_thinking_sft.jsonl \
      --rl-jsonl data/processed/interleaved_thinking_rl.jsonl \
      --heldout-jsonl data/processed/interleaved_thinking_heldout.jsonl \
      --sft-count 16384 --rl-count 14336 --heldout-count 2048
    ;;
  merge-thinking-shards)
    "${docker_base[@]}" python3 scripts/merge_thinking_shards.py \
      --input-jsonl ${THINKING_SHARD_INPUTS:-data/processed/interleaved_thinking_shard0.jsonl data/processed/interleaved_thinking_shard1.jsonl} \
      --output-jsonl data/processed/interleaved_thinking_full.jsonl \
      --expected-total 32768 --expected-train 30720 --expected-val 2048 \
      --trim-to-expected \
      --require-preserved
    ;;
  data-integrity-gate)
    "${docker_gpu[@]}" -lc "python3 scripts/eval_data_integrity_gate.py \
      --config configs/thinking_sft_base.yaml \
      --input-jsonl data/processed/interleaved_thinking_heldout.jsonl \
      --judge-endpoint \"${JUDGE_ENDPOINT}\" --judge-model \"${JUDGE_MODEL}\" \
      --num-prefixes \"${DATA_GATE_PREFIXES:-512}\" --samples-per-prefix \"${DATA_GATE_SAMPLES_PER_PREFIX:-4}\" \
      --output-dir \"outputs/data_integrity_gate/${RUN_ID}\"" \
      2>&1 | tee "logs/${RUN_ID}-data-integrity-gate.log"
    ;;
  sft-base)
    require_data_integrity_gate
    scripts/run_experiment.sh configs/thinking_sft_base.yaml "${RUN_ID}-sft-base"
    ;;
  sft-self-improved)
    require_data_integrity_gate
    INIT_FROM_CHECKPOINT=outputs/self_improving_pretraining/final.pt \
      scripts/run_experiment.sh configs/thinking_sft_self_improved.yaml "${RUN_ID}-sft-self-improved"
    ;;
  sft-cpt)
    require_data_integrity_gate
    INIT_FROM_CHECKPOINT=outputs/standard_cpt/final.pt \
      scripts/run_experiment.sh configs/thinking_sft_cpt.yaml "${RUN_ID}-sft-cpt"
    ;;
  rlmt-base)
    require_data_integrity_gate
    "${docker_gpu[@]}" -lc "python3 -m pip install --quiet wandb && python3 scripts/train_rlmt.py \
      --arm think_base \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --steps "${RLMT_STEPS:-1000}" --prefixes-per-step "${RLMT_PREFIXES_PER_STEP:-4}" \
      --samples-per-prefix 16 --enforce-stop-conditions" \
      2>&1 | tee "logs/${RUN_ID}-rlmt-base.log"
    ;;
  rlmt-self-improved)
    require_data_integrity_gate
    "${docker_gpu[@]}" -lc "python3 -m pip install --quiet wandb && python3 scripts/train_rlmt.py \
      --arm think_self_improved \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --steps "${RLMT_STEPS:-1000}" --prefixes-per-step "${RLMT_PREFIXES_PER_STEP:-4}" \
      --samples-per-prefix 16 --enforce-stop-conditions" \
      2>&1 | tee "logs/${RUN_ID}-rlmt-self-improved.log"
    ;;
  rlmt-cpt)
    require_data_integrity_gate
    "${docker_gpu[@]}" -lc "python3 -m pip install --quiet wandb && python3 scripts/train_rlmt.py \
      --arm think_cpt \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --steps "${RLMT_STEPS:-1000}" --prefixes-per-step "${RLMT_PREFIXES_PER_STEP:-4}" \
      --samples-per-prefix 16 --enforce-stop-conditions" \
      2>&1 | tee "logs/${RUN_ID}-rlmt-cpt.log"
    ;;
  smoke-rlmt)
    require_data_integrity_gate
    "${docker_gpu[@]}" -lc "python3 -m pip install --quiet wandb && python3 scripts/train_rlmt.py \
      --arm "${SMOKE_RLMT_ARM:-think_base}" \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --steps 1 --prefixes-per-step 1 --samples-per-prefix 4 \
      --gen-batch-size 4 --enforce-stop-conditions \
      --output-dir "outputs/smoke_rlmt/${RUN_ID}"" \
      2>&1 | tee "logs/${RUN_ID}-smoke-rlmt.log"
    ;;
  reward-gate-pre-rlmt)
    "${docker_gpu[@]}" -lc "python3 scripts/eval_reward_gate.py \
      --arms think_base think_cpt think_self_improved \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --num-prefixes 128 --samples-per-prefix 16 \
      --output-dir "outputs/reward_gate/pre_rlmt_${RUN_ID}"" \
      2>&1 | tee "logs/${RUN_ID}-reward-gate-pre-rlmt.log"
    ;;
  reward-gate-post-rlmt)
    "${docker_gpu[@]}" -lc "python3 scripts/eval_reward_gate.py \
      --arms think_base think_cpt think_self_improved think_base_rlmt think_cpt_rlmt think_self_improved_rlmt \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --num-prefixes 128 --samples-per-prefix 16 \
      --output-dir "outputs/reward_gate/post_rlmt_${RUN_ID}"" \
      2>&1 | tee "logs/${RUN_ID}-reward-gate-post-rlmt.log"
    ;;
  thinking-eval)
    "${docker_gpu[@]}" -lc "python3 scripts/eval_thinking.py \
      --arms raw_base think_base think_cpt think_self_improved think_base_rlmt think_cpt_rlmt think_self_improved_rlmt \
      --judge-endpoint "${JUDGE_ENDPOINT}" --judge-model "${JUDGE_MODEL}" \
      --output-dir "outputs/thinking_eval/${RUN_ID}"" \
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
  smoke-sip-dpo
  cpt-baseline
  sip-cpt
  build-rewrites
  sip-cpt-rewrite
  build-thinking
  merge-thinking-shards
  split-thinking
  data-integrity-gate
  sft-base
  sft-cpt
  sft-self-improved
  reward-gate-pre-rlmt
  smoke-rlmt
  rlmt-base
  rlmt-cpt
  rlmt-self-improved
  reward-gate-post-rlmt
  thinking-eval
  causal-probe
  selector-ablation
  reasoning-eval
EOF
    ;;
esac
