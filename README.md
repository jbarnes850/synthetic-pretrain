# Self-Improving Pretraining for Thinking Mid-Training

Public research code for a faithful small-scale experiment that combines:

1. Self-Improving Pretraining continued pretraining with Online DPO.
2. RAM-style Thinking Mid-training data augmentation and SFT.
3. RL mid-training on thought-conditioned suffix prediction.
4. Reward-variance gates, causal thought probes, and reasoning evals.

The target run uses `Qwen/Qwen3.5-0.8B-Base` as the student and a stronger
teacher/judge served through an OpenAI-compatible endpoint. The code is arranged
around the recipe, not around an older release lineage.

The corpus defaults are `mlfoundations/dclm-baseline-1.0-parquet` plus
`HuggingFaceTB/finemath`, matching the RAM paper's DCLM + FineMath source
family.

## Recipe

```text
DCLM + FineMath chunks
  -> prefix/suffix examples
  -> standard continued-pretraining control
  -> Self-Improving Pretraining continued pretraining
  -> teacher-inserted interleaved thoughts
  -> SFT/RL split
  -> Thinking SFT from base, standard-CPT, and self-improved checkpoints
  -> pre-RLMT reward-variance gate
  -> RLMT on thought + suffix generations
  -> causal thought probe and reasoning eval
```

The SIP stage follows the paper's Online DPO suffix-vs-K-rollouts branch:
sample K=16 rollouts from the current policy for each prefix, judge them in a
full pairwise pool with the original suffix, then optimize chosen versus
rejected continuations. The quality judge prompt in `prompts/judge_quality.txt`
matches the prompt printed in the SIP paper. Pairwise judge calls support
repeated voting and pointwise scores average the repeated outcomes. Teacher
rewrites are available only as a separate ablation
(`configs/self_improving_pretraining_rewrite.yaml`), where the pool is original
suffix + rewrite + K=16 policy rollouts.

The thinking stages follow the RAM
mid-training object: augment raw chunks with interleaved thoughts, train SFT on
one split, and run RLMT on a disjoint split where reward is assigned to the
predicted suffix given prefix plus generated thought. RLMT uses a Dr. GRPO-style
fixed-budget loss: centered returns within each prompt group, no reward-std
normalization in the optimizer, and response-length tracking for correct and
incorrect samples.

## Repository Map

```text
configs/
  standard_cpt.yaml                 Same-corpus continued-pretraining control
  self_improving_pretraining.yaml   SIP continued-pretraining config
  self_improving_pretraining_rewrite.yaml
  thinking_sft_base.yaml            Thinking SFT from the base model
  thinking_sft_cpt.yaml             Thinking SFT from the CPT control
  thinking_sft_self_improved.yaml   Thinking SFT from the SIP checkpoint
  thinking_sft_raw_control.yaml     Raw-token budget control

scripts/
  check_models.py                   Offline student/teacher compatibility check
  prepare_pretraining_data.py       DCLM/FineMath prefix/suffix materialization
  build_rewrite_data.py             Teacher rewrite pool builder for ablation
  build_thinking_data.py            Teacher augmentation for interleaved thoughts
  split_midtraining_data.py         Disjoint SFT/RL/heldout split builder
  train.py                          NTP, SFT, and Online DPO trainer
  train_rlmt.py                     RLMT loop with reward-variance stop rules
  eval_reward_gate.py               Pre/post-RLMT reward variance gate
  probe_thought_use.py              Causal thought-use intervention
  counterfactual_thought_bank_sglang.py
  eval_thinking.py
  eval_reasoning_sglang.py
  run_pipeline.sh                   Stage launcher for Spark/container runs
```

## Validation

Local static checks:

```bash
ruff check scripts
python3 -m py_compile scripts/*.py
bash -n scripts/run_pipeline.sh scripts/run_experiment.sh scripts/serve_judge.sh
```

Container and GPU smoke tests should be staged before full runs:

```bash
scripts/run_pipeline.sh compat
scripts/run_pipeline.sh prepare-corpus
scripts/run_pipeline.sh smoke-sip-dpo
scripts/run_pipeline.sh cpt-baseline
scripts/run_pipeline.sh sip-cpt
scripts/run_pipeline.sh build-rewrites
scripts/run_pipeline.sh sip-cpt-rewrite
scripts/run_pipeline.sh build-thinking
scripts/run_pipeline.sh split-thinking
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
scripts/run_pipeline.sh causal-probe
scripts/run_pipeline.sh selector-ablation
scripts/run_pipeline.sh reasoning-eval
```

Do not skip the reward gate. RLMT should stop before expensive training if more
than half of evaluated prefix groups have near-zero reward variance.

## Claim Boundary

This code is scoped to a faithful small-scale reproduction and failure-localized
extension of the SIP plus Thinking Mid-training lifecycle. It is not a release
of an instruction assistant, and it does not claim mature reasoning without the
post-training stages and eval evidence needed to support that.
