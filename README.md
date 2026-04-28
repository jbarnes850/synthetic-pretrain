# Self-Improving Pretraining for Thinking Mid-Training

[![Model](https://img.shields.io/badge/HuggingFace-Model-blue)](https://huggingface.co/Jarrodbarnes/qwen3-0.6B-interleaved-thinking)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue)](https://huggingface.co/datasets/Jarrodbarnes/qwen3-0.6B-interleaved-thinking-data)
[![Blog](https://img.shields.io/badge/Blog-Research%20Writeup-black)](https://jbarnes850.github.io/2026/04/27/self-improving-pretraining-thinking-midtraining/)
[![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)](LICENSE)

This repository contains the code release for a small-scale adaptation of Tan et al.'s self-improving pretraining and thinking mid-training pipeline to `Qwen/Qwen3-0.6B-Base`.

The experiment asks whether ordinary pretraining text can become a sequence of training environments before agentic post-training begins:

1. A prefix-suffix continuation-quality task for self-improving continued pretraining.
2. An interleaved-thinking SFT task that teaches a short local thought interface.
3. An RL mid-training task that rewards thought-conditioned suffix prediction.
4. A causal thought-use probe that tests whether the thought text actually steers behavior.

The released model is not an instruction-tuned assistant. It is a research artifact for studying whether a small base model can be shaped into an emerging thought-conditioned continuation interface.

## Release Links

| Artifact | Link |
| --- | --- |
| Blog post | <https://jbarnes850.github.io/2026/04/27/self-improving-pretraining-thinking-midtraining/> |
| Model | <https://huggingface.co/Jarrodbarnes/qwen3-0.6B-interleaved-thinking> |
| Dataset | <https://huggingface.co/datasets/Jarrodbarnes/qwen3-0.6B-interleaved-thinking-data> |

## Repository Map

```text
configs/
  base.yaml                         Shared training/data/runtime defaults
  self_improving_pretraining.yaml   Continued pretraining with Online DPO
  thinking_sft_*.yaml               Matched SFT arms for raw and interleaved chunks

scripts/
  prepare_pretraining_data.py       Materialize FineWeb-Edu prefix/suffix chunks
  train.py                          Shared trainer for NTP, SFT, and Online DPO
  train_dpo.py                      Online DPO helper code
  build_thinking_data.py            Teacher augmentation for interleaved thoughts
  train_rlmt.py                     Small RLMT loop
  eval_*.py                         Continuation, thinking, reward, and reasoning evals
  probe_thought_use.py              Causal thought-use intervention
  export_model.py                   Export a checkpoint as a Hugging Face model directory

docs/
  data_audit.md                     Dataset structure and caveats
  results.md                        Release-facing results summary
  thought_use_probe.md              Applied interpretability appendix
```

Large artifacts are intentionally not stored in this code repository. The model weights and dataset payload are released on Hugging Face.

## Main Results

| Stage | Primary evidence | Interpretation |
| --- | --- | --- |
| Continued pretraining | 81/128 held-out pairwise continuation wins over Qwen3-0.6B-Base | The self-improved checkpoint produced better judged continuations |
| Interleaved-thinking SFT | Thought-token NLL dropped from 4.24 to 3.16 and 3.14 | SFT installed the thought interface |
| RLMT reward gate | Self-improved RLMT reached the highest reward mean at 0.098 | RLMT made the interface rewardable under the suffix-prediction objective |
| Thought-use probe | Swapped thoughts reduced reward to 0.016-0.023 | Thought text became a causal behavioral control surface |

The downstream reasoning evaluation was mixed. This is a 0.6B model with a short 200-step RLMT run, so the right claim is narrow: the lifecycle is trainable at small scale, and the thought channel becomes behaviorally meaningful, but the model does not learn a mature agentic reasoning policy.

## Setup

The original experiments ran in a Dockerized GPU environment with local Hugging Face caches. For code inspection, linting, and small local checks:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
ruff check scripts
python3 -m py_compile scripts/*.py
```

For the containerized validation path without spending GPU time:

```bash
RUN_GPU_SMOKE=0 RUN_HARDWARE_PROBE=0 scripts/validate.sh
```

For an actual GPU smoke, set `RUN_GPU_SMOKE=1` and `RUN_HARDWARE_PROBE=1` in an environment with the expected CUDA container and mounted model/data caches.

## Minimal Reproduction Path

The release keeps the lifecycle scripts separate so each claim is tied to a stage.

```bash
# 1. Prepare prefix/suffix pretraining chunks.
python3 scripts/prepare_pretraining_data.py --config configs/self_improving_pretraining.yaml --force --validate

# 2. Run self-improving continued pretraining.
JUDGE_ENDPOINT=http://127.0.0.1:30000 JUDGE_MODEL=qwen-judge \
  python3 scripts/train.py --config configs/self_improving_pretraining.yaml

# 3. Build interleaved-thinking SFT data.
python3 scripts/build_thinking_data.py \
  --config configs/thinking_sft_base.yaml \
  --input-jsonl data/processed/phase1_raw_examples.jsonl \
  --output-jsonl data/processed/interleaved_thinking.jsonl

# 4. Train matched SFT arms.
python3 scripts/train.py --config configs/thinking_sft_base.yaml
python3 scripts/train.py --config configs/thinking_sft_self_improved.yaml

# 5. Run RLMT and the causal thought-use probe.
python3 scripts/train_rlmt.py --arm think_phase3
python3 scripts/probe_thought_use.py --arms think_base think_phase3 think_base_rlmt think_phase3_rlmt
```

The arm identifiers preserve the experiment lineage: `think_base` starts from Qwen3-0.6B-Base, `think_phase3` starts from the self-improved checkpoint, and `*_rlmt` denotes the corresponding RLMT arm.

The exact public model and dataset are available from the Hugging Face links above.

## Claim Boundaries

This repository supports the blog's cautious claim: pretraining-style text can be wrapped into continuation selection, interleaved thought insertion, and thought-conditioned reward tasks before agentic post-training.

It does not support broad claims about mature reasoning, production assistant behavior, or general RLMT scaling. The thought-use probe shows causal sensitivity to thought content, while also showing that sampled thoughts are not yet reliably better than blank or generic scaffolds at this scale.

## Citation

If this release is useful, cite the blog post and this repository. The upstream method is from Tan et al., *Self-Improving Pretraining*.
