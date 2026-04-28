# Results

These are the release-facing results behind the blog post, *Self-Improving Pretraining as a Substrate for Agentic Post-Training*.

The experiment is a small-scale adaptation of Tan et al.'s self-improving pretraining and thinking mid-training pipeline to `Qwen/Qwen3-0.6B-Base`. The right reading is lifecycle evidence at small scale, not a claim that the released 0.6B model is a mature reasoning policy.

## Stage Summary

| Stage | Primary Evidence | Interpretation |
| --- | --- | --- |
| Self-improving continued pretraining | 81/128 held-out pairwise continuation wins over Qwen3-0.6B-Base | The self-improved checkpoint produced better judged continuations |
| Interleaved-thinking SFT | Thought-token NLL dropped from 4.24 to 3.16 and 3.14 | SFT installed the thought interface |
| RLMT reward gate | Self-improved RLMT reached the highest reward mean at 0.098 | RLMT made the interface rewardable under the suffix-prediction objective |
| Reasoning eval | GSM8K, MATH-500, GPQA-Diamond, and OlympiadBench moved differently across Mean@8 and Pass@8 | Downstream reasoning did not improve uniformly |
| Causal thought-use probe | Swapped thoughts reduced reward to 0.016-0.023 | Thought text became a causal behavioral control surface |

## Continued Pretraining

The self-improving pretraining stage used Online DPO with `K=16` model rollouts per prefix and full-pairwise judging over continuation candidates.

On held-out prefix-suffix chunks, the self-improved checkpoint beat the original Qwen3-0.6B-Base checkpoint on 81 of 128 pairwise continuation judgments, a 63.28% win rate.

Raw suffix NLL moved slightly in the wrong direction, from 2.56 to 2.60. That is expected for this objective: the stage was not optimizing exact imitation of the original suffix, but judged continuation quality.

## Thinking SFT

The interleaved-thinking dataset contains 8,704 rows. It averaged 4.39 thoughts per row, had zero malformed rows, and preserved the raw text closely, with average raw word coverage of 99.98%.

Thought-token NLL dropped:

| Arm | Thought-token NLL |
| --- | ---: |
| Raw baseline | 4.24 |
| Base + interleaved-thinking SFT | 3.16 |
| Self-improved + interleaved-thinking SFT | 3.14 |

This is evidence that SFT installed the thought interface. It is not evidence by itself that thoughts are useful.

## RLMT Reward Gate

RLMT used the paper-aligned reward object:

```text
prefix -> generated thought -> predicted suffix -> judge(predicted suffix, true suffix)
```

The judge scored only the predicted suffix against the true suffix. It did not directly grade the thought.

| Arm | Reward Mean |
| --- | ---: |
| Base + Think SFT | 0.088 |
| Self-improved + Think SFT | 0.091 |
| Base + Think + RLMT | 0.094 |
| Self-improved + Think + RLMT | 0.098 |

The self-improved RLMT arm was the released model.

## Reasoning Eval

The downstream reasoning evaluation was mixed. On macro Mean@8, the self-improved thinking model was strongest before RLMT. On macro Pass@8, the self-improved RLMT model was highest.

| Arm | Macro Mean@8 | Macro Pass@8 |
| --- | ---: | ---: |
| Base + Think SFT | 0.172 | 0.500 |
| Base + Think + RLMT | 0.172 | 0.500 |
| Self-improved + Think SFT | 0.175 | 0.512 |
| Self-improved + Think + RLMT | 0.166 | 0.512 |

At this scale, the reward-object result is cleaner than the downstream benchmark result.

## Claim Boundary

These results support a cautious claim: ordinary pretraining text can be turned into a sequence of training environments, and a small base model can learn an emerging thought-conditioned continuation interface before agentic post-training.

They do not support a claim that the model learned a mature agentic reasoning policy. The experiment uses a 0.6B model, a short 200-step RLMT run, and a smaller evaluation budget than the original paper.
