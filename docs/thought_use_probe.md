# Causal Thought-Use Probe

This is the core applied interpretability experiment for the release.

The intervention is simple: replace the model's generated thought before suffix generation, then measure whether suffix reward changes. The reward object remains paper-aligned: the judge scores only the predicted suffix against the true suffix.

```text
prefix -> intervened thought -> predicted suffix -> judge(predicted suffix, true suffix)
```

## Setup

| Component | Value |
| --- | --- |
| Arms | Base+Think SFT, Self-improved+Think SFT, Base+Think+RLMT, Self-improved+Think+RLMT |
| Prefixes | 32 held-out prefixes |
| Samples | 4 per prefix per condition |
| Conditions | normal model thought, blank thought, generic thought, same-arm swapped thought |
| Total judged suffixes | 2,048 |
| Invalid judge rate | 0.0 |

## Main Result

Swapping in an unrelated thought sharply reduced suffix reward across all arms.

| Arm | Normal Thought Reward | Swapped Thought Reward |
| --- | ---: | ---: |
| Base + Think SFT | 0.1172 | 0.0156 |
| Self-improved + Think SFT | 0.1250 | 0.0234 |
| Base + Think + RLMT | 0.0859 | 0.0156 |
| Self-improved + Think + RLMT | 0.1094 | 0.0156 |

The thought channel is therefore not just a tag format. Mismatched thought content causally steers suffix generation away from rewarded continuations.

## Important Limitation

Blank and generic thoughts often matched or outperformed the model's own sampled thoughts. Generic thoughts reached reward means between 0.1797 and 0.2031 in this run.

The right interpretation is thought use is present but immature. At 0.6B parameters and 200 RLMT updates, the model learned an emerging thought-conditioned continuation interface, but it did not learn to reliably generate the best thoughts for that interface.

## Safe Claim

Use this as the interpretability claim:

> In a deliberately small 0.6B setting with only 200 RLMT updates, a causal thought-use probe shows that interleaved thought text is already a behavioral control surface: swapping in an unrelated thought sharply reduces suffix reward across all arms. However, the model's own sampled thoughts are not yet reliably better than blank or generic scaffolds, so the result supports an emerging interface rather than a fully optimized reasoning policy.
