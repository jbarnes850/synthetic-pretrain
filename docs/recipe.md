# Experiment Recipe

This repo keeps one canonical run path:

1. Build FineWeb-Edu prefix/suffix examples.
2. Run Self-Improving Pretraining continued pretraining with Online DPO.
   The mainline candidate pool is original suffix plus K=16 policy rollouts.
3. Optionally run the isolated SIP ablation with original suffix, teacher
   rewrite, and K=16 policy rollouts.
4. Ask the teacher to insert interleaved thoughts into raw chunks.
5. Split the augmented corpus into disjoint SFT, RLMT, and heldout partitions.
6. Train Thinking SFT arms from the base and self-improved checkpoints.
7. Run the reward-variance gate before RLMT.
8. Run RLMT only if the gate has enough per-prefix reward variance.
9. Run causal thought probes and reasoning evals.

The rewrite candidate pool is an ablation, not the default. Rewrites can test
whether stronger-teacher continuations improve selection pressure, but they must
not replace the mainline policy-rollout pool because that would hide whether the
student's own continuation distribution is improving.

RLMT uses Dr. GRPO-compatible optimization: advantages are centered returns
within a prompt group, token log-probs are summed and normalized by a fixed
generation budget, and per-group reward std is only a safety metric/stop gate.
Track total response length and incorrect-response length during RLMT; length
growth without reward movement is treated as a failure signal, not emergence.

Old release-result summaries are intentionally not part of this tree. Public
claims should be regenerated from source metrics after the current run completes.
