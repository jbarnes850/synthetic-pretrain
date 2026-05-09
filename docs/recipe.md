# Experiment Recipe

This repo keeps one canonical run path:

1. Build DCLM/FineMath prefix/suffix examples.
2. Run a standard continued-pretraining control on the same examples.
3. Run Self-Improving Pretraining continued pretraining with Online DPO.
   The mainline candidate pool is original suffix plus K=16 policy rollouts.
   Pairwise judge calls are repeated and pointwise scores average the votes.
4. Optionally run the isolated SIP ablation with original suffix, teacher
   rewrite, and K=16 policy rollouts.
5. Ask the teacher to insert interleaved thoughts into raw chunks using the
   Figure 19-style thinking augmentation prompt.
6. Split the augmented corpus into disjoint SFT, RLMT, and heldout partitions.
7. Train Thinking SFT arms from the base, standard-CPT, and SIP checkpoints.
8. Run the reward-variance gate before RLMT.
9. Run RLMT only if the gate has enough per-prefix reward variance.
10. Run causal thought probes and reasoning evals.
11. Run selector ablations on the scored thought bank, including oracle@16 as
    a diagnostic upper bound only.

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

Corpus handles:

- `mlfoundations/dclm-baseline-1.0-parquet`
- `HuggingFaceTB/finemath`
