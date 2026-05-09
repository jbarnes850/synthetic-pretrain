# RLMT Framework Gate

The first run keeps SIP continued pretraining and Thinking SFT in custom
PyTorch/Transformers code. The SIP objective is specialized: streaming
pretraining prefixes, original suffix pivot metrics, and full pairwise
comparison over an original-suffix-plus-rollout candidate pool.

RLMT is the only stage worth migrating to NeMo RL after the custom loop passes
local smoke tests. The current upstream NeMo RL path is not a drop-in
replacement for this first run: its GRPO advantage estimator can disable reward
std normalization, but its token-level loss normalizes by valid generated tokens,
not by a fixed generation-budget constant. That is closer to token-level GRPO /
DAPO-style aggregation than the Dr. GRPO correction we need here.

NeMo RL is a fit only if the environment and loss can preserve this contract
exactly:

- prompt: prefix plus thought opening;
- action: generated thought plus predicted suffix, with the boundary inserted
  outside the reward;
- reward: suffix judged against the raw reference suffix;
- group size: 16 samples per prefix;
- optimizer: Dr. GRPO-compatible centered returns and fixed-budget token loss,
  either by upstream support or by a small auditable loss override;
- logging: near-zero reward groups, response length, incorrect-response length,
  invalid judge rate, and artifact rate.

Do not migrate CPT/SIP or SFT for this run. Do not migrate RLMT until a zero-pod
NeMo smoke proves exact reward grouping, no reward-std normalization, and fixed
budget loss normalization. A framework migration is useful only if it removes
RLMT scaling risk without changing the method.

Primary references:

- https://arxiv.org/abs/2503.20783
- https://github.com/NVIDIA-NeMo/RL
- https://github.com/NVIDIA-NeMo/Gym
