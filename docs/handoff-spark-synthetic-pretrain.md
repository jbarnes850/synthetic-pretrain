# Spark Synthetic Pretrain Handoff

## Current State
- Project root: `/home/jarrodbarnes/synthetic-pretrain`
- Controller root used for this run: `/tmp/spark_parallel_scaffold`
- Canonical results ledger: `/home/jarrodbarnes/synthetic-pretrain/results.json` on cfd0
- Direct Spark interconnect:
  - f7e2: `192.168.100.10`
  - cfd0: `192.168.100.11`
- FinePhrase cache was synced cfd0 -> f7e2 over the direct interconnect at about 490 MB/s.

## Completed Validation
- Local compile: `python3 -m compileall scripts tools`
- Local lint: `ruff check scripts/*.py tools/*.py`
- f7e2 full smoke: `scripts/run_validation.sh`
- f7e2 smoke metrics:
  - `primary_metric=11.35820198059082`
  - `val_loss_raw=11.338806867599487`
  - `available_mem_gib_min=102.2507438659668`

## Two-Arm Sweep
- cfd0 run: `parallel_cfd0_1776475985`
  - Config: `configs/generated/parallel_lr1e3_depth4.yaml`
  - Status: `keep`
  - Reason: `primary_improved`
  - `primary_metric=7.135038414001465`
  - `val_loss_raw=7.109295272827149`
  - `tok_per_sec=14002.635458597757`
  - `available_mem_gib_min=106.15929412841797`
- f7e2 run: `parallel_f7e2_1776475985`
  - Config: `configs/generated/parallel_lr8e4_depth8.yaml`
  - Status: `discard`
  - Reason: `no_keep_rule_satisfied`
  - `primary_metric=7.22169563293457`
  - `val_loss_raw=7.202846336364746`
  - `tok_per_sec=11725.714431735754`
  - `available_mem_gib_min=96.14257431030273`

## Queue State
- `queue/done/parallel_cfd0_1776475985.json`
- `queue/done/parallel_f7e2_1776475985.json`
- No active queue/running files after authoritative queue resync.

## Judge/Eval Mode
- Added OpenAI-compatible pairwise judge selection in `scripts/select_targets.py`.
- Added config: `configs/judge_eval.yaml`
- Added role runner: `scripts/run_judge_role_mode.sh`
- Added controller role dispatch: `scripts/spark_orchestrator.py run-role`
- Default judge server model cache: `models--Qwen--Qwen3-4B-Instruct-2507`
- f7e2 verified snapshot: `/home/jarrodbarnes/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`

## Next Phase
1. Launch judge server on f7e2:
   `PYTHONUNBUFFERED=1 python3 scripts/spark_orchestrator.py run-role --node f7e2 --role serve-judge --config configs/judge_eval.yaml`
2. Run judge selection/eval on cfd0 against `http://f7e2:30000` or the f7e2 Tailscale/direct reachable endpoint.
3. Train a small judge-selected RF-NLL arm only after the judge selection artifact and stats file are present.
