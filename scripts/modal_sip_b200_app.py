#!/usr/bin/env python3
"""Modal B200 runner for synthetic-pretrain.

This file is intentionally launch-guarded. Defining/deploying it is cheap, but
the local entrypoint refuses to allocate GPUs unless `--confirm-spend true` is
passed explicitly.

Intended next-step shape after the Spark overnight readout:

  modal run scripts/modal_sip_b200_app.py --mode sip-smoke --steps 25 --confirm-spend true

The SIP function uses one B200 for the learner and one B200 for an SGLang
Qwen3.5-35B-A3B-FP8 judge inside the same Modal container. It preserves the
paper-parity training object: K=16 rollouts, original suffix in the pool,
full pairwise comparisons, judge_repeats=1, Online DPO.
"""

from __future__ import annotations

import gzip
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "synthetic-pretrain-b200"
REPO_DIR = Path("/workspace")
OUTPUTS_DIR = REPO_DIR / "outputs"

SIP_CONFIG = "configs/self_improving_pretraining_k16_r1_w64.yaml"
SIP_DATA = REPO_DIR / "data" / "processed" / "pretraining_examples.jsonl"
SIP_DATA_GZ = REPO_DIR / "data" / "processed" / "pretraining_examples.jsonl.gz"
SIP_DATA_GZ_PARTS = REPO_DIR / "data" / "processed" / "pretraining_examples.jsonl.gz.parts"

THINKING_SFT_DATA = REPO_DIR / "data" / "processed" / "interleaved_thinking_sft.jsonl"
THINKING_RL_DATA = REPO_DIR / "data" / "processed" / "interleaved_thinking_rl.jsonl"
THINKING_HELDOUT_DATA = REPO_DIR / "data" / "processed" / "interleaved_thinking_heldout.jsonl"
THINKING_LOCK_MANIFEST = REPO_DIR / "data" / "processed" / "interleaved_thinking_data_lock.json"

HF_MODEL_CACHE = Path("/hf/hub/models--Qwen--Qwen3.5-0.8B-Base")
TRAIN_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
JUDGE_MODEL_PATH = "Qwen/Qwen3.5-35B-A3B-FP8"
JUDGE_MODEL_NAME = "qwen35_35b_a3b_fp8"
JUDGE_ENDPOINT = "http://127.0.0.1:30000"
JUDGE_PORT = 30000
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.10.post1-cu130-runtime"


app = modal.App(APP_NAME)

hf_volume = modal.Volume.from_name("synthetic-pretrain-hf", create_if_missing=True)
outputs_volume = modal.Volume.from_name("synthetic-pretrain-outputs", create_if_missing=True)
data_volume = modal.Volume.from_name("synthetic-pretrain-data", create_if_missing=True)
deepgemm_volume = modal.Volume.from_name("synthetic-pretrain-deepgemm", create_if_missing=True)

image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .apt_install("curl", "git", "rsync")
    .run_commands(
        "python -m pip install --no-deps pyarrow pyyaml requests "
        "typing_extensions==4.15.0 wandb==0.27.0 gitpython sentry-sdk gitdb smmap"
    )
    .env(
        {
            "HF_HOME": "/hf",
            "HF_HUB_CACHE": "/hf/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
            "SGLANG_USE_CUDA_IPC_TRANSPORT": "1",
            "SGLANG_USE_IPC_POOL_HANDLE_CACHE": "1",
            "DEEPGEMM_CACHE_DIR": "/root/.cache/deep_gemm",
        }
    )
    .add_local_dir(
        ".",
        remote_path=str(REPO_DIR),
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".ruff_cache",
            "data",
            "logs",
            "outputs",
            "tmp",
        ],
    )
)


def _run(cmd: str, *, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(["bash", "-lc", cmd], cwd=REPO_DIR, env=merged_env, check=True)


def _write_runtime_config(
    *,
    base_config: str,
    output_dir: Path,
    max_steps: int,
    eval_every: int,
    save_every: int,
    judge_workers: int,
    init_from_checkpoint: str | None = None,
    data_overrides: dict[str, str] | None = None,
) -> Path:
    config_path = REPO_DIR / "configs" / f"modal_runtime_{output_dir.name}.yaml"
    lines = [
        f"include: {base_config}",
        "",
        "train:",
        f"  output_dir: {output_dir.relative_to(REPO_DIR)}",
        f"  max_steps: {max_steps}",
        f"  eval_every: {eval_every}",
        f"  save_every: {save_every}",
    ]
    if init_from_checkpoint:
        lines.append(f"  init_from_checkpoint: {init_from_checkpoint}")
    if data_overrides:
        lines.extend(["", "data:"])
        for key, value in data_overrides.items():
            lines.append(f"  {key}: {value}")
    lines.extend(
        [
            "",
            "selection:",
            f"  judge_endpoint: {JUDGE_ENDPOINT}",
            f"  judge_model: {JUDGE_MODEL_NAME}",
            "  judge_repeats: 1",
            f"  judge_max_workers: {judge_workers}",
            "",
        ]
    )
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def _require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing required Modal volume inputs:\n"
            + "\n".join(f"- {path}" for path in missing)
            + "\nStage these into the Modal volumes before launching spend."
        )


def _ensure_sip_data() -> None:
    if SIP_DATA.exists():
        return
    gzip_source = SIP_DATA_GZ if SIP_DATA_GZ.exists() else None
    if gzip_source is None and SIP_DATA_GZ_PARTS.exists():
        parts = sorted(SIP_DATA_GZ_PARTS.glob("part-*"))
        if parts:
            gzip_source = Path("/tmp/pretraining_examples.jsonl.gz")
            with gzip_source.open("wb") as dst:
                for part in parts:
                    with part.open("rb") as src:
                        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    if gzip_source is None:
        _require_paths([SIP_DATA])
        return

    tmp_path = SIP_DATA.with_name(f"{SIP_DATA.name}.tmp")
    tmp_path.unlink(missing_ok=True)
    with gzip.open(gzip_source, "rb") as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    tmp_path.replace(SIP_DATA)
    data_volume.commit()


def _ensure_hf_cache(repo_id: str, expected_cache_path: Path) -> None:
    if expected_cache_path.exists():
        return
    script = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({repo_id!r}, cache_dir='/hf/hub', local_files_only=False)"
    )
    _run(f"python3 -c {shlex.quote(script)}")


def _start_judge(
    *,
    gpu_index: int = 1,
    context_length: int = 32768,
    max_workers: int = 64,
    use_fp8_kv_cache: bool = False,
) -> subprocess.Popen:
    cuda_graph_bs = max(32, min(256, max_workers * 2))
    cmd_parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_index}",
        "python3 -m sglang.launch_server",
        "--model-path",
        shlex.quote(JUDGE_MODEL_PATH),
        "--served-model-name",
        shlex.quote(JUDGE_MODEL_NAME),
        "--host 0.0.0.0",
        f"--port {JUDGE_PORT}",
        "--tp 1",
        "--trust-remote-code",
        "--reasoning-parser qwen3",
        "--tool-call-parser qwen3_coder",
        "--attention-backend trtllm_mha",
        "--mem-fraction-static 0.8",
        f"--context-length {context_length}",
        f"--max-running-requests {max_workers}",
        f"--cuda-graph-max-bs {cuda_graph_bs}",
        "--enable-metrics",
        "--decode-log-interval 100",
    ]
    if use_fp8_kv_cache:
        cmd_parts.extend(["--kv-cache-dtype", "fp8_e4m3"])
    cmd = " ".join(cmd_parts)
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd=REPO_DIR, preexec_fn=os.setsid)
    deadline = time.time() + 20 * 60
    while time.time() < deadline:
        probe = subprocess.run(
            ["bash", "-lc", f"curl -fsS http://127.0.0.1:{JUDGE_PORT}/health >/dev/null && curl -fsS {JUDGE_ENDPOINT}/v1/models >/dev/null"],
            cwd=REPO_DIR,
            check=False,
        )
        if probe.returncode == 0:
            _warmup_judge()
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"judge server exited early with code={proc.returncode}")
        time.sleep(5)
    raise TimeoutError("judge server did not become healthy within 20 minutes")


def _warmup_judge() -> None:
    payload = {
        "model": JUDGE_MODEL_NAME,
        "messages": [{"role": "user", "content": "Return JSON: {\"winner\":\"A\"}"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    _run(
        "curl -fsS "
        f"{JUDGE_ENDPOINT}/v1/chat/completions "
        "-H 'Content-Type: application/json' "
        f"-d {shlex.quote(json.dumps(payload))} >/dev/null"
    )


def _stop_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _train_with_config(config_path: Path, *, run_id: str, learner_gpu: int = 0) -> None:
    env = {
        "CUDA_VISIBLE_DEVICES": str(learner_gpu),
        "HF_HOME": "/hf",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONPATH": str(REPO_DIR / "scripts"),
        "JUDGE_ENDPOINT": JUDGE_ENDPOINT,
        "JUDGE_MODEL": JUDGE_MODEL_NAME,
        "RUN_ID": run_id,
        "CONFIG_PATH": str(config_path.relative_to(REPO_DIR)),
    }
    output_dir = _config_output_dir(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"python3 scripts/train.py --config {shlex.quote(str(config_path.relative_to(REPO_DIR)))} 2>&1 | tee {shlex.quote(str(output_dir / 'train.log'))}"
    _run(cmd, env=env)


def _config_output_dir(config_path: Path) -> Path:
    code = (
        "from common import load_config; "
        f"cfg=load_config({str(config_path.relative_to(REPO_DIR))!r}); "
        "print(cfg['train']['output_dir'])"
    )
    result = subprocess.run(
        ["python3", "-c", code],
        cwd=REPO_DIR,
        env={**os.environ, "PYTHONPATH": str(REPO_DIR / "scripts")},
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return REPO_DIR / result.stdout.strip()


@app.function(
    image=image,
    gpu="B200:2",
    timeout=60 * 60 * 24,
    volumes={
        "/hf": hf_volume,
        str(OUTPUTS_DIR): outputs_volume,
        str(REPO_DIR / "data" / "processed"): data_volume,
        "/root/.cache/deep_gemm": deepgemm_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface"),
    ],
)
def run_sip_b200(
    *,
    steps: int,
    run_id: str,
    judge_workers: int = 64,
    output_suffix: str = "smoke",
    judge_fp8_kv_cache: bool = False,
) -> str:
    """Run SIP with one B200 learner and one B200 SGLang FP8 judge."""
    os.chdir(REPO_DIR)
    _ensure_sip_data()
    _ensure_hf_cache(TRAIN_MODEL_ID, HF_MODEL_CACHE)
    output_dir = OUTPUTS_DIR / f"self_improving_pretraining_k16_r1_w64_modal_{output_suffix}"
    config = _write_runtime_config(
        base_config=SIP_CONFIG,
        output_dir=output_dir,
        max_steps=steps,
        eval_every=max(1, min(100, steps)),
        save_every=max(1, min(100, steps)),
        judge_workers=judge_workers,
    )
    judge = _start_judge(gpu_index=1, max_workers=judge_workers, use_fp8_kv_cache=judge_fp8_kv_cache)
    try:
        _train_with_config(config, run_id=run_id, learner_gpu=0)
    finally:
        _stop_process_group(judge)
        outputs_volume.commit()
    return str(output_dir)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 12,
    volumes={
        "/hf": hf_volume,
        str(OUTPUTS_DIR): outputs_volume,
        "/root/.cache/deep_gemm": deepgemm_volume,
        str(REPO_DIR / "data" / "processed"): data_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface"),
    ],
)
def run_thinking_sft_b200(*, init_checkpoint: str, steps: int, run_id: str) -> str:
    """Run thinking SFT after the locked data and SIP checkpoint are staged."""
    os.chdir(REPO_DIR)
    _require_paths([THINKING_SFT_DATA, THINKING_HELDOUT_DATA, THINKING_LOCK_MANIFEST, HF_MODEL_CACHE, REPO_DIR / init_checkpoint])
    output_dir = OUTPUTS_DIR / "thinking_sft_self_improved_modal"
    config = _write_runtime_config(
        base_config="configs/thinking_sft_base.yaml",
        output_dir=output_dir,
        max_steps=steps,
        eval_every=max(1, min(256, steps)),
        save_every=max(1, min(512, steps)),
        judge_workers=1,
        init_from_checkpoint=init_checkpoint,
        data_overrides={
            "examples_jsonl": str(THINKING_SFT_DATA.relative_to(REPO_DIR)),
            "raw_examples_jsonl": str(THINKING_SFT_DATA.relative_to(REPO_DIR)),
            "interleaved_thinking_examples_jsonl": str(THINKING_SFT_DATA.relative_to(REPO_DIR)),
            "heldout_examples_jsonl": str(THINKING_HELDOUT_DATA.relative_to(REPO_DIR)),
        },
    )
    _train_with_config(config, run_id=run_id, learner_gpu=0)
    outputs_volume.commit()
    return str(output_dir)


@app.function(
    image=image,
    gpu="B200:2",
    timeout=60 * 60 * 24,
    volumes={
        "/hf": hf_volume,
        str(OUTPUTS_DIR): outputs_volume,
        "/root/.cache/deep_gemm": deepgemm_volume,
        str(REPO_DIR / "data" / "processed"): data_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface"),
    ],
)
def run_rlmt_b200(
    *,
    checkpoint: str,
    steps: int,
    run_id: str,
    judge_workers: int = 64,
    row_offset: int = 0,
    eval_every: int = 100,
    save_every: int = 100,
    judge_fp8_kv_cache: bool = False,
) -> str:
    """Run the self-improved RLMT arm after reward-variance/oracle gates pass."""
    os.chdir(REPO_DIR)
    _require_paths([THINKING_RL_DATA, THINKING_LOCK_MANIFEST, HF_MODEL_CACHE, REPO_DIR / checkpoint])
    output_dir = OUTPUTS_DIR / "rlmt_self_improved_modal"
    judge = _start_judge(gpu_index=1, max_workers=judge_workers, use_fp8_kv_cache=judge_fp8_kv_cache)
    try:
        env = {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": "/hf",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONPATH": str(REPO_DIR / "scripts"),
            "RUN_ID": run_id,
        }
        cmd = " ".join(
            [
                "python3 scripts/train_rlmt.py",
                "--arm think_self_improved",
                "--config configs/thinking_sft_base.yaml",
                "--checkpoint",
                shlex.quote(checkpoint),
                "--output-dir",
                shlex.quote(str(output_dir.relative_to(REPO_DIR))),
                "--data-path",
                shlex.quote(str(THINKING_RL_DATA.relative_to(REPO_DIR))),
                "--judge-endpoint",
                shlex.quote(JUDGE_ENDPOINT),
                "--judge-model",
                shlex.quote(JUDGE_MODEL_NAME),
                "--steps",
                str(steps),
                "--prefixes-per-step 4",
                "--samples-per-prefix 16",
                "--row-offset",
                str(row_offset),
                "--eval-every",
                str(eval_every),
                "--save-every",
                str(save_every),
                "--judge-max-workers",
                str(judge_workers),
                "--enforce-stop-conditions",
            ]
        )
        _run(cmd, env=env)
    finally:
        _stop_process_group(judge)
        outputs_volume.commit()
    return str(output_dir)


@app.local_entrypoint()
def main(
    mode: str = "sip-smoke",
    steps: int = 25,
    confirm_spend: bool = False,
    judge_workers: int = 64,
    rlmt_row_offset: int = 0,
    rlmt_eval_every: int = 100,
    rlmt_save_every: int = 100,
    judge_fp8_kv_cache: bool = False,
    output_suffix: str = "full",
    run_id: str | None = None,
    checkpoint: str = "outputs/self_improving_pretraining_k16_r1_w64_modal_full/final.pt",
) -> None:
    if not confirm_spend:
        raise SystemExit("Refusing to allocate GPUs. Re-run with --confirm-spend true after the Spark overnight gate.")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    if mode == "sip-smoke":
        resolved = run_id or f"modal-sip-k16-r1-w64-smoke-{stamp}"
        print(
            run_sip_b200.remote(
                steps=steps,
            run_id=resolved,
            judge_workers=judge_workers,
            output_suffix="smoke",
            judge_fp8_kv_cache=judge_fp8_kv_cache,
            )
        )
    elif mode == "sip-full":
        resolved = run_id or f"modal-sip-k16-r1-w64-full-{stamp}"
        call = run_sip_b200.spawn(
            steps=steps,
            run_id=resolved,
            judge_workers=judge_workers,
            output_suffix=output_suffix,
            judge_fp8_kv_cache=judge_fp8_kv_cache,
        )
        print(f"spawned sip-full run_id={resolved} call_id={call.object_id}")
    elif mode == "thinking-sft":
        resolved = run_id or f"modal-thinking-sft-self-improved-{stamp}"
        print(run_thinking_sft_b200.remote(init_checkpoint=checkpoint, steps=steps, run_id=resolved))
    elif mode == "rlmt":
        resolved = run_id or f"modal-rlmt-self-improved-{stamp}"
        print(
            run_rlmt_b200.remote(
                checkpoint=checkpoint,
                steps=steps,
                run_id=resolved,
                judge_workers=judge_workers,
                row_offset=rlmt_row_offset,
                eval_every=rlmt_eval_every,
                save_every=rlmt_save_every,
                judge_fp8_kv_cache=judge_fp8_kv_cache,
            )
        )
    else:
        raise SystemExit(f"Unknown mode={mode!r}")
