from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def init_wandb_run(
    *,
    name: str,
    job_type: str,
    config: dict[str, Any],
    output_dir: str | Path,
):
    load_dotenv()
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"wandb: disabled because import failed: {exc}", flush=True)
        return None

    project = os.environ.get("WANDB_PROJECT", "synthetic-pretrain")
    entity = os.environ.get("WANDB_ENTITY") or None
    group = os.environ.get("WANDB_GROUP") or None
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type=job_type,
        config=config,
        dir=str(output_dir),
        resume="allow",
    )
    return run


def log_wandb(run, metrics: dict[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return
    clean: dict[str, int | float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            clean[key] = int(value)
        elif isinstance(value, int | float):
            clean[key] = value
    if clean:
        run.log(clean, step=step)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()
