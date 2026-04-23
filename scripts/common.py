import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    include = cfg.pop("include", None)
    if include:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = Path.cwd() / include_path
        base = load_config(include_path)
        return deep_merge(base, cfg)
    return cfg


def latest_snapshot(repo_cache: str | Path) -> Path:
    repo_cache = Path(repo_cache)
    snapshots = repo_cache / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"Missing snapshots directory: {snapshots}")
    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No snapshots found under {snapshots}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def jsonl_iter(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_available_mem_gib() -> float:
    try:
        out = subprocess.check_output(["bash", "-lc", "free -b | awk '/Mem:/ {print $7}'"], text=True)
        return float(out.strip()) / (1024**3)
    except Exception:
        return -1.0


def now_run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def emit_metric(name: str, value: Any) -> None:
    print(f"{name}: {value}", flush=True)


def safe_mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))
