#!/usr/bin/env python3
"""Offline compatibility smoke for student and teacher model paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import resolve_hf_path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _cache_root(path: Path) -> Path:
    parts = path.parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        return Path(*parts[:idx])
    return path


def inspect_files(resolved: Path) -> dict[str, Any]:
    cache_root = _cache_root(resolved)
    incomplete = sorted(str(p) for p in (cache_root / "blobs").glob("*.incomplete")) if (cache_root / "blobs").exists() else []
    required = ["config.json"]
    missing_required = [name for name in required if not (resolved / name).exists()]
    tokenizer_present = any((resolved / name).exists() for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    if not tokenizer_present:
        missing_required.append("tokenizer.json|tokenizer.model|vocab.json")

    index_path = resolved / "model.safetensors.index.json"
    shard_paths: list[Path]
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        shard_paths = [resolved / name for name in shard_names]
    else:
        shard_paths = sorted(resolved.glob("*.safetensors"))
    missing_shards = [str(path) for path in shard_paths if not path.exists()]
    total_safetensor_bytes = sum(path.resolve().stat().st_size for path in shard_paths if path.exists())
    return {
        "missing_required": missing_required,
        "incomplete_blob_count": len(incomplete),
        "incomplete_blobs": incomplete[:20],
        "safetensor_shard_count": len(shard_paths),
        "missing_safetensor_shards": missing_shards,
        "total_safetensor_bytes": total_safetensor_bytes,
    }


def inspect_model(path: str, load_causal_lm: bool) -> dict[str, Any]:
    resolved = resolve_hf_path(path)
    cfg = AutoConfig.from_pretrained(resolved, local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(resolved, local_files_only=True, trust_remote_code=True)
    files = inspect_files(resolved)
    out: dict[str, Any] = {
        "input_path": path,
        "resolved_path": str(resolved),
        "exists": Path(resolved).exists(),
        **files,
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "tokenizer_len": len(tok),
        "tokenizer_ok": len(tok) <= int(getattr(cfg, "vocab_size", len(tok))),
    }
    if load_causal_lm:
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
        )
        out["causal_lm_class"] = model.__class__.__name__
        out["num_parameters"] = sum(p.numel() for p in model.parameters())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--teacher-path", default="")
    parser.add_argument("--load-student-causal-lm", action="store_true")
    args = parser.parse_args()

    result = {
        "student": inspect_model(args.student_path, args.load_student_causal_lm),
        "teacher": inspect_model(args.teacher_path, False) if args.teacher_path else None,
    }
    if not result["student"]["tokenizer_ok"]:
        raise RuntimeError(f"student tokenizer exceeds vocab: {result['student']}")
    for name in ("student", "teacher"):
        model = result.get(name)
        if not model:
            continue
        if model["missing_required"] or model["incomplete_blob_count"] or model["missing_safetensor_shards"]:
            raise RuntimeError(f"{name} model files are incomplete: {model}")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
