#!/usr/bin/env python3
"""Materialize a checkpoint as a Hugging Face model directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from common import load_config
from eval_thinking import ARM_SPECS, load_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(ARM_SPECS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    spec = ARM_SPECS[args.arm]
    cfg = load_config(spec["config"])
    cfg.setdefault("runtime", {})["device"] = "cpu"
    cfg["runtime"]["dtype"] = args.dtype
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, Path(spec["checkpoint"]), torch.device("cpu"))
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    metadata = {
        "arm": args.arm,
        "config": spec["config"],
        "checkpoint": spec["checkpoint"],
        "dtype": args.dtype,
    }
    (out_dir / "arm_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), **metadata}), flush=True)


if __name__ == "__main__":
    main()
