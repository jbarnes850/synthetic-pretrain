#!/usr/bin/env python3
"""Cache the bounded DCLM/FineMath parquet shard set referenced by a config."""
from __future__ import annotations

import argparse
import json
from typing import Any

from common import load_config
from huggingface_hub import HfApi, hf_hub_download


def source_files(api: HfApi, source: dict[str, Any]) -> list[str]:
    repo_cache = str(source["repo_cache"])
    marker = "/datasets--"
    if marker not in repo_cache:
        raise ValueError(f"Cannot infer repo id from repo_cache={repo_cache!r}")
    repo_id = repo_cache.split(marker, 1)[1].replace("--", "/")
    subset = str(source.get("subset") or "").strip("/")
    pattern_suffix = str(source.get("path_glob") or "**/*.parquet").split("*")[-1] or ".parquet"
    limit = int(source.get("max_parquet_files") or 0)

    files = []
    for path in api.list_repo_files(repo_id, repo_type="dataset"):
        if subset and not path.startswith(subset + "/"):
            continue
        if path.endswith(pattern_suffix):
            files.append(path)
    files = sorted(files)
    if limit > 0:
        files = files[:limit]
    return repo_id, files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/self_improving_pretraining.yaml")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    api = HfApi()
    summary = {"config": args.config, "sources": []}
    for source in cfg["data"]["pretraining_sources"]:
        repo_id, files = source_files(api, source)
        source_summary = {
            "name": source.get("name", repo_id),
            "repo_id": repo_id,
            "requested_files": len(files),
            "files": [],
        }
        print(json.dumps({k: source_summary[k] for k in ["name", "repo_id", "requested_files"]}), flush=True)
        for idx, filename in enumerate(files, start=1):
            print(json.dumps({"source": source_summary["name"], "idx": idx, "total": len(files), "file": filename}), flush=True)
            path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                cache_dir=args.cache_dir,
            )
            source_summary["files"].append(path)
        summary["sources"].append(source_summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
