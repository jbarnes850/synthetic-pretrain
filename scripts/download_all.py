"""Download SmolLM2 + FineWeb-Edu shards + FinePhrase-math shards with clean logging."""
import time
import traceback
from pathlib import Path
from huggingface_hub import snapshot_download

LOG = Path("logs/downloads.log")
LOG.parent.mkdir(exist_ok=True)

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}\n"
    with open(LOG, "a") as f:
        f.write(line)
    print(line, end="", flush=True)

def dl(repo_id, repo_type="model", allow_patterns=None):
    log(f"START {repo_id} ({repo_type}) patterns={allow_patterns}")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
        )
        log(f"DONE  {repo_id} -> {path}")
        return True
    except Exception as e:
        log(f"FAIL  {repo_id}: {type(e).__name__}: {e}")
        traceback.print_exc(file=open(LOG, "a"))
        return False

log("=== driver start ===")
results = {}
results["smollm2"] = dl("HuggingFaceTB/SmolLM2-1.7B-Instruct", "model")
results["fineweb"] = dl("HuggingFaceFW/fineweb-edu", "dataset",
                        ["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"])
results["finephrase"] = dl("HuggingFaceFW/finephrase", "dataset",
                           ["math/000_00000_*.parquet"])
log(f"=== driver end: {results} ===")
