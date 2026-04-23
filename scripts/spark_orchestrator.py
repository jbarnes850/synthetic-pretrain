#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
QUEUE = ROOT / "queue"
LOCKS = ROOT / "locks"
WORKERS = ROOT / "workers"
LOCAL_HF_STAGE = Path(os.environ.get("SPARK_LOCAL_HF_STAGE", "/tmp/spark_hf_cache")) / "hub"

NODES = {
    "cfd0": {
        "host": "jarrodbarnes@100.113.207.120",
        "direct_ip": "192.168.100.11",
        "project": "/home/jarrodbarnes/synthetic-pretrain",
    },
    "f7e2": {
        "host": "jarrodbarnes@100.70.91.108",
        "direct_ip": "192.168.100.10",
        "project": "/home/jarrodbarnes/synthetic-pretrain",
    },
}


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def sh(host: str, command: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return run(["ssh", host, command], check=check, capture=capture)


def ensure_dirs() -> None:
    for path in [QUEUE / "pending", QUEUE / "running", QUEUE / "done", QUEUE / "failed", LOCKS, WORKERS]:
        path.mkdir(parents=True, exist_ok=True)


def write_job(run_id: str, config: str, node: str | None = None) -> Path:
    ensure_dirs()
    job = {"run_id": run_id, "config": config, "node": node, "created_at": time.time()}
    path = QUEUE / "pending" / f"{run_id}.json"
    path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    return path


def sync_code_to(node: str) -> None:
    meta = NODES[node]
    excludes = [
        "--exclude",
        ".git/",
        "--exclude",
        ".venv/",
        "--exclude",
        "__pycache__/",
        "--exclude",
        "logs/",
        "--exclude",
        "outputs/",
        "--exclude",
        "workers/",
        "--exclude",
        "queue/",
        "--exclude",
        "locks/",
    ]
    run(["rsync", "-az", *excludes, str(ROOT) + "/", f"{meta['host']}:{meta['project']}/"])


def sync_data_to_f7e2() -> None:
    # Prefer the direct Spark interconnect for bulk data. Fall back to a
    # local-mediated Tailscale copy only if cfd0 cannot SSH to f7e2 directly.
    dataset_dir = "datasets--HuggingFaceFW--finephrase"
    f7_check = sh(
        NODES["f7e2"]["host"],
        f"test -d ~/.cache/huggingface/hub/{dataset_dir}/snapshots",
        check=False,
    )
    if f7_check.returncode == 0:
        print("FinePhrase cache already present on f7e2", flush=True)
        return

    sh(
        NODES["cfd0"]["host"],
        f"test -d ~/.cache/huggingface/hub/{dataset_dir}/snapshots",
    )

    direct_check = sh(
        NODES["cfd0"]["host"],
        f"ssh -o BatchMode=yes -o ConnectTimeout=5 jarrodbarnes@{NODES['f7e2']['direct_ip']} true",
        check=False,
    )
    if direct_check.returncode == 0:
        sh(
            NODES["cfd0"]["host"],
            "ssh -o BatchMode=yes "
            f"jarrodbarnes@{NODES['f7e2']['direct_ip']} mkdir -p ~/.cache/huggingface/hub && "
            f"rsync -a --partial --info=progress2 "
            f"~/.cache/huggingface/hub/{dataset_dir} "
            f"jarrodbarnes@{NODES['f7e2']['direct_ip']}:~/.cache/huggingface/hub/",
        )
        return

    print("Direct cfd0->f7e2 sync unavailable; falling back to local staging", flush=True)
    sh(NODES["f7e2"]["host"], "mkdir -p ~/.cache/huggingface/hub")
    LOCAL_HF_STAGE.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-az", f"{NODES['cfd0']['host']}:~/.cache/huggingface/hub/{dataset_dir}", str(LOCAL_HF_STAGE) + "/"])
    run(["rsync", "-az", str(LOCAL_HF_STAGE / dataset_dir), f"{NODES['f7e2']['host']}:~/.cache/huggingface/hub/"])


def sync_results_from_cfd0() -> None:
    meta = NODES["cfd0"]
    run(["rsync", "-az", f"{meta['host']}:{meta['project']}/results.json", str(ROOT) + "/"], check=False)


def sync_controller_state_to_cfd0() -> None:
    meta = NODES["cfd0"]
    for rel in ["results.json", "queue/", "workers/"]:
        path = ROOT / rel
        if path.exists():
            src = str(path) + ("/" if rel.endswith("/") else "")
            cmd = ["rsync", "-az"]
            if rel == "queue/":
                cmd.append("--delete")
            run([*cmd, src, f"{meta['host']}:{meta['project']}/{rel}"], check=False)


def acquire_lock(node: str, run_id: str) -> Path:
    ensure_dirs()
    path = LOCKS / f"{node}.lock"
    if path.exists():
        raise RuntimeError(f"worker lock already exists: {path}")
    path.write_text(json.dumps({"node": node, "run_id": run_id, "locked_at": time.time()}, indent=2) + "\n", encoding="utf-8")
    return path


def release_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def move_job(run_id: str, source: str, target: str, **updates: Any) -> None:
    src = QUEUE / source / f"{run_id}.json"
    dst = QUEUE / target / f"{run_id}.json"
    data: dict[str, Any] = {}
    if src.exists():
        data = json.loads(src.read_text(encoding="utf-8"))
        src.unlink()
    data.update(updates)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def node_free(node: str) -> bool:
    meta = NODES[node]
    cmd = "docker ps -q | wc -l && pgrep -af 'scripts/train.py|scripts/run_experiment.sh' || true"
    out = sh(meta["host"], cmd, capture=True).stdout.strip().splitlines()
    running_containers = int(out[0]) if out else 0
    return running_containers == 0 and len(out) == 1


def dispatch_job(job_path: Path, node: str) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    meta = NODES[node]
    run_id = job["run_id"]
    config = job["config"]
    sync_code_to(node)
    remote = (
        f"cd {meta['project']} && "
        f"mkdir -p queue/running queue/done queue/failed workers/{node}/{run_id} logs && "
        f"SPARK_WORKER_NAME={node} nohup scripts/run_experiment.sh {config} {run_id} "
        f"> logs/{run_id}.worker.out 2>&1; "
        f"status=$?; "
        f"if [ $status -eq 0 ]; then mv queue/running/{run_id}.json queue/done/{run_id}.json 2>/dev/null || true; "
        f"else mv queue/running/{run_id}.json queue/failed/{run_id}.json 2>/dev/null || true; fi; "
        f"exit $status"
    )
    running_path = QUEUE / "running" / job_path.name
    job["node"] = node
    job["started_at"] = time.time()
    running_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    job_path.unlink()
    sh(meta["host"], f"mkdir -p {meta['project']}/queue/running && cat > {meta['project']}/queue/running/{run_id}.json <<'EOF'\n{json.dumps(job)}\nEOF")
    sh(meta["host"], remote, check=False)


def fetch_worker_result(node: str, run_id: str) -> Path:
    meta = NODES[node]
    local_dir = WORKERS / node / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-az", f"{meta['host']}:{meta['project']}/workers/{node}/{run_id}/", str(local_dir) + "/"], check=False)
    return local_dir


def log_result(node: str, run_id: str, config: str, wall_clock: float, status_hint: str) -> dict[str, Any]:
    from result_decider import decide_status

    local_dir = fetch_worker_result(node, run_id)
    metrics_file = local_dir / "metrics.json"
    if not metrics_file.exists():
        metrics = {}
        status = "crash"
        reason = "missing_metrics"
    else:
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if status_hint != "ok":
            status = "crash"
            reason = status_hint
        else:
            history = json.loads((ROOT / "results.json").read_text(encoding="utf-8")) if (ROOT / "results.json").exists() else []
            decision = decide_status(history, metrics, config)
            status = decision["status"]
            reason = decision["reason"]

    payload = json.dumps(metrics)
    cfg_payload = json.dumps({"config": config, "node": node, "run_id": run_id, "decision_reason": reason})
    run(
        [
            "python3",
            "tools/experiment_log.py",
            "log",
            "--commit",
            git_commit(),
            "--metrics",
            payload,
            "--config",
            cfg_payload,
            "--cost",
            "0",
            "--wall-clock",
            str(wall_clock),
            "--gpu",
            f"DGX Spark GB10 {node}",
            "--status",
            status,
            "--description",
            f"parallel worker {run_id} on {node}: {reason}",
        ],
        check=False,
    )
    return {"run_id": run_id, "node": node, "status": status, "reason": reason, "metrics": metrics}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "nogit"


def run_pair(config_a: str, config_b: str) -> None:
    ensure_dirs()
    sync_results_from_cfd0()
    sync_data_to_f7e2()
    jobs = [
        ("cfd0", f"parallel_cfd0_{int(time.time())}", config_a),
        ("f7e2", f"parallel_f7e2_{int(time.time())}", config_b),
    ]
    procs = []
    started = {}
    for node, run_id, config in jobs:
        write_job(run_id, config, node)
        lock = acquire_lock(node, run_id)
        sync_code_to(node)
        meta = NODES[node]
        started[run_id] = time.time()
        move_job(run_id, "pending", "running", node=node, started_at=started[run_id])
        cmd = (
            f"cd {meta['project']} && "
            f"mkdir -p workers/{node}/{run_id} logs && "
            f"SPARK_WORKER_NAME={node} scripts/run_experiment.sh {config} {run_id}"
        )
        procs.append((node, run_id, config, lock, subprocess.Popen(["ssh", meta["host"], cmd], text=True)))
        sync_controller_state_to_cfd0()
    for node, run_id, config, lock, proc in procs:
        try:
            rc = proc.wait()
            wall = time.time() - started[run_id]
            status_hint = "ok" if rc == 0 else f"rc_{rc}"
            result = log_result(node, run_id, config, wall, status_hint)
            target = "done" if result["status"] in {"keep", "discard"} else "failed"
            move_job(run_id, "running", target, completed_at=time.time(), result_status=result["status"], reason=result["reason"])
            print(json.dumps(result, indent=2), flush=True)
        finally:
            release_lock(lock)
            sync_controller_state_to_cfd0()


def smoke_f7e2() -> None:
    ensure_dirs()
    sync_data_to_f7e2()
    sync_code_to("f7e2")
    meta = NODES["f7e2"]
    cmd = f"cd {meta['project']} && scripts/run_validation.sh"
    sh(meta["host"], cmd)


def run_role(node: str, role: str, config: str, run_id: str | None) -> None:
    ensure_dirs()
    sync_code_to(node)
    meta = NODES[node]
    role_run_id = run_id or f"{role}_{node}_{int(time.time())}"
    cmd = (
        f"cd {meta['project']} && "
        f"mkdir -p logs workers/{node}/{role_run_id} && "
        f"SPARK_WORKER_NAME={node} scripts/run_judge_role_mode.sh {role} {config} {role_run_id}"
    )
    sh(meta["host"], cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    sub.add_parser("init").set_defaults(func=lambda _args: ensure_dirs())
    sub.add_parser("sync-f7-data").set_defaults(func=lambda _args: sync_data_to_f7e2())
    sub.add_parser("smoke-f7e2").set_defaults(func=lambda _args: smoke_f7e2())
    pair = sub.add_parser("run-pair")
    pair.add_argument("--config-a", required=True)
    pair.add_argument("--config-b", required=True)
    pair.set_defaults(func=lambda args: run_pair(args.config_a, args.config_b))
    role = sub.add_parser("run-role")
    role.add_argument("--node", required=True, choices=sorted(NODES))
    role.add_argument("--role", required=True, choices=["serve-judge", "eval-select", "trainer"])
    role.add_argument("--config", default="configs/judge_eval.yaml")
    role.add_argument("--run-id")
    role.set_defaults(func=lambda args: run_role(args.node, args.role, args.config, args.run_id))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
