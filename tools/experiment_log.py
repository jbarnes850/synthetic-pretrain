#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path


RESULTS = Path("results.json")


def load_rows() -> list[dict]:
    if not RESULTS.exists():
        return []
    return json.loads(RESULTS.read_text(encoding="utf-8") or "[]")


def save_rows(rows: list[dict]) -> None:
    RESULTS.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def cmd_log(args) -> None:
    metrics = json.loads(args.metrics)
    if args.status != "crash":
        from result_decider import invalid_metrics

        invalid = invalid_metrics(metrics)
        if invalid:
            raise ValueError(f"Cannot log {args.status} with invalid metrics: {invalid}")
    rows = load_rows()
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": args.commit,
        "metrics": metrics,
        "config": json.loads(args.config),
        "cost": float(args.cost),
        "wall_clock": float(args.wall_clock),
        "gpu": args.gpu,
        "status": args.status,
        "description": args.description,
    }
    rows.append(row)
    save_rows(rows)
    print(json.dumps(row, indent=2))


def cmd_history(_args) -> None:
    print(json.dumps(load_rows(), indent=2))


def cmd_best(_args) -> None:
    rows = [r for r in load_rows() if r.get("status") == "keep"]
    if not rows:
        print("{}")
        return
    best = min(rows, key=lambda r: r.get("metrics", {}).get("primary_metric", float("inf")))
    print(json.dumps(best, indent=2))


def cmd_decide(args) -> None:
    from result_decider import decide_status

    history = load_rows()
    metrics = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
    decision = decide_status(history, metrics, args.config)
    print(json.dumps(decision, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    log = sub.add_parser("log")
    log.add_argument("--commit", required=True)
    log.add_argument("--metrics", required=True)
    log.add_argument("--config", required=True)
    log.add_argument("--cost", required=True)
    log.add_argument("--wall-clock", required=True)
    log.add_argument("--gpu", required=True)
    log.add_argument("--status", required=True, choices=["keep", "discard", "crash"])
    log.add_argument("--description", required=True)
    log.set_defaults(func=cmd_log)
    hist = sub.add_parser("history")
    hist.set_defaults(func=cmd_history)
    best = sub.add_parser("best")
    best.set_defaults(func=cmd_best)
    decide = sub.add_parser("decide")
    decide.add_argument("--metrics-file", required=True)
    decide.add_argument("--config", required=True)
    decide.set_defaults(func=cmd_decide)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
