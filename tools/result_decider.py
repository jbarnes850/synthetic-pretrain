import math
from typing import Any


REQUIRED_METRICS = [
    "primary_metric",
    "val_loss_raw",
    "val_loss_selected",
    "judge_win_rate_vs_raw",
    "chosen_original_rate",
    "chosen_finephrase_rate",
    "chosen_rollout_rate",
    "repetition_4gram_rate",
    "tokens_seen",
    "tok_per_sec",
    "available_mem_gib_min",
]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def invalid_metrics(metrics: dict[str, Any]) -> list[str]:
    return [key for key in REQUIRED_METRICS if key not in metrics or not finite(metrics.get(key))]


def best_kept(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    kept = [
        row
        for row in history
        if row.get("status") == "keep" and finite(row.get("metrics", {}).get("primary_metric"))
    ]
    if not kept:
        return None
    return min(kept, key=lambda row: float(row["metrics"]["primary_metric"]))


def is_raw_baseline_config(config: Any) -> bool:
    text = str(config)
    return "baseline_raw" in text or "blog_raw" in text or "raw_ntp" in text


def research_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ignore earlier scaffold/infra rows once corrected real-data runs exist."""
    rows = [
        row
        for row in history
        if row.get("config", {}).get("data") or "_real_" in str(row.get("config", {}).get("run_id", ""))
    ]
    return rows or history


def raw_baseline(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw_rows = [
        row
        for row in history
        if row.get("status") == "keep"
        and is_raw_baseline_config(row.get("config", {}).get("config", ""))
        and finite(row.get("metrics", {}).get("val_loss_raw"))
    ]
    if raw_rows:
        return raw_rows[-1]


def raw_anchor(history: list[dict[str, Any]]) -> float | None:
    raw_row = raw_baseline(history)
    if raw_row is not None:
        return float(raw_row["metrics"]["val_loss_raw"])
    kept = [row for row in history if row.get("status") == "keep" and finite(row.get("metrics", {}).get("val_loss_raw"))]
    if not kept:
        return None
    return min(float(row["metrics"]["val_loss_raw"]) for row in kept)


def decide_status(history: list[dict[str, Any]], metrics: dict[str, Any], config: str) -> dict[str, Any]:
    missing = invalid_metrics(metrics)
    if missing:
        return {"status": "crash", "reason": f"missing_or_nonfinite_metrics:{','.join(missing)}"}

    if float(metrics["available_mem_gib_min"]) < 4.0:
        return {"status": "crash", "reason": "memory_floor_breach"}

    if is_raw_baseline_config(config):
        return {"status": "keep", "reason": "raw_baseline_anchor"}

    scoped_history = research_history(history)
    best = best_kept(scoped_history)
    raw_row = raw_baseline(scoped_history)
    anchor = raw_anchor(scoped_history)
    primary = float(metrics["primary_metric"])
    val_raw = float(metrics["val_loss_raw"])

    if anchor is not None and val_raw > anchor * 1.05:
        return {
            "status": "discard",
            "reason": "raw_val_loss_regression",
            "anchor_raw": anchor,
            "val_loss_raw": val_raw,
        }

    rep = float(metrics["repetition_4gram_rate"])
    kept_reps = []
    if raw_row is not None and finite(raw_row.get("metrics", {}).get("repetition_4gram_rate")):
        kept_reps = [float(raw_row["metrics"]["repetition_4gram_rate"])]
    else:
        kept_reps = [
            float(row["metrics"]["repetition_4gram_rate"])
            for row in scoped_history
            if row.get("status") == "keep" and finite(row.get("metrics", {}).get("repetition_4gram_rate"))
        ]
    if kept_reps:
        best_rep = min(kept_reps)
        if (best_rep > 0 and rep > best_rep * 1.25) or (best_rep == 0 and rep > 0.01):
            return {
                "status": "discard",
                "reason": "repetition_regression",
                "best_repetition_4gram_rate": best_rep,
                "repetition_4gram_rate": rep,
            }

    if best is None:
        return {"status": "keep", "reason": "first_finite_result"}

    best_primary = float(best["metrics"]["primary_metric"])
    if primary < best_primary:
        return {
            "status": "keep",
            "reason": "primary_improved",
            "best_primary": best_primary,
            "primary": primary,
        }

    win_rate = float(metrics.get("judge_win_rate_vs_raw", 0.0) or 0.0)
    raw_win = 0.0
    if raw_row is not None:
        raw_win = float(raw_row.get("metrics", {}).get("judge_win_rate_vs_raw", 0.0) or 0.0)
    best_win = max(
        [
            raw_win,
            *[
                float(row.get("metrics", {}).get("judge_win_rate_vs_raw", 0.0) or 0.0)
                for row in scoped_history
                if row.get("status") == "keep"
            ],
        ],
        default=raw_win,
    )
    if win_rate >= best_win + 0.03:
        return {
            "status": "keep",
            "reason": "judge_win_rate_improved",
            "best_win_rate": best_win,
            "win_rate": win_rate,
        }

    return {
        "status": "discard",
        "reason": "no_keep_rule_satisfied",
        "best_primary": best_primary,
        "primary": primary,
    }
