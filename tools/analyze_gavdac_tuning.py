"""Analyze completed direct_main runs from exact evaluation logs and final JSON."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"([-+\d.eE]+)"


def evaluation_history(stdout, stderr):
    steps = [int(s) for s in re.findall(r"root t_env: (\d+) / \d+", stderr)]
    values = {
        source: [float(x) for x in re.findall(rf"^{label} eval\s+{NUMBER}\s*$",
                                             stdout, flags=re.MULTILINE)]
        for source, label in (("rl", "RL"), ("ea", "EA"))
    }
    if not steps or any(len(v) != len(steps) for v in values.values()):
        raise ValueError("Evaluation step/return counts differ or logs are incomplete")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Evaluation steps must strictly increase")
    if not all(math.isfinite(x) for v in values.values() for x in v):
        raise ValueError("Non-finite evaluation return")
    return [{"step": step, **{source: v[i] for source, v in values.items()}}
            for i, step in enumerate(steps)]


def summarize_series(history, source, window_steps=500000):
    x = np.array([row["step"] for row in history])
    y = np.array([row[source] for row in history])
    index = int(np.argmax(y))
    windows = []
    for lower in range(0, int(x[-1]) + 1, window_steps):
        selected = y[(x >= lower) & (x < lower + window_steps)]
        if selected.size:
            windows.append({
                "from_step": lower, "to_step": lower + window_steps,
                "count": int(selected.size), "mean": float(selected.mean()),
                "std": float(selected.std()), "min": float(selected.min()),
                "max": float(selected.max()),
            })
    tail = x >= max(0, x[-1] - window_steps)
    tail_x, tail_y = x[tail], y[tail]
    slope = (float(np.polyfit((tail_x - tail_x[0]) / 1e6, tail_y, 1)[0])
             if tail_x.size >= 2 else None)
    return {
        "best_return": float(y[index]), "best_step": int(x[index]),
        "last_return": float(y[-1]), "last_step": int(x[-1]),
        "best_in_last_10pct": bool(x[index] >= 0.9 * x[-1]),
        "tail_return_slope_per_million_steps": slope,
        "windows": windows,
        "scope": "Fixed validation reward, not a convergence test or independent seed estimate",
    }


def summarize_candidates(report):
    candidates = report["candidates"]
    if not candidates or len({row["row_idx"] for row in candidates}) != len(candidates):
        raise ValueError("Candidate days must be nonempty and distinct")
    baseline = sum(row["baseline_cost"] for row in candidates)
    cost = sum(row["operating_cost"] for row in candidates)
    correction = sum(row["inventory_adjustment_cu"] for row in candidates)
    if baseline <= 0 or not all(math.isfinite(x) for x in (baseline, cost, correction)):
        raise ValueError("Invalid cost totals")
    return {
        "checkpoint_step": report["checkpoint_step"],
        "rows": [row["row_idx"] for row in candidates],
        "day_count": len(candidates),
        "mean_cost": cost / len(candidates),
        "mean_baseline": baseline / len(candidates),
        "weighted_saving_fraction": 1 - cost / baseline,
        "inventory_adjusted_saving_fraction": 1 - (cost + correction) / baseline,
        "qualified_count": sum(bool(row["qualified"]) for row in candidates),
        "max_terminal_soc_error": max(row["max_terminal_soc_error"] for row in candidates),
        "min_bidirectional_battery_fraction": min(
            row["bidirectional_battery_fraction"] for row in candidates
        ),
        "min_day_saving_fraction": min(row["cost_saving_ratio"] for row in candidates),
        "scope": "All fixed candidate validation days; not selected best-day averages",
    }


def analyze(directory, recovery=None):
    launch_path = directory / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8")) if launch_path.exists() else {}
    stdout_path = directory / launch.get("stdout", "stdout.log")
    stderr_path = directory / launch.get("stderr", "stderr.log")
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    reports_root = directory / "dispatch_showcases"
    recovery_record = None
    if recovery is not None:
        from recover_gavdac_showcase import reporting_failure_step
        recovery_record = json.loads(recovery.read_text(encoding="utf-8"))
        expected_step = reporting_failure_step(stderr, recovery_record["requested_steps"])
        if (Path(recovery_record["original_run"]).resolve() != directory.resolve()
                or recovery_record["actual_steps"] != expected_step
                or recovery_record["reporting_status"] !=
                "recovered; W&B finish returned successfully"
                or recovery_record["original_stderr_sha256"] !=
                hashlib.sha256(stderr_path.read_bytes()).hexdigest()):
            raise ValueError("Recovery evidence does not match original training")
        reports_root = Path(recovery_record["reports"])
    else:
        if "root Finished Training" not in stderr or "Exiting script" not in stdout:
            raise ValueError("Training or W&B shutdown has not completed successfully")
        if "Traceback (most recent call last)" in stdout + stderr:
            raise ValueError("Training logs contain a traceback")
    history = evaluation_history(stdout, stderr)
    summaries = {}
    for path in sorted(reports_root.rglob("selection.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        source = report["policy_source"]
        if source in summaries:
            raise ValueError("Multiple final reports for a policy; select a unique run")
        summaries[source] = summarize_candidates(report)
        if recovery_record is not None and (
            report["policy_sha256"] != recovery_record["policy_sha256"][source]
            or report["logged_step"] != recovery_record["actual_steps"]
        ):
            raise ValueError("Recovered report checkpoint/step mismatch")
    if recovery_record is not None and set(summaries) != {"rl", "ea"}:
        raise ValueError("Both recovered policy reports are required")
    return {
        "run_directory": str(directory.resolve()),
        "log_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (stdout_path, stderr_path)
        },
        "series": {source: summarize_series(history, source) for source in ("rl", "ea")},
        "candidate_economics": summaries, "history": history,
        "reporting_recovery": recovery_record,
        "warning": "Best reward selection and cyclic-day cost ranking differ; compare both. "
                   "Single-run late improvement is evidence against declaring convergence, "
                   "not proof of what longer training will achieve.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery", type=Path)
    args = parser.parse_args()
    directory = args.run_dir.resolve(strict=True)
    output = args.output.resolve()
    results = (ROOT / "results").resolve()
    if not directory.is_relative_to(results) or not output.is_relative_to(results):
        raise ValueError("Inputs and generated reports must stay under results/")
    payload = analyze(directory, args.recovery)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "history"},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
