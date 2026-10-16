"""Post-selection oracle diagnostics; never modifies or retrains PPO policies."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tools.run_physical_microgrid_milp import solve_day
from tools.run_sb3_reward_experiment import audit_snapshot, digest, save_json
from tools.run_sb3_reward_tuning import assessment
from tools.sb3_dispatch_reward_env import load_environment_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    manifest = audit_snapshot(experiment)
    holdout = json.loads((experiment / "holdout/summary.json").read_text(encoding="utf-8"))
    if holdout["rows"] != manifest["reserved_test_rows"]:
        raise ValueError("Held-out dates differ from their preregistered reservation")
    output = experiment / "holdout/oracle"
    output.mkdir(exist_ok=False)
    env_class = load_environment_class(experiment / "snapshot")
    config = deepcopy(manifest["env_args"])
    settings = deepcopy(config)
    settings.update(result_dir=str(output), log_trajectories=False)
    env = env_class(settings, "holdout_oracle")
    days, all_records = [], []
    for row in holdout["rows"]:
        enabled = solve_day(env, config, row, "test", config["initial_soc"])
        disabled = solve_day(env, config, row, "test", config["initial_soc"],
                             battery_enabled=False)
        if not enabled["all_solved"] or not disabled["all_solved"]:
            raise RuntimeError("Holdout reference did not reach optimal solver status")
        days.append({
            "row_idx": row, "cyclic_cost": enabled["full_cost"],
            "no_battery_cost": disabled["full_cost"],
            "max_replay_error": max(enabled["max_replay_error"], disabled["max_replay_error"]),
            "interpolation_error_bound": enabled["interpolation_error_bound"],
            "oracle_agent_metrics": enabled["agents"],
        })
        all_records.extend(enabled["records"])
    oracle_cost = float(np.mean([d["cyclic_cost"] for d in days]))
    no_battery_cost = float(np.mean([d["no_battery_cost"] for d in days]))
    evaluated = [
        {"mode": r["mode"], "seed": r["seed"], **assessment(
            r["test"], oracle_cost, manifest["acceptance"],
        )} for r in holdout["runs"]
    ]
    aggregate = {}
    for mode in manifest["modes"]:
        results = [r for r in holdout["runs"] if r["mode"] == mode]
        cost = float(np.mean([r["test"]["mean_full_cost"] for r in results]))
        aggregate[mode] = {
            "mean_operating_cost": cost,
            "cost_excess_vs_cyclic_reference": cost / oracle_cost - 1,
            "cost_reduction_vs_no_battery_reference": 1 - cost / no_battery_cost,
            "passing_seeds": sum(r["passed"] for r in evaluated if r["mode"] == mode),
            "seed_count": len(results),
            "mean_terminal_error_kwh": float(np.mean([
                r["test"]["mean_terminal_error_kwh"] for r in results
            ])),
        }
    save_json(output / "summary.json", {
        "rows": holdout["rows"], "days": days, "oracle_mean_cost": oracle_cost,
        "no_battery_mean_cost": no_battery_cost,
        "policy_acceptance": evaluated, "aggregate": aggregate,
        "scope": "Post-selection explanation only; no reward or model selection from this result",
        "caveat": "Soft inventory tolerance differs from exact cyclic constraint; not a strict optimality gap",
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): digest(p) for p in (
                ROOT / "tools/run_physical_microgrid_milp.py",
                ROOT / "tools/run_current_model_milp.py", Path(__file__),
            )
        },
    })
    pd.DataFrame(all_records).to_csv(output / "trajectories.csv", index=False)
    audit_snapshot(experiment)
    print(json.dumps({"oracle": oracle_cost, "no_battery": no_battery_cost,
                      "aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
