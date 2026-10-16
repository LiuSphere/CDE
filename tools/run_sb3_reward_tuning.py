"""Validation-only inventory reward tuning, followed by a frozen confirmation."""

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_sb3_reward_experiment import (
    audit_snapshot, digest, launch_matrix, prepare, save_json,
)

RECIPES = {
    "terminal_ref": {"mode": "terminal", "settings": {}},
    "deadline_l1": {"mode": "deadline", "settings": {}},
    "deadline_q16": {"mode": "deadline", "settings": {"inventory_quadratic": 16}},
    "deadline_q64": {"mode": "deadline", "settings": {"inventory_quadratic": 64}},
    "market_q16": {"mode": "deadline", "settings": {
        "inventory_quadratic": 16, "deadline_start_hour": 20,
        "future_price_potential": True, "deficit_only_potential": True,
    }},
    "market_q64": {"mode": "deadline", "settings": {
        "inventory_quadratic": 64, "deadline_start_hour": 20,
        "future_price_potential": True, "deficit_only_potential": True,
    }},
}
NEW_TEST_ROWS = [922, 934, 958, 970, 994, 1006, 1030, 1042, 1074, 1086]
ACCEPTANCE = {
    "max_terminal_soc_error": 0.03,
    "max_mean_cost_above_cyclic_oracle": 0.05,
    "minimum_soc_span": 0.48,
    "minimum_bidirectional_fraction": 1.0,
}


def assessment(evaluation, oracle_cost, requirements=ACCEPTANCE):
    agents = [a for day in evaluation["days"] for a in day["agents"]]
    maximum_error = max(a["terminal_soc_error"] for a in agents)
    minimum_span = min(a["soc_span"] for a in agents)
    fraction = float(np.mean([a["bidirectional"] for a in agents]))
    excess = evaluation["mean_full_cost"] / oracle_cost - 1
    checks = {
        "per_battery_terminal_inventory": maximum_error <= requirements["max_terminal_soc_error"],
        "operating_cost": excess <= requirements["max_mean_cost_above_cyclic_oracle"],
        "meaningful_soc_span": minimum_span >= requirements["minimum_soc_span"],
        "bidirectional_use": fraction >= requirements["minimum_bidirectional_fraction"],
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "max_terminal_soc_error": maximum_error, "min_soc_span": minimum_span,
        "bidirectional_fraction": fraction, "cost_excess_vs_cyclic_reference": excess,
        "note": "Cyclic perfect-foresight cost is a reference, not a matched feasible-policy optimality gap",
    }


def collect(experiment):
    manifest = audit_snapshot(experiment)
    results = []
    for seed in manifest["seeds"]:
        weights = []
        warmup_weights = []
        for name in manifest["modes"]:
            directory = experiment / "runs" / f"{name}_seed{seed}"
            run = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
            settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
            if (run["mode"] != name or run["seed"] != seed
                    or run["actual_steps"] != int(np.ceil(manifest["requested_steps"] / 2400)) * 2400
                    or run["reload_check"] != "exact"
                    or digest(directory / "final_policy.zip") != run["policy_sha256"]):
                raise RuntimeError("Invalid final training run")
            if run["validation"]["rows"] != manifest["validation_rows"]:
                raise RuntimeError("Validation dates changed")
            if bool(run["test"]) != bool(manifest["test_rows"]):
                raise RuntimeError("Test evaluation protocol changed")
            if run["test"] and run["test"]["rows"] != manifest["test_rows"]:
                raise RuntimeError("Test dates changed")
            if settings["reward_recipe"] != manifest["recipes"][name]:
                raise RuntimeError("Reward recipe changed")
            weights.append(settings["initial_policy_sha256"])
            if manifest.get("curriculum"):
                warmup = json.loads((directory / "warmup.json").read_text(encoding="utf-8"))
                if warmup["steps"] != manifest["curriculum"]["warmup_steps"]:
                    raise RuntimeError("Reward warmup length changed")
                warmup_weights.append(warmup["policy_sha256"])
            verdict = assessment(
                run["validation"], manifest["oracle_mean_cost"], manifest["acceptance"],
            )
            results.append({**run, "acceptance": verdict})
        if len(set(weights)) != 1:
            raise RuntimeError("Paired initial policy weights differ")
        if warmup_weights and len(set(warmup_weights)) != 1:
            raise RuntimeError("Reward variants diverged before their shared warmup ended")
    grouped = {}
    for name in manifest["modes"]:
        selected = [r for r in results if r["mode"] == name]
        grouped[name] = {
            "passing_seeds": sum(r["acceptance"]["passed"] for r in selected),
            "seed_count": len(selected),
            "mean_cost": float(np.mean([r["validation"]["mean_full_cost"] for r in selected])),
            "mean_inventory_error_kwh": float(np.mean([
                r["validation"]["mean_terminal_error_kwh"] for r in selected
            ])),
            "worst_terminal_soc_error": max(r["acceptance"]["max_terminal_soc_error"] for r in selected),
            "minimum_soc_span": min(r["acceptance"]["min_soc_span"] for r in selected),
        }
    analysis = experiment / "analysis"
    analysis.mkdir(exist_ok=False)
    payload = {"runs": results, "aggregate": grouped, "acceptance": manifest["acceptance"]}
    save_json(analysis / "summary.json", payload)
    print(json.dumps(grouped, indent=2), flush=True)
    return payload


def reuse_completed_runs(experiment, previous):
    manifest, old = audit_snapshot(experiment), audit_snapshot(previous)
    for key in ("env_args", "recipes", "ppo", "policy_kwargs", "curriculum",
                "requested_steps", "source_sha256", "validation_rows", "test_rows",
                "acceptance"):
        if manifest.get(key) != old.get(key):
            raise ValueError(f"Cannot reuse a changed experiment field: {key}")
    reused = []
    for seed in set(manifest["seeds"]) & set(old["seeds"]):
        for mode in manifest["modes"]:
            source = previous / "runs" / f"{mode}_seed{seed}"
            completed = json.loads((source / "completed.json").read_text(encoding="utf-8"))
            if (completed["mode"] != mode or completed["seed"] != seed
                    or completed["reload_check"] != "exact"
                    or completed["policy_sha256"] != digest(source / "final_policy.zip")):
                raise RuntimeError("Original run is not a valid completed policy")
            target = experiment / "runs" / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copytree(source, target)
            reused.append({"run": source.name, "source": str(source.resolve()),
                           "policy_sha256": completed["policy_sha256"]})
    save_json(experiment / "reused_runs.json", reused)


def confirm_frozen_policies(experiment, selected):
    import torch
    from stable_baselines3 import PPO
    from tools.run_sb3_reward_experiment import write_records

    manifest = audit_snapshot(experiment)
    summary = json.loads((experiment / "analysis/summary.json").read_text(encoding="utf-8"))
    if selected not in summary["aggregate"]:
        raise ValueError("Select a reward present in this frozen matrix")
    verdict = summary["aggregate"][selected]
    if verdict["seed_count"] < 3 or verdict["passing_seeds"] != verdict["seed_count"]:
        raise ValueError("Holdout requires at least three passing validation seeds")
    rows = manifest["reserved_test_rows"]
    output = experiment / "holdout"
    output.mkdir(exist_ok=False)
    path = experiment / "snapshot/tools/sb3_dispatch_reward_env.py"
    spec = importlib.util.spec_from_file_location("frozen_reward_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    env_class = adapter.load_environment_class(experiment / "snapshot")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = []
    for seed in manifest["seeds"]:
        for name in manifest["modes"]:
            run_dir = experiment / "runs" / f"{name}_seed{seed}"
            original = json.loads((run_dir / "completed.json").read_text(encoding="utf-8"))
            if digest(run_dir / "final_policy.zip") != original["policy_sha256"]:
                raise RuntimeError("Final policy changed before holdout")
            recipe = manifest["recipes"][name]
            model = PPO.load(run_dir / "final_policy.zip", device="cpu")
            validation = adapter.DispatchRewardEnv(
                env_class, manifest["env_args"], recipe["mode"], output,
                gamma=manifest["ppo"]["gamma"], source="validation", **recipe["settings"],
            )
            repeated, _ = adapter.evaluate(
                model, validation, manifest["validation_rows"], collect_records=True,
            )
            # Numeric endpoint equality must hold before observing new dates.
            if repeated != original["validation"]:
                raise RuntimeError("Frozen policy validation endpoint changed")
            test = adapter.DispatchRewardEnv(
                env_class, manifest["env_args"], recipe["mode"], output,
                gamma=manifest["ppo"]["gamma"], source="test", **recipe["settings"],
            )
            evaluation, trajectory = adapter.evaluate(model, test, rows, collect_records=True)
            write_records(output / f"{name}_seed{seed}_trajectories.csv", trajectory)
            results.append({"seed": seed, "mode": name, "test": evaluation})
            validation.close()
            test.close()
    audit_snapshot(experiment)
    payload = {
        "selected": selected, "rows": rows, "runs": results,
        "selection": "all models and reward parameters frozen before this evaluation",
    }
    save_json(output / "summary.json", payload)
    print(json.dumps({
        name: {
            "mean_cost": float(np.mean([r["test"]["mean_full_cost"] for r in results
                                       if r["mode"] == name])),
            "max_terminal_soc_error": max(
                a["terminal_soc_error"] for r in results if r["mode"] == name
                for d in r["test"]["days"] for a in d["agents"]
            ),
        } for name in manifest["modes"]
    }, indent=2), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipes", nargs="+", choices=RECIPES,
                        default=["terminal_ref", "deadline_q16", "deadline_q64"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--timesteps", type=int, default=240_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=120_000)
    parser.add_argument("--ramp-steps", type=int, default=120_000)
    parser.add_argument("--noise-decay-steps", type=int, default=240_000)
    parser.add_argument("--final-max-std", type=float, default=0.08)
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--selected", choices=RECIPES)
    parser.add_argument("--oracle", type=Path, default=Path(
        "results/oracle_milp/battery_activity_diagnosis_20261017_a/summary.json",
    ))
    args = parser.parse_args()
    if not np.isfinite(args.log_std_init):
        raise ValueError("log-std-init must be finite")
    experiment = args.output.resolve()
    if args.evaluate_holdout:
        if args.selected is None:
            raise ValueError("Holdout needs an explicitly frozen selected reward")
        confirm_frozen_policies(experiment, args.selected)
        return
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    for relative, expected in oracle["source_sha256"].items():
        if relative.replace("\\", "/").startswith("src/") and digest(ROOT / relative) != expected:
            raise RuntimeError("Oracle no longer matches the physical environment")
    extra = {
        "stage": "confirmation" if args.confirm else "validation_tuning",
        "reserved_test_rows": NEW_TEST_ROWS,
        "acceptance": ACCEPTANCE, "oracle_mean_cost": oracle["mean_full_cost"],
        "oracle_summary_sha256": digest(args.oracle),
        "selection_rule": "validation-only; all final seeds reported; no checkpoint or test selection",
    }
    if args.curriculum:
        if (min(args.warmup_steps, args.ramp_steps, args.noise_decay_steps) <= 0
                or any(value % 2400 for value in (
                    args.warmup_steps, args.ramp_steps, args.noise_decay_steps))
                or args.timesteps < args.warmup_steps + args.ramp_steps + 2400
                or not 0 < args.final_max_std <= np.exp(args.log_std_init)):
            raise ValueError("Curriculum needs aligned, positive stages and a valid noise cap")
        extra["curriculum"] = {
            "warmup_steps": args.warmup_steps, "ramp_steps": args.ramp_steps,
            "noise_decay_steps": args.noise_decay_steps, "final_max_std": args.final_max_std,
        }
    prepare(
        experiment, args.seeds, args.timesteps,
        recipes={name: RECIPES[name] for name in args.recipes},
        test_rows=NEW_TEST_ROWS if args.confirm else [],
        extra_manifest=extra, extra_files=[Path(__file__)],
        policy_kwargs={
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "log_std_init": args.log_std_init,
        },
    )
    if args.reuse_from:
        reuse_completed_runs(experiment, args.reuse_from.resolve())
    started = time.perf_counter()
    count = launch_matrix(experiment, args.workers, reuse_completed=bool(args.reuse_from))
    result = collect(experiment)
    save_json(experiment / "completed.json", {
        "run_count": count, "elapsed_seconds": time.perf_counter() - started,
        "source_audit": "passed", "aggregate": result["aggregate"],
    })


if __name__ == "__main__":
    main()
