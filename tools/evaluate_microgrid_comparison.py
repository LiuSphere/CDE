"""Evaluate a fresh policy or frozen rule under explicit inventory boundaries."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_runtime(checkpoint, output):
    """Use the checkpoint's actual source, never bypass its source checks."""
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    source = Path(metadata["source_snapshot"])
    for name, expected in metadata["source_sha256"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or sha256(path) != expected:
            raise ValueError(f"Checkpoint source changed: {name}")
    for name, expected in metadata["data_sha256"].items():
        path = (ROOT / "src/envs/data" / name).resolve()
        if not path.is_relative_to((ROOT / "src/envs/data").resolve()) or sha256(path) != expected:
            raise ValueError(f"Checkpoint input data changed: {name}")
    runtime = output / "runtime"
    shutil.copytree(source, runtime)
    data = runtime / "envs/data"
    data.mkdir(exist_ok=True)
    for name in metadata["data_sha256"]:
        shutil.copyfile(ROOT / "src/envs/data" / name, data / name)
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--rule", action="store_true")
    policy.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--mode", choices=["continuous", "daily_reset"], required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--rows", nargs="+", type=int)
    selection.add_argument("--all-days", action="store_true")
    parser.add_argument("--retrospective-holdout", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.retrospective_holdout:
        raise ValueError("Holdout evaluation requires --retrospective-holdout after protocol freeze")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()) or output == ROOT / "results":
        raise ValueError("Evaluation output must be a new directory below results/")
    if any(name in sys.modules for name in ("direct_main", "envs.maenv", "utils.policy_evaluation")):
        raise RuntimeError("Run this entry point in a fresh process for source isolation")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    source = checkpoint_runtime(checkpoint, output) if checkpoint else ROOT / "src"
    sys.path.insert(0, str(source))

    import numpy as np
    import torch
    from direct_main import load_yaml
    import direct_main
    from utils.comparison_evaluation import evaluate_inventory, make_evaluation_runner
    from utils.policy_evaluation import validate_policy_checkpoint
    from utils.rule_policy import TariffSOCPolicy

    if checkpoint:
        config = load_yaml(checkpoint / "config.yaml")
    else:
        configs = []
        with patch.object(sys, "argv", ["direct_main.py", "--config", "gavdac_ea_main"]), \
             patch.object(direct_main, "run", lambda config, logger: configs.append(config)):
            direct_main.main()
        config = configs[0]
    config["env_args"].update(result_dir=str(output), run_id="comparison", log_trajectories=False)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    runner, mac = make_evaluation_runner(config)
    metadata = None
    if checkpoint:
        metadata = validate_policy_checkpoint(
            runner, SimpleNamespace(**{**config, "policy_checkpoint": str(checkpoint)}),
        )
        mac.load_models(str(checkpoint))
    else:
        settings = load_yaml(source / "config/baselines/microgrid_rule.yaml")
        mac = TariffSOCPolicy(runner.env, settings)
    rows = (getattr(runner.env, f"{args.split}_indices").tolist()
            if args.all_days else args.rows)
    if metadata and args.split == "test" and not set(rows).isdisjoint(metadata["validation"]["rows"]):
        raise ValueError("Holdout overlaps checkpoint-selection rows")
    summary, records = evaluate_inventory(
        runner, mac, rows, config["evaluation_initial_soc"], args.split, args.mode,
        policy_source=metadata["policy_source"] if metadata else "rule",
        checkpoint_step=metadata["checkpoint_step"] if metadata else 0,
    )
    summary.update(
        split=args.split,
        source_sha256={path.relative_to(source).as_posix(): sha256(path)
                       for path in sorted(source.rglob("*")) if path.suffix in {".py", ".yaml"}},
        data_sha256={path.name: sha256(path) for path in (source / "envs/data").glob("*.csv")},
        rule_specification=mac.specification() if args.rule else None,
        checkpoint=str(checkpoint) if checkpoint else None,
        policy_sha256=metadata["policy_sha256"] if metadata else None,
        evaluation_entry_sha256=sha256(Path(__file__)),
        test_limitation="Retrospective holdout, historically used for diagnostics"
        if args.split == "test" else "Validation-only development check",
    )
    with (output / "trajectories.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({
        "output": str(output), "mode": args.mode, "days": len(rows),
        "physical_checks_passed": summary["physical_checks_passed"],
        "mean_economic_cost": summary["mean_economic_cost"],
    }))


if __name__ == "__main__":
    main()
