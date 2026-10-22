"""Fixed-day validation and immutable inference checkpoints for microgrid policies."""

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml


def fixed_rows(env, source, configured, count):
    indices = getattr(env, f"{source}_indices")
    rows = list(configured) if configured else sorted(indices.tolist())[:count]
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("Fixed evaluation rows must be nonempty and unique")
    if count < 1:
        raise ValueError("fixed_eval_days must be positive")
    if any(isinstance(row, bool) or int(row) != row or row not in indices for row in rows):
        raise ValueError(f"Fixed rows must belong to the {source} split")
    return [int(row) for row in rows]


def evaluate_fixed(runner, mac, rows, initial_soc, source, policy_source, step,
                   record=False):
    returns = []
    logging = runner.env.log_trajectories
    logging_mode = runner.env.log_trajectories_mode
    runner.env.log_trajectories = record
    if record:
        runner.env.log_trajectories_mode = "test"
    try:
        with torch.no_grad():
            options = [
                {"row_idx": row, "source": source, "initial_soc": initial_soc,
                 "trajectory_context": {
                     "policy_source": policy_source, "checkpoint_step": step,
                 }}
                for row in rows
            ]
            if getattr(runner.args, "rollout_workers", 0) and len(rows) > 1:
                returns = [
                    value for _, value in runner.run_group(
                        [mac] * len(rows), test_mode=True, test_options=options,
                    )
                ]
                if not all(math.isfinite(value) for value in returns):
                    raise ValueError("Non-finite fixed evaluation return")
                return {"rows": rows, "returns": returns,
                        "mean_return": float(np.mean(returns))}
            for row in rows:
                _, value = runner.run(
                    mac, test_mode=True,
                    test_options={
                        "row_idx": row, "source": source,
                        "initial_soc": initial_soc,
                        "trajectory_context": {
                            "policy_source": policy_source, "checkpoint_step": step,
                        },
                    },
                )
                returns.append(value)
    finally:
        runner.env.log_trajectories = logging
        runner.env.log_trajectories_mode = logging_mode
    if not all(math.isfinite(value) for value in returns):
        raise ValueError("Non-finite fixed evaluation return")
    return {"rows": rows, "returns": returns, "mean_return": float(np.mean(returns))}


class BestPolicyStore:
    def __init__(self, root, config):
        self.root = Path(root)
        self.config = config
        self.best = {}
        self.root.mkdir(parents=True, exist_ok=False)
        src = Path(__file__).resolve().parents[1]
        snapshot = self.root / "source"
        snapshot.mkdir()
        for path in src.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml"}:
                target = snapshot / path.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        self.data_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (src / "envs/data").glob("*.csv")
        }
        self.source_hashes = {
            str(path.relative_to(src)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in src.rglob("*.py")
        }

    def consider(self, mac, source, step, evaluation, env):
        score = evaluation["mean_return"]
        if source in self.best and score <= self.best[source]["mean_return"]:
            return None
        path = self.root / source / str(step)
        path.mkdir(parents=True, exist_ok=False)
        mac.save_models(str(path))
        with (path / "config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.config, file, sort_keys=False)
        metadata = {
            "policy_source": source, "checkpoint_step": step,
            "mean_return": score, "validation": evaluation,
            "initial_soc": self.config["evaluation_initial_soc"],
            "battery_model_version": env.battery_model_version,
            "data_sha256": self.data_hashes,
            "policy_sha256": hashlib.sha256((path / "agent.th").read_bytes()).hexdigest(),
            "checkpoint": str(path.resolve()),
            "source_snapshot": str((self.root / "source").resolve()),
            "source_sha256": self.source_hashes,
            "battery_parameters": {
                key: env.params[key].tolist() for key in
                ("battery_caps", "battery_charge_max", "battery_discharge_max",
                 "raw_ch", "raw_dis")
            },
            "time_step_hours": env.time_step_hours,
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
        self.best[source] = metadata
        (self.root / f"best_{source}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
        return path


def validate_policy_checkpoint(runner, args):
    checkpoint = Path(args.policy_checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    with (checkpoint / "config.yaml").open(encoding="utf-8") as file:
        stored_config = yaml.safe_load(file)
    runtime_keys = {"run_id", "result_dir", "log_trajectories", "log_trajectories_mode"}
    expected_env = {key: value for key, value in stored_config["env_args"].items()
                    if key not in runtime_keys}
    actual_env = {key: value for key, value in args.env_args.items()
                  if key not in runtime_keys}
    if expected_env != actual_env or stored_config["seed"] != args.seed:
        raise ValueError("Checkpoint environment/configuration changed")
    if metadata["battery_model_version"] != runner.env.battery_model_version:
        raise ValueError("Checkpoint battery model differs from the current environment")
    src = Path(__file__).resolve().parents[1]
    for name, expected in metadata["source_sha256"].items():
        if hashlib.sha256((src / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint source changed: {name}")
    if metadata["time_step_hours"] != runner.env.time_step_hours:
        raise ValueError("Checkpoint time step differs from the environment")
    for key, expected in metadata["battery_parameters"].items():
        if not np.array_equal(runner.env.params[key], expected):
            raise ValueError(f"Checkpoint battery parameters changed: {key}")
    for name, expected in metadata["data_sha256"].items():
        if hashlib.sha256((src / "envs/data" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint input data changed: {name}")
    if hashlib.sha256((checkpoint / "agent.th").read_bytes()).hexdigest() != metadata["policy_sha256"]:
        raise ValueError("Checkpoint policy hash mismatch")
    return metadata


def evaluate_saved_policy(runner, mac, args):
    checkpoint = Path(args.policy_checkpoint)
    metadata = validate_policy_checkpoint(runner, args)
    mac.load_models(str(checkpoint))
    rows = fixed_rows(runner.env, "test", args.typical_day_rows, args.fixed_eval_days)
    if not set(rows).isdisjoint(metadata["validation"]["rows"]):
        raise ValueError("Test days overlap checkpoint selection days")
    result = evaluate_fixed(
        runner, mac, rows, args.evaluation_initial_soc, "test",
        metadata["policy_source"], metadata["checkpoint_step"], record=True,
    )
    result.update(
        checkpoint=str(checkpoint.resolve()), policy_sha256=metadata["policy_sha256"],
        policy_source=metadata["policy_source"], initial_soc=args.evaluation_initial_soc,
        battery_model_version=runner.env.battery_model_version,
        selection_validation=metadata["validation"],
    )
    (runner.env.render_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result
