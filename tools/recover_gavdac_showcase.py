"""Recover final reports after the known Windows path-length reporting failure."""

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from controllers.cqmix_controller import CQMixMAC
from runners.episode_runner import EpisodeRunner
from utils.dispatch_showcase import DispatchShowcase
from utils.logging import Logger, get_logger


def reporting_failure_step(stderr, requested):
    steps = [int(x) for x in re.findall(r"EA cycle \d+ at t_env=(\d+)", stderr)]
    report_steps = [int(x) for x in re.findall(r"checkpoint_\d+_logged_(\d+)", stderr)]
    if ("[WinError 206]" not in stderr
            or "showcase.publish_final" not in stderr
            or not steps or not report_steps or len(set(report_steps)) != 1
            or report_steps[0] <= requested
            or not 0 <= report_steps[0] - max(steps) <= 2400):
        raise ValueError("Not the verified post-budget Windows showcase failure")
    return report_steps[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-id", required=True)
    args = parser.parse_args()
    run, output = args.run_dir.resolve(strict=True), args.output.resolve()
    if not all(p.is_relative_to(ROOT / "results") for p in (run, output)):
        raise ValueError("Recovery paths must be under results/")
    if output.exists():
        raise FileExistsError(output)
    stores = list((run / "models").glob("*/best_policies"))
    if len(stores) != 1:
        raise ValueError("Expected one immutable policy store")
    best = {source: json.loads((stores[0] / f"best_{source}.json").read_text())
            for source in ("rl", "ea")}
    with (Path(best["rl"]["checkpoint"]) / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    stderr_path = run / "stderr.log"
    step = reporting_failure_step(
        stderr_path.read_text(encoding="utf-8", errors="replace"), config["t_max"],
    )
    for item in best.values():
        for name, digest in item["source_sha256"].items():
            if hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Saved training source differs: {name}")
        for name, digest in item["data_sha256"].items():
            if hashlib.sha256((ROOT / "src/envs/data" / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Training inputs differ: {name}")
        if hashlib.sha256((Path(item["checkpoint"]) / "agent.th").read_bytes()).hexdigest() != item["policy_sha256"]:
            raise ValueError("Actor hash mismatch")
    output.mkdir(parents=True)
    runtime = deepcopy(config)
    runtime["env_args"].update(result_dir=str(output), run_id="recovery",
                               log_trajectories=False)
    logger = Logger(get_logger())
    logger.wandb_run = wandb.init(
        project=config["wandb_project"], entity=config["wandb_entity"],
        id=args.wandb_id, resume="must", dir=str(output),
        settings=wandb.Settings(console="off"),
    )
    logger.wandb_run.define_metric("environment_step")
    logger.wandb_run.define_metric("*", step_metric="environment_step")
    runner = EpisodeRunner(SimpleNamespace(**runtime), logger)
    info = runner.get_env_info()
    for key in ("n_agents", "n_actions", "state_shape", "obs_shape", "action_spaces"):
        setattr(runner.args, key, info[key])
    scheme = {
        "state": {"vshape": info["state_shape"]},
        "obs": {"vshape": info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (info["n_actions"],), "group": "agents", "dtype": torch.float},
        "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    mac = CQMixMAC(scheme, {"agents": info["n_agents"]}, runner.args)
    runner.setup(scheme, {"agents": info["n_agents"]}, {}, mac)
    runner.batch = runner.new_batch()
    try:
        DispatchShowcase(output / "reports", logger, runner.args).publish_final(
            runner, mac, SimpleNamespace(best=best), step,
        )
        logger.wandb_run.summary["report_recovered_after_path_failure"] = True
        logger.finish()
    finally:
        runner.close_env()
    manifest = {
        "original_run": str(run), "actual_steps": step, "requested_steps": config["t_max"],
        "original_error": "post-training WinError 206 during showcase construction",
        "original_stderr_sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
        "reports": str(output / "reports"), "wandb_id": args.wandb_id,
        "policy_sha256": {source: item["policy_sha256"] for source, item in best.items()},
        "training_status": "budget reached; original process failed in final reporting",
        "reporting_status": "recovered; W&B finish returned successfully",
    }
    with (output / "recovery.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
