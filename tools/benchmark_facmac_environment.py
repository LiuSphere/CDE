"""Bounded, fixed-validation learning check through the real training entry point."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--config", choices=["facmac_ea", "facmac_no_ea"], default="facmac_ea")
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(source / "src"))
    import direct_main
    import run as training
    import yaml
    import fastrand
    import torch

    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    random.seed(args.seed)
    fastrand.pcg32_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)

    out = ROOT / "results/facmac_environment" / args.output_id
    out.mkdir(parents=True, exist_ok=False)
    history = []
    references = {}
    phase = "scheduled"
    original = training.evaluate_fixed
    original_runner = training.EpisodeRunner
    original_epoch = training.utils_ne.SSNE.epoch

    def capture_runner(*positional, **keywords):
        runner = original_runner(*positional, **keywords)
        for index, agent in enumerate(runner.env.possible_agents):
            runner.env.spaces[agent]["action"].seed(args.seed + index)
        references["runner"] = runner
        return runner

    def capture_epoch(evolver, pop, *positional, **keywords):
        elite = original_epoch(evolver, pop, *positional, **keywords)
        references["ea"] = pop[int(elite)]
        return elite

    def record_evaluation(runner, mac, rows, initial_soc, source, policy_source, step,
                          record=False):
        returns, metrics = [], []
        if policy_source == "rl":
            references["rl"] = mac
        for row in rows:
            result = original(runner, mac, [row], initial_soc, source, policy_source,
                              step, record=record)
            returns.extend(result["returns"])
            stats = dict(runner.env._stats)
            stats["mean_soc"] /= runner.env._ep_steps
            stats["operating_cost"] = stats["full_cost"] + stats["battery_punishment_reward"]
            stats.update(row_idx=int(row), terminal_soc=runner.env.params["socs"].tolist())
            metrics.append(stats)
        history.append({
            "step": int(step), "policy": policy_source, "rows": metrics,
            "phase": phase,
            "mean_full_cost": float(np.mean([r["full_cost"] for r in metrics])),
            "mean_operating_cost": float(np.mean([r["operating_cost"] for r in metrics])),
            "mean_return": float(np.mean(returns)),
        })
        return {"rows": rows, "returns": returns, "mean_return": float(np.mean(returns))}

    config = direct_main.recursive_dict_update(
        direct_main.load_yaml(str(source / "src/config/default.yaml")),
        direct_main.load_yaml(str(source / "src/config/envs/microgrid.yaml")),
    )
    sequential = training.run_sequential

    def capture_config(args, logger):
        nonlocal phase
        random.seed(args.seed)
        fastrand.pcg32_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        with (out / "config_snapshot.yaml").open("x", encoding="utf-8") as file:
            yaml.safe_dump(vars(args), file, sort_keys=False)
        result = sequential(args, logger)
        phase = "final"
        runner = references["runner"]
        for policy in ("rl", "ea") if args.EA else ("rl",):
            if policy not in references:
                raise RuntimeError(f"No final {policy} policy was captured")
            record_evaluation(runner, references[policy], rows, args.evaluation_initial_soc,
                              "validation", policy, runner.t_env)
        return result
    rows = [740, 766, 808, 862, 880, 904]
    argv = [
        "direct_main.py", "--config", args.config, "--env-config", "microgrid",
        "--override", "wandb_mode=disabled", "--override", "save_model=False",
        "--override", f"seed={args.seed}", "--override", f"t_max={args.timesteps}",
        "--override", "test_interval=4000", "--override", "fixed_eval_days=6",
        "--override", f"validation_rows={rows}",
        "--override", f"env_args.result_dir={out}",
        "--override", f"wandb_dir={out}/wandb",
    ]
    start = time.perf_counter()
    with patch.object(sys, "argv", argv), \
         patch.object(training, "evaluate_fixed", record_evaluation), \
         patch.object(training, "run_sequential", capture_config), \
         patch.object(training, "EpisodeRunner", capture_runner), \
         patch.object(training.utils_ne.SSNE, "epoch", capture_epoch):
        direct_main.main()
    elapsed = time.perf_counter() - start
    payload = {
        "model_version": config["env_args"]["model_version"],
        "algorithm": args.config, "seed": args.seed, "requested_steps": args.timesteps,
        "seconds": elapsed, "validation_rows": rows,
        "argv": argv,
        "source_root": str(source),
        "cpu_threads": args.cpu_threads,
        "rng_protocol": "entrypoint numpy/torch; benchmark python/fastrand/gym action spaces",
        "deterministic_algorithms": True,
        "pre_rollout_reseed": True,
        "actual_steps": references["runner"].t_env,
        "scope": "short learning/compatibility check, not algorithm superiority evidence",
        "history": history,
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source / "src/envs/maenv.py", source / "src/run.py",
                      source / "src/runners/episode_runner.py",
                      source / "src/config/envs/microgrid.yaml")
        },
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with (out / "learning_check.json").open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    print(f"LEARNING_CHECK={out}", flush=True)


if __name__ == "__main__":
    main()
