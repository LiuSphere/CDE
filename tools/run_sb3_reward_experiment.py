"""Paired, frozen-source PPO reward ablation on current physical microgrids."""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sb3_dispatch_reward_env import (
    REWARD_MODES, VALIDATION_ROWS, DispatchRewardEnv, evaluate, load_environment_class,
)


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)


def write_records(path, records):
    with Path(path).open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_same_replay(original, restored, records, replay_records):
    physical_records = [{k: v for k, v in r.items() if k != "episode"} for r in records]
    physical_replay = [{k: v for k, v in r.items() if k != "episode"} for r in replay_records]
    if original != restored or physical_records != physical_replay:
        raise RuntimeError("Saved final policy does not reproduce validation exactly")


def retryable_launch_error(message, run_directory):
    return (
        ("Failed to spawn:" in message or "uv trampoline failed to spawn Python child process" in message)
        and "os error 5" in message and "Traceback" not in message
        and not Path(run_directory).exists()
    )


def audit_snapshot(experiment):
    manifest = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    for name, expected in manifest["source_sha256"].items():
        if digest(experiment / "snapshot" / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    return manifest


def train_one(experiment, mode, seed):
    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    manifest = audit_snapshot(experiment)
    if mode not in manifest["modes"] or seed not in manifest["seeds"]:
        raise ValueError("Worker identity is outside the frozen experiment")
    recipe = manifest.get("recipes", {}).get(mode, {"mode": mode, "settings": {}})
    output = experiment / "runs" / f"{mode}_seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    config = manifest["env_args"]
    env_class = load_environment_class(experiment / "snapshot")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gamma = manifest["ppo"]["gamma"]

    def make_env(source):
        return DispatchRewardEnv(
            env_class, config, recipe["mode"], output, gamma=gamma, source=source,
            **recipe["settings"],
        )

    interface = make_env("train")
    check_env(interface, warn=True)
    interface.close()
    train = make_env("train")
    train.action_space.seed(seed)
    validation = make_env("validation")
    test = make_env("test")
    vec = VecNormalize(
        DummyVecEnv([lambda: Monitor(train, filename=str(output / "train.monitor.csv"))]),
        norm_obs=False, norm_reward=True, clip_reward=np.inf, gamma=gamma,
    )
    model = PPO(
        "MlpPolicy", vec, device="cpu", seed=seed, verbose=0,
        policy_kwargs=manifest.get("policy_kwargs", {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}),
        **manifest["ppo"],
    )
    model.set_logger(configure(str(output / "logs"), ["csv"]))
    initial_hash = hashlib.sha256(b"".join(
        value.detach().numpy().tobytes() for value in model.policy.state_dict().values()
    )).hexdigest()
    save_json(output / "settings.json", {
        "mode": mode, "seed": seed, "ppo": manifest["ppo"],
        "policy_kwargs": manifest.get("policy_kwargs"),
        "reward_recipe": recipe,
        "initial_policy_sha256": initial_hash,
        "inventory_price": train.inventory_price.tolist(),
        "terminal_price": train.terminal_price.tolist(),
        "sb3_version": stable_baselines3.__version__, "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__, "numpy_version": np.__version__,
        "python_version": sys.version, "device": "cpu", "torch_threads": 1,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "deterministic_algorithms": True,
    })

    class ValidationCurve(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_evaluation = 0
            self.warmup_saved = False

        def _on_step(self):
            return True

        def record(self):
            result, _ = evaluate(self.model, validation, manifest["validation_rows"])
            entry = {"steps": self.num_timesteps, **result}
            with (output / "validation.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, allow_nan=False) + "\n")

        def _on_rollout_start(self):
            curriculum = manifest.get("curriculum")
            if curriculum:
                warmup = curriculum["warmup_steps"]
                ramp = curriculum["ramp_steps"]
                train.reward_strength = float(np.clip(
                    (self.num_timesteps - warmup) / ramp, 0, 1,
                ))
                if self.num_timesteps >= warmup:
                    if not self.warmup_saved:
                        weight_hash = hashlib.sha256(b"".join(
                            value.detach().numpy().tobytes()
                            for value in self.model.policy.state_dict().values()
                        )).hexdigest()
                        save_json(output / "warmup.json", {
                            "steps": self.num_timesteps, "policy_sha256": weight_hash,
                        })
                        self.warmup_saved = True
                    fraction = min(
                        (self.num_timesteps - warmup) / curriculum["noise_decay_steps"], 1,
                    )
                    maximum_std = np.exp(
                        (1 - fraction) * manifest["policy_kwargs"].get("log_std_init", 0)
                        + fraction * np.log(curriculum["final_max_std"]),
                    )
                    # Applied before collecting a fresh rollout, never between
                    # collection and PPO updates on that rollout.
                    with torch.no_grad():
                        self.model.policy.log_std.clamp_(max=float(np.log(maximum_std)))
            if self.num_timesteps >= self.next_evaluation:
                self.record()
                self.next_evaluation = self.num_timesteps + 24_000

    callback = ValidationCurve()
    started = time.perf_counter()
    model.learn(total_timesteps=manifest["requested_steps"], callback=callback)
    model.save(output / "final_policy")
    vec.save(output / "final_vecnormalize.pkl")
    final, records = evaluate(
        model, validation, manifest["validation_rows"], collect_records=True,
    )
    write_records(output / "validation_trajectories.csv", records)
    # The test split is observed only once, after training; it selects nothing.
    held_out = None
    if manifest["test_rows"]:
        held_out, test_records = evaluate(
            model, test, manifest["test_rows"], collect_records=True,
        )
        write_records(output / "test_trajectories.csv", test_records)
    restored = PPO.load(output / "final_policy.zip", device="cpu")
    replay, replay_records = evaluate(
        restored, validation, manifest["validation_rows"], collect_records=True,
    )
    assert_same_replay(final, replay, records, replay_records)
    audit_snapshot(experiment)
    save_json(output / "completed.json", {
        "mode": mode, "seed": seed, "actual_steps": model.num_timesteps,
        "elapsed_seconds": time.perf_counter() - started,
        "validation": final, "test": held_out, "reload_check": "exact",
        "policy_sha256": digest(output / "final_policy.zip"),
    })
    vec.close()
    validation.close()
    test.close()
    print(f"DONE {mode} seed={seed} steps={model.num_timesteps}", flush=True)


def prepare(experiment, seeds, steps, *, recipes=None, test_rows=None,
            extra_manifest=None, extra_files=(), policy_kwargs=None):
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0 or steps < 2400:
        raise ValueError("Expected unique nonnegative seeds and at least one PPO rollout")
    experiment.mkdir(parents=True, exist_ok=False)
    snapshot = experiment / "snapshot"
    files = [
        ROOT / "src/envs/maenv.py", ROOT / "src/config/default.yaml",
        ROOT / "src/config/envs/microgrid.yaml",
        ROOT / "src/config/algs/gavdac_no_ea.yaml",
        Path(__file__), ROOT / "tools/sb3_dispatch_reward_env.py",
        ROOT / "pyproject.toml", ROOT / "uv.lock", *extra_files,
        *sorted((ROOT / "src/envs/data").glob("*.csv")),
    ]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[relative.as_posix()] = digest(target)
        if digest(path) != hashes[relative.as_posix()]:
            raise RuntimeError(f"Source changed during snapshot: {path}")
    # Use the project's actual entry-point merge helpers for configuration.
    sys.path.insert(0, str(ROOT / "src"))
    from direct_main import load_yaml, normalize_scalars, recursive_dict_update
    config = {}
    for relative in ("src/config/default.yaml", "src/config/envs/microgrid.yaml",
                     "src/config/algs/gavdac_no_ea.yaml"):
        config = recursive_dict_update(config, load_yaml(str(snapshot / relative)))
    config = normalize_scalars(config)
    save_json(experiment / "merged_config.json", config)
    (experiment / "logs").mkdir()
    manifest = {
        "seeds": seeds, "modes": list(recipes) if recipes else list(REWARD_MODES),
        "requested_steps": steps,
        "env_args": config["env_args"], "validation_rows": VALIDATION_ROWS,
        "test_rows": [913, 940, 980, 1020, 1060, 1090] if test_rows is None else test_rows,
        "source_sha256": hashes,
        "policy_kwargs": policy_kwargs or {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        "ppo": {
            "n_steps": 2400, "batch_size": 240, "n_epochs": 10,
            "learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "scope": "centralized PPO reward diagnosis; not GAVDAC algorithm comparison",
        "boundaries": "all modes: random training day, initial SOC 0.35, true terminal at 24h",
        "selection": "final endpoint only; no best-seed/checkpoint/test selection",
        "reward_formulas": {
            "current": "mean((baseline_cost-full_cost)*reward_scale)",
            "terminal": "current - reward_scale/N * terminal_inventory_cost at t=23",
            "potential": "terminal + reward_scale/N * (gamma*Phi(next)-Phi(current))",
            "Phi": "sum(capacity*(SOC-initial_SOC)*eta_discharge*train_mean_tariff); Phi(terminal)=0",
            "terminal_inventory_cost": "sum(capacity*abs(SOC-initial_SOC)*1.25*(train_max_tariff+cycle_cost)/eta_charge)",
        },
    }
    if recipes:
        manifest["recipes"] = recipes
    if extra_manifest:
        if set(manifest) & set(extra_manifest):
            raise ValueError("Extra protocol fields cannot replace core experiment settings")
        manifest.update(extra_manifest)
    save_json(experiment / "experiment.json", manifest)


def summarize(experiment):
    manifest = audit_snapshot(experiment)
    runs = []
    for seed in manifest["seeds"]:
        initial_hashes = []
        for mode in manifest["modes"]:
            directory = experiment / "runs" / f"{mode}_seed{seed}"
            result = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
            settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
            if result["mode"] != mode or result["seed"] != seed:
                raise RuntimeError("Run identity mismatch")
            expected_steps = int(np.ceil(manifest["requested_steps"] / 2400)) * 2400
            if result["actual_steps"] != expected_steps or result["reload_check"] != "exact":
                raise RuntimeError("Incomplete run endpoint")
            if digest(directory / "final_policy.zip") != result["policy_sha256"]:
                raise RuntimeError("Final policy changed")
            for split in ("validation", "test"):
                if result[split]["rows"] != manifest[f"{split}_rows"]:
                    raise RuntimeError("Evaluation rows changed")
            initial_hashes.append(settings["initial_policy_sha256"])
            runs.append(result)
        if len(set(initial_hashes)) != 1:
            raise RuntimeError("Paired rewards must start from identical policy weights")
    metrics = (
        "mean_full_cost", "mean_inventory_adjusted_cost", "mean_terminal_error_kwh",
        "mean_mean_terminal_soc_error", "mean_charge_kwh", "mean_discharge_kwh",
        "mean_bidirectional_agent_fraction", "mean_mean_soc_span",
        "mean_mean_generator_hours", "mean_generator_kwh", "mean_grid_kwh", "mean_peer_kwh",
    )
    aggregate = {}
    for split in ("validation", "test"):
        aggregate[split] = {}
        for mode in manifest["modes"]:
            selected = [r[split] for r in runs if r["mode"] == mode]
            aggregate[split][mode] = {
                metric: {
                    "mean": float(np.mean([r[metric] for r in selected])),
                    "std": float(np.std([r[metric] for r in selected], ddof=1))
                    if len(selected) > 1 else 0.0,
                } for metric in metrics
            }
    pairs = []
    for seed in manifest["seeds"]:
        baseline = next(r for r in runs if r["seed"] == seed and r["mode"] == "current")
        for mode in manifest["modes"][1:]:
            other = next(r for r in runs if r["seed"] == seed and r["mode"] == mode)
            pairs.append({
                "seed": seed, "mode": mode,
                **{split: {
                    "raw_cost_improvement_pct": 100 * (
                        1 - other[split]["mean_full_cost"] / baseline[split]["mean_full_cost"]
                    ),
                    "adjusted_cost_improvement_pct": 100 * (
                        1 - other[split]["mean_inventory_adjusted_cost"]
                        / baseline[split]["mean_inventory_adjusted_cost"]
                    ),
                } for split in ("validation", "test")},
            })
    analysis = experiment / "analysis"
    analysis.mkdir(exist_ok=False)
    save_json(analysis / "summary.json", {
        "seeds": manifest["seeds"], "aggregate": aggregate, "paired": pairs,
        "runs": runs, "actual_steps": sum(r["actual_steps"] for r in runs),
        "note": "Unconstrained terminal SOC; inventory-adjusted cost is a penalized score, not an electricity bill",
    })
    make_plots(experiment, runs)
    return aggregate


def make_plots(experiment, runs, *, output=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    modes = list(REWARD_MODES)
    labels = ["Current", "Terminal cost", "Terminal + potential"]
    colors = ["#777777", "#226e9b", "#20875b"]
    seeds = sorted({r["seed"] for r in runs})
    output = experiment / "analysis" if output is None else Path(output)
    if not output.is_dir():
        output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    for ax, metric, ylabel in zip(
        axes,
        ("mean_full_cost", "mean_terminal_error_kwh", "mean_inventory_adjusted_cost"),
        ("Operating cost (CU/day)", "Terminal inventory error (kWh)",
         "Operating cost + boundary penalty (CU/day)"),
    ):
        for j, seed in enumerate(seeds):
            values = [
                next(r for r in runs if r["mode"] == m and r["seed"] == seed)["validation"][metric]
                for m in modes
            ]
            positions = np.arange(3) + (j - (len(seeds) - 1) / 2) * 0.075
            ax.plot(positions, values, color="#b7b7b7", linewidth=0.8, zorder=1)
            for i, value in enumerate(values):
                ax.scatter(positions[i], value, color=colors[i], s=36, zorder=2)
                ax.annotate(str(seed), (positions[i], value), xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=9)
        ax.set_xticks(range(3), labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.12, y=0.15)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("SB3 PPO, 3 paired seeds | final validation | lower is better in every panel")
    fig.savefig(output / "reward_comparison.png", dpi=150)
    plt.close(fig)
    # Show a predeclared seed and every validation day, never a selected winner.
    seed = seeds[0]
    rows = VALIDATION_ROWS
    config = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    capacity = np.array([p["battery_cap"] for p in config["env_args"]["data"]])
    initial_soc = config["env_args"]["initial_soc"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout="constrained", sharey=True)
    for mode, label, color in zip(modes, labels, colors):
        frame = pd.read_csv(experiment / "runs" / f"{mode}_seed{seed}/validation_trajectories.csv")
        for ax, row in zip(axes.flat, rows):
            daily = frame[frame.row_idx == row]
            soc = daily.pivot(index="col_idx", columns="agent", values="soc").to_numpy()
            avg = soc @ capacity / capacity.sum()
            ax.plot(np.arange(25), [initial_soc, *avg], label=label, color=color)
            ax.set(title=f"Row {row}", xlabel="Hour", ylabel="Capacity-weighted SOC",
                   ylim=(0.18, 0.82), xlim=(0, 24))
            ax.grid(alpha=0.2)
    for ax in axes.flat:
        ax.axhline(initial_soc, color="#aaaaaa", linestyle="--", linewidth=0.8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Seed {seed}, declared in advance | all six validation days")
    fig.savefig(output / f"soc_seed{seed}.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained", sharey=True)
    row = rows[0]
    for mode, label, color in zip(modes, labels, colors):
        frame = pd.read_csv(experiment / "runs" / f"{mode}_seed{seed}/validation_trajectories.csv")
        for ax, agent in zip(axes, sorted(frame.agent.unique())):
            data = frame[(frame.row_idx == row) & (frame.agent == agent)]
            ax.plot(np.arange(25), [initial_soc, *data.soc.tolist()], label=label, color=color)
            ax.set(title=agent, xlabel="Hour", ylabel="SOC", ylim=(0.18, 0.82), xlim=(0, 24))
    for ax in axes:
        ax.axhline(initial_soc, color="#aaaaaa", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Seed {seed}, first declared validation row {row} | individual battery inventory")
    fig.savefig(output / f"individual_soc_seed{seed}.png", dpi=150)
    plt.close(fig)


def launch_matrix(experiment, workers, *, reuse_completed=False):
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest = audit_snapshot(experiment)

    def launch(item):
        mode, seed = item
        name = f"{mode}_seed{seed}"
        directory = experiment / "runs" / name
        if reuse_completed and directory.is_dir():
            result = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
            settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
            if (result["mode"] != mode or result["seed"] != seed
                    or result["actual_steps"] != int(np.ceil(manifest["requested_steps"] / 2400)) * 2400
                    or result["policy_sha256"] != digest(directory / "final_policy.zip")
                    or settings["ppo"] != manifest["ppo"]
                    or settings["policy_kwargs"] != manifest["policy_kwargs"]
                    or settings["reward_recipe"] != manifest["recipes"][mode]):
                raise RuntimeError("Reused run failed verification")
            print(f"REUSED {name}", flush=True)
            return
        command = [
            "uv", "run", "--no-sync", sys.executable,
            str(experiment / "snapshot/tools/run_sb3_reward_experiment.py"),
            "--output", str(experiment), "--worker-mode", mode, "--worker-seed", str(seed),
        ]
        environment = dict(os.environ, PYTHONHASHSEED=str(seed))
        for attempt in range(8):
            log_path = experiment / "logs" / f"{name}_attempt{attempt}.log"
            with log_path.open("x", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            if result.returncode == 0:
                break
            message = log_path.read_text(encoding="utf-8", errors="replace")
            retryable = retryable_launch_error(message, experiment / "runs" / name)
            if not retryable or attempt == 7:
                raise subprocess.CalledProcessError(result.returncode, command)
            print(f"RETRY pre-Python uv launch: {name}, attempt={attempt + 1}", flush=True)
            time.sleep(min(attempt + 1, 5))
        print(f"COMPLETED {name}", flush=True)

    jobs = [(mode, seed) for seed in manifest["seeds"] for mode in manifest["modes"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(launch, jobs))
    return len(jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-mode")
    parser.add_argument("--worker-seed", type=int)
    args = parser.parse_args()
    experiment = args.output.resolve()
    if args.worker_mode is not None:
        train_one(experiment, args.worker_mode, args.worker_seed)
        return
    prepare(experiment, args.seeds, args.timesteps)
    started = time.perf_counter()
    count = launch_matrix(experiment, args.workers)
    summary = summarize(experiment)
    save_json(experiment / "completed.json", {
        "run_count": count, "elapsed_seconds": time.perf_counter() - started,
        "source_audit": "passed", "aggregate": summary,
    })
    print(f"RESULTS={experiment}", flush=True)


if __name__ == "__main__":
    main()
