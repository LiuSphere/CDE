"""Formal centralized SB3 PPO training using the unedited maenv_sep reference."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sb3_microgrid_env import (
    ReferenceMicrogridEnv, evaluate_days, snapshot_reference, write_records,
)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def plot_typical_days(directory, records):
    for row in sorted({r["row_idx"] for r in records}):
        subset = [r for r in records if r["row_idx"] == row]
        first = [r for r in subset if r["agent"] == "agent1"]
        hours = np.arange(24)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        axes[0, 0].plot(hours, [r["load_raw"] for r in first], label="Raw load")
        axes[0, 0].plot(hours, [r["pv_raw"] for r in first], label="Raw PV")
        axes[0, 0].set(title="Source inputs (CSV units unverified)", xlabel="Hour")
        for name, label in (("price", "Price"), ("load", "Load"), ("pv", "PV")):
            axes[0, 1].plot(hours, [r[name] for r in first], label=label)
        axes[0, 1].set(title="Standardized model inputs", xlabel="Hour")
        for agent in ("agent1", "agent2", "agent3"):
            data = [r for r in subset if r["agent"] == agent]
            axes[1, 0].plot(np.arange(25), [data[0]["soc_init"]] + [r["soc"] for r in data], label=agent)
            axes[1, 1].plot(hours, [r["pb"] for r in data], label=agent)
        axes[1, 0].set(title="SOC (initial 0.35, no terminal target)", ylim=(0.18, 0.82), xlabel="Hour")
        axes[1, 1].set(title="Actual battery power (+ discharge)", xlabel="Hour")
        for ax in axes.flat:
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle(f"SB3 PPO | best validation checkpoint | fixed test row {row}", fontsize=14)
        fig.savefig(directory / f"row_{row}_overview.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout="constrained")
        for ax, agent in zip(axes, ("agent1", "agent2", "agent3")):
            data = [r for r in subset if r["agent"] == agent]
            for name in ("pd", "pb", "pg"):
                ax.plot(hours, [r[name] for r in data], label=name)
            ax.set(title=agent, ylabel="Model power")
            ax.grid(alpha=0.2)
            ax.legend(ncol=3)
        axes[-1].set_xlabel("Hour")
        fig.suptitle(f"SB3 PPO dispatch | fixed test row {row}", fontsize=14)
        fig.savefig(directory / f"row_{row}_dispatch.png", dpi=150)
        plt.close(fig)


class BestTypicalDays(BaseCallback):
    def __init__(self, output, bundle, settings):
        super().__init__()
        self.output, self.settings = output, settings
        self.validation = ReferenceMicrogridEnv(bundle, record=True)
        self.test = ReferenceMicrogridEnv(bundle, record=True)
        self.validation_rows = list(range(731, 739))
        self.test_rows = list(range(913, 921))
        self.next_eval = 0
        self.best_score = -np.inf
        self.best_path = None
        self.history = []
        self.last_eval = -1
        self.started_at = time.time()
        self.heartbeat_step = 0

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        if self.num_timesteps >= self.next_eval:
            self.evaluate()
            self.next_eval = self.num_timesteps + self.settings["eval_interval"]
        if self.num_timesteps - self.heartbeat_step >= 2400:
            self.heartbeat_step = self.num_timesteps
            self.status("running")

    def status(self, state):
        save_json(self.output / "status.json", {
            "state": state, "pid": os.getpid(), "steps": self.num_timesteps,
            "target_steps": self.settings["total_timesteps"],
            "elapsed_seconds": time.time() - self.started_at,
            "best_validation_return": self.best_score if np.isfinite(self.best_score) else None,
            "best_checkpoint": str(self.best_path) if self.best_path else None,
            "typical_day_rows": self.test_rows,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    def evaluate(self):
        if self.last_eval == self.num_timesteps:
            return
        self.last_eval = self.num_timesteps
        validation, _ = evaluate_days(
            self.model, self.validation, self.validation_rows, self.settings["evaluation_initial_soc"],
        )
        entry = {"steps": self.num_timesteps, **validation}
        self.history.append(entry)
        with (self.output / "validation_history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, allow_nan=False) + "\n")
        if validation["mean_return"] > self.best_score:
            path = self.output / "best_checkpoints" / f"step_{self.num_timesteps:09d}"
            path.mkdir(parents=True, exist_ok=False)
            self.model.save(path / "policy")
            self.model.get_vec_normalize_env().save(path / "vecnormalize.pkl")
            test, records = evaluate_days(
                self.model, self.test, self.test_rows, self.settings["evaluation_initial_soc"],
            )
            write_records(path / "typical_day_trajectories.csv", records)
            for row in self.test_rows:
                write_records(
                    path / f"row_{row}_trajectory.csv",
                    [r for r in records if r["row_idx"] == row],
                )
            metadata = {
                "steps": self.num_timesteps, "selection": "highest fixed validation mean raw reward",
                "validation": validation, "fixed_test": test, "checkpoint": str(path.resolve()),
                "initial_soc": self.settings["evaluation_initial_soc"],
                "policy_sha256": hashlib.sha256((path / "policy.zip").read_bytes()).hexdigest(),
                "vecnormalize_sha256": hashlib.sha256((path / "vecnormalize.pkl").read_bytes()).hexdigest(),
                "source_sha256": self.settings["reference_sha256"],
                "reward": "mean(0.1 * (eco_reward + env_reward + punishment))",
            }
            save_json(path / "metadata.json", metadata)
            save_json(self.output / "best.json", metadata)
            self.best_score, self.best_path = validation["mean_return"], path
            print(f"BEST steps={self.num_timesteps} validation={self.best_score:.6f} test={test['mean_return']:.6f}", flush=True)
        self.logger.record("fixed_validation/raw_mean_return", validation["mean_return"])
        self.status("running")

    def _on_training_end(self):
        self.evaluate()
        self.model.save(self.output / "final_policy")
        self.model.get_vec_normalize_env().save(self.output / "final_vecnormalize.pkl")
        # Verify the immutable winning files restore the exact typical-day data.
        winner = PPO.load(self.best_path / "policy.zip", device="cpu")
        original_hash = hashlib.sha256((self.best_path / "policy.zip").read_bytes()).hexdigest()
        evaluation, records = evaluate_days(
            winner, self.test, self.test_rows, self.settings["evaluation_initial_soc"],
        )
        with (self.best_path / "typical_day_trajectories.csv").open(encoding="utf-8-sig", newline="") as file:
            import csv
            saved = list(csv.DictReader(file))
        if len(saved) != len(records):
            raise RuntimeError("Winning checkpoint trajectory count changed on reload")
        for old, new in zip(saved, records):
            for key in ("soc", "pd", "pb", "pg", "reward", "team_reward"):
                if not np.isclose(float(old[key]), new[key], rtol=0, atol=1e-7):
                    raise RuntimeError(f"Winning checkpoint replay differs: {key}")
        plot_typical_days(self.best_path, records)
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        ax.plot([h["steps"] for h in self.history], [h["mean_return"] for h in self.history])
        ax.set(xlabel="Environment steps", ylabel="Raw mean daily return", title="SB3 PPO fixed validation curve")
        ax.grid(alpha=0.2)
        fig.savefig(self.output / "validation_curve.png", dpi=150)
        plt.close(fig)
        save_json(self.output / "completed.json", {
            "actual_steps": self.num_timesteps, "best_checkpoint": str(self.best_path),
            "best_test": evaluation, "reload_trajectory_check": "passed",
            "policy_sha256": original_hash, "elapsed_seconds": time.time() - self.started_at,
        })
        self.status("completed")
        print(f"COMPLETED steps={self.num_timesteps} best={self.best_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("C:/Users/xyx/Desktop/MG/MG/maenv_sep.py"))
    parser.add_argument("--reference-config", type=Path, default=Path("C:/Users/xyx/Desktop/IPPO4CPT/envs/MG/env_config.json"))
    args = parser.parse_args()
    output = args.output or (
        ROOT / "results/sb3_ppo" /
        ("formal_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "status.json", {
        "state": "initializing", "pid": os.getpid(), "target_steps": args.timesteps, "steps": 0,
    })
    hashes = snapshot_reference(args.reference, args.reference_config, output / "reference")
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    settings = {
        "total_timesteps": args.timesteps, "seed": args.seed, "device": "cpu",
        "algorithm": "centralized SB3 PPO, not IPPO",
        "n_steps": 2400, "batch_size": 240, "n_epochs": 10,
        "learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "eval_interval": 24_000, "evaluation_initial_soc": 0.35,
        "reward_normalization_training_only": True, "observation_normalization": False,
        "reference_sha256": hashes, "sb3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__, "run_directory": str(output),
        "source_config_multi_target": True, "selected_reference_branch": "single_objective",
        "observation_space_adaptation": "correct declared bounds only; do not clamp or rescale observations",
        "action_adaptation": "symmetric joint Box maps linearly to original per-agent action bounds",
        "training_rows": [0, 730], "validation_rows": list(range(731, 739)),
        "test_rows": list(range(913, 921)),
        "daily_truncation": "24 steps; original train SOC and dates continue across reset",
    }
    save_json(output / "settings.json", settings)
    # Interface check only, never a short training run. It uses a separate
    # instance and cannot advance the formal training environment.
    interface_env = ReferenceMicrogridEnv(output / "reference")
    check_env(interface_env, warn=True)
    interface_env.close()
    vec_env = VecNormalize(
        DummyVecEnv([lambda: Monitor(
            ReferenceMicrogridEnv(output / "reference"),
            filename=str(output / "train.monitor.csv"),
        )]),
        norm_obs=False, norm_reward=True, gamma=settings["gamma"],
    )
    model = PPO(
        "MlpPolicy", vec_env, device="cpu", seed=args.seed,
        n_steps=settings["n_steps"], batch_size=settings["batch_size"],
        n_epochs=settings["n_epochs"], learning_rate=settings["learning_rate"],
        gamma=settings["gamma"], gae_lambda=settings["gae_lambda"],
        clip_range=settings["clip_range"], ent_coef=settings["ent_coef"],
        vf_coef=settings["vf_coef"], max_grad_norm=settings["max_grad_norm"],
        policy_kwargs={"net_arch": settings["net_arch"]}, verbose=1,
    )
    model.set_logger(configure(str(output / "logs"), ["stdout", "csv"]))
    callback = BestTypicalDays(output, output / "reference", settings)
    print(f"FORMAL_TRAINING target={args.timesteps} output={output}", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=callback, log_interval=10)
    vec_env.close()


if __name__ == "__main__":
    main()
