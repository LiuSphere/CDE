"""Independent CTDE MAPPO training with SB3's proven on-policy PPO core."""

import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import stable_baselines3
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from modules.mappo import CentralizedPolicy, MicrogridAgentVecEnv
from utils.comparison_evaluation import make_evaluation_runner
from utils.dispatch_showcase import DispatchShowcase
from utils.logging import Logger
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed, fixed_rows


class MAPPOProgress(BaseCallback):
    def __init__(self, vector_env, runner, mac, store, logger, config):
        super().__init__()
        self.vector_env, self.runner, self.mac = vector_env, runner, mac
        self.store, self.metrics_logger, self.config = store, logger, config
        self.started = time.monotonic()
        self.last_evaluation = 0
        self.last_metrics = 0
        self.evaluation_interactions = 0
        self.finite_update_checks = 0
        self.last_update_metrics = {}
        self.validation_rows = fixed_rows(
            runner.env, "validation", config["validation_rows"], config["fixed_eval_days"])

    def evaluate(self, step):
        before = self.vector_env.joint_steps
        result = evaluate_fixed(
            self.runner, self.mac, self.validation_rows,
            self.config["evaluation_initial_soc"], "validation", "rl", step)
        assert self.vector_env.joint_steps == before
        self.evaluation_interactions += len(self.validation_rows) * self.runner.episode_limit
        self.store.consider(self.mac, "rl", step, result, self.runner.env)
        self.metrics_logger.log_stat("eval/rl_reward", result["mean_return"], step)
        self.metrics_logger.log_stat("eval/best_reward", self.store.best["rl"]["mean_return"], step)
        self.last_evaluation = step

    def log_losses(self, step, emit=True):
        for key, value in self.model.logger.name_to_value.items():
            if key.startswith("train/") and isinstance(value, (float, int, np.number)):
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite MAPPO update: {key}")
                self.last_update_metrics[key] = float(value)
                if emit:
                    self.metrics_logger.log_stat("loss/" + key.removeprefix("train/"), value, step)
        if self.last_update_metrics:
            self.finite_update_checks += 1

    def _on_rollout_start(self):
        step = self.vector_env.joint_steps
        emit = step - self.last_metrics >= self.config["log_interval"]
        self.log_losses(step, emit=emit)
        if emit:
            self.last_metrics = step

    def _on_step(self):
        step = self.vector_env.joint_steps
        if self.model.num_timesteps != step * self.vector_env.n_agents:
            raise ValueError("SB3 agent sample counter differs from physical joint interactions")
        if step - self.last_evaluation >= self.config["test_interval"]:
            self.evaluate(step)
            stats = self.vector_env.completed_stats
            if stats:
                for key in stats[0]:
                    self.metrics_logger.log_stat(
                        "train/" + key, float(np.mean([item[key] for item in stats])), step)
                stats.clear()
            elapsed = time.monotonic() - self.started
            remaining = elapsed * max(self.config["t_max"] - step, 0) / step
            self.metrics_logger.log_stat("train/joint_steps", step, step)
            self.metrics_logger.log_stat("train/agent_samples", self.model.num_timesteps, step)
            self.metrics_logger.log_stat("train/optimizer_updates", self.model._n_updates, step)
            print(json.dumps({
                "joint_steps": step, "budget": self.config["t_max"],
                "agent_samples": self.model.num_timesteps,
                "optimizer_updates": self.model._n_updates,
                "elapsed_seconds": elapsed, "remaining_seconds": remaining,
            }), flush=True)
            self.metrics_logger.flush()
        return True


def run_mappo(config, console):
    if (config["use_cuda"] or config["EA"] or config["env"] != "microgrid"
            or config["evaluate"] or config["policy_checkpoint"]):
        raise ValueError("MAPPO training requires CPU microgrid, no EA or inference overrides")
    torch.set_num_threads(1)
    config = deepcopy(config)
    token = config["unique_token"]
    config["env_args"].update(run_id=token, log_trajectories=False)
    out = Path(config["local_results_path"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "merged_config.yaml").open("x", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    logger = Logger(console)
    logger.setup_wandb(config, token, config["wandb_project"], config["wandb_entity"],
                       config["wandb_mode"], config["wandb_dir"])
    vector_env = MicrogridAgentVecEnv(config)
    n = vector_env.n_agents
    rollout = config["mappo_n_steps"]
    if rollout % vector_env.env.episode_limit:
        raise ValueError("MAPPO rollouts must contain complete daily episodes")
    model = PPO(
        CentralizedPolicy, vector_env, device="cpu", seed=config["seed"],
        learning_rate=config["mappo_learning_rate"], n_steps=rollout,
        batch_size=rollout * n, n_epochs=config["mappo_epochs"], gamma=config["gamma"],
        gae_lambda=config["mappo_gae_lambda"], clip_range=config["mappo_clip_range"],
        clip_range_vf=config["mappo_clip_range_vf"], normalize_advantage=True,
        ent_coef=config["mappo_ent_coef"], vf_coef=config["mappo_vf_coef"],
        max_grad_norm=config["mappo_max_grad_norm"],
        policy_kwargs=dict(width=config["mappo_width"],
                           log_std_init=config["mappo_log_std_init"]),
        verbose=0)
    model.set_logger(configure(format_strings=[]))
    # Inference adapter shares only the stateless policy. Evaluation environment
    # and EpisodeBatch are separate from the live on-policy rollout.
    runner, mac = make_evaluation_runner(config)
    mac.agent = model.policy
    store = BestPolicyStore(out / "models" / token / "best_policies", config)
    progress = MAPPOProgress(vector_env, runner, mac, store, logger, config)
    progress.evaluate(0)
    started = time.monotonic()
    model.learn(total_timesteps=int(config["t_max"]) * n, callback=progress)
    steps = vector_env.joint_steps
    progress.log_losses(steps)
    if steps != progress.last_evaluation:
        progress.evaluate(steps)
    if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
        raise ValueError("Non-finite final MAPPO parameters")
    if config["save_model"]:
        model.save(out / "final_ppo")
    if config["showcase_enabled"]:
        showcase = DispatchShowcase(out / "dispatch_showcases" / token, logger, runner.args)
        showcase.publish_final(runner, mac, store, steps)
    completion = {
        "algorithm": "feedforward_parameter_sharing_MAPPO",
        "core": "stable_baselines3.PPO", "core_version": stable_baselines3.__version__,
        "actual_joint_interactions": steps, "requested_joint_interactions": config["t_max"],
        "agent_samples": model.num_timesteps, "n_agents": n,
        "optimizer_updates": model._n_updates, "completed_episodes": vector_env.episodes,
        "finite_update_checks": progress.finite_update_checks,
        "last_update_metrics": progress.last_update_metrics,
        "evaluation_interactions": progress.evaluation_interactions,
        "showcase_interactions": len(config["showcase_candidate_rows"]) * runner.episode_limit
        if config["showcase_enabled"] else 0,
        "wall_seconds": time.monotonic() - started,
        "finite_parameters": True, "device": "cpu", "dtype": "float32",
        "training_budget_includes_evaluation": False,
        "sampling": "One world; n_agents synchronized agent trajectories per joint step",
        "actor_memory": "Feedforward; own previous action reset daily",
        "checkpoint_scope": "Inference bundle plus separate PPO archive; no exact-resume claim",
    }
    with (out / "completion_checks.json").open("x", encoding="utf-8") as file:
        json.dump(completion, file, indent=2, allow_nan=False)
    logger.log_stat("train/joint_steps", steps, steps)
    logger.log_stat("train/optimizer_updates", model._n_updates, steps)
    logger.finish()
    vector_env.close()
    runner.close_env()
    console.info("Finished Training")
    print("Exiting script", flush=True)
