"""CTDE information boundaries, agent counting and current-environment replay."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import direct_main
from modules.mappo import CentralizedPolicy, MicrogridAgentVecEnv
from utils.comparison_evaluation import evaluate_inventory, make_evaluation_runner


def config():
    loaded = []
    with patch.object(sys, "argv", ["direct_main", "--config", "gavdac_mappo"]), \
            patch.object(direct_main, "run", lambda config, logger: loaded.append(config)):
        direct_main.main()
    return loaded[0]


class MAPPOTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(self.directory.cleanup)
        self.config = config()
        self.config["env_args"].update(
            result_dir=self.directory.name, run_id="mappo_test", log_trajectories=False)
        self.config["local_results_path"] = self.directory.name

    def test_actor_local_invariance_critic_global_and_gaussian_ratios(self):
        env = MicrogridAgentVecEnv(self.config)
        obs = {k: torch.from_numpy(v) for k, v in env.reset().items()}
        policy = CentralizedPolicy(env.observation_space, env.action_space, lambda _: .0003)
        actions, values, logp = policy(obs)
        again, new_logp, entropy = policy.evaluate_actions(obs, actions)
        torch.testing.assert_close(values, again)
        torch.testing.assert_close(torch.exp(new_logp - logp), torch.ones(env.n_agents))
        self.assertTrue(torch.isfinite(entropy).all())
        changed = {**obs, "central": obs["central"] + 10}
        torch.testing.assert_close(policy.get_distribution(obs).mode(),
                                   policy.get_distribution(changed).mode(), rtol=0, atol=0)
        self.assertFalse(torch.allclose(policy.predict_values(obs), policy.predict_values(changed)))
        self.assertEqual(actions.shape, (env.n_agents, 2))
        self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.float32
                            for p in policy.parameters()))

    def test_joint_rewards_counter_truncation_and_previous_action_reset(self):
        env = MicrogridAgentVecEnv(self.config)
        env.reset()
        self.assertEqual(env.joint_steps, 0)
        for hour in range(24):
            actions = np.full((env.n_agents, 2), .1, dtype=np.float32)
            obs, rewards, done, infos = env.step(actions)
            np.testing.assert_array_equal(rewards, np.full(env.n_agents, rewards[0]))
            self.assertEqual(env.joint_steps, hour + 1)
        self.assertTrue(done.all())
        self.assertTrue(all(i["TimeLimit.truncated"] for i in infos))
        dim = env.env.get_env_info()["obs_shape"]
        np.testing.assert_array_equal(obs["local"][:, dim:dim + 2], 0)
        terminal = np.stack([i["terminal_observation"]["local"] for i in infos])
        np.testing.assert_allclose(terminal[:, dim:dim + 2], .1)
        self.assertEqual(env.episodes, 1)
        self.assertTrue(env.env.idx[0] in env.env.train_indices)

    def test_ppo_core_updates_and_bootstraps_time_limit(self):
        env = MicrogridAgentVecEnv(self.config)
        model = PPO(CentralizedPolicy, env, n_steps=24, batch_size=72, n_epochs=2,
                    seed=0, device="cpu", policy_kwargs=dict(width=64))
        initial = [p.detach().clone() for p in model.policy.parameters()]
        terminal_values, terminal_rewards = [], []

        class Capture(BaseCallback):
            def _on_step(self):
                if self.locals["dones"].all():
                    terminal_rewards.append(self.locals["rewards"].copy())
                    values = []
                    for info in self.locals["infos"]:
                        terminal = {k: torch.as_tensor(v).unsqueeze(0)
                                    for k, v in info["terminal_observation"].items()}
                        with torch.no_grad():
                            values.append(float(self.model.policy.predict_values(terminal)[0, 0]))
                    terminal_values.append(values)
                return True

        model.learn(total_timesteps=72, callback=Capture())
        self.assertEqual(env.joint_steps, 24)
        self.assertEqual(model.num_timesteps, 24 * env.n_agents)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial, model.policy.parameters())))
        np.testing.assert_allclose(
            model.rollout_buffer.rewards[-1], terminal_rewards[0]
            + model.gamma * np.array(terminal_values[0]), atol=1e-5)

    def test_runner_checkpoint_and_repeatable_inventory_evaluation(self):
        runner, mac = make_evaluation_runner(self.config)
        first, records = evaluate_inventory(
            runner, mac, [731, 732], .35, "validation", "continuous")
        second, other = evaluate_inventory(
            runner, mac, [731, 732], .35, "validation", "continuous")
        physical = lambda items: [{k: v for k, v in r.items() if k != "episode"}
                                   for r in items]
        self.assertEqual(physical(records), physical(other))
        self.assertTrue(first["physical_checks_passed"])
        self.assertEqual(first["totals"], second["totals"])
        saved = Path(self.directory.name) / "actor"
        saved.mkdir()
        mac.save_models(saved)
        old = [p.detach().clone() for p in mac.agent.parameters()]
        with torch.no_grad():
            for p in mac.agent.parameters():
                p.add_(1)
        mac.load_models(saved)
        for a, b in zip(old, mac.agent.parameters()):
            torch.testing.assert_close(a, b)
        daily, _ = evaluate_inventory(
            runner, mac, [731, 732], .35, "validation", "daily_reset")
        self.assertTrue(daily["physical_checks_passed"])

    def test_entry_training_and_strict_checkpoint_evaluation(self):
        out = Path(self.directory.name) / "integration"
        command = [
            "uv", "run", "python", "src/direct_main.py", "--config", "gavdac_mappo",
        ]
        overrides = {
            "unique_token": "mappo_test", "t_max": 480, "wandb_mode": "disabled",
            "showcase_enabled": True, "save_model": False, "test_interval": 240,
            "local_results_path": str(out), "env_args.result_dir": str(out),
            "wandb_dir": str(out / "wandb"),
        }
        for key, value in overrides.items():
            command += ["--override", key + "=" + json.dumps(value)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        completion = json.loads((out / "completion_checks.json").read_text())
        self.assertEqual(completion["actual_joint_interactions"], 480)
        self.assertEqual(completion["agent_samples"], 1440)
        self.assertEqual(completion["optimizer_updates"], 10)
        self.assertGreater(completion["finite_update_checks"], 0)
        self.assertEqual(len(list((out / "dispatch_showcases").rglob("*.png"))), 10)
        metadata = json.loads((out / "models/mappo_test/best_policies/best_rl.json").read_text())
        evaluation = subprocess.run(
            ["uv", "run", "python", "tools/evaluate_microgrid_comparison.py",
             "--checkpoint", metadata["checkpoint"], "--mode", "continuous",
             "--rows", "731", "732", "--output", str(out / "verify")],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=90)
        self.assertEqual(evaluation.returncode, 0, evaluation.stdout + evaluation.stderr)


if __name__ == "__main__":
    unittest.main()
