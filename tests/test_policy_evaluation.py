import csv
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from controllers.cqmix_controller import CQMixMAC
from direct_main import load_yaml, recursive_dict_update
from runners.episode_runner import EpisodeRunner
from utils.policy_evaluation import (
    BestPolicyStore, evaluate_fixed, evaluate_saved_policy, fixed_rows,
)


class QuietLogger:
    def log_stat(self, *args):
        pass


def setup_runner(directory):
    config = load_yaml(str(ROOT / "src/config/default.yaml"))
    config = recursive_dict_update(config, load_yaml(str(ROOT / "src/config/envs/microgrid.yaml")))
    config.update(name="policy_tests", env="microgrid", seed=0, device="cpu")
    config["env_args"].update(seed=0, result_dir=str(directory), run_id="fixed_days")
    args = SimpleNamespace(**deepcopy(config))
    runner = EpisodeRunner(args, QuietLogger())
    info = runner.get_env_info()
    for key in ("n_agents", "n_actions", "state_shape", "obs_shape", "action_spaces"):
        setattr(args, key, info[key])
    scheme = {
        "state": {"vshape": info["state_shape"]},
        "obs": {"vshape": info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (info["n_actions"],), "group": "agents", "dtype": torch.float},
        "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    mac = CQMixMAC(scheme, {"agents": 3}, args)
    runner.setup(scheme, {"agents": 3}, {}, mac)
    return config, runner, mac


class PolicyEvaluationTest(unittest.TestCase):
    def test_ea_rollouts_share_training_scenario_without_action_noise(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            _, runner, mac = setup_runner(directory)
            scenario = runner.env.training_scenario()
            first, first_return = runner.run(mac, training_options=scenario, explore=False)
            first_actions = first["actions"].clone()
            second, second_return = runner.run(mac, training_options=scenario, explore=False)
            self.assertEqual(first_return, second_return)
            torch.testing.assert_close(first_actions, second["actions"], rtol=0, atol=0)
            self.assertEqual(runner.t_env, 48)
            self.assertEqual(runner.env.idx, [scenario["row_idx"] + 1, 0])

    def test_fixed_days_do_not_consume_training_rng_and_are_repeatable(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            _, runner, mac = setup_runner(directory)
            rows = fixed_rows(runner.env, "validation", [], 2)
            state = deepcopy(runner.env.rng.bit_generator.state)
            first = evaluate_fixed(runner, mac, rows, 0.35, "validation", "ea", 123)
            second = evaluate_fixed(runner, mac, rows, 0.35, "validation", "rl", 123)
            self.assertEqual(first, second)
            self.assertEqual(state, runner.env.rng.bit_generator.state)
            self.assertEqual(runner.t_env, 0)
            self.assertEqual(runner.batch["obs"].shape, (1, 25, 3, 20))
            self.assertEqual(runner.batch["state"].shape, (1, 25, 60))
            self.assertEqual(runner.batch["actions"].shape, (1, 25, 3, 2))
            self.assertEqual(runner.batch["reward"].shape, (1, 25, 1))
            self.assertEqual(runner.batch["obs"].device.type, "cpu")
            self.assertTrue(set(rows).isdisjoint(runner.env.test_indices))
            with self.assertRaises(ValueError):
                fixed_rows(runner.env, "test", rows, 2)

    def test_best_policy_is_immutable_and_restores_exact_actor_and_trajectories(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            config, runner, mac = setup_runner(directory)
            rows = fixed_rows(runner.env, "validation", [], 2)
            evaluation = evaluate_fixed(runner, mac, rows, 0.35, "validation", "ea", 123)
            store = BestPolicyStore(Path(directory) / "best", config)
            checkpoint = store.consider(mac, "ea", 123, evaluation, runner.env)
            original = {key: value.clone() for key, value in mac.agent.state_dict().items()}
            self.assertIsNone(store.consider(mac, "ea", 124, evaluation, runner.env))
            with torch.no_grad():
                for parameter in mac.parameters():
                    parameter.add_(1)
            runner.args.policy_checkpoint = str(checkpoint)
            runner.args.fixed_eval_days = 2
            output = evaluate_saved_policy(runner, mac, runner.args)
            self.assertEqual(output["policy_source"], "ea")
            for key, value in mac.agent.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            csv_path = next(runner.env.render_dir.glob("episodes_test_*.csv"))
            with csv_path.open(encoding="utf-8", newline="") as file:
                records = list(csv.DictReader(file))
            self.assertEqual(len(records), 2 * 24 * 3)
            self.assertEqual({r["policy_source"] for r in records}, {"ea"})
            self.assertEqual({r["checkpoint_step"] for r in records}, {"123"})
            agents = {a["agent_name"]: a for a in config["env_args"]["data"]}
            for r in records:
                executed = float(r["pb"])
                before = float(r["soc_init"])
                agent = agents[r["agent"]]
                energy = (executed * agent["raw_ch"] if executed < 0
                          else executed / agent["raw_dis"])
                after = before - energy / agent["battery_cap"]
                self.assertAlmostEqual(float(r["soc"]), after)
                self.assertEqual(r["battery_model_version"], "gavdac_dispatch_v3")
            manifest = json.loads((Path(directory) / "best/best_ea.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["checkpoint"], str(checkpoint.resolve()))
            runner.env.time_step_hours = 0.25
            with self.assertRaisesRegex(ValueError, "time step"):
                evaluate_saved_policy(runner, mac, runner.args)


if __name__ == "__main__":
    unittest.main()
