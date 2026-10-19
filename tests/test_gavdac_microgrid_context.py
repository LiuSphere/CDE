"""Learnability changes must retain the physical objective and information rules."""

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from envs.maenv import MGEnv


class DispatchContextTest(unittest.TestCase):
    def make_env(self):
        config = yaml.safe_load(
            (ROOT / "src/config/envs/microgrid.yaml").read_text(encoding="utf-8"),
        )["env_args"]
        directory = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(directory.cleanup)
        config.update(result_dir=directory.name, seed=0)
        return MGEnv(deepcopy(config), "gavdac_context_tests")

    def test_fraction_actions_are_feasible_and_invertible(self):
        env = self.make_env()
        rng = np.random.default_rng(5)
        for _ in range(100):
            soc = rng.uniform(env.soc_min, env.soc_max, 3)
            actions = rng.uniform([0, -1], [1, 1], (3, 2))
            result = env.dispatch(actions, soc, (740, 12))
            lower, upper = env.battery_power_limits(soc)
            expected = actions[:, 1] * np.where(actions[:, 1] >= 0, upper, -lower)
            np.testing.assert_allclose(result["pb"], expected, atol=1e-10)
            np.testing.assert_allclose(result["battery_action_clip"], 0)
            np.testing.assert_allclose(result["punishment"], 0)
            inverse = env.battery_action_from_power(result["pb"], soc)
            np.testing.assert_allclose(inverse, actions[:, 1], atol=1e-10)
        with self.assertRaises(ValueError):
            env.battery_action_from_power(np.ones(3), np.full(3, env.soc_min))

    def test_centering_preserves_same_scenario_discounted_policy_differences(self):
        env = self.make_env()
        rng = np.random.default_rng(7)
        offsets = []
        for policy in range(3):
            soc = np.full(3, 0.35)
            offset = 0.0
            for t in range(24):
                actions = (np.zeros((3, 2)) if policy == 0
                           else rng.uniform([0, -1], [1, 1], (3, 2)))
                r = env.dispatch(actions, soc, (740, t))
                soc = r["soc"]
                np.testing.assert_allclose(
                    r["reward"], (r["baseline_cost"] - r["full_cost"]) * env.reward_scale,
                )
                offset += 0.99**t * np.mean(r["reward"] - r["cost_reward"])
            offsets.append(offset)
        np.testing.assert_allclose(offsets, offsets[0], atol=1e-10)

    def test_zero_dispatch_has_zero_centered_reward(self):
        env = self.make_env()
        for hour in range(24):
            r = env.dispatch(np.zeros((3, 2)), np.full(3, 0.35), (740, hour))
            np.testing.assert_allclose(r["reward"], 0, atol=1e-12)
            self.assertGreater(r["full_cost"].sum(), 0)

    def test_forecasts_do_not_read_future_realizations(self):
        env = self.make_env()
        env.reset_test(source="validation", row_idx=740, initial_soc=0.35)
        before = env._observations()
        self.assertEqual(before.shape, (3, 20))
        for key in ("load", "pv"):
            env.agent_data[key][740, 1:] *= 1000
            env.agent_data[key][env.test_indices] *= 1000
        np.testing.assert_array_equal(env._observations(), before)
        for key in ("load", "pv"):
            np.testing.assert_allclose(
                env.forecast_templates[key], env.agent_data[key][env.train_indices].mean(axis=0),
            )

    def test_observation_exposes_actual_available_battery_power(self):
        env = self.make_env()
        env.params["socs"][:] = [0.2, 0.5, 0.8]
        obs = env._observations()
        lower, upper = env.battery_power_limits(env.params["socs"])
        np.testing.assert_allclose(obs[:, 12], -lower / env.observation_power_scale)
        np.testing.assert_allclose(obs[:, 13], upper / env.observation_power_scale)
        for a, row in enumerate(obs):
            self.assertTrue(env.observation_space(a + 1).contains(row))

    def test_training_scenario_replay_is_repeatable_and_split_safe(self):
        env = self.make_env()
        env.idx = [10, 0]
        env.params["socs"][:] = [0.3, 0.4, 0.5]
        scenario = env.training_scenario()
        rng = deepcopy(env.rng.bit_generator.state)
        returns = []
        for _ in range(5):
            env.reset(options=scenario)
            total = 0
            for t in range(24):
                total += env.step(np.tile([0.5, 0.2], (3, 1)))[0]
            returns.append(total)
        np.testing.assert_array_equal(returns, np.full(5, returns[0]))
        self.assertEqual(env.idx, [11, 0])
        self.assertEqual(env.rng.bit_generator.state, rng)
        with self.assertRaises(ValueError):
            env.reset(options={"row_idx": 740, "initial_soc": [0.35] * 3})
        env.reset_test(source="validation", row_idx=740, initial_soc=0.35)
        saved = env.training_scenario()
        self.assertEqual(saved["row_idx"], 11)
        env.reset(options=saved)
        self.assertEqual(env.idx, [11, 0])


if __name__ == "__main__":
    unittest.main()
