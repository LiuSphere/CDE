import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from envs.maenv import MGEnv, standardize_data
from tools.run_perfect_prediction_oracle import simulate_episode


class BatteryTest(unittest.TestCase):
    def test_equipment_parameters_match_supplied_ippo_config(self):
        env = self.make_env()
        expected = {
            "dg_max": [225, 225, 180],
            "battery_caps": [100, 100, 100],
            "battery_punishment_params": [27.5, 27.5, 100],
            "raw_ch": [0.98, 0.98, 0.98],
            "raw_dis": [0.98, 0.98, 0.98],
            "bessa": [0.0152, 0.0158, 0.0172],
            "bessb": [5.61, 5.63, 5.72],
            "bessc": [24, 34, 37],
            "costa": [0.0083, 0.0079, 0.0096],
            "costb": [5.76, 5.67, 5.79],
            "costc": [64, 357, 104],
            "env_param": [1.6481, 1.6481, 1.6481],
        }
        for key, values in expected.items():
            np.testing.assert_array_equal(env.params[key], values)
        for i in range(3):
            np.testing.assert_array_equal(
                env.observation_space(i + 1).low, [0, 0, 0, -1],
            )
            np.testing.assert_array_equal(env.observation_space(i + 1).high, 1)
            np.testing.assert_array_equal(env.action_space(i + 1).low, [0, -1])
            np.testing.assert_array_equal(env.action_space(i + 1).high, [1, 1])

    def make_env(self):
        config = yaml.safe_load((ROOT / "src/config/envs/microgrid.yaml").read_text(encoding="utf-8"))
        config = deepcopy(config["env_args"])
        directory = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(directory.cleanup)
        config.update(result_dir=directory.name, run_id="battery_tests", seed=0)
        env = MGEnv(config, "tests")
        env.reset()
        return env

    def test_reference_discharge_multiplies_efficiency(self):
        env = self.make_env()
        env.params["socs"][:] = 0.5
        env.step(np.tile([0, 0.1], (3, 1)).astype(float))
        np.testing.assert_allclose(env.params["socs"], 0.402)

    def test_reference_low_soc_projection_and_pre_projection_penalty(self):
        env = self.make_env()
        env.params["socs"][:] = 0.25
        env.step(np.tile([0, 0.1], (3, 1)))
        for row in env.ep_buffer:
            self.assertAlmostEqual(row["pb"], 5)
            self.assertAlmostEqual(row["soc"], 0.2)
            self.assertAlmostEqual(row["punishment"], 0)

    def test_reference_high_soc_projection(self):
        env = self.make_env()
        env.params["socs"][:] = 0.75
        env.step(np.tile([0, -1], (3, 1)))
        for i, row in enumerate(env.ep_buffer):
            self.assertAlmostEqual(row["pb"], -5)
            self.assertAlmostEqual(row["soc"], 0.8)
            self.assertAlmostEqual(
                row["punishment"], -75 * env.params["battery_punishment_params"][i],
            )

    def test_standardized_data_is_used_without_agent_splitting(self):
        env = self.make_env()
        obs = env.get_obs()
        self.assertEqual(env.get_env_info()["obs_shape"], 4)
        self.assertEqual(env.get_env_info()["state_shape"], 12)
        self.assertEqual(env.get_state().shape, (12,))
        for i in range(3):
            np.testing.assert_array_equal(obs[0], obs[i])
        # The reference's declared [0, 1] bounds do not contain its standardized
        # negative inputs. Reproduce both without silently clipping observations.
        self.assertFalse(env.observation_space(1).contains(obs[0]))
        for name in ("price", "load", "pv"):
            expected = standardize_data(env.raw_data[name])
            np.testing.assert_array_equal(env.data_dict[name], expected)
        self.assertAlmostEqual(float(obs[0][0]), -1.6367058, places=6)
        self.assertAlmostEqual(float(obs[0][1]), -0.4398988, places=6)
        self.assertAlmostEqual(float(obs[0][2]), -0.7584983, places=6)
        np.testing.assert_array_equal(standardize_data(np.ones((2, 24))), 0)

    def test_independent_grid_settlement_and_exact_reward_components(self):
        env = self.make_env()
        env.params["socs"][:] = 0.5
        env.data_dict["price"][0, 0] = 2
        env.data_dict["load"][0, 0] = 100
        env.data_dict["pv"][0, 0] = 0
        actions = np.array([[0.5, 0.1], [0.25, -0.1], [0, 0]])
        team, _, info = env.step(actions)
        expected = []
        for i, row in enumerate(env.ep_buffer):
            pd, pb, soc = row["pd"], row["pb"], row["soc"]
            pg = 100 - pd - pb
            params = env.params
            x = pb + 3 * params["battery_caps"][i] * (1 - soc)
            bess = -(
                params["bessa"][i] * x**2 + params["bessb"][i] * x
                + params["bessc"][i]
            ) * (0.3 if pb > 0 else 1)
            gen = -(
                params["costa"][i] * pd**2 + params["costb"][i] * pd
                + params["costc"][i]
            )
            buy = -pg * 2 if pg > 0 else 0
            eco = 0.5 * gen + bess + buy
            reward = (eco - pd * params["env_param"][i] + row["punishment"]) * 0.1
            self.assertAlmostEqual(row["pg"], pg)
            self.assertAlmostEqual(row["eco_reward"], eco)
            self.assertAlmostEqual(row["reward"], reward)
            self.assertEqual(row["team_reward"], team)
            expected.append(reward)
        self.assertAlmostEqual(team, float(np.mean(np.array(expected, dtype=np.float32))))
        self.assertAlmostEqual(info["grid_purchase_cost"], 2 * (0 + 53.75 + 100))

    def test_negative_standardized_prices_are_not_clamped_and_export_is_free(self):
        env = self.make_env()
        env.data_dict["price"][0, :2] = -2
        env.data_dict["load"][0, :2] = [10, 0]
        env.data_dict["pv"][0, :2] = [0, 10]
        env.step(np.zeros((3, 2)))
        self.assertEqual(env.ep_buffer[0]["buy_cost"], -20)
        env.step(np.zeros((3, 2)))
        self.assertEqual(env.ep_buffer[-1]["buy_cost"], 0)

    def test_polynomial_is_not_replaced_by_absolute_value(self):
        env = self.make_env()
        env.params["bessa"][:] = 0
        env.params["bessb"][:] = 0
        env.params["bessc"][:] = -5
        env.step(np.zeros((3, 2)))
        self.assertEqual(env.ep_buffer[0]["bess_cost"], -5)

    def test_repeated_dispatch_matches_reference_scalar_equations(self):
        env = self.make_env()
        rng = np.random.default_rng(17)
        for _ in range(72):
            before = env.params["socs"].copy()
            actions = rng.uniform([0, -1], [1, 1], size=(3, 2))
            _, terminated, _ = env.step(actions)
            for i, row in enumerate(env.ep_buffer[-3:]):
                pb = actions[i, 1] * 100
                soc = before[i] - 0.98 * pb / 100
                if soc < 0.2:
                    soc, pb = 0.2, (before[i] - 0.2) * 100
                if soc > 0.8:
                    soc, pb = 0.8, (before[i] - 0.8) * 100
                self.assertAlmostEqual(row["pb"], pb)
                self.assertAlmostEqual(row["soc"], soc)
            self.assertTrue(np.all((env.params["socs"] >= 0.2) & (env.params["socs"] <= 0.8)))
            if terminated:
                env.reset()

    def test_daily_reset_preserves_soc_and_advances_the_day(self):
        env = self.make_env()
        env.params["socs"][:] = 0.5
        for _ in range(24):
            _, terminated, info = env.step(np.tile([0, -0.01], (3, 1)))
        self.assertTrue(terminated)
        self.assertTrue(info["episode_limit"])
        final = env.params["socs"].copy()
        self.assertEqual(env.idx, [1, 0])
        obs = env.reset()
        np.testing.assert_array_equal(env.params["socs"], final)
        np.testing.assert_allclose(np.array(obs)[:, 3], final)
        self.assertEqual(env.idx, [1, 0])
        env.idx = [730, 23]
        env.step(np.zeros((3, 2)))
        self.assertEqual(env.idx, [0, 0])

    def test_returned_observation_keeps_reference_current_hour_timing(self):
        env = self.make_env()
        first = np.array(env.get_obs())[:, :3]
        env.step(np.zeros((3, 2)))
        np.testing.assert_array_equal(np.array(env.get_obs())[:, :3], first)
        env.step(np.zeros((3, 2)))
        expected = [env.data_dict[key][0, 1] for key in ("price", "load", "pv")]
        np.testing.assert_array_equal(env.get_obs()[0][:3], expected)

    def test_evaluation_does_not_replace_the_training_soc_or_clock(self):
        env = self.make_env()
        for _ in range(24):
            env.step(np.tile([0, -0.01], (3, 1)))
        idx, socs = env.idx.copy(), env.params["socs"].copy()
        env.reset_test(source="validation", initial_soc=0.7)
        for _ in range(24):
            env.step(np.tile([0, 0.1], (3, 1)))
        env.reset_test(source="test", initial_soc=0.4)
        env.step(np.zeros((3, 2)))
        env.reset()
        self.assertEqual(env.idx, idx)
        np.testing.assert_array_equal(env.params["socs"], socs)

    def test_numerical_simulator_matches_environment_rewards_and_costs(self):
        env = self.make_env()
        initial = np.array([0.25, 0.5, 0.75])
        actions = np.random.default_rng(22).uniform(
            [0, -1], [1, 1], size=(24, 3, 2),
        )
        before = env.params["socs"].copy()
        simulated = simulate_episode(actions.flatten(), env, 0, initial)
        np.testing.assert_array_equal(env.params["socs"], before)
        env.reset_test(source="all", row_idx=0, initial_soc=initial)
        total = 0
        for action in actions:
            reward, _, info = env.step(action)
            total += reward
        self.assertEqual(simulated["episode_return"], total)
        for key in (
            "grid_purchase_cost", "baseline_grid_purchase_cost",
            "grid_purchase_saving", "grid_purchase_saving_ratio", "mean_soc",
        ):
            self.assertAlmostEqual(simulated[key], info[key])


if __name__ == "__main__":
    unittest.main()
