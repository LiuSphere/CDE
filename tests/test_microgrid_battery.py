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
from envs.maenv import MGEnv
from tools.run_perfect_prediction_oracle import simulate_episode


class BatteryTest(unittest.TestCase):
    def make_env(self, overrides=None):
        config = yaml.safe_load(
            (ROOT / "src/config/envs/microgrid.yaml").read_text(encoding="utf-8"),
        )["env_args"]
        directory = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(directory.cleanup)
        config.update(result_dir=directory.name, run_id="battery_tests", seed=0)
        config.update(overrides or {})
        return MGEnv(deepcopy(config), "tests")

    def test_observation_contract_is_finite_and_inside_declared_bounds(self):
        env = self.make_env()
        for row in (0, 730, 912, 1094):
            env.reset_test(source="all", row_idx=row, initial_soc=0.35)
            for _ in range(24):
                for i, obs in enumerate(env.get_obs()):
                    self.assertTrue(env.observation_space(i + 1).contains(obs))
                env.step(np.zeros((3, 2)))
        self.assertEqual(env.get_env_info()["obs_shape"], 20)
        self.assertEqual(env.get_env_info()["state_shape"], 60)
        self.assertEqual(env.get_state().dtype, np.float32)

    def test_repeated_dispatch_matches_independent_scalar_physics(self):
        env = self.make_env()
        rng = np.random.default_rng(17)
        for _ in range(72):
            before = env.params["socs"].copy()
            actions = rng.uniform([0, -1], [1, 1], size=(3, 2))
            _, done, _ = env.step(actions)
            for i, record in enumerate(env.ep_buffer[-3:]):
                p = env.params
                cap, eta_c, eta_d = p["battery_caps"][i], p["raw_ch"][i], p["raw_dis"][i]
                lower = -min(p["battery_charge_max"][i],
                             (env.soc_max - before[i]) * cap / eta_c)
                upper = min(p["battery_discharge_max"][i],
                            (before[i] - env.soc_min) * cap * eta_d)
                pb = actions[i, 1] * (-lower if actions[i, 1] < 0 else upper)
                soc = before[i] - (pb * eta_c if pb < 0 else pb / eta_d) / cap
                self.assertAlmostEqual(record["pb"], pb)
                self.assertAlmostEqual(record["soc"], soc)
                self.assertAlmostEqual(
                    record["pv"] + record["pd"] + pb + record["pg"] + record["peer_import"],
                    record["load"] + record["spill"] + record["peer_export"],
                )
                self.assertAlmostEqual(record["bess_cost"], abs(pb) * p["cycle_cost"][i])
                self.assertGreaterEqual(record["buy_cost"], 0)
                self.assertAlmostEqual(
                    record["full_cost"],
                    record["economic_cost"] - record["env_reward"] - record["punishment"],
                )
            if done:
                env.reset()

    def test_saturated_charge_respects_rating_and_energy_balance(self):
        env = self.make_env()
        env.params["socs"][:] = 0.79
        env.step(np.tile([0, -1], (3, 1)))
        np.testing.assert_allclose(env.params["socs"], env.soc_max)
        for i, r in enumerate(env.ep_buffer):
            expected = -(env.soc_max - 0.79) * env.params["battery_caps"][i] / env.params["raw_ch"][i]
            self.assertAlmostEqual(r["pb"], expected)
            self.assertEqual(r["punishment"], 0)

    def test_daily_reset_preserves_soc_and_next_day(self):
        env = self.make_env()
        for _ in range(24):
            _, done, info = env.step(np.tile([0, -0.01], (3, 1)))
        self.assertTrue(done)
        self.assertTrue(info["episode_limit"])
        final = env.params["socs"].copy()
        env.reset()
        np.testing.assert_array_equal(env.params["socs"], final)
        self.assertEqual(env.idx, [1, 0])
        env.idx = [730, 23]
        env.step(np.zeros((3, 2)))
        self.assertEqual(env.idx, [0, 0])

    def test_observation_advances_to_next_hour(self):
        env = self.make_env()
        first = env.get_obs()[0].copy()
        env.step(np.zeros((3, 2)))
        self.assertNotEqual(float(first[0]), float(env.get_obs()[0][0]))
        self.assertAlmostEqual(
            float(env.get_obs()[0][0]),
            env.data_dict["price"][0, 1] / env.observation_price_scale, places=6,
        )

    def test_evaluation_restores_training_clock_soc_and_rng(self):
        env = self.make_env()
        for _ in range(24):
            env.step(np.tile([0, -0.01], (3, 1)))
        idx, socs = env.idx.copy(), env.params["socs"].copy()
        rng = deepcopy(env.rng.bit_generator.state)
        env.reset_test(source="validation", initial_soc=0.7)
        for _ in range(24):
            env.step(np.tile([0, 0.1], (3, 1)))
        env.reset_test(source="test", initial_soc=0.4)
        env.step(np.zeros((3, 2)))
        env.reset()
        self.assertEqual(env.idx, idx)
        np.testing.assert_array_equal(env.params["socs"], socs)
        self.assertEqual(env.rng.bit_generator.state, rng)

    def test_numerical_simulator_matches_without_mutating_environment(self):
        env = self.make_env()
        initial = np.array([0.25, 0.5, 0.75])
        actions = np.random.default_rng(22).uniform([0, -1], [1, 1], size=(24, 3, 2))
        before, index = env.params["socs"].copy(), env.idx.copy()
        simulated = simulate_episode(actions.flatten(), env, 0, initial)
        np.testing.assert_array_equal(env.params["socs"], before)
        self.assertEqual(env.idx, index)
        env.reset_test(source="all", row_idx=0, initial_soc=initial)
        total = 0
        for action in actions:
            reward, _, info = env.step(action)
            total += reward
        self.assertEqual(simulated["episode_return"], total)
        for key in ("grid_purchase_cost", "baseline_grid_purchase_cost",
                    "grid_purchase_saving", "grid_purchase_saving_ratio", "mean_soc"):
            self.assertAlmostEqual(simulated[key], info[key])

    def test_invalid_parameters_fail_instead_of_falling_back(self):
        with self.assertRaises(ValueError):
            self.make_env({"price_scale": -1})
        with self.assertRaises(ValueError):
            self.make_env({"model_version": "maenv_sep_v1"})
        with self.assertRaises(ValueError):
            self.make_env({"initial_soc": 0.99})


if __name__ == "__main__":
    unittest.main()
