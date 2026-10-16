"""Reward accounting and SB3 interfaces must preserve the physical transitions."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3.common.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sb3_dispatch_reward_env import DispatchRewardEnv, load_environment_class
from tools.sb3_dispatch_reward_env import evaluate
from tools.run_sb3_reward_experiment import assert_same_replay, retryable_launch_error
from tools.run_sb3_reward_tuning import assessment


class DispatchRewardsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(self.directory.cleanup)
        self.config = yaml.safe_load(
            (ROOT / "src/config/envs/microgrid.yaml").read_text(encoding="utf-8"),
        )["env_args"]
        self.env_class = load_environment_class(ROOT)

    def env(self, mode, source="train", **settings):
        return DispatchRewardEnv(
            self.env_class, self.config, mode, self.directory.name, source=source, **settings,
        )

    def test_gym_contract_and_true_terminal(self):
        env = self.env("potential")
        check_env(env)
        env.reset(seed=3)
        for t in range(24):
            obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
            self.assertEqual(terminated, t == 23)
            self.assertFalse(truncated)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
        self.assertIn("day", info)
        json.dumps(info["day"], allow_nan=False)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(6, dtype=np.float32))

    def test_reward_modes_have_identical_physics(self):
        environments = [self.env(m) for m in ("current", "terminal", "potential", "deadline")]
        for env in environments:
            env.reset(options={"row_idx": 10})
        rng = np.random.default_rng(1)
        raw_total, terminal_total, boundary = 0.0, 0.0, 0.0
        for t in range(24):
            action = rng.uniform(-1, 1, 6).astype(np.float32)
            results = [e.step(action) for e in environments]
            for result in results[1:]:
                np.testing.assert_array_equal(result[0], results[0][0])
                self.assertEqual(result[4]["full_cost"], results[0][4]["full_cost"])
            raw_total += results[0][1]
            terminal_total += results[1][1]
            boundary += results[1][4]["terminal_inventory_cost"]
        scale = environments[0].core.reward_scale / 3
        self.assertAlmostEqual(raw_total - terminal_total, scale * boundary, places=10)
        self.assertGreater(boundary, 0)

    def test_discounted_potential_telescopes_to_zero(self):
        for direction in (-0.6, 0.6):
            terminal, potential = self.env("terminal"), self.env("potential")
            for env in (terminal, potential):
                env.reset(options={"row_idx": 10})
            discounted_difference = 0.0
            shaping = []
            for t in range(24):
                action = np.tile([0.0, direction], 3).astype(np.float32)
                left, right = terminal.step(action), potential.step(action)
                discounted_difference += potential.gamma**t * (right[1] - left[1])
                shaping.append(right[4]["shaping_cu"])
            self.assertAlmostEqual(discounted_difference, 0.0, places=10)
            self.assertTrue(any(abs(x) > 1e-5 for x in shaping))

    def test_deadline_dense_reward_has_exact_terminal_objective(self):
        for quadratic in (0, 16, 64):
            env = self.env("deadline", inventory_quadratic=quadratic)
            env.reset(options={"row_idx": 10})
            actual, raw, penalty = 0.0, 0.0, 0.0
            rng = np.random.default_rng(5)
            for t in range(24):
                action = rng.uniform(-1, 1, 6).astype(np.float32)
                _, reward, _, _, info = env.step(action)
                actual += env.gamma**t * reward
                raw += env.gamma**t * info["raw_reward"]
                penalty += env.gamma**t * info["training_inventory_cost"]
            scale = env.core.reward_scale / 3
            self.assertAlmostEqual(actual, raw - scale * penalty, places=9)
            self.assertGreaterEqual(env.metrics()["training_inventory_cost"],
                                    env.metrics()["terminal_inventory_cost"])

    def test_market_potential_is_causal_and_has_the_same_terminal_objective(self):
        env = self.env(
            "deadline", inventory_quadratic=16, deadline_start_hour=20,
            future_price_potential=True, deficit_only_potential=True,
        )
        env.reset(options={"row_idx": 10})
        env.steps_in_day = 5
        before = env.potential([0.7] * 3)
        env.core.data_dict["price"][env.core.test_indices] *= 100
        env.core.agent_data["load"][:, 6:] *= 100
        env.core.agent_data["pv"][:, 6:] *= 100
        self.assertEqual(before, env.potential([0.7] * 3))
        env.reset(options={"row_idx": 10})
        actual, base, penalty = 0.0, 0.0, 0.0
        for t in range(24):
            action = np.tile([0.1, -0.2 if t < 6 else 0.3], 3).astype(np.float32)
            _, reward, _, _, info = env.step(action)
            actual += env.gamma**t * reward
            base += env.gamma**t * info["raw_reward"]
            penalty += env.gamma**t * info["training_inventory_cost"]
        self.assertAlmostEqual(actual, base - env.core.reward_scale / 3 * penalty, places=8)

    def test_curriculum_zero_strength_preserves_original_reward(self):
        current = self.env("current")
        candidate = self.env("deadline", inventory_quadratic=16, future_price_potential=True)
        candidate.reward_strength = 0.0
        for env in (current, candidate):
            env.reset(options={"row_idx": 10})
        rng = np.random.default_rng(12)
        for _ in range(24):
            action = rng.uniform(-1, 1, 6).astype(np.float32)
            left, right = current.step(action), candidate.step(action)
            self.assertEqual(left[1], right[1])
            np.testing.assert_array_equal(left[0], right[0])

    def test_deadline_pressure_is_time_dependent_and_inventory_cost_is_symmetric(self):
        env = self.env("deadline", inventory_quadratic=16)
        env.reset(options={"row_idx": 10})
        below, above = [0.25] * 3, [0.45] * 3
        self.assertAlmostEqual(env.objective_inventory_cost(below),
                               env.objective_inventory_cost(above))
        self.assertEqual(env.potential(below), 0.0)
        env.steps_in_day = 12
        halfway = env.potential(below)
        env.steps_in_day = 23
        self.assertLess(env.potential(below), halfway)
        self.assertLess(halfway, 0)
        for settings in ({"inventory_quadratic": -1}, {"inventory_weight": 0},
                         {"deadline_power": float("nan")}):
            with self.assertRaises(ValueError):
                self.env("deadline", **settings)

    def test_acceptance_rejects_idle_depleted_and_expensive_policies(self):
        good = {"days": [{"agents": [{
            "terminal_soc_error": 0.01, "soc_span": 0.55, "bidirectional": True,
        }]}], "mean_full_cost": 103}
        self.assertTrue(assessment(good, 100)["passed"])
        for key, bad in (("terminal_soc_error", 0.15), ("soc_span", 0.0),
                         ("bidirectional", False)):
            data = json.loads(json.dumps(good))
            data["days"][0]["agents"][0][key] = bad
            self.assertFalse(assessment(data, 100)["passed"])
        expensive = {**good, "mean_full_cost": 110}
        self.assertFalse(assessment(expensive, 100)["passed"])

    def test_no_free_throughput_reward(self):
        env = self.env("potential")
        env.reset(options={"row_idx": 0})
        self.assertEqual(env.potential([env.initial_soc] * 3), 0.0)
        self.assertEqual(env.terminal_inventory_cost([env.initial_soc] * 3), 0.0)
        charge_phi = env.potential([0.4] * 3)
        discharge_phi = env.potential([0.3] * 3)
        self.assertGreater(charge_phi, 0)
        self.assertLess(discharge_phi, 0)

    def test_profitable_cyclic_arbitrage_beats_idle_without_activity_bonus(self):
        for mode in ("terminal", "potential", "deadline"):
            idle, cyclic = self.env(mode), self.env(mode)
            for env in (idle, cyclic):
                env.reset(options={"row_idx": 10})
            idle_return, cyclic_return = 0.0, 0.0
            for hour in range(24):
                idle_action = np.tile([-1.0, 0.0], 3).astype(np.float32)
                action = idle_action.copy()
                soc = cyclic.core.params["socs"]
                lower, upper = cyclic.core.battery_power_limits(soc)
                if hour <= 4:
                    power = lower
                elif 18 <= hour <= 20:
                    power = upper
                elif hour >= 21:
                    power = -np.minimum(
                        np.maximum(cyclic.initial_soc - soc, 0)
                        * cyclic.capacities / cyclic.core.params["raw_ch"], -lower,
                    )
                else:
                    power = np.zeros(3)
                action[1::2] = cyclic.core.battery_action_from_power(power, soc)
                idle_return += idle.gamma**hour * idle.step(idle_action)[1]
                cyclic_return += cyclic.gamma**hour * cyclic.step(action)[1]
            self.assertLess(cyclic.metrics()["full_cost"], idle.metrics()["full_cost"])
            self.assertGreater(cyclic_return, idle_return)
            self.assertAlmostEqual(cyclic.metrics()["terminal_error_kwh"], 0.0, places=5)
            self.assertEqual(cyclic.metrics()["bidirectional_agent_fraction"], 1.0)

    def test_seeded_days_and_initial_inventory_match(self):
        first, second = self.env("current"), self.env("potential")
        for i in range(5):
            opts = {"seed": 4} if i == 0 else {}
            np.testing.assert_array_equal(first.reset(**opts)[0], second.reset(**opts)[0])
            self.assertEqual(first.row, second.row)
            np.testing.assert_array_equal(first.core.params["socs"], [0.35] * 3)
        with self.assertRaises(ValueError):
            first.reset(options={"row_idx": 740})
        with self.assertRaises(ValueError):
            first.reset(options={"row_idx": 0, "initial_soc": 0.8})
        validation = self.env("potential", "validation")
        validation.reset(options={"row_idx": 740})
        with self.assertRaises(ValueError):
            validation.reset(options={"row_idx": 10})

    def test_saturation_cannot_count_as_cycling(self):
        for action, expected_soc in ((-1, 0.8), (1, 0.2)):
            env = self.env("potential")
            env.reset(options={"row_idx": 10})
            for _ in range(24):
                env.step(np.tile([0.0, action], 3).astype(np.float32))
            metrics = env.metrics()
            self.assertEqual(metrics["bidirectional_agent_fraction"], 0.0)
            self.assertAlmostEqual(metrics["terminal_error_kwh"],
                                   abs(expected_soc - 0.35) * 1000)
            for a in metrics["agents"]:
                self.assertAlmostEqual(a["terminal_soc"], expected_soc)

    def test_reward_coefficients_ignore_heldout_inputs(self):
        env = self.env("potential")
        price, boundary = env.inventory_price.copy(), env.terminal_price.copy()
        env.core.data_dict["price"][env.core.validation_indices] *= 100
        env.core.data_dict["price"][env.core.test_indices] *= 100
        np.testing.assert_array_equal(price, env.inventory_price)
        np.testing.assert_array_equal(boundary, env.terminal_price)

    def test_replay_ignores_only_runtime_episode_counter(self):
        class FixedPolicy:
            def predict(self, obs, deterministic):
                return np.zeros(6, dtype=np.float32), None

        env = self.env("terminal", "validation")
        first, records = evaluate(FixedPolicy(), env, [740], collect_records=True)
        second, replay = evaluate(FixedPolicy(), env, [740], collect_records=True)
        self.assertNotEqual(records[0]["episode"], replay[0]["episode"])
        assert_same_replay(first, second, records, replay)
        replay[0]["soc"] += 1e-6
        with self.assertRaises(RuntimeError):
            assert_same_replay(first, second, records, replay)
        with self.assertRaises(RuntimeError):
            assert_same_replay(first, second, records, replay[:-1])

    def test_retry_is_limited_to_pre_python_permission_failure(self):
        missing = Path(self.directory.name) / "missing"
        messages = [
            "error: Failed to spawn: python.exe\nCaused by: access denied (os error 5)",
            "error: uv trampoline failed to spawn Python child process\n"
            "Caused by: permission denied (os error 5)",
        ]
        for message in messages:
            self.assertTrue(retryable_launch_error(message, missing))
            self.assertFalse(retryable_launch_error(message, Path(self.directory.name)))
            self.assertFalse(retryable_launch_error("Traceback\n" + message, missing))
        self.assertFalse(retryable_launch_error("ValueError: invalid training tensor", missing))


if __name__ == "__main__":
    unittest.main()
