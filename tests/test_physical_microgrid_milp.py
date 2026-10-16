import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_physical_microgrid_milp import (
    acceptance, make_env, solve_agent, solve_day, solve_network,
)


PARAMS = {
    "battery_cap": 100, "battery_charge_max": 25, "battery_discharge_max": 25,
    "raw_ch": 0.93, "raw_dis": 0.93, "dg_max": 30,
    "costa": 0.00015, "costb": 0.14, "costc": 0, "env_param": 0.015,
    "cycle_cost": 0.01,
}


class PhysicalMilpTest(unittest.TestCase):
    def test_real_cycling_is_profitable_and_energy_conserving(self):
        load, pv = np.full(24, 80.0), np.zeros(24)
        price = np.array([0.06] * 6 + [0.30] * 15 + [0.06] * 3)
        result = solve_agent(load, pv, price, PARAMS, 0.35, terminal_soc=0.35)
        baseline = solve_agent(load, pv, price, PARAMS, 0.35, terminal_soc=0.35,
                               battery_enabled=False)
        self.assertEqual(result["status"], 0)
        self.assertLess(result["objective"], baseline["objective"])
        s = {k: np.array(v) for k, v in result["schedule"].items()}
        self.assertGreaterEqual(np.ptp(s["soc"]), 0.6 - 1e-8)
        self.assertGreaterEqual(s["discharge"].sum(), 0.9 * 60 * 0.93)
        np.testing.assert_allclose(
            np.diff(s["soc"]) * 100, 0.93 * s["charge"] - s["discharge"] / 0.93,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            s["pg"] + s["pd"] + s["discharge"] + pv,
            load + s["charge"] + s["spill"], atol=1e-8,
        )
        self.assertTrue(np.all(s["charge"] * s["discharge"] < 1e-8))
        self.assertAlmostEqual(s["soc"][-1], 0.35)

    def test_flat_price_does_not_reward_useless_battery_cycles(self):
        p = {**PARAMS, "dg_max": 0}
        result = solve_agent(np.full(24, 80), np.zeros(24), np.full(24, 0.1),
                             p, 0.35, terminal_soc=0.35)
        self.assertAlmostEqual(sum(result["schedule"]["charge"]), 0, places=6)
        self.assertAlmostEqual(sum(result["schedule"]["discharge"]), 0, places=6)

    def test_real_representative_day_uses_all_devices_and_saves_full_cost(self):
        config = yaml.safe_load(
            (ROOT / "src/config/envs/microgrid.yaml").read_text(encoding="utf-8"),
        )["env_args"]
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            env = make_env(config, Path(directory))
            result = solve_day(env, config, 927, "test", 0.35)
            baseline = solve_day(env, config, 927, "test", 0.35, battery_enabled=False)
            isolated = solve_day(env, config, 927, "test", 0.35, exchange_enabled=False)
            self.assertTrue(acceptance([result])["passed"])
            self.assertLess(result["full_cost"], baseline["full_cost"] * 0.99)
            self.assertLess(result["full_cost"], isolated["full_cost"] * 0.995)
            self.assertLess(result["max_replay_error"], 1e-6)

    def test_joint_network_limits_losses_and_no_grid_resale(self):
        params = [{**PARAMS, "dg_max": 0}] * 3
        load = np.tile([0.0, 80.0, 60.0], (24, 1))
        pv = np.tile([100.0, 0.0, 0.0], (24, 1))
        solution = solve_network(load, pv, np.full(24, 0.3), params, [40, 25, 25],
                                 0.98, 0.003, 0.35, terminal_soc=0.35,
                                 battery_enabled=False)
        schedules = solution["schedules"]
        np.testing.assert_allclose(schedules[0]["peer_export"], 40)
        np.testing.assert_allclose(schedules[0]["pg"], 0)
        received = sum(np.array(s["peer_import"]) for s in schedules)
        np.testing.assert_allclose(received, 39.2)
        for s in schedules:
            self.assertTrue(np.all(np.array(s["pg"]) * s["peer_export"] < 1e-9))

    def test_prohibitive_degradation_cost_does_not_force_storage_activity(self):
        p = {**PARAMS, "cycle_cost": 1.0}
        result = solve_agent(
            np.full(24, 80), np.zeros(24),
            np.array([0.06] * 6 + [0.30] * 15 + [0.06] * 3),
            p, 0.35, terminal_soc=0.35,
        )
        self.assertAlmostEqual(sum(result["schedule"]["charge"]), 0, places=6)
        self.assertAlmostEqual(sum(result["schedule"]["discharge"]), 0, places=6)


if __name__ == "__main__":
    unittest.main()
