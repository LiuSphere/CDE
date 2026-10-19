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


class PhysicalMicrogridTest(unittest.TestCase):
    def make_env(self):
        config = yaml.safe_load(
            (ROOT / "src/config/envs/microgrid.yaml").read_text(encoding="utf-8"),
        )["env_args"]
        directory = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(directory.cleanup)
        config.update(result_dir=directory.name, seed=0)
        return MGEnv(deepcopy(config), "physics_tests")

    def test_physical_version_and_positive_settlement_inputs(self):
        env = self.make_env()
        self.assertEqual(env.battery_model_version, "gavdac_dispatch_v3")
        self.assertTrue(np.all(env.data_dict["price"] > 0))
        self.assertTrue(np.all(env.data_dict["load"] >= 0))
        self.assertTrue(np.all(env.data_dict["pv"] >= 0))

    def test_discharge_divides_by_efficiency(self):
        env = self.make_env()
        env.params["socs"][:] = 0.5
        env.step(np.tile([0, 0.1], (env.n_agents, 1)))
        available = np.minimum(
            env.params["battery_discharge_max"],
            (0.5 - env.soc_min) * env.params["battery_caps"] * env.params["raw_dis"],
        )
        expected = 0.5 - (
            0.1 * available * env.time_step_hours
            / (env.params["raw_dis"] * env.params["battery_caps"])
        )
        np.testing.assert_allclose(env.params["socs"], expected, atol=1e-12)

    def test_idle_storage_is_free_and_tiny_discharge_has_no_discount(self):
        env = self.make_env()
        soc = np.full(env.n_agents, 0.6)
        idle = env.dispatch(np.zeros((env.n_agents, 2)), soc, (0, 0))
        tiny = np.zeros((env.n_agents, 2))
        tiny[:, 1] = 1e-7
        moving = env.dispatch(tiny, soc, (0, 0))
        np.testing.assert_array_equal(idle["bess_cost"], 0)
        self.assertTrue(np.all(moving["bess_cost"] > 0))
        self.assertLess(float(moving["bess_cost"].sum()), 1e-4)

    def test_split_conserves_source_power_and_balance(self):
        env = self.make_env()
        result = env.dispatch(np.zeros((env.n_agents, 2)), np.full(3, 0.5), (0, 0))
        self.assertAlmostEqual(result["load"].sum(), env.data_dict["load"][0, 0])
        self.assertAlmostEqual(result["pv"].sum(), env.data_dict["pv"][0, 0])
        np.testing.assert_allclose(
            result["pv"] + result["pd"] + result["pb"] + result["pg"] + result["peer_import"],
            result["load"] + result["spill"] + result["peer_export"], atol=1e-10,
        )

    def test_charge_discharge_round_trip_loses_energy(self):
        env = self.make_env()
        env.params["socs"][:] = 0.5
        before = env.params["socs"].copy()
        env.step(np.tile([0, -0.1], (3, 1)))
        env.step(np.tile([0, 0.1], (3, 1)))
        self.assertTrue(np.all(env.params["socs"] < before))

    def test_bounds_and_rejected_invalid_requests(self):
        env = self.make_env()
        env.params["socs"][:] = env.soc_min
        env.step(np.tile([0, 1], (3, 1)))
        np.testing.assert_allclose(env.params["socs"], env.soc_min)
        self.assertTrue(all(r["pb"] == 0 for r in env.ep_buffer))
        self.assertTrue(all(r["punishment"] == 0 for r in env.ep_buffer))
        with self.assertRaises(ValueError):
            env.step(np.full((3, 2), np.nan))
        with self.assertRaises(ValueError):
            env.step(np.full((3, 2), 2))


if __name__ == "__main__":
    unittest.main()
