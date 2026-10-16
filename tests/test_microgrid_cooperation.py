"""Physical cooperation, not just parameter labels or shared rewards."""

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


class CooperationTest(unittest.TestCase):
    def make_env(self, **overrides):
        config = yaml.safe_load(
            (ROOT / "src/config/envs/microgrid.yaml").read_text(encoding="utf-8"),
        )["env_args"]
        directory = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(directory.cleanup)
        config.update(result_dir=directory.name, seed=0, **overrides)
        return MGEnv(deepcopy(config), "cooperation_tests")

    def test_profiles_and_equipment_are_not_proportional_copies(self):
        env = self.make_env()
        load = env.agent_data["load"][740]
        shares = load / load.sum(axis=1, keepdims=True)
        self.assertGreater(np.ptp(shares[:, 0]), 0.1)
        self.assertFalse(np.allclose(env.params["load_share"], env.params["pv_share"]))
        duration = env.params["battery_caps"] / env.params["battery_discharge_max"]
        self.assertGreater(np.ptp(duration), 0.5)
        for key in ("load", "pv"):
            np.testing.assert_allclose(env.agent_data[key].sum(axis=2), env.data_dict[key])

    def test_one_agent_changes_another_agents_physical_grid_purchase(self):
        env = self.make_env()
        env.agent_data["load"][0, 0] = [120, 20, 50]
        env.agent_data["pv"][0, 0] = 0
        soc = np.full(3, 0.5)
        idle = env.dispatch(np.zeros((3, 2)), soc, (0, 0))
        actions = np.zeros((3, 2))
        actions[2, 0] = 1
        helping = env.dispatch(actions, soc, (0, 0))
        self.assertLess(helping["pg"][0], idle["pg"][0])
        self.assertGreater(helping["peer_export"][2], 0)
        self.assertEqual(helping["pg"][2], 0)
        np.testing.assert_allclose(
            helping["pv"] + helping["pd"] + helping["pb"] + helping["pg"]
            + helping["peer_import"],
            helping["load"] + helping["peer_export"] + helping["spill"], atol=1e-10,
        )
        self.assertAlmostEqual(
            helping["peer_import"].sum(),
            helping["peer_export"].sum() * env.exchange_efficiency,
        )

    def test_zero_links_recover_isolated_settlement(self):
        env = self.make_env()
        env.params["exchange_max"][:] = 0
        result = env.dispatch(np.tile([0.8, 0.2], (3, 1)), np.full(3, 0.5), (740, 12))
        np.testing.assert_array_equal(result["peer_export"], 0)
        np.testing.assert_array_equal(result["peer_import"], 0)
        np.testing.assert_allclose(
            result["pg"], np.maximum(result["net_grid_request"], 0),
        )
        np.testing.assert_array_equal(result["exchange_cost"], 0)

    def test_arbitrary_actions_keep_link_capacity_and_no_grid_reexport(self):
        env = self.make_env()
        rng = np.random.default_rng(37)
        for t in range(24):
            r = env.dispatch(rng.uniform([0, -1], [1, 1], (3, 2)),
                             rng.uniform(0.2, 0.8, 3), (740, t))
            self.assertTrue(np.all(r["peer_export"] <= env.params["exchange_max"] + 1e-9))
            self.assertTrue(np.all(r["peer_import"] <= env.params["exchange_max"] + 1e-9))
            np.testing.assert_allclose(r["peer_import"] * r["peer_export"], 0, atol=1e-9)
            np.testing.assert_allclose(r["pg"] * r["peer_export"], 0, atol=1e-9)
            np.testing.assert_allclose(r["spill"] * r["peer_import"], 0, atol=1e-9)
            self.assertAlmostEqual(
                r["peer_export"].sum() - r["peer_import"].sum(), r["exchange_loss"].sum(),
            )

    def test_broadcast_observations_change_with_peer_soc(self):
        env = self.make_env()
        before = env._observations()[0].copy()
        env.params["socs"][2] += 0.1
        after = env._observations()[0]
        np.testing.assert_array_equal(before[:4], after[:4])
        self.assertFalse(np.array_equal(before[4:], after[4:]))


if __name__ == "__main__":
    unittest.main()
