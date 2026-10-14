import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from envs.maenv import MGEnv


class BatteryTest(unittest.TestCase):
    def make_env(self, dt=1.0, charge=200.0, discharge=200.0):
        config = yaml.safe_load((ROOT / "src/config/envs/microgrid.yaml").read_text(encoding="utf-8"))
        config = deepcopy(config["env_args"])
        config.update(result_dir=str(ROOT / "results"), run_id="battery_tests",
                      seed=0, time_step_hours=dt)
        for agent in config["data"]:
            agent.update(battery_charge_max=charge, battery_discharge_max=discharge)
        env = MGEnv(config, "tests")
        env.reset()
        return env

    def test_discharge_efficiency(self):
        env = self.make_env()
        env.params["socs"][:] = 0.5
        env.step(np.tile([0, 0.1], (3, 1)))
        np.testing.assert_allclose(env.params["socs"], 0.5 - 20 / (0.95 * 200))

    def test_low_soc_projection_conserves_energy(self):
        env = self.make_env()
        env.params["socs"][:] = 0.25
        env.step(np.tile([0, 1], (3, 1)))
        for row in env.ep_buffer:
            self.assertAlmostEqual(row["pb"], 9.5)
            self.assertAlmostEqual(row["soc"], 0.2)

    def test_high_soc_projection_conserves_energy(self):
        env = self.make_env()
        env.params["socs"][:] = 0.75
        env.step(np.tile([0, -1], (3, 1)))
        for row in env.ep_buffer:
            self.assertAlmostEqual(row["pb"], -10 / 0.95)
            self.assertAlmostEqual(row["soc"], 0.8)

    def test_time_step_and_separate_power_ratings(self):
        env = self.make_env(dt=0.25, charge=40, discharge=20)
        env.params["socs"][:] = 0.5
        env.step(np.tile([0, 1], (3, 1)))
        self.assertAlmostEqual(env.ep_buffer[0]["pb"], 20)
        self.assertAlmostEqual(env.params["socs"][0], 0.5 - 20 * 0.25 / (0.95 * 200))

    def test_repeated_random_dispatch_conserves_energy(self):
        env = self.make_env(dt=0.25, charge=80, discharge=40)
        rng = np.random.default_rng(17)
        for _ in range(100):
            before = env.params["socs"].copy()
            env.step(rng.uniform([-0.0, -1.0], [1.0, 1.0], size=(3, 2)))
            power = np.array([r["pb"] for r in env.ep_buffer[-3:]])
            delta = np.where(power < 0, -0.95 * power, -power / 0.95) * 0.25 / 200
            np.testing.assert_allclose(env.params["socs"] - before, delta, atol=1e-14)
            self.assertTrue(np.all((env.params["socs"] >= 0.2) & (env.params["socs"] <= 0.8)))


if __name__ == "__main__":
    unittest.main()
