import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sb3_microgrid_env import ReferenceMicrogridEnv, snapshot_reference


class SB3ReferenceEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir=ROOT / "results")
        cls.bundle = Path(cls.directory.name) / "reference"
        cls.source = Path("C:/Users/xyx/Desktop/MG/MG/maenv_sep.py")
        config = Path("C:/Users/xyx/Desktop/IPPO4CPT/envs/MG/env_config.json")
        snapshot_reference(cls.source, config, cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_unchanged_source_and_gymnasium_contract(self):
        self.assertEqual(
            hashlib.sha256(self.source.read_bytes()).hexdigest(),
            hashlib.sha256((self.bundle / "maenv_sep.py").read_bytes()).hexdigest(),
        )
        env = ReferenceMicrogridEnv(self.bundle)
        check_env(env, warn=True)
        self.assertEqual(env.observation_space.shape, (12,))
        self.assertEqual(env.action_space.shape, (6,))
        self.assertTrue(json.loads((self.bundle / "env_config.json").read_text())["multi_target"] == "True")
        self.assertTrue(all(not c.multi_target for c in env.cores))

    def test_native_records_and_single_objective_reward(self):
        env = ReferenceMicrogridEnv(self.bundle, record=True)
        obs, _ = env.reset(options={"row_idx": 913, "initial_soc": 0.35})
        self.assertTrue(env.observation_space.contains(obs))
        for _ in range(24):
            obs, reward, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            latest = env.records[-3:]
            expected = []
            for row in latest:
                raw = (
                    -row["economic_cost"] + row["env_reward"] + row["punishment"]
                ) * 0.1
                self.assertAlmostEqual(row["reward"], raw, delta=1e-4)
                expected.append(np.float32(row["reward"]))
            self.assertEqual(reward, float(np.mean(expected)))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(len(env.records), 72)
        self.assertEqual({r["row_idx"] for r in env.records}, {913})
        self.assertEqual({r["col_idx"] for r in env.records}, set(range(24)))

    def test_training_stream_preserves_native_soc_and_next_day(self):
        env = ReferenceMicrogridEnv(self.bundle)
        env.reset(seed=0)
        for _ in range(24):
            env.step(np.array([-1, -0.01] * 3, dtype=np.float32))
        final = [core._MyEnv__soc for core in env.cores]
        obs, _ = env.reset()
        np.testing.assert_allclose(obs.reshape(3, 4)[:, 3], final)
        self.assertEqual([core._MyEnv__node for core in env.cores], [[1, 0]] * 3)

    def test_symmetric_actions_map_to_original_bounds(self):
        env = ReferenceMicrogridEnv(self.bundle)
        np.testing.assert_array_equal(
            env.scale_action(np.full(6, -1, dtype=np.float32)), env.action_lows,
        )
        np.testing.assert_array_equal(
            env.scale_action(np.full(6, 1, dtype=np.float32)), env.action_highs,
        )


if __name__ == "__main__":
    unittest.main()
