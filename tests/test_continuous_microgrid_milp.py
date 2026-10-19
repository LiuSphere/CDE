import sys
import json
import io
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tools.run_continuous_microgrid_milp import main, solve_continuous
from tools.run_physical_microgrid_milp import make_env, solve_network


class ContinuousMilpTest(unittest.TestCase):
    def test_entry_writes_complete_serializable_evidence(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            output = Path(directory) / "pilot"
            with patch.object(sys, "argv", ["pilot", "--rows", "731",
                                           "--output", str(output)]), redirect_stdout(io.StringIO()):
                main()
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["acceptance_passed"])
            self.assertGreater(summary["process_peak_working_set_bytes"], 0)
            self.assertIn("tools/run_continuous_microgrid_milp.py", summary["source_sha256"])
            self.assertEqual(len((output / "trajectories.csv").read_text().splitlines()), 73)
            self.assertEqual(json.loads((output / "solver.json").read_text())["status"], 0)

    def test_joint_validation_replay_and_cost_certificate(self):
        config = yaml.safe_load((ROOT / "src/config/envs/microgrid.yaml").read_text(
            encoding="utf-8"))["env_args"]
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            env = make_env(config, Path(directory))
            summary, solution, records = solve_continuous(env, config, [731, 732])
            self.assertTrue(summary["acceptance_passed"])
            json.dumps(summary, allow_nan=False)
            json.dumps(solution, allow_nan=False)
            self.assertEqual(solution["objective_name"], "economic_cost")
            self.assertEqual(len(records), 2 * 24 * env.n_agents)
            np.testing.assert_allclose(summary["days"][0]["terminal_soc"],
                                       summary["days"][1]["initial_soc"], atol=1e-10)
            self.assertTrue(summary["physical_checks_passed"])
            self.assertGreaterEqual(summary["certified_economic_suboptimality_upper"], 0)
            with self.assertRaises(ValueError):
                solve_continuous(env, config, [731, 733])

    def test_objective_does_not_include_emissions_when_economic(self):
        p = dict(battery_cap=100, battery_charge_max=25, battery_discharge_max=25,
                 raw_ch=.93, raw_dis=.93, dg_max=30, costa=0, costb=.14, costc=0,
                 cycle_cost=.01, env_param=.2)
        inputs = (np.full((1, 1), 80), np.zeros((1, 1)), np.array([.2]),
                  [p], [0], 1, 0, .35)
        economic = solve_network(*inputs, battery_enabled=False, objective="economic_cost")
        full = solve_network(*inputs, battery_enabled=False)
        self.assertAlmostEqual(economic["schedules"][0]["pd"][0], 30)
        self.assertAlmostEqual(full["schedules"][0]["pd"][0], 0)
        with self.assertRaises(ValueError):
            solve_network(*inputs, objective="reward")


if __name__ == "__main__":
    unittest.main()
