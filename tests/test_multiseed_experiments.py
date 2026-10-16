"""Statistical units and endpoint checks for the paired environment experiment."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_microgrid_multiseed import compare_runs, final_policies


def make_run(seed, version, algorithm):
    evaluations = []
    for policy in ("rl", "ea") if algorithm == "facmac_ea" else ("rl",):
        cost = (100 if policy == "rl" else 80) if version == "v2" else (
            90 if policy == "rl" else 84
        )
        evaluations.append({
            "phase": "final", "policy": policy, "step": 20016,
            "rows": [{"row_idx": row, "full_cost": cost + 5,
                      "battery_punishment_reward": -5, "terminal_soc": [0.35] * 3,
                      "battery_charge_energy": 20, "battery_discharge_energy": 10}
                     for row in (740, 766)],
            "mean_operating_cost": cost,
        })
    return {
        "seed": seed, "model_version": ("heterogeneous_exchange_v2" if version == "v2"
                                      else "facmac_dispatch_v3"),
        "algorithm": algorithm, "actual_steps": 20016, "requested_steps": 20000,
        "validation_rows": [740, 766], "cpu_threads": 1, "rng_protocol": "test",
        "deterministic_algorithms": True, "pre_rollout_reseed": True,
        "history": evaluations,
    }


def runs():
    return [make_run(seed, version, algorithm) for seed in range(5)
            for version in ("v2", "v3") for algorithm in ("facmac_ea", "facmac_no_ea")]


class MultiSeedTest(unittest.TestCase):
    def test_paired_costs_exclude_penalties_and_keep_policies_separate(self):
        summary = compare_runs(runs())
        for category in ("facmac_only", "actor_with_ea"):
            stats = summary["aggregate"][category]
            self.assertAlmostEqual(stats["mean_paired_improvement_pct"], 10)
            self.assertEqual(stats["improved_seeds"], 5)
            self.assertEqual(stats["seed_count"], 5)
        self.assertAlmostEqual(summary["aggregate"]["ea_elite"]["mean_paired_improvement_pct"], -5)
        self.assertEqual(summary["aggregate"]["ea_elite"]["improved_seeds"], 0)
        self.assertAlmostEqual(summary["aggregate"]["best_with_ea"]["mean_paired_improvement_pct"], -5)

    def test_incomplete_duplicate_and_unseeded_runs_are_rejected(self):
        data = runs()
        with self.assertRaises(ValueError):
            compare_runs(data[:-1])
        with self.assertRaises(ValueError):
            compare_runs(data + [data[0]])
        changed = deepcopy(data)
        changed[0]["pre_rollout_reseed"] = False
        with self.assertRaises(ValueError):
            compare_runs(changed)

    def test_budgets_and_validation_dates_must_match(self):
        for field, value in (("actual_steps", 20160), ("requested_steps", 30000),
                             ("validation_rows", [740])):
            data = runs()
            data[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_runs(data)

    def test_final_endpoint_and_cost_decomposition_are_checked(self):
        run = make_run(0, "v2", "facmac_ea")
        run["history"][0]["phase"] = "scheduled"
        with self.assertRaises(ValueError):
            final_policies(run)
        run = make_run(0, "v2", "facmac_ea")
        run["history"][0]["mean_operating_cost"] = 999
        with self.assertRaises(ValueError):
            final_policies(run)


if __name__ == "__main__":
    unittest.main()
