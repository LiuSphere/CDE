"""Exact log analysis and horizon-only parameter isolation."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import direct_main
from analyze_facmac_tuning import (
    evaluation_history, summarize_candidates, summarize_series,
)


class TuningTest(unittest.TestCase):
    def test_default_entry_uses_v03_and_preserves_explicit_config_selection(self):
        def load(arguments):
            configs = []
            with patch.object(sys, "argv", ["direct_main.py", *arguments]), \
                 patch.object(direct_main, "run", lambda config, logger: configs.append(config)):
                direct_main.main()
            return configs[0]

        default = load([])
        self.assertEqual(default, load(["--config", "facmac_ea_tune_v03_long"]))
        self.assertEqual(default["t_max"], 6000000)
        self.assertEqual(default["lr"], 3e-4)
        self.assertEqual(default["env"], "microgrid")
        self.assertEqual(
            load(["--config", "facmac_ea_tune_v04_actor_lr"])["lr"], 1e-4
        )
        self.assertEqual(load(["--override", "t_max=24"])["t_max"], 24)

    def test_recovery_only_accepts_verified_post_budget_showcase_path_failure(self):
        from recover_facmac_showcase import reporting_failure_step
        log = ("root EA cycle 10 at t_env=5997984\n"
               "showcase.publish_final(runner, mac, policy_store, runner.t_env)\n"
               "FileNotFoundError: [WinError 206] checkpoint_100_logged_6000144\n")
        self.assertEqual(reporting_failure_step(log, 6000000), 6000144)
        for bad in (log.replace("206", "5"), log.replace("6000144", "5000144"),
                    log.replace("showcase.publish_final", "learner.train")):
            with self.assertRaises(ValueError):
                reporting_failure_step(bad, 6000000)

    def test_exact_history_matches_validation_steps_not_recent_stat_averages(self):
        history = evaluation_history(
            "RL eval  1.0\nEA eval  2.0\nRL eval  3.5\nEA eval  -1.0\n",
            "[INFO 01:00:00] root t_env: 24 / 100\n"
            "eval/rl_reward: 999\n[INFO 01:00:01] root t_env: 96 / 100\n",
        )
        self.assertEqual(history, [
            {"step": 24, "rl": 1.0, "ea": 2.0},
            {"step": 96, "rl": 3.5, "ea": -1.0},
        ])
        summary = summarize_series(history, "rl", window_steps=50)
        self.assertEqual(summary["best_step"], 96)
        self.assertTrue(summary["best_in_last_10pct"])
        self.assertGreater(summary["tail_return_slope_per_million_steps"] or 0, -1)

    def test_incomplete_or_duplicate_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "counts"):
            evaluation_history("RL eval  1\n", "root t_env: 24 / 100\n")
        with self.assertRaisesRegex(ValueError, "increase"):
            evaluation_history(
                "RL eval  1\nEA eval  2\nRL eval  1\nEA eval  2\n",
                "root t_env: 24 / 100\nroot t_env: 24 / 100\n",
            )
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            evaluation_history("RL eval  1e999\nEA eval  2\n", "root t_env: 24 / 100\n")

    def test_economics_uses_all_days_and_ratio_of_totals(self):
        rows = [
            {"row_idx": 1, "baseline_cost": 100, "operating_cost": 50,
             "inventory_adjustment_cu": 10, "qualified": True,
             "max_terminal_soc_error": 0, "bidirectional_battery_fraction": 1,
             "cost_saving_ratio": 0.5},
            {"row_idx": 2, "baseline_cost": 900, "operating_cost": 810,
             "inventory_adjustment_cu": 20, "qualified": False,
             "max_terminal_soc_error": 0.15, "bidirectional_battery_fraction": 2 / 3,
             "cost_saving_ratio": 0.1},
        ]
        result = summarize_candidates({"checkpoint_step": 24, "candidates": rows})
        self.assertAlmostEqual(result["weighted_saving_fraction"], 0.14)
        self.assertAlmostEqual(result["inventory_adjusted_saving_fraction"], 0.11)
        self.assertEqual(result["qualified_count"], 1)
        self.assertEqual(result["mean_cost"], 430)
        with self.assertRaisesRegex(ValueError, "distinct"):
            summarize_candidates({"checkpoint_step": 24, "candidates": [rows[0], rows[0]]})

    def test_v03_changes_only_horizon_and_name_from_compact_v2(self):
        def load(name):
            configs = []
            with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
                 patch.object(direct_main, "run", lambda config, logger: configs.append(config)):
                direct_main.main()
            return configs[0]

        before = load("facmac_ea_compact_v2")
        expected = deepcopy(before)
        expected.update(name="facmac_ea_tune_v03_long", t_max=6000000)
        self.assertEqual(load("facmac_ea_tune_v03_long"), expected)
        expected.update(name="facmac_ea_tune_v04_actor_lr", lr=1e-4)
        self.assertEqual(load("facmac_ea_tune_v04_actor_lr"), expected)
        expected.update(name="facmac_ea_tune_v05_sparse_ea", frac=0.1)
        self.assertEqual(load("facmac_ea_tune_v05_sparse_ea"), expected)
        expected.update(name="facmac_ea_tune_v06_gentle_reset", frac=1.0,
                        prob_reset_and_sup=0.005)
        self.assertEqual(load("facmac_ea_tune_v06_gentle_reset"), expected)


if __name__ == "__main__":
    unittest.main()
