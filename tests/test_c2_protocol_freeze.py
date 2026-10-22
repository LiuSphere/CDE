"""Retained no-training guards for the archived H02 protocol-freeze tool."""

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.freeze_c2_protocol import (
    compare_host, entry_config, validate_candidate, validate_evaluation,
)


class ProtocolFreezeTest(unittest.TestCase):
    def test_actual_entry_config_only_enables_registered_intervention(self):
        parent = entry_config("gavdac_ea_main")
        # The failed opt-in config is retired; exercise the retained guard synthetically.
        candidate = {**parent, "name": "gavdac_c2_h02", "clipped_double_critic": True}
        validate_candidate(parent, candidate)
        for name, value in (("lr", .02), ("EA", False), ("t_max", 500000),
                            ("crossover_prob", 1.), ("checkpoint_path", "old_weights")):
            changed = {**candidate, name: value}
            with self.assertRaises(ValueError):
                validate_candidate(parent, changed)

    def test_fresh_host_rejects_old_token_occupied_slot_and_changed_bytes(self):
        previous = {"files": {"src/run.py": {"raw": "exact_original_bytes"}}, "platform": "frozen_os"}
        current = {
            "files": copy.deepcopy(previous["files"]), "platform": "frozen_os",
            "slots": [{"slot": 0, "state": "free"}], "dirty_scope": "", "existing_tokens": [],
        }
        compare_host(previous, current)
        for key, value in (
            ("existing_tokens", ["h02s0a"]), ("dirty_scope", " M src/run.py"),
            ("slots", [{"slot": 0, "state": "stale_lock"}]),
            ("files", {"src/run.py": {"raw": "changed"}}), ("platform", "new_os"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare_host(previous, {**current, key: value})

    def test_per_seed_entry_export_keeps_frozen_budget_and_seeds_environment(self):
        config = entry_config("gavdac_ea_main", {
            "seed": 2, "unique_token": "h02_export_test", "wandb_mode": "online"})
        self.assertEqual(config["seed"], 2)
        self.assertEqual(config["env_args"]["seed"], 2)
        self.assertEqual(config["t_max"], 6000000)
        self.assertEqual(config["unique_token"], "h02_export_test")
        self.assertFalse(config["checkpoint_path"])
        self.assertFalse(config["policy_checkpoint"])

    def test_evaluation_cannot_relax_threshold_or_change_inventory_or_days(self):
        evaluation = {
            "rows": list(range(913, 1095)), "mode": "continuous",
            "actor": "validation-selected rl", "initial_soc": .35,
            "terminal_soc": "free within device bounds", "minimum_saving_fraction": .03,
        }
        validate_evaluation(evaluation)
        for key, value in (("minimum_saving_fraction", .01), ("mode", "daily_reset"),
                           ("rows", [913]), ("initial_soc", .5), ("actor", "ea")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_evaluation({**evaluation, key: value})


if __name__ == "__main__":
    unittest.main()
