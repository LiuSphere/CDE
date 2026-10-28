"""H13's new dispatch token does not launch an experiment in these checks."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.experiment_slot import validate_job, validate_worker_options
from tools.freeze_h11_protocol import entry_config


class H13LaunchTest(unittest.TestCase):
    def test_entry_config_has_only_name_and_norm_delta_from_actual_H12(self):
        overrides = {"rollout_workers": 2, "learner_workers": 2}
        parent = entry_config("gavdac_c2_h12", overrides)
        child = entry_config("gavdac_c2_h13", overrides)
        self.assertEqual(child, {**parent, "name": "gavdac_c2_h13",
                                 "actor_feature_layer_norm": True})

    def test_fresh_training_and_primary_without_training_workers(self):
        validate_job("gavdac_c2_h13", 0, "h13guard", 0, 2)
        validate_worker_options("gavdac_c2_h13", 1, 2, 2)
        validate_worker_options("gavdac_c2_h13", 1, 0, 0, "newmanifest.json")
        for options in ((1, 2, 2, "manifest.json"), (2, 2, 2, None),
                        (1, True, 2, None)):
            with self.assertRaises(ValueError):
                validate_worker_options("gavdac_c2_h13", *options)


if __name__ == "__main__":
    unittest.main()
