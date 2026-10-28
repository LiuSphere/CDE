"""Actual H12 slot dispatch contract; no experiment is launched by these tests."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.experiment_slot import validate_job, validate_worker_options


class H12LaunchTest(unittest.TestCase):
    def test_candidate_training_and_primary_manifest_worker_contract(self):
        validate_job("gavdac_c2_h12", 0, "h12guard", 0, 2)
        validate_worker_options("gavdac_c2_h12", 1, 2, 2)
        validate_worker_options("gavdac_c2_h12", 1, 0, 0, "newmanifest.json")
        with self.assertRaises(ValueError):
            validate_worker_options("gavdac_c2_h12", 1, 2, 2, "newmanifest.json")

    def test_historical_child_parent_and_bad_dispatch_are_rejected(self):
        for name in ("gavdac_c2_h04", "gavdac_c2_h11", "gavdac_c2_h11_parent"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_job(name, 0, "h12guard", 0, 2)
        for seed, token, slot, slots in ((3, "h12guard", 0, 2), (0, "../old", 0, 2),
                                         (0, "h12guard", 2, 2)):
            with self.assertRaises(ValueError):
                validate_job("gavdac_c2_h12", seed, token, slot, slots)


if __name__ == "__main__":
    unittest.main()
