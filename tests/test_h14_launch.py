"""H14 direct-entry and supervisor contracts; no formal job is launched."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.experiment_slot import validate_job, validate_worker_options
from tools.freeze_h11_protocol import entry_config
from learners.signed_mixer_huber_learner import signed_mixer_enabled
from run import args_sanity_check


class H14LaunchTest(unittest.TestCase):
    def test_actual_entry_only_adds_name_and_true_flag(self):
        parent, child = [entry_config(n, {"rollout_workers": 2, "learner_workers": 2})
                         for n in ("gavdac_c2_h12", "gavdac_c2_h14")]
        self.assertEqual(child, {**parent, "name": "gavdac_c2_h14", "signed_mixer_hyperweights": True})
        self.assertEqual(args_sanity_check(copy.deepcopy(child), SimpleNamespace(warning=lambda *_: None)), child)
        self.assertTrue(signed_mixer_enabled(SimpleNamespace(**child)))

    def test_training_and_evaluation_options_remain_separate(self):
        validate_job("gavdac_c2_h14", 0, "h14guard", 0, 2)
        validate_worker_options("gavdac_c2_h14", 1, 2, 2)
        validate_worker_options("gavdac_c2_h14", 1, 0, 0, "newmanifest.json")
        for threads, rollout, learner, manifest in (
            (1, 2, 2, "newmanifest.json"), (2, 2, 2, None),
            (1, True, 2, None), (1, 2, 2., None), (1, -1, 2, None)):
            with self.subTest(options=(threads, rollout, learner)), self.assertRaises(ValueError):
                validate_worker_options("gavdac_c2_h14", threads, rollout, learner, manifest)

    def test_retired_or_invalid_slot_dispatch_rejected(self):
        for name in ("gavdac_c2_h11", "gavdac_c2_h13", "gavdac_c2_h11_parent"):
            with self.subTest(config=name), self.assertRaises(ValueError):
                validate_job(name, 0, "h14guard", 0, 2)
        for seed, token, slot, slots in ((3, "h14guard", 0, 2), (0, "../old", 0, 2),
                                         (0, "h14guard", 2, 2)):
            with self.assertRaises(ValueError):
                validate_job("gavdac_c2_h14", seed, token, slot, slots)


if __name__ == "__main__":
    unittest.main()
