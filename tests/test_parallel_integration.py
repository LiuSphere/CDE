"""Main-branch compatibility gates for the optional execution pools."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from learners.clipped_double_gavdac_learner import ClippedDoubleGAVDACLearner
from learners.gavdac_learner import GAVDACLearner
from test_gavdac_sequence import LossLog, fixture
from tools.experiment_slot import validate_worker_options


class ParallelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_h02_rejects_single_critic_sharding_at_construction_and_dispatch(self):
        _, mac, args, scheme = fixture(
            clipped_double_critic=True, learner_workers=2, use_cuda=False,
        )
        with self.assertRaisesRegex(ValueError, "H02 learner sharding"):
            ClippedDoubleGAVDACLearner(mac, scheme, LossLog(), args)
        with self.assertRaisesRegex(ValueError, "H02 learner sharding"):
            validate_worker_options("gavdac_c2_h02", 1, 0, 2)
        validate_worker_options("gavdac_c2_h02", 1, 0, 0)
        validate_worker_options("gavdac_c2_h02", 1, 2, 0)

    def test_baseline_keeps_main_actor_context_gradient(self):
        batch, mac, args, scheme = fixture(use_cuda=False)
        learner = GAVDACLearner(mac, scheme, LossLog(), args)
        contexts = []
        hook = learner.mixer.register_forward_pre_hook(
            lambda module, items, kwargs: contexts.append(
                kwargs["hidden_states"].requires_grad,
            ), with_kwargs=True,
        )
        try:
            learner.train(batch, 0, 0)
            self.assertEqual(contexts, [False, True])
            self.assertIsNone(learner.gradient_pool)
        finally:
            hook.remove()
            learner.close()


if __name__ == "__main__":
    unittest.main()
