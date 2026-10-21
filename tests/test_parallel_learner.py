"""Float32 equivalence of synchronous batch-sharded critic/actor updates."""

import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from learners.gavdac_learner import GAVDACLearner
from learners import gradient_pool
from test_gavdac_sequence import LossLog, fixture


class ParallelLearnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"

    def test_complete_uneven_updates_gradients_optimizers_and_targets(self):
        for agents, optimizer, context in (
            (3, "adam", True), (3, "rmsprop", False), (1, "adam", True),
        ):
            with self.subTest(agents=agents, optimizer=optimizer, context=context):
                batch, mac, args, scheme = fixture(
                    n_agents=agents, batch_size=5, length=7, optimizer=optimizer,
                    actor_mixer_context_grad=context, use_cuda=False,
                )
                # Empty shards must not produce NaN or dilute the global critic mask.
                batch["filled"][:2] = 0
                serial = GAVDACLearner(mac, scheme, LossLog(), args)
                parallel = copy.deepcopy(serial)
                parallel.args.learner_workers = 3
                try:
                    for step in range(4):
                        serial.train(batch, step + 1, step)
                        parallel.train(batch, step + 1, step)
                        self.assertIsNone(parallel.gradient_pool.pending_batch)
                        self.assertEqual(parallel.gradient_pool.last_phase_pids["critic"],
                                         parallel.gradient_pool.last_phase_pids["actor"])
                        for name in ("critic", "mixer", "target_critic", "target_mixer",
                                     "agent_optimiser", "critic_optimiser"):
                            if not hasattr(serial, name) or getattr(serial, name) is None:
                                continue
                            torch.testing.assert_close(
                                getattr(serial, name).state_dict(),
                                getattr(parallel, name).state_dict(),
                                rtol=1e-4, atol=1e-5,
                            )
                        torch.testing.assert_close(
                            serial.mac.agent.state_dict(), parallel.mac.agent.state_dict(),
                            rtol=1e-4, atol=1e-5,
                        )
                        torch.testing.assert_close(
                            serial.target_mac.agent.state_dict(),
                            parallel.target_mac.agent.state_dict(), rtol=1e-4, atol=1e-5,
                        )
                        torch.testing.assert_close(
                            [p.grad for p in serial.agent_params + serial.critic_params],
                            [p.grad for p in parallel.agent_params + parallel.critic_params],
                            rtol=1e-4, atol=1e-5,
                        )
                        for key, value in serial.logger.values.items():
                            self.assertAlmostEqual(value, parallel.logger.values[key], places=5)
                        self.assertTrue(all(
                            param.dtype == torch.float32 and param.device.type == "cpu"
                            for param in parallel.agent_params + parallel.critic_params
                        ))
                        self.assertTrue(all(
                            param.grad is None for param in parallel.target_mac.parameters()
                        ))
                    self.assertGreater(len(parallel.gradient_pool.worker_pids), 1)
                finally:
                    serial.close()
                    parallel.close()
                self.assertIsNone(parallel.gradient_pool)

    def test_zero_valid_mask_raises_before_optimizer_changes(self):
        batch, mac, args, scheme = fixture(use_cuda=False, learner_workers=2)
        learner = GAVDACLearner(mac, scheme, LossLog(), args)
        before = copy.deepcopy(learner.mac.agent.state_dict())
        batch["filled"].zero_()
        try:
            with self.assertRaisesRegex(ValueError, "no valid transitions"):
                learner.train(batch, 1, 1)
            torch.testing.assert_close(before, learner.mac.agent.state_dict(), rtol=0, atol=0)
        finally:
            learner.close()

    def test_cached_worker_actor_uses_one_history_and_rejects_wrong_phase(self):
        batch, mac, args, scheme = fixture(use_cuda=False)
        learner = GAVDACLearner(mac, scheme, LossLog(), args)
        weights = {
            name: {key: value.numpy().copy() for key, value in module.state_dict().items()}
            for name, module in gradient_pool._modules(learner).items()
        }
        data = {key: value.numpy().copy()
                for key, value in batch.data.transition_data.items()}
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] *= 1 - batch["terminated"][:, :-2].float()
        actor_weights = {key: weights[key] for key in ("critic", "mixer")}
        with patch.multiple(gradient_pool, _learner=learner, _scheme=scheme,
                            _groups=batch.groups, _pending=None), \
             patch.object(mac, "forward_sequence", wraps=mac.forward_sequence) as forward:
            with self.assertRaisesRegex(RuntimeError, "matching cached"):
                gradient_pool._gradients("actor", actor_weights, None, None, None, 1)
            gradient_pool._gradients("critic", weights, data, float(mask.sum()), 1, 1)
            with self.assertRaisesRegex(RuntimeError, "still pending"):
                gradient_pool._gradients("critic", weights, data, float(mask.sum()), 1, 2)
            with self.assertRaisesRegex(RuntimeError, "matching cached"):
                gradient_pool._gradients("actor", actor_weights, None, None, None, 2)
            gradient_pool._gradients("actor", actor_weights, None, None, None, 1)
            self.assertEqual(forward.call_count, 1)
            self.assertIsNone(gradient_pool._pending)


if __name__ == "__main__":
    unittest.main()
