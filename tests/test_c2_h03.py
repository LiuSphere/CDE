"""H03 cadence isolation and synchronous shard lifecycle checks."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from learners.delayed_actor_learner import DelayedActorGAVDACLearner
from learners.gavdac_learner import GAVDACLearner
from learners import gradient_pool
from test_gavdac_sequence import LossLog, fixture
from direct_main import load_yaml, recursive_dict_update
from tools.experiment_slot import validate_job, validate_worker_options


class H03Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "1"

    def assert_models(self, first, second, exact=False):
        tolerance = dict(rtol=0, atol=0) if exact else dict(rtol=1e-4, atol=1e-5)
        for name in ("critic", "mixer", "target_critic", "target_mixer",
                     "agent_optimiser", "critic_optimiser"):
            torch.testing.assert_close(
                getattr(first, name).state_dict(), getattr(second, name).state_dict(),
                **tolerance,
            )
        for name in ("mac", "target_mac"):
            torch.testing.assert_close(
                getattr(first, name).agent.state_dict(),
                getattr(second, name).agent.state_dict(), **tolerance,
            )

    def test_missing_or_one_interval_is_baseline_full_update(self):
        for explicit in (False, True):
            batch, mac, args, scheme = fixture(use_cuda=False)
            if explicit:
                args.actor_update_interval = 1
            initial_rng = torch.get_rng_state()
            baseline = GAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
            torch.set_rng_state(initial_rng)
            candidate = DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
            for step in range(3):
                rng = torch.get_rng_state()
                baseline.train(batch, step + 1, step)
                expected_rng = torch.get_rng_state()
                torch.set_rng_state(rng)
                candidate.train(batch, step + 1, step)
                self.assert_models(baseline, candidate, exact=True)
                torch.testing.assert_close(expected_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_cadence_and_targets_across_three_call_bursts(self):
        batch, mac, args, scheme = fixture(use_cuda=False, actor_update_interval=2)
        learner = DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
        with patch.object(learner, "_update_targets_soft",
                          wraps=learner._update_targets_soft) as targets:
            for step in range(1, 8):
                actor = copy.deepcopy(mac.agent.state_dict())
                optimizer = copy.deepcopy(learner.agent_optimiser.state_dict())
                before_target = copy.deepcopy(learner.target_critic.state_dict())
                learner.train(batch, step, step // 3)
                if step % 2:
                    torch.testing.assert_close(actor, mac.agent.state_dict(), rtol=0, atol=0)
                    torch.testing.assert_close(
                        optimizer, learner.agent_optimiser.state_dict(), rtol=0, atol=0,
                    )
                expected = {key: value.lerp(
                    learner.critic.state_dict()[key], args.target_update_tau,
                ) for key, value in before_target.items()}
                torch.testing.assert_close(expected, learner.target_critic.state_dict())
                self.assertEqual(learner.actor_steps, step // 2)
                self.assertEqual(learner.critic_steps, step)
                self.assertEqual(learner.target_updates, step)
                self.assertEqual(learner.finite_checks, step)
            self.assertEqual(targets.call_count, 7)
        self.assertTrue(all(param.grad is None for param in learner.target_mac.parameters()))

    def test_parallel_skips_release_graphs_with_changing_participation(self):
        batch, mac, args, scheme = fixture(
            use_cuda=False, batch_size=5, length=7, actor_update_interval=2,
        )
        serial = DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
        parallel = copy.deepcopy(serial)
        parallel.args.learner_workers = 2
        try:
            for step, size in enumerate((5, 1, 1, 5, 3, 2), start=1):
                batch, _, _, _ = fixture(
                    use_cuda=False, batch_size=size, length=7,
                )
                serial.train(batch, step, step)
                parallel.train(batch, step, step)
                self.assert_models(serial, parallel)
                torch.testing.assert_close(
                    [param.grad for param in serial.agent_params + serial.critic_params],
                    [param.grad for param in parallel.agent_params + parallel.critic_params],
                    rtol=1e-4, atol=1e-5,
                )
                self.assertIsNone(parallel.gradient_pool.pending_batch)
                self.assertEqual(parallel.actor_steps, step // 2)
            with self.assertRaisesRegex(RuntimeError, "same critic batch"):
                parallel.gradient_pool.discard_actor(batch)
        finally:
            serial.close()
            parallel.close()

    def test_invalid_interval_and_nonfinite_fail_before_update(self):
        for value in (True, False, 0, -1, 2.5, "2"):
            batch, mac, args, scheme = fixture(actor_update_interval=value)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
        batch, mac, args, scheme = fixture(actor_update_interval=2)
        learner = DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
        original = copy.deepcopy(learner.critic.state_dict())
        batch["reward"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            learner.train(batch, 1, 1)
        torch.testing.assert_close(original, learner.critic.state_dict(), rtol=0, atol=0)
        self.assertEqual(learner.critic_steps, 0)

    def test_worker_discard_requires_matching_pending_round(self):
        _, mac, args, scheme = fixture(actor_update_interval=2)
        learner = DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
        mac.hidden_states = torch.ones(1, requires_grad=True) * 2
        with patch.object(gradient_pool, "_pending", None), \
                patch.object(gradient_pool, "_learner", learner):
            with self.assertRaisesRegex(RuntimeError, "matching cached"):
                gradient_pool._discard_actor(1)
            gradient_pool._pending = (2, object())
            with self.assertRaisesRegex(RuntimeError, "matching cached"):
                gradient_pool._discard_actor(1)
            gradient_pool._discard_actor(2)
            self.assertIsNone(gradient_pool._pending)
            self.assertIsNone(mac.hidden_states.grad_fn)

    def test_even_actor_retains_action_context_gradients_and_cannot_change_critic(self):
        batch, mac, args, scheme = fixture(actor_update_interval=2)
        learner = DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
        captures = {}
        original = learner._actor_loss

        def actor_loss(batch, policy, inputs):
            for name in ("actions", "hidden_states"):
                policy[name].retain_grad()
                captures[name] = policy[name]
            captures["critic"] = copy.deepcopy(learner.critic.state_dict())
            captures["mixer"] = copy.deepcopy(learner.mixer.state_dict())
            return original(batch, policy, inputs)

        with patch.object(learner, "_actor_loss", side_effect=actor_loss) as loss:
            learner.train(batch, 1, 1)
            self.assertEqual(loss.call_count, 0)
            self.assertIsNone(mac.hidden_states.grad_fn)
            learner.train(batch, 2, 1)
            self.assertEqual(loss.call_count, 1)
        for name in ("actions", "hidden_states"):
            gradient = captures[name].grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
        for name in ("critic", "mixer"):
            torch.testing.assert_close(
                captures[name], getattr(learner, name).state_dict(), rtol=0, atol=0,
            )

    def test_config_difference_and_diagnostic_artifact(self):
        validate_job("gavdac_c2_h03", 2, "h03guard", 0, 1)
        validate_worker_options("gavdac_c2_h03", 1, 2, 2)
        def merged(name):
            config = load_yaml(ROOT / "src/config/default.yaml")
            recursive_dict_update(config, load_yaml(ROOT / "src/config/envs/microgrid.yaml"))
            recursive_dict_update(config, load_yaml(ROOT / f"src/config/algs/{name}.yaml"))
            return config
        baseline, candidate = merged("gavdac_ea_main"), merged("gavdac_c2_h03")
        self.assertEqual(
            {key for key in baseline.keys() | candidate.keys()
             if baseline.get(key) != candidate.get(key)},
            {"name", "actor_update_interval"},
        )
        batch, mac, args, scheme = fixture(actor_update_interval=2)
        learner = DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
        for step in range(3):
            learner.train(batch, step + 1, step)
        with tempfile.TemporaryDirectory() as directory:
            learner.save_models(directory)
            payload = json.loads((Path(directory) / "h03_diagnostics.json").read_text())
            self.assertEqual(payload["critic_optimizer_steps"], 3)
            self.assertEqual(payload["actor_optimizer_steps"], 1)
            self.assertEqual(payload["finite_update_checks"], 3)
            self.assertFalse(payload["exact_resume_supported"])


if __name__ == "__main__":
    unittest.main()
