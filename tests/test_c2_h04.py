"""H04 loss isolation, baseline equivalence, sharded updates and saved source."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from learners.gavdac_learner import GAVDACLearner
from learners.huber_learner import HuberGAVDACLearner, huber_enabled
from test_gavdac_sequence import LossLog, fixture
from test_policy_evaluation import setup_runner
from tools.freeze_h03_protocol import entry_config
from tools.experiment_slot import validate_job
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed


class H04Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"

    def assert_updates_equal(self, first, second, exact=False):
        tolerance = dict(rtol=0, atol=0) if exact else dict(rtol=1e-4, atol=1e-5)
        for name in ("critic", "mixer", "target_critic", "target_mixer",
                     "agent_optimiser", "critic_optimiser"):
            if getattr(first, name, None) is not None:
                torch.testing.assert_close(getattr(first, name).state_dict(),
                                           getattr(second, name).state_dict(), **tolerance)
        for name in ("mac", "target_mac"):
            torch.testing.assert_close(getattr(first, name).agent.state_dict(),
                                       getattr(second, name).agent.state_dict(), **tolerance)

    def test_analytic_loss_derivative_and_small_error_scale(self):
        _, mac, args, scheme = fixture(use_cuda=False, scaled_huber_critic=True)
        learner = HuberGAVDACLearner(mac, scheme, LossLog(), args)
        errors = torch.tensor([-100., -1., -.5, 0., .5, 1., 100.], requires_grad=True)
        loss = learner._td_error_loss(errors, 1)
        torch.testing.assert_close(loss, torch.tensor(400.5), rtol=0, atol=0)
        loss.backward()
        torch.testing.assert_close(errors.grad, torch.tensor([-2., -2., -1., 0., 1., 2., 2.]), rtol=0, atol=0)
        small = torch.tensor([-.5, -.125, 0., .125, .5], requires_grad=True)
        torch.testing.assert_close(learner._td_error_loss(small, 5), small.square().sum()/5, rtol=0, atol=0)
        learner.close()

    def test_missing_false_matches_baseline_full_updates_and_rng(self):
        for explicit in (False, True):
            batch, mac, args, scheme = fixture(use_cuda=False)
            if explicit:
                args.scaled_huber_critic = False
            state = torch.get_rng_state()
            baseline = GAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
            torch.set_rng_state(state)
            candidate = HuberGAVDACLearner(mac, scheme, LossLog(), args)
            for step in range(3):
                rng = torch.get_rng_state()
                baseline.train(batch, step + 1, step)
                after = torch.get_rng_state()
                torch.set_rng_state(rng)
                candidate.train(batch, step + 1, step)
                self.assert_updates_equal(baseline, candidate, exact=True)
                torch.testing.assert_close(after, torch.get_rng_state(), rtol=0, atol=0)
            baseline.close()
            candidate.close()

    def test_original_target_termination_padding_and_global_denominator(self):
        batch, mac, args, scheme = fixture(n_agents=1, batch_size=1, length=5,
                                          gamma=.5, use_cuda=False, scaled_huber_critic=True)
        learner = HuberGAVDACLearner(mac, scheme, LossLog(), args)
        with torch.no_grad():
            for module in (learner.critic, learner.target_critic):
                for param in module.parameters():
                    param.zero_()
            learner.target_critic.fc3.bias.fill_(2.)
        batch["reward"][0, :, 0] = torch.tensor([.25, 2., 100., 100., 100.])
        batch["terminated"].zero_()
        batch["terminated"][0, 1] = 1
        batch["filled"][:, 2:] = 0
        captured = []
        original = learner._td_error_loss
        with patch.object(learner, "_td_error_loss",
                          side_effect=lambda e, d: (captured.append((e.detach().clone(), d)), original(e, d))[1]):
            _, _, loss = learner._critic_loss(batch)
        torch.testing.assert_close(captured[0][0].flatten(), torch.tensor([1.25, 2., 0., 0.]))
        self.assertEqual(float(captured[0][1]), 2.)
        torch.testing.assert_close(loss, torch.tensor(2.25))
        _, _, global_loss = learner._critic_loss(batch, denominator=4.)
        torch.testing.assert_close(global_loss, loss / 2)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in learner.target_critic.parameters()))
        learner.close()

    def test_actor_live_context_critic_immutability_and_every_step_targets(self):
        batch, mac, args, scheme = fixture(use_cuda=False, scaled_huber_critic=True)
        learner = HuberGAVDACLearner(mac, scheme, LossLog(), args)
        contexts, actions, critic_at_step = [], [], []
        def context_hook(module, inputs, keywords):
            hidden = keywords["hidden_states"]
            if hidden.requires_grad:
                hidden.retain_grad()
                contexts.append(hidden)
        hook = learner.mixer.register_forward_pre_hook(context_hook, with_kwargs=True)
        original_actor_loss = learner._actor_loss
        def actor_loss(batch, policy, inputs):
            policy["actions"].retain_grad()
            actions.append(policy["actions"])
            return original_actor_loss(batch, policy, inputs)
        def record_critic(optimizer, arguments, keywords):
            critic_at_step.append(copy.deepcopy(learner.critic.state_dict()))
        spy = learner.critic_optimiser.register_step_post_hook(record_critic)
        try:
            with patch.object(learner, "_actor_loss", side_effect=actor_loss):
                for step in range(2):
                    targets = [p.detach().clone() for p in learner.target_mac.parameters()]
                    learner.train(batch, step + 1, step)
                    torch.testing.assert_close(critic_at_step[-1], learner.critic.state_dict(), rtol=0, atol=0)
                    for old, live, target in zip(targets, learner.mac.parameters(), learner.target_mac.parameters()):
                        torch.testing.assert_close(target, old.lerp(live, args.target_update_tau), rtol=0, atol=0)
            self.assertTrue(all(v.grad is not None and v.grad.abs().sum() > 0 for v in actions + contexts))
            self.assertEqual(learner.diagnostics()["actor_optimizer_steps"], 2)
            self.assertEqual(learner.diagnostics()["target_updates"], 2)
            self.assertEqual(learner.finite_update_checks, 2)
        finally:
            spy.remove()
            hook.remove()
            learner.close()

    def test_sharded_uneven_batches_gradients_and_complete_updates(self):
        _, mac, args, scheme = fixture(use_cuda=False, scaled_huber_critic=True)
        rng = torch.get_rng_state()
        serial = HuberGAVDACLearner(mac, scheme, LossLog(), args)
        torch.set_rng_state(rng)
        parallel = HuberGAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
        parallel.args.learner_workers = 2
        try:
            for step, size in enumerate((5, 1, 3)):
                batch, _, _, _ = fixture(batch_size=size, length=7, use_cuda=False)
                batch["reward"].mul_(20)
                if size > 1:
                    batch["filled"][0] = 0
                serial.train(batch, step + 1, step)
                parallel.train(batch, step + 1, step)
                self.assert_updates_equal(serial, parallel)
                torch.testing.assert_close(
                    [p.grad for p in serial.agent_params + serial.critic_params],
                    [p.grad for p in parallel.agent_params + parallel.critic_params],
                    rtol=1e-4, atol=1e-5)
                self.assertEqual(serial.diagnostics(), parallel.diagnostics())
                self.assertIsNone(parallel.gradient_pool.pending_batch)
            self.assertEqual(len(parallel.gradient_pool.worker_pids), 2)
        finally:
            serial.close()
            parallel.close()

    def test_invalid_flags_nonfinite_and_diagnostic_roundtrip(self):
        for value in (1, 0, None, "true", []):
            _, mac, args, scheme = fixture(use_cuda=False, scaled_huber_critic=value)
            with self.assertRaisesRegex(ValueError, "boolean"):
                HuberGAVDACLearner(mac, scheme, LossLog(), args)
        batch, mac, args, scheme = fixture(use_cuda=False, scaled_huber_critic=True)
        learner = HuberGAVDACLearner(mac, scheme, LossLog(), args)
        for error, denominator in ((float("nan"), 1), (float("inf"), 1), (1., 0), (1., float("nan"))):
            with self.assertRaises(ValueError):
                learner._td_error_loss(torch.tensor([error]), denominator)
        learner.train(batch, 1, 1)
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            learner.save_models(directory)
            _, fresh_mac, fresh_args, fresh_scheme = fixture(use_cuda=False, scaled_huber_critic=True)
            restored = HuberGAVDACLearner(fresh_mac, fresh_scheme, LossLog(), fresh_args)
            restored.load_models(directory)
            self.assertEqual(restored.diagnostics(), learner.diagnostics())
            torch.testing.assert_close(restored.mac.agent.state_dict(), learner.mac.agent.state_dict(), rtol=0, atol=0)
            path = Path(directory) / "h04_diagnostics.json"
            value = json.loads(path.read_text())
            value["target_updates"] += 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "counters"):
                restored.load_models(directory)
            restored.close()
        with torch.no_grad():
            learner.agent_params[0].flatten()[0] = float("inf")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            learner._check_parameters(learner.agent_params)
        learner.close()

    def test_entry_config_and_dispatch_are_explicit(self):
        baseline = entry_config("gavdac_ea_main")
        candidate = entry_config("gavdac_c2_h04")
        self.assertEqual(candidate, {**baseline, "name": "gavdac_c2_h04", "scaled_huber_critic": True})
        validate_job("gavdac_c2_h04", 2, "h04test", 0, 1)
        self.assertFalse(huber_enabled(type("Args", (), {})()))

    def test_strict_saved_source_checkpoint_inference(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            config, runner, mac = setup_runner(directory)
            config.update(name="gavdac_c2_h04", scaled_huber_critic=True,
                          rollout_workers=2, learner_workers=2)
            try:
                evaluation = evaluate_fixed(runner, mac, [731, 732], .35, "validation", "rl", 24)
                checkpoint = BestPolicyStore(Path(directory) / "best", config).consider(
                    mac, "rl", 24, evaluation, runner.env)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                self.assertIn("learners/huber_learner.py", metadata["source_sha256"])
            finally:
                runner.close_env()
            output = Path(directory) / "validation"
            result = subprocess.run(
                ["uv", "run", "--frozen", "python", "tools/evaluate_microgrid_comparison.py",
                 "--checkpoint", str(checkpoint), "--split", "validation", "--mode", "continuous",
                 "--rows", "731", "732", "--output", str(output)],
                cwd=ROOT, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["physical_checks_passed"])
            self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
            for name, digest in metadata["source_sha256"].items():
                self.assertEqual(hashlib.sha256((output / "runtime" / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
