"""Five-step target oracle, isolation, default equivalence and strict provenance."""
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
from learners.five_step_learner import FiveStepGAVDACLearner, five_step_enabled, n_step_targets
from learners import gradient_pool
from test_gavdac_sequence import LossLog, fixture
from test_policy_evaluation import setup_runner
from tools.freeze_h04_protocol import entry_config
from tools.experiment_slot import validate_job
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed


def scalar_oracle(rewards, terminated, mask, target_qs, gamma):
    answer = torch.zeros_like(target_qs)
    for row in range(rewards.shape[0]):
        for start in range(rewards.shape[1]):
            total, discount = 0., 1.
            if not mask[row, start, 0]:
                continue
            for index in range(start, min(start + 5, rewards.shape[1])):
                if not mask[row, index, 0]:
                    break
                total += discount * float(rewards[row, index, 0])
                discount *= gamma
                if terminated[row, index, 0]:
                    break
                if index == start + 4 or index + 1 == rewards.shape[1] or not mask[row, index + 1, 0]:
                    answer[row, start] = discount * target_qs[row, index]
                    break
            answer[row, start] += total
    return answer


class H05Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"

    def assert_updates_equal(self, first, second, exact=False):
        tolerance = dict(rtol=0, atol=0) if exact else dict(rtol=1e-4, atol=1e-5)
        for name in ("critic", "mixer", "target_critic", "target_mixer", "agent_optimiser", "critic_optimiser"):
            if getattr(first, name, None) is not None:
                torch.testing.assert_close(getattr(first, name).state_dict(),
                                           getattr(second, name).state_dict(), **tolerance)
        for name in ("mac", "target_mac"):
            torch.testing.assert_close(getattr(first, name).agent.state_dict(),
                                       getattr(second, name).agent.state_dict(), **tolerance)

    def test_independent_target_oracle_all_terminal_offsets_short_tails_and_gaps(self):
        for length in range(1, 8):
            rewards = torch.arange(1, length + 1, dtype=torch.float32)[None, :, None].repeat(4, 1, 1)
            target = (torch.arange(length * 3, dtype=torch.float32).reshape(1, length, 3) + 10).repeat(4, 1, 1)
            for terminal in range(-1, length):
                terminated, mask = torch.zeros_like(rewards), torch.ones_like(rewards)
                if terminal >= 0:
                    terminated[0, terminal] = 1
                if length > 1:
                    mask[1, length // 2:] = 0
                    mask[2, 1] = 0
                mask[3] = 0
                actual = n_step_targets(rewards, terminated, mask, target, .5)
                torch.testing.assert_close(actual, scalar_oracle(rewards, terminated, mask, target, .5),
                                           rtol=0, atol=0)
                one_step = n_step_targets(rewards, terminated, mask, target, .5, n_steps=1)
                original = (rewards.expand_as(target) + .5 * (1 - terminated).expand_as(target) * target) * mask
                torch.testing.assert_close(one_step, original, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device.type, "cpu")
        rewards = torch.arange(1., 8.)[None, :, None].requires_grad_()
        target = torch.arange(10., 17.)[None, :, None].requires_grad_()
        result = n_step_targets(rewards, torch.zeros_like(rewards), torch.ones_like(rewards), target, .99)
        self.assertFalse(result.requires_grad)

    def test_no_padded_or_postterminal_reward_or_bootstrap_leak(self):
        rewards = torch.arange(1., 8.)[None, :, None].repeat(2, 1, 1)
        target = rewards * 10
        terminated, mask = torch.zeros_like(rewards), torch.ones_like(rewards)
        terminated[0, 2] = 1
        mask[1, 3:] = 0
        before = n_step_targets(rewards, terminated, mask, target, .99)
        rewards[:, 3:] = 1e6
        target[0, 2:] = -1e6
        target[1, 3:] = -1e6
        after = n_step_targets(rewards, terminated, mask, target, .99)
        torch.testing.assert_close(after[:, :3], before[:, :3], rtol=0, atol=0)

    def test_missing_false_exact_complete_updates_and_rng(self):
        for explicit in (False, True):
            batch, mac, args, scheme = fixture(use_cuda=False)
            if explicit:
                args.five_step_critic = False
            rng = torch.get_rng_state()
            baseline = GAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
            torch.set_rng_state(rng)
            candidate = FiveStepGAVDACLearner(mac, scheme, LossLog(), args)
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

    def test_terminal_reward_time_limit_bootstrap_and_original_MSE_denominator(self):
        batch, mac, args, scheme = fixture(n_agents=1, batch_size=1, length=5,
                                          gamma=.5, use_cuda=False, five_step_critic=True)
        learner = FiveStepGAVDACLearner(mac, scheme, LossLog(), args)
        with torch.no_grad():
            for module in (learner.critic, learner.target_critic):
                for param in module.parameters():
                    param.zero_()
            learner.target_critic.fc3.bias.fill_(2.)
        batch["reward"][0, :, 0] = torch.tensor([1., 2., 100., 100., 100.])
        batch["terminated"].zero_()
        batch["filled"][:, 2:] = 0
        _, _, loss = learner._critic_loss(batch)
        torch.testing.assert_close(loss, torch.tensor(7.625), rtol=0, atol=0)
        _, _, global_loss = learner._critic_loss(batch, denominator=4.)
        torch.testing.assert_close(global_loss, loss / 2, rtol=0, atol=0)
        batch["terminated"][0, 1] = 1
        _, _, terminal_loss = learner._critic_loss(batch)
        torch.testing.assert_close(terminal_loss, torch.tensor(4.), rtol=0, atol=0)
        terminal_loss.backward()
        self.assertTrue(all(param.grad is None for param in learner.target_critic.parameters()))
        learner.close()

    def test_actor_live_context_action_critic_immutability_and_target_order(self):
        batch, mac, args, scheme = fixture(use_cuda=False, five_step_critic=True)
        learner = FiveStepGAVDACLearner(mac, scheme, LossLog(), args)
        contexts, actions, critic_steps = [], [], []
        def context_hook(module, inputs, keywords):
            hidden = keywords["hidden_states"]
            if hidden.requires_grad:
                hidden.retain_grad()
                contexts.append(hidden)
        hook = learner.mixer.register_forward_pre_hook(context_hook, with_kwargs=True)
        original = learner._actor_loss
        def actor_loss(batch, policy, inputs):
            policy["actions"].retain_grad()
            actions.append(policy["actions"])
            return original(batch, policy, inputs)
        spy = learner.critic_optimiser.register_step_post_hook(
            lambda *unused: critic_steps.append(copy.deepcopy(learner.critic.state_dict())))
        try:
            with patch.object(learner, "_actor_loss", side_effect=actor_loss):
                for step in range(2):
                    targets = [param.detach().clone() for param in learner.target_mac.parameters()]
                    learner.train(batch, step + 1, step)
                    torch.testing.assert_close(critic_steps[-1], learner.critic.state_dict(), rtol=0, atol=0)
                    for old, live, target in zip(targets, learner.mac.parameters(), learner.target_mac.parameters()):
                        torch.testing.assert_close(target, old.lerp(live, args.target_update_tau), rtol=0, atol=0)
            self.assertTrue(all(value.grad is not None and value.grad.abs().sum() > 0 for value in actions + contexts))
            self.assertEqual(learner.actor_optimizer_steps, 2)
            self.assertEqual(learner.target_updates, learner.finite_update_checks)
        finally:
            spy.remove()
            hook.remove()
            learner.close()

    def test_two_worker_uneven_varying_batches_complete_updates_and_gradients(self):
        _, mac, args, scheme = fixture(use_cuda=False, five_step_critic=True)
        rng = torch.get_rng_state()
        serial = FiveStepGAVDACLearner(mac, scheme, LossLog(), args)
        torch.set_rng_state(rng)
        parallel = FiveStepGAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
        parallel.args.learner_workers = 2
        try:
            for step, (size, length) in enumerate(((5, 7), (1, 4), (3, 8))):
                batch, _, _, _ = fixture(batch_size=size, length=length, use_cuda=False)
                if size > 1:
                    batch["filled"][0] = 0
                    batch["filled"][-1, -2:] = 0
                serial.train(batch, step + 1, step)
                parallel.train(batch, step + 1, step)
                self.assert_updates_equal(serial, parallel)
                torch.testing.assert_close([param.grad for param in serial.agent_params + serial.critic_params],
                                           [param.grad for param in parallel.agent_params + parallel.critic_params],
                                           rtol=1e-4, atol=1e-5)
                self.assertEqual(serial.diagnostics(), parallel.diagnostics())
                self.assertIsNone(parallel.gradient_pool.pending_batch)
            self.assertEqual(len(parallel.gradient_pool.worker_pids), 2)
        finally:
            serial.close()
            parallel.close()

    def test_invalid_options_shapes_nonfinite_and_diagnostic_roundtrip(self):
        for value in (1, 0, None, "true", []):
            _, mac, args, scheme = fixture(use_cuda=False, five_step_critic=value)
            with self.assertRaisesRegex(ValueError, "boolean"):
                FiveStepGAVDACLearner(mac, scheme, LossLog(), args)
        values = torch.ones(1, 2, 1)
        for bad in (0, True, 2.5):
            with self.assertRaises(ValueError):
                n_step_targets(values, values, values, values, .99, bad)
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                n_step_targets(values * value, values, values, values, .99)
        with self.assertRaisesRegex(ValueError, "B,T"):
            n_step_targets(values[:, :1], values, values, values, .99)
        batch, mac, args, scheme = fixture(use_cuda=False, five_step_critic=True)
        learner = FiveStepGAVDACLearner(mac, scheme, LossLog(), args)
        learner.train(batch, 1, 1)
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            learner.save_models(directory)
            _, fresh_mac, fresh_args, fresh_scheme = fixture(use_cuda=False, five_step_critic=True)
            restored = FiveStepGAVDACLearner(fresh_mac, fresh_scheme, LossLog(), fresh_args)
            restored.load_models(directory)
            self.assertEqual(restored.diagnostics(), learner.diagnostics())
            torch.testing.assert_close(restored.mac.agent.state_dict(), learner.mac.agent.state_dict(), rtol=0, atol=0)
            path = Path(directory) / "h05_diagnostics.json"
            value = json.loads(path.read_text())
            value["target_updates"] += 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "counters"):
                restored.load_models(directory)
            restored.close()
        bad = copy.deepcopy(batch)
        bad["reward"].fill_(float("nan"))
        before = copy.deepcopy(learner.critic.state_dict())
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            learner.train(bad, 2, 2)
        torch.testing.assert_close(before, learner.critic.state_dict(), rtol=0, atol=0)
        learner.close()

    def test_entry_and_worker_dispatch_explicit_default_unchanged(self):
        baseline, candidate = entry_config("gavdac_ea_main"), entry_config("gavdac_c2_h05")
        self.assertEqual(candidate, {**baseline, "name": "gavdac_c2_h05", "five_step_critic": True})
        validate_job("gavdac_c2_h05", 2, "h05test", 0, 1)
        self.assertFalse(five_step_enabled(type("Args", (), {})()))
        _, _, args, scheme = fixture(use_cuda=False, five_step_critic=True)
        barrier = type("Barrier", (), {"wait": lambda self, **kwargs: None})()
        with patch.object(torch, "set_num_interop_threads"), patch.multiple(
            gradient_pool, _learner=None, _scheme=None, _groups=None, _transport=None,
        ):
            gradient_pool._initialize(args, scheme, barrier)
            self.assertIsInstance(gradient_pool._learner, FiveStepGAVDACLearner)

    def test_strict_saved_source_validation_inference(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            config, runner, mac = setup_runner(directory)
            config.update(name="gavdac_c2_h05", five_step_critic=True, rollout_workers=2, learner_workers=2)
            try:
                evaluation = evaluate_fixed(runner, mac, [731, 732], .35, "validation", "rl", 24)
                checkpoint = BestPolicyStore(Path(directory) / "best", config).consider(
                    mac, "rl", 24, evaluation, runner.env)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                self.assertIn("learners/five_step_learner.py", metadata["source_sha256"])
            finally:
                runner.close_env()
            output = Path(directory) / "validation"
            response = subprocess.run(
                ["uv", "run", "--frozen", "python", "tools/evaluate_microgrid_comparison.py",
                 "--checkpoint", str(checkpoint), "--split", "validation", "--mode", "continuous",
                 "--rows", "731", "732", "--output", str(output)],
                cwd=ROOT, capture_output=True, timeout=60)
            self.assertEqual(response.returncode, 0, response.stderr.decode("utf-8", errors="replace"))
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["physical_checks_passed"])
            self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
            for name, digest in metadata["source_sha256"].items():
                self.assertEqual(hashlib.sha256((output / "runtime" / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
