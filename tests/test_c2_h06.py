"""H06 representation oracle, gradient contracts and complete sharded updates."""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from learners.gavdac_learner import GAVDACLearner
from learners.layer_norm_learner import (
    LayerNormGAVDACCritic, LayerNormGAVDACLearner, critic_layer_norm_enabled,
)
from learners import gradient_pool
from modules.critic import GAVDACCritic
from test_gavdac_sequence import LossLog, fixture
from tools.freeze_h05_protocol import entry_config
from tools.experiment_slot import validate_job


def learner(normalized=True, **options):
    batch, mac, args, scheme = fixture(use_cuda=False, critic_layer_norm=normalized, **options)
    return batch, LayerNormGAVDACLearner(mac, scheme, LossLog(), args), args, scheme


def worker_identity():
    from learners import gradient_pool
    value = gradient_pool._learner
    return type(value).__name__, type(value.critic).__name__, sorted(value.critic.state_dict())


class H06Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"

    def test_explicit_formula_pre_relu_gradients_and_parameter_increment(self):
        batch, value, args, scheme = learner()
        critic = value.critic
        x = torch.linspace(-2, 3, 8 * 20).reshape(8, 20).requires_grad_()
        action = torch.linspace(-1, 1, 16).reshape(8, 2).requires_grad_()

        def norm(tensor, module):
            centered = tensor - tensor.mean(-1, keepdim=True)
            return centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-5) * module.weight + module.bias

        original = critic(x, action)[0]
        manual = F.linear(torch.cat([x, action], -1), critic.fc1.weight, critic.fc1.bias)
        manual = F.relu(norm(manual, critic.norm1))
        manual = F.linear(manual, critic.fc2.weight, critic.fc2.bias)
        manual = F.relu(norm(manual, critic.norm2))
        manual = F.linear(manual, critic.fc3.weight, critic.fc3.bias)
        torch.testing.assert_close(original, manual, rtol=1e-5, atol=1e-6)
        params = [x, action, *critic.parameters()]
        left = torch.autograd.grad(original.square().sum(), params)
        right = torch.autograd.grad(manual.square().sum(), params)
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
        plain = GAVDACCritic(scheme, args)
        self.assertEqual(sum(p.numel() for p in critic.parameters()) - sum(p.numel() for p in plain.parameters()),
                         4 * args.rnn_hidden_dim)
        self.assertTrue(all(g.abs().sum() > 0 for g in left[-4:]))
        for name in ("norm1", "norm2"):
            module = getattr(critic, name)
            self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
            self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
            self.assertEqual(module.eps, 1e-5)
        self.assertEqual(original.shape, (8, 1))
        self.assertEqual(original.dtype, torch.float32)

    def test_per_example_independence_train_eval_constant_and_large_inputs(self):
        _, value, _, _ = learner()
        critic = value.critic
        x, a = torch.randn(6, 20), torch.randn(6, 2)
        whole = critic(x, a)[0]
        pieces = torch.cat([critic(x[i:i+1], a[i:i+1])[0] for i in range(6)])
        torch.testing.assert_close(whole, pieces, rtol=1e-5, atol=1e-6)
        changed = x.clone()
        changed[1:] = 1e6
        torch.testing.assert_close(critic(changed, a)[0][0], whole[0], rtol=0, atol=0)
        critic.eval()
        torch.testing.assert_close(critic(x, a)[0], whole, rtol=0, atol=0)
        self.assertFalse(any("running" in key for key in critic.state_dict()))
        for scalar in (0., 1., 1e6):
            q = critic(torch.full_like(x, scalar), torch.full_like(a, scalar))[0]
            self.assertTrue(torch.isfinite(q).all())
            gradients = torch.autograd.grad(q.sum(), list(critic.parameters()))
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_original_initialization_RNG_and_missing_false_complete_updates(self):
        batch, mac, args, scheme = fixture(use_cuda=False)
        rng = torch.get_rng_state()
        baseline = GAVDACLearner(mac, scheme, LossLog(), args)
        after = torch.get_rng_state()
        for flag in (None, False, True):
            torch.set_rng_state(rng)
            candidate_args = copy.deepcopy(args)
            if flag is not None:
                candidate_args.critic_layer_norm = flag
            candidate = LayerNormGAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), candidate_args)
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            for name, weight in baseline.critic.state_dict().items():
                torch.testing.assert_close(weight, candidate.critic.state_dict()[name], rtol=0, atol=0)
            torch.testing.assert_close(baseline.mixer.state_dict(), candidate.mixer.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(baseline.mac.agent.state_dict(), candidate.mac.agent.state_dict(), rtol=0, atol=0)
            if flag is not True:
                self.assertIs(type(candidate.critic), GAVDACCritic)
        # Fresh instances preserve optimizer post-step hooks, unlike optimizer deepcopy.
        torch.set_rng_state(rng)
        left = GAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
        torch.set_rng_state(rng)
        right_args = copy.deepcopy(args)
        right_args.critic_layer_norm = False
        right = LayerNormGAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), right_args)
        for step in range(3):
            before = torch.get_rng_state()
            left.train(batch, step, step)
            consumed = torch.get_rng_state()
            torch.set_rng_state(before)
            right.train(batch, step, step)
            self.assertTrue(torch.equal(consumed, torch.get_rng_state()))
            for name in ("critic", "mixer", "target_critic", "target_mixer", "agent_optimiser", "critic_optimiser"):
                torch.testing.assert_close(getattr(left, name).state_dict(), getattr(right, name).state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(left.mac.agent.state_dict(), right.mac.agent.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(left.target_mac.agent.state_dict(), right.target_mac.agent.state_dict(), rtol=0, atol=0)

    def test_original_one_step_MSE_terminal_padding_and_global_denominator(self):
        batch, value, _, _ = learner(n_agents=1, batch_size=3, length=6)
        with torch.no_grad():
            value.critic.fc3.weight.zero_()
            value.critic.fc3.bias.zero_()
            value.target_critic.fc3.weight.zero_()
            value.target_critic.fc3.bias.fill_(2)
        mask = batch["filled"][:, :-1].float().clone()
        mask[:, 1:] *= 1 - batch["terminated"][:, :-2].float()
        target = batch["reward"][:, :-1] + .99 * (1 - batch["terminated"][:, :-1].float()) * 2
        _, _, loss = value._critic_loss(batch)
        torch.testing.assert_close(loss, (target * mask).square().sum() / mask.sum())
        _, _, global_loss = value._critic_loss(batch, denominator=100)
        torch.testing.assert_close(global_loss, loss * mask.sum() / 100)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in value.target_critic.parameters()))
        self.assertTrue(all(p.grad is None for p in value.target_mac.parameters()))
        self.assertTrue(all(p.grad is None for p in value.mac.parameters()))
        batch["reward"][-1, -2] += 1000
        # The last transition of this row was masked in fixture; explicitly mask before sentinel change.
        batch["filled"][-1, -2] = 0
        _, _, masked = value._critic_loss(batch)
        batch["reward"][-1, -2] -= 2000
        _, _, masked_again = value._critic_loss(batch)
        torch.testing.assert_close(masked, masked_again, rtol=0, atol=0)

    def test_live_Actor_context_action_and_every_step_target_order(self):
        batch, value, _, _ = learner()
        policy, inputs, critic_loss = value._critic_loss(batch)
        policy["actions"].retain_grad()
        policy["hidden_states"].retain_grad()
        value._set_critic_grad(False)
        try:
            actor_loss = value._actor_loss(batch, policy, inputs)
            actor_loss.backward()
        finally:
            value._set_critic_grad(True)
        self.assertGreater(float(policy["actions"].grad.abs().sum()), 0)
        self.assertGreater(float(policy["hidden_states"].grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in value.critic_params))
        before_target = copy.deepcopy(value.target_critic.state_dict())
        before_norm = value.critic.norm1.weight.detach().clone()
        events = []
        value.critic_optimiser.register_step_post_hook(lambda *unused: events.append("critic"))
        value.agent_optimiser.register_step_post_hook(lambda *unused: events.append("actor"))
        update = value._update_targets_soft

        def target(tau):
            events.append("target")
            update(tau)

        with patch.object(value, "_update_targets_soft", target):
            value.train(batch, 1, 1)
        self.assertEqual(events, ["critic", "actor", "target"])
        self.assertFalse(torch.equal(before_norm, value.critic.norm1.weight))
        for name, current in value.critic.state_dict().items():
            expected = before_target[name].lerp(current, value.args.target_update_tau)
            torch.testing.assert_close(expected, value.target_critic.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(value.diagnostics()["finite_update_checks"], 1)
        # Isolate an Actor optimizer step and prove no Critic parameter is stepped.
        saved = copy.deepcopy(value.critic.state_dict())
        policy = value.mac.forward_sequence(batch, end=-1)
        value._set_critic_grad(False)
        try:
            value.agent_optimiser.zero_grad(set_to_none=True)
            value._actor_loss(batch, policy, value._build_inputs_sequence(batch, end=-1)).backward()
            value.agent_optimiser.step()
        finally:
            value._set_critic_grad(True)
        torch.testing.assert_close(saved, value.critic.state_dict(), rtol=0, atol=0)

    def test_two_workers_uneven_varying_full_updates_and_new_affine_transport(self):
        batch, serial, args, scheme = learner(batch_size=5, length=7)
        _, parallel, _, _ = learner(batch_size=5, length=7)
        parallel.args.learner_workers = 2
        try:
            for step, (size, length) in enumerate(((5, 7), (2, 4), (7, 8))):
                batch, _, _, _ = fixture(batch_size=size, length=length, use_cuda=False)
                batch["filled"][:1] = 0
                serial.train(batch, step, step)
                parallel.train(batch, step, step)
                for name in ("critic", "mixer", "target_critic", "target_mixer", "agent_optimiser", "critic_optimiser"):
                    torch.testing.assert_close(getattr(serial, name).state_dict(), getattr(parallel, name).state_dict(),
                                               rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(serial.mac.agent.state_dict(), parallel.mac.agent.state_dict(), rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(serial.target_mac.agent.state_dict(), parallel.target_mac.agent.state_dict(),
                                           rtol=1e-4, atol=1e-5)
                torch.testing.assert_close([p.grad for p in serial.agent_params + serial.critic_params],
                                           [p.grad for p in parallel.agent_params + parallel.critic_params],
                                           rtol=1e-4, atol=1e-5)
                self.assertEqual(serial.diagnostics(), parallel.diagnostics())
            self.assertEqual(len(parallel.gradient_pool.worker_pids), 2)
            for executor in parallel.gradient_pool.executors:
                actual = executor.submit(worker_identity).result()
                self.assertEqual(actual[:2], ("LayerNormGAVDACLearner", "LayerNormGAVDACCritic"))
                self.assertIn("norm2.bias", actual[2])
        finally:
            serial.close()
            parallel.close()

    def test_invalid_flags_shapes_nonfinite_and_diagnostic_roundtrip(self):
        for option in (1, 0, None, "true", []):
            with self.assertRaises(ValueError):
                critic_layer_norm_enabled(type("Args", (), {"critic_layer_norm": option})())
        batch, value, _, _ = learner()
        for x, a in ((torch.zeros(2, 21), None), (torch.zeros(2, 20), torch.full((2, 2), float("nan")))):
            with self.assertRaises(ValueError):
                value.critic(x, a)
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            value.train(batch, 1, 1)
            value.save_models(directory)
            _, restored, _, _ = learner()
            restored.load_models(directory)
            self.assertEqual(value.diagnostics(), restored.diagnostics())
            self.assertFalse(restored.diagnostics()["exact_resume_supported"])
            path = Path(directory) / "h06_diagnostics.json"
            saved = json.loads(path.read_text())
            saved["critic_optimizer_steps"] = -1
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaises(ValueError):
                restored.load_models(directory)

    def test_entry_and_explicit_dispatch_are_only_flag_name_change(self):
        parent = entry_config("gavdac_ea_main", {"rollout_workers": 2, "learner_workers": 2})
        candidate = entry_config("gavdac_c2_h06", {"rollout_workers": 2, "learner_workers": 2})
        self.assertEqual(candidate, {**parent, "name": "gavdac_c2_h06", "critic_layer_norm": True})
        validate_job("gavdac_c2_h06", 2, "h06test", 0, 1)
        import run
        self.assertIn("LayerNormGAVDACLearner", inspect.getsource(run._run_with_runner))
        self.assertIn("critic_layer_norm_enabled", inspect.getsource(gradient_pool._initialize))
        self.assertEqual(inspect.getsource(GAVDACLearner._make_critic).strip(),
                         "def _make_critic(self, scheme, args):\n        return GAVDACCritic(scheme, args)")

    def test_strict_saved_source_validation_inference_731_732(self):
        from utils.policy_evaluation import BestPolicyStore, evaluate_fixed, fixed_rows
        from test_policy_evaluation import setup_runner
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            config, runner, mac = setup_runner(directory)
            config.update(name="gavdac_c2_h06", critic_layer_norm=True)
            rows = fixed_rows(runner.env, "validation", [], 2)
            self.assertEqual(rows, [731, 732])
            evaluation = evaluate_fixed(runner, mac, rows, .35, "validation", "rl", 123)
            store = BestPolicyStore(Path(directory) / "best", config)
            checkpoint = store.consider(mac, "rl", 123, evaluation, runner.env)
            output = Path(directory) / "strict"
            response = subprocess.run([
                "uv", "run", "--frozen", "python", "tools/evaluate_microgrid_comparison.py",
                "--checkpoint", str(checkpoint), "--split", "validation", "--rows", "731", "732",
                "--mode", "continuous", "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(response.returncode, 0, response.stdout + response.stderr)
            summary = json.loads((output / "summary.json").read_text())
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            self.assertTrue(summary["physical_checks_passed"])
            self.assertEqual(summary["rows"], [731, 732])
            self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
            for name, digest in metadata["source_sha256"].items():
                self.assertEqual(summary["source_sha256"][name], digest)


if __name__ == "__main__":
    unittest.main()
