"""H07 independent target-action oracle, replay/RNG boundaries and full sharded updates."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from learners.gavdac_learner import GAVDACLearner
from learners.target_smoothing_learner import (
    NOISE_FIELD, TargetSmoothedGAVDACLearner, target_policy_smoothing_enabled,
)
from learners import gradient_pool
from test_gavdac_sequence import LossLog, fixture
from tools.freeze_h06_protocol import entry_config
from tools.experiment_slot import validate_job


def learner(enabled=True, **options):
    batch, mac, args, scheme = fixture(use_cuda=False, seed=0, target_policy_smoothing=enabled, **options)
    return batch, TargetSmoothedGAVDACLearner(mac, scheme, LossLog(), args), args, scheme


def worker_identity():
    from learners import gradient_pool
    value = gradient_pool._learner
    return type(value).__name__, value.target_noise_batches, value.noise_generator.get_state()


class H07Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"

    def test_independent_asymmetric_Box_formula_noise_and_action_clips(self):
        batch, value, _, _ = learner()
        prepared = value._prepare_batch(batch)
        policy = value.target_mac.forward_sequence(batch)
        actions = policy["actions"]
        noise = prepared[NOISE_FIELD]
        generator = torch.Generator().manual_seed(0)
        expected_noise = torch.randn(actions.shape, generator=generator).mul(.2).clamp(-.5, .5)
        torch.testing.assert_close(noise, expected_noise, rtol=0, atol=0)
        self.assertTrue(torch.equal(generator.get_state(), value.noise_generator.get_state()))
        low, high = value.target_mac._get_action_bounds(actions)
        for scalar in (-.5, 0., .5):
            noise.fill_(scalar)
            result = value._target_actions(prepared, policy)
            expected = torch.minimum(torch.maximum(actions + scalar*(high-low)/2, low), high)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        noise.fill_(.5)
        policy["actions"] = high.expand_as(actions).clone()
        torch.testing.assert_close(value._target_actions(prepared, policy), policy["actions"], rtol=0, atol=0)
        noise.fill_(-.5)
        policy["actions"] = low.expand_as(actions).clone()
        torch.testing.assert_close(value._target_actions(prepared, policy), policy["actions"], rtol=0, atol=0)
        self.assertEqual(noise.dtype, torch.float32)
        self.assertEqual(noise.device.type, "cpu")

    def test_no_replay_history_mutation_and_only_target_action_changes(self):
        batch, value, _, _ = learner()
        original = {key: tensor.clone() for key, tensor in batch.data.transition_data.items()}
        rng = torch.get_rng_state()
        prepared = value._prepare_batch(batch)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertNotIn(NOISE_FIELD, batch.data.transition_data)
        torch.testing.assert_close(original, batch.data.transition_data, rtol=0, atol=0)
        for key in original:
            self.assertEqual(batch[key].data_ptr(), prepared[key].data_ptr())
        plain = value.target_mac.forward_sequence(batch)
        other = value.target_mac.forward_sequence(prepared)
        torch.testing.assert_close(plain, other, rtol=0, atol=0)
        prepared[NOISE_FIELD].zero_()
        torch.testing.assert_close(value._target_actions(prepared, other),
                                   value.target_mac._clamp_actions_to_space(plain["actions"]), rtol=0, atol=0)
        prepared[NOISE_FIELD].fill_(.25)
        self.assertFalse(torch.equal(value._target_actions(prepared, other), plain["actions"]))
        batch, value, _, _ = learner()
        original = {key: tensor.clone() for key, tensor in batch.data.transition_data.items()}
        rng = torch.get_rng_state()
        value.train(batch, 1, 1)
        torch.testing.assert_close(original, batch.data.transition_data, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertNotIn(NOISE_FIELD, batch.data.transition_data)

    def test_missing_false_initialization_full_update_and_global_RNG_exact(self):
        batch, mac, args, scheme = fixture(use_cuda=False, seed=0)
        rng = torch.get_rng_state()
        baseline = GAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
        after = torch.get_rng_state()
        for enabled in (None, False, True):
            torch.set_rng_state(rng)
            options = copy.deepcopy(args)
            if enabled is not None:
                options.target_policy_smoothing = enabled
            value = TargetSmoothedGAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), options)
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            for name in ("critic", "mixer", "target_critic", "target_mixer"):
                torch.testing.assert_close(getattr(baseline, name).state_dict(), getattr(value, name).state_dict(),
                                           rtol=0, atol=0)
            if enabled is not True:
                left = baseline
                for step in range(3):
                    before = torch.get_rng_state()
                    left.train(batch, step, step)
                    consumed = torch.get_rng_state()
                    torch.set_rng_state(before)
                    value.train(batch, step, step)
                    self.assertTrue(torch.equal(consumed, torch.get_rng_state()))
                    for name in ("critic", "mixer", "target_critic", "target_mixer",
                                 "agent_optimiser", "critic_optimiser"):
                        torch.testing.assert_close(getattr(left, name).state_dict(), getattr(value, name).state_dict(),
                                                   rtol=0, atol=0)
                    torch.testing.assert_close(left.mac.agent.state_dict(), value.mac.agent.state_dict(), rtol=0, atol=0)
                    torch.testing.assert_close(left.target_mac.agent.state_dict(), value.target_mac.agent.state_dict(),
                                               rtol=0, atol=0)
                # Reset baseline for the next flag, without changing the tested global-RNG contract.
                torch.set_rng_state(rng)
                baseline = GAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))

    def test_one_step_MSE_terminal_padding_global_denominator_and_target_detach(self):
        batch, value, _, _ = learner(n_agents=1, batch_size=3, length=6)
        prepared = value._prepare_batch(batch)
        with torch.no_grad():
            value.critic.fc3.weight.zero_()
            value.critic.fc3.bias.zero_()
            value.target_critic.fc3.weight.zero_()
            value.target_critic.fc3.bias.fill_(2)
        mask = batch["filled"][:, :-1].float().clone()
        mask[:, 1:] *= 1-batch["terminated"][:, :-2].float()
        targets = batch["reward"][:, :-1] + .99*(1-batch["terminated"][:, :-1].float())*2
        _, _, loss = value._critic_loss(prepared)
        torch.testing.assert_close(loss, (targets*mask).square().sum()/mask.sum())
        _, _, global_loss = value._critic_loss(prepared, denominator=100)
        torch.testing.assert_close(global_loss, loss*mask.sum()/100)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in value.target_critic.parameters()))
        self.assertTrue(all(p.grad is None for p in value.target_mac.parameters()))
        self.assertTrue(all(p.grad is None for p in value.mac.parameters()))
        batch["filled"][-1, -2] = 0
        batch["reward"][-1, -2] += 1000
        first = value._critic_loss(prepared)[2]
        batch["reward"][-1, -2] -= 2000
        torch.testing.assert_close(first, value._critic_loss(prepared)[2], rtol=0, atol=0)

    def test_live_Actor_context_action_and_original_optimizer_target_order(self):
        batch, value, _, _ = learner()
        prepared = value._prepare_batch(batch)
        policy, inputs, _ = value._critic_loss(prepared)
        policy["actions"].retain_grad()
        policy["hidden_states"].retain_grad()
        value._set_critic_grad(False)
        try:
            value._actor_loss(prepared, policy, inputs).backward()
        finally:
            value._set_critic_grad(True)
        self.assertGreater(float(policy["actions"].grad.abs().sum()), 0)
        self.assertGreater(float(policy["hidden_states"].grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in value.critic_params))
        # The preceding explicit preparation was only an oracle, not a completed update.
        batch, value, _, _ = learner()
        before = copy.deepcopy(value.target_critic.state_dict())
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
        for name, current in value.critic.state_dict().items():
            torch.testing.assert_close(before[name].lerp(current, value.args.target_update_tau),
                                       value.target_critic.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(value.diagnostics()["finite_update_checks"], 1)
        self.assertEqual(value.target_noise_batches, 1)

    def test_two_worker_uneven_varying_full_updates_shared_noise_and_no_worker_draw(self):
        _, serial, _, _ = learner(batch_size=5, length=7)
        _, parallel, _, _ = learner(batch_size=5, length=7)
        parallel.args.learner_workers = 2
        try:
            for step, (size, length) in enumerate(((5, 7), (2, 4), (7, 8))):
                batch, _, _, _ = fixture(batch_size=size, length=length, use_cuda=False)
                batch["filled"][:1] = 0
                serial.train(batch, step, step)
                parallel.train(batch, step, step)
                self.assertTrue(torch.equal(serial.noise_generator.get_state(), parallel.noise_generator.get_state()))
                self.assertIn(NOISE_FIELD, parallel.gradient_pool.batch_buffers)
                self.assertNotIn(NOISE_FIELD, batch.data.transition_data)
                for name in ("critic", "mixer", "target_critic", "target_mixer", "agent_optimiser", "critic_optimiser"):
                    torch.testing.assert_close(getattr(serial, name).state_dict(), getattr(parallel, name).state_dict(),
                                               rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(serial.mac.agent.state_dict(), parallel.mac.agent.state_dict(),
                                           rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(serial.target_mac.agent.state_dict(), parallel.target_mac.agent.state_dict(),
                                           rtol=1e-4, atol=1e-5)
                torch.testing.assert_close([p.grad for p in serial.agent_params+serial.critic_params],
                                           [p.grad for p in parallel.agent_params+parallel.critic_params],
                                           rtol=1e-4, atol=1e-5)
                self.assertEqual(serial.diagnostics(), parallel.diagnostics())
            for executor in parallel.gradient_pool.executors:
                name, draws, state = executor.submit(worker_identity).result()
                self.assertEqual(name, "TargetSmoothedGAVDACLearner")
                self.assertEqual(draws, 0)
                self.assertTrue(torch.equal(state, torch.Generator().manual_seed(0).get_state()))
        finally:
            serial.close()
            parallel.close()

    def test_reject_bad_flags_noise_shapes_nonfinite_bounds_and_diagnostics(self):
        for flag in (0, 1, None, "true", []):
            with self.assertRaises(ValueError):
                target_policy_smoothing_enabled(type("Args", (), {"target_policy_smoothing": flag})())
        batch, value, _, _ = learner()
        prepared = value._prepare_batch(batch)
        with self.assertRaisesRegex(ValueError, "already exists"):
            value._prepare_batch(prepared)
        policy = value.target_mac.forward_sequence(batch)
        for bad in (torch.zeros(1), torch.full_like(prepared[NOISE_FIELD], float("nan")),
                    torch.full_like(prepared[NOISE_FIELD], .6)):
            prepared.data.transition_data[NOISE_FIELD] = bad
            with self.assertRaises(ValueError):
                value._target_actions(prepared, policy)
        with self.assertRaisesRegex(ValueError, "parent-generated"):
            value._target_actions(batch, policy)
        batch, value, _, _ = learner()
        value.target_mac._action_highs.copy_(value.target_mac._action_lows)
        with self.assertRaisesRegex(ValueError, "positive finite Box"):
            value._target_actions(value._prepare_batch(batch), value.target_mac.forward_sequence(batch))
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            batch, value, _, _ = learner()
            value.train(batch, 1, 1)
            value.save_models(directory)
            _, restored, _, _ = learner()
            restored.load_models(directory)
            self.assertEqual(value.diagnostics(), restored.diagnostics())
            self.assertFalse(restored.diagnostics()["exact_resume_supported"])
            path = Path(directory) / "h07_diagnostics.json"
            saved = json.loads(path.read_text())
            saved["target_noise_batches"] = -1
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaises(ValueError):
                restored.load_models(directory)

    def test_entry_only_name_flag_and_explicit_run_worker_construction(self):
        parent = entry_config("gavdac_ea_main", {"rollout_workers": 2, "learner_workers": 2})
        candidate = entry_config("gavdac_c2_h07", {"rollout_workers": 2, "learner_workers": 2})
        self.assertEqual(candidate, {**parent, "name": "gavdac_c2_h07", "target_policy_smoothing": True})
        validate_job("gavdac_c2_h07", 2, "h07test", 0, 1)
        import run
        self.assertIn("TargetSmoothedGAVDACLearner", inspect.getsource(run._run_with_runner))
        self.assertIn("target_policy_smoothing_enabled", inspect.getsource(gradient_pool._initialize))
        self.assertEqual(inspect.getsource(GAVDACLearner._target_actions).strip(),
                         'def _target_actions(self, batch, target_policy):\n'
                         '        return self.target_mac._clamp_actions_to_space(target_policy["actions"])')

    def test_strict_saved_source_validation_inference_731_732(self):
        from utils.policy_evaluation import BestPolicyStore, evaluate_fixed, fixed_rows
        from test_policy_evaluation import setup_runner
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            config, runner, mac = setup_runner(directory)
            config.update(name="gavdac_c2_h07", target_policy_smoothing=True)
            rows = fixed_rows(runner.env, "validation", [], 2)
            self.assertEqual(rows, [731, 732])
            evaluation = evaluate_fixed(runner, mac, rows, .35, "validation", "rl", 123)
            checkpoint = BestPolicyStore(Path(directory) / "best", config).consider(
                mac, "rl", 123, evaluation, runner.env)
            output = Path(directory) / "strict"
            response = subprocess.run([
                "uv", "run", "--frozen", "python", "tools/evaluate_microgrid_comparison.py",
                "--checkpoint", str(checkpoint), "--split", "validation", "--rows", "731", "732",
                "--mode", "continuous", "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(response.returncode, 0, response.stdout+response.stderr)
            summary = json.loads((output / "summary.json").read_text())
            metadata = json.loads((checkpoint / "metadata.json").read_text())
            self.assertTrue(summary["physical_checks_passed"])
            self.assertEqual(summary["rows"], rows)
            self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
            for name, digest in metadata["source_sha256"].items():
                self.assertEqual(summary["source_sha256"][name], digest)


if __name__ == "__main__":
    unittest.main()
