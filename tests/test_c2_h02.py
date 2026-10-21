"""Mechanical H02 gates; no retained-data performance assertions."""

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch as th
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import direct_main
import run
from ea.ssne_base import SSNEBase
from learners.clipped_double_gavdac_learner import ClippedDoubleGAVDACLearner
from learners.gavdac_learner import GAVDACLearner
from test_gavdac_sequence import LossLog, fixture
from tools.experiment_slot import validate_job


def loaded_config(name):
    configs = []
    with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
         patch.object(direct_main, "run", lambda config, logger: configs.append(config)):
        direct_main.main()
    return configs[0]


def learner_fixture(**options):
    batch, mac, args, scheme = fixture(clipped_double_critic=True, **options)
    return batch, ClippedDoubleGAVDACLearner(mac, scheme, LossLog(), args)


class ConstantCritic(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = th.tensor(values)
        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = None

    def forward(self, inputs, actions, hidden):
        return self.values.repeat(actions.shape[0] * actions.shape[1]).view(-1, 1), None


class SumMixer(nn.Module):
    def forward(self, q, states, **kwargs):
        value = q.view(states.shape[0], states.shape[1], 3).sum(-1, keepdim=True)
        return value, None, None


class H02MechanicalTest(unittest.TestCase):
    def test_entry_config_is_one_intervention_and_supervisor_accepts_it(self):
        parent = loaded_config("gavdac_ea_main")
        self.assertEqual(loaded_config("gavdac_c2_h02"), {
            **parent, "name": "gavdac_c2_h02", "clipped_double_critic": True})
        validate_job("gavdac_c2_h02", 0, "h02checka", 0, 2)

    def test_minimum_is_after_team_mixing_and_target_has_no_graph(self):
        batch, learner = learner_fixture()
        learner.target_critic = ConstantCritic([0., 10., 0.])
        learner.target_critic2 = ConstantCritic([5., 0., 5.])
        learner.target_mixer = SumMixer()
        learner.target_mixer2 = SumMixer()
        target = learner.target_mac.forward_sequence(batch)
        values = learner._target_team_values(batch, target["actions"], target["hidden_states"])
        th.testing.assert_close(values, th.full((4, 4, 1), 10.))
        self.assertFalse(values.requires_grad)
        self.assertIsNone(values.grad_fn)

    def test_target_reward_terminal_and_padding_loss(self):
        batch, learner = learner_fixture()
        with th.no_grad():
            target_policy = learner.target_mac.forward_sequence(batch)
            target_actions = learner.target_mac._clamp_actions_to_space(target_policy["actions"])
            target_q = learner._target_team_values(batch, target_actions, target_policy["hidden_states"])
            expected_target = batch["reward"][:, :-1] + learner.args.gamma * (
                1 - batch["terminated"][:, :-1].float()) * target_q
            current = learner.mac.forward_sequence(batch, end=-1)
            q = learner._team_values(
                learner.critic, learner.mixer, learner._build_inputs_sequence(batch, end=-1),
                batch["actions"][:, :-1], batch["state"][:, :-1], batch["obs"][:, :-1],
                current["hidden_states"], batch,
            )
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] *= 1 - batch["terminated"][:, :-2].float()
            expected_loss = (((expected_target - q) * mask) ** 2).sum() / mask.sum()
        learner.train(batch, 0, 0)
        self.assertAlmostEqual(learner.last_update_metrics["critic1_loss"], expected_loss.item(), places=6)
        q = th.zeros(1, 3, 1, requires_grad=True)
        loss = learner._critic_loss(q, th.tensor([[[2.], [999.], [4.]]]),
                                    th.tensor([[[1.], [0.], [1.]]]))
        self.assertEqual(loss.item(), 10.)
        loss.backward()
        self.assertEqual(q.grad[0, 1, 0].item(), 0.)

    def test_independent_initialization_updates_and_targets(self):
        for mode, optimizer in (("soft", "adam"), ("hard", "rmsprop")):
            with self.subTest(mode=mode):
                batch, learner = learner_fixture(target_update_mode=mode, optimizer=optimizer)
                self.assertNotEqual(learner.critic.fc1.weight.data_ptr(), learner.critic2.fc1.weight.data_ptr())
                self.assertFalse(th.equal(learner.critic.fc1.weight, learner.critic2.fc1.weight))
                for target, online in ((learner.target_critic, learner.critic),
                                       (learner.target_critic2, learner.critic2),
                                       (learner.target_mixer2, learner.mixer2)):
                    th.testing.assert_close(target.state_dict(), online.state_dict(), rtol=0, atol=0)
                before = [p.detach().clone() for p in learner.critic2_params]
                old_targets = [p.detach().clone() for p in learner.target_critic2.parameters()]
                learner.train(batch, 0, 0)
                self.assertTrue(any(not th.equal(a, b) for a, b in zip(before, learner.critic2_params)))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in learner.critic2_params))
                tau = learner.args.target_update_tau if mode == "soft" else 1.
                for old, target, online in zip(old_targets, learner.target_critic2.parameters(),
                                              learner.critic2.parameters(), strict=True):
                    th.testing.assert_close(target, old.lerp(online, tau), rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in learner.target_critic2.parameters()))
                self.assertEqual(learner.optimizer_steps, {"actor": 1, "critic1": 1, "critic2": 1})
                self.assertEqual(learner.finite_update_checks, 1)

    def test_actor_uses_q1_and_live_context_without_changing_critics(self):
        batch, learner = learner_fixture()
        calls, contexts, captured = [], [], {}
        hook = learner.critic2.register_forward_hook(lambda *unused: calls.append(1))
        context_hook = learner.mixer.register_forward_pre_hook(
            lambda module, args, kwargs: contexts.append(kwargs["hidden_states"].requires_grad),
            with_kwargs=True)
        second_step = learner.critic2_optimiser.step

        def capture_critics(*args, **kwargs):
            result = second_step(*args, **kwargs)
            captured["weights"] = [p.detach().clone()
                                   for p in learner.critic_params + learner.critic2_params]
            captured["grads"] = [None if p.grad is None else p.grad.clone()
                                 for p in learner.critic_params + learner.critic2_params]
            return result

        try:
            with patch.object(learner.critic2_optimiser, "step", capture_critics):
                learner.train(batch, 0, 0)
        finally:
            hook.remove()
            context_hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(contexts, [False, True])
        params = learner.critic_params + learner.critic2_params
        for before, param, grad in zip(captured["weights"], params, captured["grads"], strict=True):
            th.testing.assert_close(param, before, rtol=0, atol=0)
            th.testing.assert_close(param.grad, grad, rtol=0, atol=0)
        self.assertTrue(learner.mac.agent.rnn.weight_hh.grad.abs().sum() > 0)
        self.assertTrue(all(p.requires_grad for p in params))

    def test_baseline_missing_and_false_flag_complete_updates_identical(self):
        batch, mac, args, scheme = fixture()
        left = GAVDACLearner(mac, scheme, LossLog(), args)
        right = copy.deepcopy(left)
        right.args.clipped_double_critic = False
        for update in range(3):
            left.train(batch, update, update)
            right.train(batch, update, update)
        for name in ("critic", "mixer", "target_critic", "target_mixer"):
            th.testing.assert_close(getattr(left, name).state_dict(),
                                    getattr(right, name).state_dict(), rtol=0, atol=0)
        th.testing.assert_close(left.mac.agent.state_dict(), right.mac.agent.state_dict(), rtol=0, atol=0)
        th.testing.assert_close(left.agent_optimiser.state_dict(),
                                right.agent_optimiser.state_dict(), rtol=0, atol=0)

    def test_saved_learner_state_roundtrips_and_remains_cpu_float32(self):
        batch, learner = learner_fixture()
        learner.train(batch, 0, 0)
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            learner.save_models(directory)
            _, restored = learner_fixture()
            restored.load_models(directory)
        for name in ("critic", "critic2", "mixer", "mixer2", "target_critic2", "target_mixer2"):
            th.testing.assert_close(getattr(learner, name).state_dict(),
                                    getattr(restored, name).state_dict(), rtol=0, atol=0)
        self.assertEqual(learner.optimizer_steps, restored.optimizer_steps)
        for params in (restored.agent_params, restored.critic_params, restored.critic2_params):
            self.assertTrue(all(p.dtype == th.float32 and p.device.type == "cpu" for p in params))

    def test_nonfinite_update_is_fatal_without_incrementing_counter(self):
        _, learner = learner_fixture()
        with self.assertRaises(FloatingPointError):
            learner._step(th.tensor(float("nan")), learner.agent_optimiser,
                          learner.agent_params, "actor")
        self.assertEqual(learner.optimizer_steps["actor"], 0)

    def test_real_smoke_crosses_warmup_ea_and_saved_source_inference(self):
        counts, instances = {"ea": 0}, []
        actual_epoch = SSNEBase.epoch

        class CaptureLearner(ClippedDoubleGAVDACLearner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                instances.append(self)

        def evolve(self, *args, **kwargs):
            counts["ea"] += 1
            return actual_epoch(self, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            argv = ["direct_main.py", "--config", "gavdac_c2_h02"]
            for key, value in {
                "t_max": 12500, "wandb_mode": "disabled", "showcase_enabled": False,
                "local_results_path": directory, "env_args.result_dir": directory,
                "wandb_dir": f"{directory}/wandb", "unique_token": "h02mechanical",
            }.items():
                argv += ["--override", f"{key}={value}"]
            with patch.object(sys, "argv", argv), \
                 patch.object(run, "ClippedDoubleGAVDACLearner", CaptureLearner), \
                 patch.object(SSNEBase, "epoch", evolve):
                direct_main.main()
            completion = json.loads((Path(directory) / "completion_checks.json").read_text())
            self.assertGreater(counts["ea"], 0)
            self.assertGreater(completion["finite_update_checks"], 0)
            self.assertEqual(completion["optimizer_steps"]["critic2"],
                             completion["finite_update_checks"])
            self.assertTrue(all(math.isfinite(v) for v in completion["last_update_metrics"].values()))
            best = Path(directory) / "models/h02mechanical/best_policies/best_rl.json"
            metadata = json.loads(best.read_text())
            output = Path(directory) / "saved_source_validation"
            result = subprocess.run([
                "uv", "run", "--frozen", "python", "tools/evaluate_microgrid_comparison.py",
                "--checkpoint", metadata["checkpoint"], "--split", "validation",
                "--mode", "continuous", "--rows", "731", "732", "--output", str(output),
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
            self.assertTrue(summary["physical_checks_passed"])
            self.assertEqual(summary["rows"], [731, 732])
            print("H02_ENGINEERING", json.dumps(completion))


if __name__ == "__main__":
    unittest.main()
