"""NEW H13 oracles and whole-update checks against committed H12 parent."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from controllers.cqmix_controller import CQMixMAC
from ea.mod_neuro_evo import SSNE
from learners import gradient_pool
from learners.actor_norm_huber_learner import ActorNormHuberGAVDACLearner
from learners.delayed_actor_huber_learner import HuberDelayedActorGAVDACLearner
from learners.delayed_actor_learner import DelayedActorGAVDACLearner
from learners.gavdac_learner import GAVDACLearner
from modules.actor_feature_norm import LayerNormRNNAgent, actor_feature_norm_enabled
from modules.policy import RNNAgent
from runners.episode_runner import EpisodeRunner
from test_gavdac_sequence import fixture, LossLog, stepwise_sequence
from test_h11_parent import git_source, import_source, states
from test_policy_evaluation import QuietLogger
from tools.freeze_h11_protocol import entry_config
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed

PARENT = "8883790a9508cc6144c578bb3f6cf11571844961"


def child_fixture(**options):
    settings = dict(use_cuda=False, rnn_hidden_dim=48, actor_update_interval=2,
                    target_update_interval=1, scaled_huber_critic=True,
                    actor_feature_layer_norm=True)
    settings.update(options)
    batch, mac, args, scheme = fixture(**settings)
    return batch, ActorNormHuberGAVDACLearner(mac, scheme, LossLog(), args)


def explicit_sequence(agent, inputs, hidden):
    z = agent.fc1(inputs)
    mean = z.mean(-1, keepdim=True)
    variance = (z - mean).square().mean(-1, keepdim=True)
    x = torch.relu((z - mean) / (variance + 1e-5).sqrt()
                   * agent.lnorm.weight + agent.lnorm.bias)
    values = []
    for step in x.unbind(0):
        hidden = agent.rnn(step, hidden)
        values.append(hidden)
    history = torch.stack(values)
    return torch.tanh(agent.fc2(history)), history, hidden


class H13ActorNormTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        cls.directory = tempfile.TemporaryDirectory(prefix="h13_actual_parent_")
        cls.parent_controller = import_source(
            "h13_actual_parent_controller", git_source(PARENT, "src/controllers/cqmix_controller.py"),
            cls.directory.name)
        cls.parent_huber = import_source(
            "h13_actual_parent_huber", git_source(PARENT, "src/learners/delayed_actor_huber_learner.py"),
            cls.directory.name)
        cls.parent_delayed = import_source(
            "h13_actual_parent_delayed", git_source(PARENT, "src/learners/delayed_actor_learner.py"),
            cls.directory.name)
        cls.parent_root = import_source(
            "h13_actual_parent_root", git_source(PARENT, "src/learners/gavdac_learner.py"),
            cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_independent_pre_relu_oracle_forward_gradients_constant_and_large(self):
        _, learner = child_fixture()
        agent = learner.mac.agent
        for scale in (0., .5, 1e4):
            inputs = (torch.randn(6, 9, 25) * scale).requires_grad_()
            hidden = torch.zeros(9, 48, requires_grad=True)
            reference = copy.deepcopy(agent)
            other = inputs.detach().clone().requires_grad_()
            other_hidden = hidden.detach().clone().requires_grad_()
            actual = agent.forward_sequence(inputs, hidden)
            expected = explicit_sequence(reference, other, other_hidden)
            for value, target in zip(
                (actual["actions"], actual["hidden_states"], actual["hidden_state"]), expected, strict=True
            ):
                torch.testing.assert_close(value, target, rtol=1e-4, atol=1e-5)
                self.assertTrue(torch.isfinite(value).all())
                self.assertEqual((value.dtype, value.device.type), (torch.float32, "cpu"))
            actual_loss = actual["actions"].square().sum() + .03 * actual["hidden_states"].square().sum()
            expected_loss = expected[0].square().sum() + .03 * expected[1].square().sum()
            gradients = torch.autograd.grad(actual_loss, [inputs, hidden, *agent.parameters()])
            targets = torch.autograd.grad(expected_loss, [other, other_hidden, *reference.parameters()])
            torch.testing.assert_close(gradients, targets, rtol=2e-4, atol=2e-5)
        self.assertEqual(agent.lnorm.eps, 1e-5)
        torch.testing.assert_close(agent.lnorm.weight, torch.ones(48), rtol=0, atol=0)
        torch.testing.assert_close(agent.lnorm.bias, torch.zeros(48), rtol=0, atol=0)

    def test_original_initialization_draws_and_exact_parameter_delta(self):
        batch, mac, args, scheme = fixture(use_cuda=False, rnn_hidden_dim=48,
            actor_update_interval=2, target_update_interval=1, scaled_huber_critic=True)
        before = torch.get_rng_state()
        parent = self.parent_controller.CQMixMAC(scheme, batch.groups, args)
        after = torch.get_rng_state()
        args.actor_feature_layer_norm = True
        torch.set_rng_state(before)
        child = CQMixMAC(scheme, batch.groups, args)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        for key, value in parent.agent.state_dict().items():
            torch.testing.assert_close(child.agent.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in parent.parameters()), 15458)
        self.assertEqual(sum(p.numel() for p in child.parameters()), 15554)
        self.assertEqual(set(child.agent.state_dict()) - set(parent.agent.state_dict()),
                         {"lnorm.weight", "lnorm.bias"})
        rng = torch.get_rng_state()
        original = HuberDelayedActorGAVDACLearner(parent, scheme, LossLog(), args)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        new = ActorNormHuberGAVDACLearner(child, scheme, LossLog(), args)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in original.critic_params),
                         sum(p.numel() for p in new.critic_params))
        torch.testing.assert_close(original.critic.state_dict(), new.critic.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(original.mixer.state_dict(), new.mixer.state_dict(), rtol=0, atol=0)

    def test_actual_parent_missing_false_complete_multistep_updates_and_rng(self):
        references = (self.parent_root.GAVDACLearner, self.parent_delayed.DelayedActorGAVDACLearner,
                      self.parent_huber.HuberDelayedActorGAVDACLearner)
        current = (GAVDACLearner, DelayedActorGAVDACLearner, HuberDelayedActorGAVDACLearner)
        for kind in range(3):
            for flag in (None, False):
                for optimizer, mode in (("adam", "soft"), ("rmsprop", "hard")):
                    options = dict(use_cuda=False, rnn_hidden_dim=48, batch_size=5, length=7,
                                   optimizer=optimizer, target_update_mode=mode)
                    if kind:
                        options.update(actor_update_interval=2, target_update_interval=1)
                    if kind == 2:
                        options["scaled_huber_critic"] = True
                    if flag is not None:
                        options["actor_feature_layer_norm"] = flag
                    batch, mac, args, scheme = fixture(**options)
                    self.assertIs(type(mac.agent), RNNAgent)
                    rng = torch.get_rng_state()
                    ref_mac = self.parent_controller.CQMixMAC(scheme, batch.groups, copy.deepcopy(args))
                    after = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    new_mac = CQMixMAC(scheme, batch.groups, args)
                    self.assertTrue(torch.equal(after, torch.get_rng_state()))
                    rng = torch.get_rng_state()
                    ref = references[kind](ref_mac, scheme, LossLog(), copy.deepcopy(args))
                    after = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    candidate = current[kind](new_mac, scheme, LossLog(), args)
                    self.assertTrue(torch.equal(after, torch.get_rng_state()))
                    for step in range(1, 8):
                        if step == 4:
                            with torch.no_grad():
                                for learner in (ref, candidate):
                                    for p in learner.mac.parameters():
                                        p.add_(.0007)
                        rng = torch.get_rng_state()
                        ref.train(batch, step, step)
                        after = torch.get_rng_state()
                        torch.set_rng_state(rng)
                        candidate.train(batch, step, step)
                        self.assertTrue(torch.equal(after, torch.get_rng_state()))
                        torch.testing.assert_close(states(candidate), states(ref), rtol=0, atol=0)

    def test_timestep_sequence_outputs_gradients_full_update_and_past_history(self):
        batch, learner = child_fixture(batch_size=5, length=7)
        other = copy.deepcopy(learner.mac)
        reference = copy.deepcopy(learner)
        sequence = learner.mac.forward_sequence(batch)
        stepwise = stepwise_sequence(other, batch)
        torch.testing.assert_close(sequence, stepwise, rtol=1e-5, atol=1e-6)
        left = torch.autograd.grad(sequence["actions"].square().sum(), list(learner.mac.parameters()))
        right = torch.autograd.grad(stepwise["actions"].square().sum(), list(other.parameters()))
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
        def slow(ep_batch, end=None):
            return stepwise_sequence(reference.mac, ep_batch, end)
        def slow_target(ep_batch, end=None):
            return stepwise_sequence(reference.target_mac, ep_batch, end)
        with patch.object(reference.mac, "forward_sequence", side_effect=slow), \
             patch.object(reference.target_mac, "forward_sequence", side_effect=slow_target):
            for step in range(1, 5):
                reference.train(batch, step, step)
                learner.train(batch, step, step)
                torch.testing.assert_close(states(reference), states(learner), rtol=1e-4, atol=1e-5)
        inputs = torch.randn(6, 3, 25, requires_grad=True)
        value = learner.mac.agent.forward_sequence(inputs, torch.zeros(3, 48))
        gradient, = torch.autograd.grad(value["actions"][-1].square().sum(), inputs)
        self.assertGreater(float(gradient[0].abs().sum()), 0)

    def test_per_example_independence_train_eval_and_unrelated_sentinels(self):
        _, learner = child_fixture()
        agent = learner.mac.agent
        x = torch.randn(6, 9, 25, requires_grad=True)
        h = torch.randn(9, 48)
        agent.train()
        train = agent.forward_sequence(x, h)
        agent.eval()
        test = agent.forward_sequence(x, h)
        torch.testing.assert_close(train, test, rtol=0, atol=0)
        changed = x.detach().clone()
        changed[:, 1:] = 1e4
        isolated = agent.forward_sequence(changed, h)
        torch.testing.assert_close(test["actions"][:, 0], isolated["actions"][:, 0], rtol=0, atol=0)
        grad, = torch.autograd.grad(test["actions"][:, 0].sum(), x)
        self.assertEqual(int(torch.count_nonzero(grad[:, 1:])), 0)
        self.assertFalse(any("running" in key for key in agent.state_dict()))
        z = torch.randn(3, 48)
        first = agent.lnorm(z)
        extra = agent.lnorm(torch.cat((z, torch.full((2, 48), 1e6))))
        torch.testing.assert_close(first, extra[:3], rtol=0, atol=0)

    def test_live_context_action_norm_recurrent_gradients_and_critic_immutable(self):
        batch, learner = child_fixture(batch_size=5, length=7)
        policy, inputs, _ = learner._critic_loss(batch)
        learner.critic_optimiser.zero_grad(set_to_none=True)
        policy["actions"].retain_grad()
        policy["hidden_states"].retain_grad()
        before = [p.detach().clone() for p in learner.critic_params]
        learner._set_critic_grad(False)
        try:
            learner._actor_loss(batch, policy, inputs).backward()
        finally:
            learner._set_critic_grad(True)
        for name in ("actions", "hidden_states"):
            self.assertGreater(float(policy[name].grad.abs().sum()), 0)
        for name in ("lnorm.weight", "lnorm.bias", "rnn.weight_hh"):
            param = dict(learner.mac.agent.named_parameters())[name]
            self.assertGreater(float(param.grad.abs().sum()), 0)
        for param, old in zip(learner.critic_params, before, strict=True):
            self.assertIsNone(param.grad)
            torch.testing.assert_close(param, old, rtol=0, atol=0)

    def test_parent_loss_mask_global_denominator_and_target_detach_unchanged(self):
        batch, learner = child_fixture(batch_size=5, length=7)
        batch["reward"].mul_(12)
        parent = HuberDelayedActorGAVDACLearner(
            copy.deepcopy(learner.mac), learner.scheme, LossLog(), copy.deepcopy(learner.args))
        parent.critic.load_state_dict(learner.critic.state_dict())
        parent.target_critic.load_state_dict(learner.target_critic.state_dict())
        parent.mixer.load_state_dict(learner.mixer.state_dict())
        parent.target_mixer.load_state_dict(learner.target_mixer.state_dict())
        self.assertIs(ActorNormHuberGAVDACLearner._critic_loss, HuberDelayedActorGAVDACLearner._critic_loss)
        self.assertIs(ActorNormHuberGAVDACLearner._actor_loss, HuberDelayedActorGAVDACLearner._actor_loss)
        self.assertIs(ActorNormHuberGAVDACLearner.train, HuberDelayedActorGAVDACLearner.train)
        _, _, a = learner._critic_loss(batch, denominator=100.)
        _, _, b = parent._critic_loss(batch, denominator=100.)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.backward()
        self.assertTrue(all(p.grad is None for p in learner.target_mac.parameters()))
        self.assertTrue(all(p.grad is None for p in learner.target_critic.parameters()))

    def test_cadence_order_posthooks_norm_optimizer_and_all_target_affines(self):
        for mode in ("soft", "hard"):
            batch, learner = child_fixture(batch_size=5, length=7, target_update_mode=mode)
            norm_before = learner.mac.agent.lnorm.weight.detach().clone()
            events = []
            critic_hook = learner.critic_optimiser.register_step_post_hook(
                lambda *_: events.append("critic"))
            actor_hook = learner.agent_optimiser.register_step_post_hook(
                lambda *_: events.append("actor"))
            original_target = learner._target_phase
            def target_phase():
                events.append("target")
                original_target()
            try:
                with patch.object(learner, "_target_phase", side_effect=target_phase):
                    for step in range(1, 8):
                        if step == 4:
                            from run import rl_to_evo
                            incoming = copy.deepcopy(learner.mac)
                            with torch.no_grad():
                                for p in incoming.parameters():
                                    p.add_(.0007)
                            rl_to_evo(incoming, learner.mac)
                        modules = gradient_pool._modules(learner)
                        before = {n: [p.detach().clone() for p in modules["target_" + n].parameters()]
                                  for n in ("actor", "critic", "mixer")}
                        events.clear()
                        learner.train(batch, step, step)
                        self.assertEqual(events, ["critic", "actor", "target"] if step % 2 == 0
                                         else ["critic", "target"])
                        for name in before:
                            for old, p, t in zip(before[name], modules[name].parameters(),
                                                 modules["target_" + name].parameters(), strict=True):
                                expected = old.lerp(p.detach(), .002) if mode == "soft" else p.detach()
                                torch.testing.assert_close(t, expected, rtol=0, atol=0)
                                self.assertIsNone(t.grad)
                        self.assertEqual((learner.critic_steps, learner.actor_steps,
                                          learner.target_updates, learner.finite_checks),
                                         (step, step // 2, step, step))
                        self.assertEqual(learner.optimizer_finite_checks,
                                         {"critic": step, "actor": step // 2})
                        if step % 2:
                            self.assertFalse(learner.mac.hidden_states.requires_grad)
            finally:
                critic_hook.remove()
                actor_hook.remove()
            self.assertFalse(torch.equal(norm_before, learner.mac.agent.lnorm.weight))

    def test_EA_complete_clone_transfer_and_existing_norm_mutation_skip(self):
        _, learner = child_fixture()
        first, other = copy.deepcopy(learner.mac), copy.deepcopy(learner.mac)
        with torch.no_grad():
            first.agent.lnorm.weight.add_(.123)
            first.agent.lnorm.bias.add_(.456)
        args = SimpleNamespace(prob_reset_and_sup=.05, frac=1., pop_size=5,
                               elite_fraction=.2, mutation_prob=.9, crossover_prob=0.)
        evolution = SSNE(args)
        evolution.clone(first, other, 0)
        torch.testing.assert_close(first.agent.state_dict(), other.agent.state_dict(), rtol=0, atol=0)
        from run import rl_to_evo
        rl_to_evo(first, learner.mac)
        torch.testing.assert_close(first.agent.state_dict(), learner.mac.agent.state_dict(), rtol=0, atol=0)
        before = copy.deepcopy(other.agent.state_dict())
        evolution.mutate_inplace(other, 0, agent_level=True)
        for key in ("lnorm.weight", "lnorm.bias"):
            torch.testing.assert_close(before[key], other.agent.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(before["fc1.weight"], other.agent.fc1.weight))
        evolution.mutate_inplace(other, 0, agent_level=False)
        for key in ("lnorm.weight", "lnorm.bias"):
            torch.testing.assert_close(before[key], other.agent.state_dict()[key], rtol=0, atol=0)

    def test_real_two_workers_uneven_varying_participation_full_update_transport(self):
        batch, serial = child_fixture(batch_size=5, length=7)
        args = copy.deepcopy(serial.args)
        args.learner_workers = 2
        rng = torch.get_rng_state()
        parallel = ActorNormHuberGAVDACLearner(
            copy.deepcopy(serial.mac), serial.scheme, LossLog(), args)
        torch.set_rng_state(rng)
        parallel.critic.load_state_dict(serial.critic.state_dict())
        parallel.target_critic.load_state_dict(serial.target_critic.state_dict())
        parallel.mixer.load_state_dict(serial.mixer.state_dict())
        parallel.target_mixer.load_state_dict(serial.target_mixer.state_dict())
        try:
            for step, (size, length) in enumerate(((5, 7), (1, 4), (3, 6), (2, 5), (5, 7), (1, 4)), 1):
                batch, _, _, _ = fixture(use_cuda=False, batch_size=size, length=length)
                batch["reward"].mul_(12)
                serial.train(batch, step, step)
                parallel.train(batch, step, step)
                torch.testing.assert_close(states(serial), states(parallel), rtol=1e-4, atol=1e-5)
                pool = parallel.gradient_pool
                self.assertEqual(len(pool.worker_pids), 2)
                self.assertIsNone(pool.pending_batch)
                self.assertEqual(pool.last_phase_pids["actor" if step % 2 == 0 else "discard"],
                                 pool.last_phase_pids["critic"])
                self.assertEqual(len(pool.last_phase_pids["critic"]), min(2, size))
        finally:
            serial.close()
            parallel.close()

    def test_explicit_worker_construction_all_norm_keys_and_discard_round(self):
        batch, learner = child_fixture()
        class Barrier:
            def wait(self, timeout):
                pass
        with patch.multiple(gradient_pool, _learner=None, _pending=None,
                            _scheme=None, _groups=None, _transport=None), \
             patch.object(torch, "set_num_interop_threads"):
            gradient_pool._initialize(copy.deepcopy(learner.args), learner.scheme, Barrier())
            self.assertIs(type(gradient_pool._learner), ActorNormHuberGAVDACLearner)
            self.assertIs(type(gradient_pool._learner.mac.agent), LayerNormRNNAgent)
            weights = {n: {k: v.numpy().copy() for k, v in m.state_dict().items()}
                       for n, m in gradient_pool._modules(learner).items()}
            data = {k: v.numpy().copy() for k, v in batch.data.transition_data.items()}
            mask = batch["filled"][:, :-1].float().clone()
            mask[:, 1:] *= 1 - batch["terminated"][:, :-2].float()
            gradient_pool._gradients("critic", weights, data, float(mask.sum()), 1., 1)
            with self.assertRaisesRegex(RuntimeError, "matching cached"):
                gradient_pool._discard_actor(2)
            gradient_pool._discard_actor(1)
            self.assertIsNone(gradient_pool._pending)
            self.assertFalse(gradient_pool._learner.mac.hidden_states.requires_grad)

    def test_invalid_flags_parent_width_wrong_actor_and_runtime_finite_guards(self):
        for value in (None, 0, 1, 1., "true"):
            with self.subTest(flag=value), self.assertRaises(ValueError):
                actor_feature_norm_enabled(SimpleNamespace(actor_feature_layer_norm=value))
        _, learner = child_fixture()
        for name, value in (("actor_update_interval", 1), ("actor_update_interval", True),
                            ("target_update_interval", 2), ("scaled_huber_critic", False),
                            ("rnn_hidden_dim", 16), ("use_cuda", True)):
            args = copy.deepcopy(learner.args)
            setattr(args, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                actor_feature_norm_enabled(args)
        bad_mac = copy.deepcopy(learner.mac)
        bad_mac.agent = RNNAgent(25, learner.args)
        with self.assertRaisesRegex(ValueError, "explicit normalized"):
            ActorNormHuberGAVDACLearner(bad_mac, learner.scheme, LossLog(), learner.args)
        with torch.no_grad():
            learner.mac.agent.lnorm.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            learner.agent_optimiser.step()
        batch, learner = child_fixture()
        with torch.no_grad():
            learner.target_mac.agent.lnorm.bias.fill_(float("inf"))
        with self.assertRaises(ValueError):
            learner.train(batch, 1, 1)

    def test_diagnostics_saved_norm_actor_optimizer_and_configuration_rejection(self):
        batch, learner = child_fixture()
        for step in range(1, 3):
            learner.train(batch, step, step)
        with tempfile.TemporaryDirectory(prefix="h13diagnostic_") as directory:
            learner.save_models(directory)
            saved = json.loads((Path(directory) / "h13_diagnostics.json").read_text())
            self.assertEqual(saved["lineage_role"], "H13_child_of_retained_H12")
            self.assertEqual(saved["optimizer_post_step_finite_checks"], {"critic": 2, "actor": 1})
            self.assertFalse(saved["exact_resume_supported"])
            _, restored = child_fixture()
            restored.load_models(directory)
            torch.testing.assert_close(restored.mac.agent.state_dict(), learner.mac.agent.state_dict(),
                                       rtol=0, atol=0)
            torch.testing.assert_close(restored.agent_optimiser.state_dict(),
                                       learner.agent_optimiser.state_dict(), rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                learner.write_diagnostics(Path(directory) / "h13_diagnostics.json")
            saved["actor_feature_norm_eps"] = 1e-4
            (Path(directory) / "h13_diagnostics.json").write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                restored.load_models(directory)

    def test_strict_saved_source_validation_731_732_physical_Actor_norm_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results", prefix="h13strict_") as directory:
            config = entry_config("gavdac_c2_h13", {"rollout_workers": 2, "learner_workers": 2})
            config.update(device="cpu", local_results_path=directory)
            config["env_args"].update(seed=config["seed"], result_dir=directory, run_id="h13strict")
            args = SimpleNamespace(**copy.deepcopy(config))
            runner = EpisodeRunner(args, QuietLogger())
            info = runner.get_env_info()
            for key in ("n_agents", "n_actions", "state_shape", "obs_shape", "action_spaces"):
                setattr(args, key, info[key])
            scheme = {
                "state": {"vshape": info["state_shape"]},
                "obs": {"vshape": info["obs_shape"], "group": "agents"},
                "actions": {"vshape": (info["n_actions"],), "group": "agents", "dtype": torch.float},
                "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8}}
            mac = CQMixMAC(scheme, {"agents": 3}, args)
            runner.setup(scheme, {"agents": 3}, {}, mac)
            learner = ActorNormHuberGAVDACLearner(mac, scheme, LossLog(), args)
            self.assertEqual(sum(p.numel() for p in learner.agent_params), 15554)
            self.assertEqual(sum(p.numel() for p in learner.critic_params), 65915)
            try:
                evaluation = evaluate_fixed(runner, mac, [731, 732], .35, "validation", "rl", 24)
                store = BestPolicyStore(Path(directory) / "best", config)
                checkpoint = store.consider(mac, "rl", 24, evaluation, runner.env)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
            finally:
                runner.close_env()
                learner.close()
            output = Path(directory) / "inference"
            result = subprocess.run(
                ["uv", "run", "--frozen", "python", "tools/evaluate_microgrid_comparison.py",
                 "--checkpoint", str(checkpoint), "--split", "validation", "--mode", "continuous",
                 "--rows", "731", "732", "--output", str(output)],
                cwd=ROOT, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["physical_checks_passed"])
            self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
            for name, raw in metadata["source_sha256"].items():
                self.assertEqual(hashlib.sha256((output / "runtime" / name).read_bytes()).hexdigest(), raw)
            self.assertIn("modules/actor_feature_norm.py", metadata["source_sha256"])
            self.assertEqual(summary["data_sha256"], metadata["data_sha256"])


if __name__ == "__main__":
    unittest.main()
