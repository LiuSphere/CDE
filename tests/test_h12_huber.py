"""NEW H12 loss-only and complete-update checks against actual parent Git code."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from controllers.cqmix_controller import CQMixMAC
from learners import gradient_pool
from learners.delayed_actor_huber_learner import HuberDelayedActorGAVDACLearner, huber_enabled
from learners.delayed_actor_learner import DelayedActorGAVDACLearner
from learners.gavdac_learner import GAVDACLearner
from runners.episode_runner import EpisodeRunner
from test_gavdac_sequence import fixture, LossLog
from test_h11_parent import git_source, import_source, states
from test_policy_evaluation import QuietLogger
from tools.freeze_h11_protocol import entry_config
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed

PARENT = "4e3aa00eacf814e3d328be781e4af8632ec0488f"


class H12HuberTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        cls.saved = tempfile.TemporaryDirectory(prefix="h12_reference_")
        cls.original_root = import_source("h12_original_root", git_source(
            PARENT, "src/learners/gavdac_learner.py"), cls.saved.name)
        cls.original_parent = import_source("h12_original_parent", git_source(
            PARENT, "src/learners/delayed_actor_learner.py"), cls.saved.name)

    @classmethod
    def tearDownClass(cls):
        cls.saved.cleanup()

    def child(self, **options):
        batch, mac, args, scheme = fixture(
            use_cuda=False, actor_update_interval=2, target_update_interval=1,
            scaled_huber_critic=True, **options)
        return batch, HuberDelayedActorGAVDACLearner(mac, scheme, LossLog(), args)

    def test_analytic_loss_gradient_unit_boundaries_and_global_denominator(self):
        errors = torch.tensor([-8., -1., -.5, 0., .5, 1., 8.], requires_grad=True)
        loss = HuberDelayedActorGAVDACLearner._td_error_loss(None, errors, 9.)
        expected = torch.where(errors.abs() <= 1, errors.square(), 2 * errors.abs() - 1).sum() / 9
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        grad, = torch.autograd.grad(loss, errors)
        torch.testing.assert_close(grad, 2 * errors.detach().clamp(-1, 1) / 9, rtol=0, atol=0)
        small = torch.tensor([-.5, 0., .5], requires_grad=True)
        torch.testing.assert_close(
            HuberDelayedActorGAVDACLearner._td_error_loss(None, small, 3),
            GAVDACLearner._td_error_loss(None, small, 3), rtol=0, atol=0)

    def test_missing_false_root_and_parent_multistep_optimizer_target_RNG_exact(self):
        for delayed in (False, True):
            for explicit_false in (False, True):
                for optimizer, mode in (("adam", "soft"), ("rmsprop", "hard")):
                    batch, mac, args, scheme = fixture(
                        use_cuda=False, batch_size=5, length=7, optimizer=optimizer,
                        target_update_mode=mode)
                    if delayed:
                        args.actor_update_interval, args.target_update_interval = 2, 1
                    if explicit_false:
                        args.scaled_huber_critic = False
                    rng = torch.get_rng_state()
                    reference_type = (self.original_parent.DelayedActorGAVDACLearner if delayed
                                      else self.original_root.GAVDACLearner)
                    reference = reference_type(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
                    reference._critic_loss = MethodType(
                        self.original_root.GAVDACLearner._critic_loss, reference)
                    after = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    candidate_type = DelayedActorGAVDACLearner if delayed else GAVDACLearner
                    candidate = candidate_type(mac, scheme, LossLog(), args)
                    self.assertTrue(torch.equal(after, torch.get_rng_state()))
                    for step in range(1, 8):
                        if step == 4:
                            with torch.no_grad():
                                for learner in (reference, candidate):
                                    for p in learner.mac.parameters():
                                        p.add_(.0007)
                        rng = torch.get_rng_state()
                        reference.train(batch, step, step)
                        after = torch.get_rng_state()
                        torch.set_rng_state(rng)
                        candidate.train(batch, step, step)
                        self.assertTrue(torch.equal(after, torch.get_rng_state()))
                        torch.testing.assert_close(states(reference), states(candidate), rtol=0, atol=0)
                        if delayed:
                            self.assertEqual(candidate.actor_steps, step // 2)
                            self.assertEqual(candidate.target_updates, step)

    def test_child_initial_draws_network_counts_and_policy_values_equal_parent(self):
        batch, mac, args, scheme = fixture(
            use_cuda=False, actor_update_interval=2, target_update_interval=1)
        rng = torch.get_rng_state()
        parent = DelayedActorGAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
        after = torch.get_rng_state()
        args.scaled_huber_critic = True
        torch.set_rng_state(rng)
        child = HuberDelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        for key, module in gradient_pool._modules(parent).items():
            torch.testing.assert_close(module.state_dict(), gradient_pool._modules(child)[key].state_dict(),
                                       rtol=0, atol=0)
        torch.testing.assert_close(parent.mac.forward_sequence(batch), child.mac.forward_sequence(batch),
                                   rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in parent.critic_params),
                         sum(p.numel() for p in child.critic_params))

    def test_original_target_mask_termination_and_global_reduction(self):
        batch, learner = self.child(batch_size=5, length=7)
        batch["reward"].mul_(16)
        mixed = []
        original_mix = learner._mix_q_values
        def mix(*args, **kwargs):
            value = original_mix(*args, **kwargs)
            mixed.append(value)
            return value
        with patch.object(learner, "_mix_q_values", side_effect=mix), \
             patch.object(learner, "_td_error_loss", wraps=learner._td_error_loss) as reduce:
            _, _, loss = learner._critic_loss(batch, denominator=100.)
        self.assertEqual(len(mixed), 2)
        taken, target = mixed
        self.assertFalse(target.requires_grad)
        terminal = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float().clone()
        mask[:, 1:] *= 1 - terminal[:, :-1]
        errors = (batch["reward"][:, :-1].expand_as(target) +
                  learner.args.gamma * (1 - terminal.expand_as(target)) * target -
                  taken) * mask.expand_as(taken)
        torch.testing.assert_close(reduce.call_args.args[0], errors, rtol=0, atol=0)
        self.assertEqual(reduce.call_args.args[1], 100.)
        self.assertTrue((errors.abs() > 1).any())
        torch.testing.assert_close(loss, 2 * F.huber_loss(
            errors, torch.zeros_like(errors), reduction="sum", delta=1.) / 100., rtol=0, atol=0)

    def test_live_actor_action_context_recurrence_and_critic_immutability(self):
        batch, learner = self.child()
        policy, inputs, _ = learner._critic_loss(batch)
        policy["actions"].retain_grad()
        policy["hidden_states"].retain_grad()
        before = [p.detach().clone() for p in learner.critic_params]
        learner._set_critic_grad(False)
        try:
            learner._actor_loss(batch, policy, inputs).backward()
        finally:
            learner._set_critic_grad(True)
        self.assertGreater(float(policy["actions"].grad.abs().sum()), 0)
        self.assertGreater(float(policy["hidden_states"].grad.abs().sum()), 0)
        self.assertTrue(any("weight_hh" in n and p.grad is not None and bool(p.grad.abs().sum())
                            for n, p in learner.mac.agent.named_parameters()))
        for p, old in zip(learner.critic_params, before, strict=True):
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p, old, rtol=0, atol=0)

    def test_global_cadence_every_call_original_tau_and_EA_transfer(self):
        batch, learner = self.child(batch_size=5, length=7)
        batch["reward"].mul_(12)
        for step in range(1, 8):
            if step == 4:
                with torch.no_grad():
                    for p in learner.mac.parameters():
                        p.add_(.0007)
            modules = gradient_pool._modules(learner)
            before = {n: [p.detach().clone() for p in modules["target_" + n].parameters()]
                      for n in ("actor", "critic", "mixer")}
            learner.train(batch, step, step)
            for name in before:
                for old, source, target in zip(before[name], modules[name].parameters(),
                                               modules["target_" + name].parameters(), strict=True):
                    torch.testing.assert_close(target, old.lerp(source.detach(), .002), rtol=0, atol=0)
                    self.assertIsNone(target.grad)
            self.assertEqual((learner.critic_steps, learner.actor_steps, learner.target_updates,
                              learner.finite_checks), (step, step // 2, step, step))
            self.assertEqual(learner.optimizer_finite_checks, {"critic": step, "actor": step // 2})
            if step % 2:
                self.assertFalse(learner.mac.hidden_states.requires_grad)

    def test_real_two_workers_varying_participation_full_updates(self):
        batch, mac, args, scheme = fixture(
            use_cuda=False, actor_update_interval=2, target_update_interval=1,
            scaled_huber_critic=True, batch_size=5, length=7)
        rng = torch.get_rng_state()
        serial = HuberDelayedActorGAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
        torch.set_rng_state(rng)
        args.learner_workers = 2
        parallel = HuberDelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
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

    def test_explicit_worker_class_and_matching_discard_no_graph_leak(self):
        batch, learner = self.child()
        class Barrier:
            def wait(self, timeout):
                pass
        with patch.multiple(gradient_pool, _learner=None, _pending=None,
                            _scheme=None, _groups=None, _transport=None), \
             patch.object(torch, "set_num_interop_threads"):
            gradient_pool._initialize(copy.deepcopy(learner.args), learner.scheme, Barrier())
            self.assertIsInstance(gradient_pool._learner, HuberDelayedActorGAVDACLearner)
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

    def test_invalid_flag_intervals_cpu_dtype_finite_and_wrong_classes(self):
        for value in (None, 1, 0, 1., "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                huber_enabled(SimpleNamespace(scaled_huber_critic=value))
        for key, value in (("actor_update_interval", 1), ("actor_update_interval", True),
                           ("target_update_interval", 2), ("use_cuda", True)):
            args = SimpleNamespace(scaled_huber_critic=True, actor_update_interval=2,
                                   target_update_interval=1, use_cuda=False)
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                huber_enabled(args)
        batch, mac, args, scheme = fixture(
            use_cuda=False, actor_update_interval=2, target_update_interval=1, scaled_huber_critic=True)
        for klass in (GAVDACLearner, DelayedActorGAVDACLearner):
            with self.subTest(klass=klass), self.assertRaisesRegex(ValueError, "explicit H12"):
                klass(mac, scheme, LossLog(), args)
        for tensor, denominator in ((torch.tensor([float("nan")]), 1), (torch.ones(1, dtype=torch.float64), 1),
                                    (torch.ones(1), 0), (torch.ones(1), float("inf"))):
            with self.assertRaises(ValueError):
                HuberDelayedActorGAVDACLearner._td_error_loss(None, tensor, denominator)

    def test_diagnostics_model_optimizer_load_and_actual_posthook_guard(self):
        batch, learner = self.child()
        learner.train(batch, 1, 1)
        learner.train(batch, 2, 2)
        with tempfile.TemporaryDirectory(prefix="h12_load_") as directory:
            learner.save_models(directory)
            path = Path(directory) / "h12_diagnostics.json"
            value = json.loads(path.read_text())
            self.assertEqual(value["lineage_role"], "H12_child_of_reconstructed_H03")
            self.assertEqual(value["optimizer_post_step_finite_checks"], {"critic": 2, "actor": 1})
            self.assertEqual(value["target_updates"], 2)
            self.assertFalse(value["exact_resume_supported"])
            _, loaded = self.child()
            loaded.load_models(directory)
            torch.testing.assert_close(learner.mac.agent.state_dict(), loaded.mac.agent.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(learner.agent_optimiser.state_dict(),
                                       loaded.agent_optimiser.state_dict(), rtol=0, atol=0)
            value["target_update_interval"] = 2
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "configuration"):
                loaded.load_models(directory)
        with torch.no_grad():
            learner.critic_params[0].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            learner.critic_optimiser.step()

    def test_actual_entry_only_changes_name_and_Huber_flag(self):
        workers = {"rollout_workers": 2, "learner_workers": 2}
        parent, child = [entry_config(n, workers) for n in ("gavdac_c2_h11_parent", "gavdac_c2_h12")]
        self.assertEqual(child, {**parent, "name": "gavdac_c2_h12", "scaled_huber_critic": True})
        from run import args_sanity_check
        self.assertEqual(args_sanity_check(copy.deepcopy(child), SimpleNamespace(warning=lambda *_: None)), child)

    def test_H12_strict_saved_source_validation_physical_Actor_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results", prefix="h12strict_") as directory:
            config = entry_config("gavdac_c2_h12", {"rollout_workers": 2, "learner_workers": 2})
            config.update(device="cpu", local_results_path=directory)
            config["env_args"].update(seed=config["seed"], result_dir=directory, run_id="h12strict")
            args = SimpleNamespace(**copy.deepcopy(config))
            runner = EpisodeRunner(args, QuietLogger())
            info = runner.get_env_info()
            for k in ("n_agents", "n_actions", "state_shape", "obs_shape", "action_spaces"):
                setattr(args, k, info[k])
            scheme = {
                "state": {"vshape": info["state_shape"]},
                "obs": {"vshape": info["obs_shape"], "group": "agents"},
                "actions": {"vshape": (info["n_actions"],), "group": "agents", "dtype": torch.float},
                "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8}}
            mac = CQMixMAC(scheme, {"agents": 3}, args)
            runner.setup(scheme, {"agents": 3}, {}, mac)
            learner = HuberDelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
            self.assertEqual(sum(p.numel() for p in learner.agent_params), 15458)
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
            self.assertEqual(summary["data_sha256"], metadata["data_sha256"])


if __name__ == "__main__":
    unittest.main()
