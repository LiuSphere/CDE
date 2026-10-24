"""H08 initialization, unchanged update rules, worker and saved-source gates."""

import copy
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import direct_main
import run
from learners import gradient_pool
from learners.gavdac_learner import GAVDACLearner
from learners.small_init_learner import (
    SmallInitGAVDACCritic, SmallInitGAVDACLearner, small_init_enabled,
)
from modules.critic import GAVDACCritic
from test_gavdac_sequence import LossLog, fixture, reference_update
from utils.comparison_evaluation import make_evaluation_runner
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed
from experiment_slot import validate_job, validate_worker_options


def entry(name):
    with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
         patch.object(direct_main, "run") as execute:
        direct_main.main()
    return execute.call_args.args[0]


def make_learner(candidate=True, **options):
    options.setdefault("rnn_hidden_dim", 48)
    if candidate:
        options["critic_small_output_init"] = True
    batch, mac, args, scheme = fixture(**options)
    cls = SmallInitGAVDACLearner if candidate else GAVDACLearner
    return batch, cls(mac, scheme, LossLog(), args)


def modules(learner):
    return gradient_pool._modules(learner)


class H08Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"

    def assert_learners_close(self, left, right, rtol=0, atol=0):
        for name in modules(left):
            torch.testing.assert_close(
                modules(left)[name].state_dict(), modules(right)[name].state_dict(),
                rtol=rtol, atol=atol,
            )
        for name in ("agent_optimiser", "critic_optimiser"):
            torch.testing.assert_close(
                getattr(left, name).state_dict(), getattr(right, name).state_dict(),
                rtol=rtol, atol=atol,
            )
        torch.testing.assert_close(
            [p.grad for p in left.agent_params + left.critic_params],
            [p.grad for p in right.agent_params + right.critic_params],
            rtol=rtol, atol=atol,
        )

    def test_initialization_exact_formula_rng_and_other_networks(self):
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        _, baseline = make_learner(False)
        rng = torch.get_rng_state().clone()
        _, candidate = make_learner()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        other_numpy_rng = np.random.get_state()
        for a, b in zip(numpy_rng, other_numpy_rng, strict=True):
            np.testing.assert_equal(a, b)
        factor = 0.003 * math.sqrt(48)
        for name in modules(baseline):
            left, right = modules(baseline)[name], modules(candidate)[name]
            self.assertEqual(left.state_dict().keys(), right.state_dict().keys())
            self.assertEqual(sum(p.numel() for p in left.parameters()),
                             sum(p.numel() for p in right.parameters()))
            for key, actual in right.state_dict().items():
                expected = left.state_dict()[key]
                if name in ("critic", "target_critic") and key.startswith("fc3."):
                    expected = expected * factor
                    self.assertLessEqual(actual.abs().max().item(), 0.003)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device.type, "cpu")
                self.assertTrue(torch.isfinite(actual).all())
        self.assertIsInstance(candidate.critic, SmallInitGAVDACCritic)
        self.assertEqual(candidate.diagnostics()["finite_updates"], 0)

    def test_local_forward_action_gradient_shapes_and_independence(self):
        batch, mac, args, scheme = fixture(rnn_hidden_dim=48)
        rng = torch.get_rng_state()
        baseline = GAVDACCritic(scheme, args)
        torch.set_rng_state(rng)
        candidate = SmallInitGAVDACCritic(scheme, args)
        factor = 0.003 * math.sqrt(48)
        for shape in ((1, 1, 3, 2), (4, 5, 3, 2)):
            actions = torch.linspace(-0.4, 0.5, math.prod(shape)).reshape(shape)
            actions.requires_grad_()
            obs = torch.ones(*shape[:-1], 20)
            expected, _ = baseline(obs.reshape(-1, 20), actions)
            actual, _ = candidate(obs.reshape(-1, 20), actions)
            self.assertEqual(actual.shape, (math.prod(shape[:-1]), 1))
            torch.testing.assert_close(actual, expected * factor, rtol=2e-5, atol=1e-9)
            expected_grad = torch.autograd.grad(expected.sum(), actions)[0]
            actual_grad = torch.autograd.grad(actual.sum(), actions)[0]
            self.assertGreater(actual_grad.abs().sum().item(), 0)
            torch.testing.assert_close(actual_grad, expected_grad * factor, rtol=2e-5, atol=1e-9)
            single, _ = candidate(obs.reshape(-1, 20)[:1], actions.reshape(-1, 2)[:1])
            torch.testing.assert_close(single, actual[:1], rtol=2e-5, atol=1e-9)
        with self.assertRaises(RuntimeError):
            candidate(torch.zeros(3, 19), torch.zeros(3, 2))

    def test_missing_false_multistep_full_update_and_rng_exact(self):
        batch, missing = make_learner(False)
        _, explicit = make_learner(False, critic_small_output_init=False)
        self.assertFalse(small_init_enabled(missing.args))
        self.assertFalse(small_init_enabled(explicit.args))
        self.assert_learners_close(missing, explicit)
        for step in range(3):
            rng = torch.get_rng_state().clone()
            missing.train(batch, step, step)
            expected_rng = torch.get_rng_state().clone()
            torch.set_rng_state(rng)
            explicit.train(batch, step, step)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assert_learners_close(missing, explicit)
            self.assertEqual(missing.logger.values, explicit.logger.values)

    def test_entry_only_name_and_flag_supervisor_and_invalid_flags(self):
        baseline, candidate = entry("gavdac_ea_main"), entry("gavdac_c2_h08")
        self.assertEqual({k for k in baseline.keys() | candidate.keys()
                          if baseline.get(k) != candidate.get(k)},
                         {"name", "critic_small_output_init"})
        validate_job("gavdac_c2_h08", 2, "h08fixture", 0, 1)
        validate_worker_options("gavdac_c2_h08", 1, 2, 2)
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                run.args_sanity_check({**candidate, "critic_small_output_init": value}, Mock())
        with self.assertRaisesRegex(ValueError, "requires"):
            batch, mac, args, scheme = fixture()
            SmallInitGAVDACLearner(mac, scheme, LossLog(), args)

    def test_runner_explicit_class_dispatch_without_training(self):
        class Selected(Exception):
            pass
        for flag, ea, selected in ((None, True, "EA_GAVDACLearner"),
                                   (False, False, "GAVDACLearner"),
                                   (False, True, "EA_GAVDACLearner"),
                                   (True, True, "SmallInitGAVDACLearner")):
            config = entry("gavdac_ea_main")
            config.update(EA=ea, save_model=False, buffer_size=2)
            if flag is not None:
                config["critic_small_output_init"] = flag
            with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
                config["env_args"].update(result_dir=directory, run_id="h08_dispatch")
                runner, _ = make_evaluation_runner(config)
                try:
                    with ExitStack() as resources:
                        mocks = {name: resources.enter_context(
                            patch.object(run, name, side_effect=Selected))
                            for name in ("GAVDACLearner", "EA_GAVDACLearner",
                                         "SmallInitGAVDACLearner")}
                        with self.assertRaises(Selected):
                            run._run_with_runner(runner.args, LossLog(), runner, resources)
                        self.assertEqual(mocks[selected].call_count, 1)
                        self.assertEqual(sum(m.call_count for m in mocks.values()), 1)
                        self.assertEqual(runner.t_env, 0)
                finally:
                    runner.close_env()

    def test_h08_original_reference_full_updates_masks_targets_and_order(self):
        for agents, optimizer, mode in ((3, "adam", "soft"), (1, "rmsprop", "hard")):
            batch, candidate = make_learner(n_agents=agents, optimizer=optimizer,
                                            target_update_mode=mode)
            _, reference = make_learner(n_agents=agents, optimizer=optimizer,
                                       target_update_mode=mode)
            events = []
            handles = [
                candidate.critic_optimiser.register_step_post_hook(
                    lambda *unused: events.append("critic")),
                candidate.agent_optimiser.register_step_post_hook(
                    lambda *unused: events.append("actor")),
            ]
            try:
                for step in range(3):
                    expected = reference_update(reference, batch)
                    candidate.train(batch, step, step)
                    self.assert_learners_close(candidate, reference, rtol=1e-4, atol=1e-5)
                    np.testing.assert_allclose(
                        [candidate.logger.values["loss/critic_loss"],
                         candidate.logger.values["loss/policy_loss"]],
                        expected, rtol=1e-4, atol=1e-5,
                    )
                    self.assertEqual(events, ["critic", "actor"] * (step + 1))
                    self.assertEqual([candidate.critic_steps, candidate.actor_steps,
                                      candidate.target_steps, candidate.finite_updates],
                                     [step + 1] * 4)
                    for module in (candidate.target_mac.agent, candidate.target_critic,
                                   *([candidate.target_mixer] if agents > 1 else [])):
                        self.assertTrue(all(p.grad is None for p in module.parameters()))
            finally:
                for handle in handles:
                    handle.remove()

    def test_actor_live_context_recurrence_action_and_critic_immutability(self):
        batch, learner = make_learner()
        policy, inputs, _ = learner._critic_loss(batch)
        for key in ("hidden_states", "actions"):
            policy[key].retain_grad()
        critic_before = copy.deepcopy(learner.critic.state_dict())
        learner._set_critic_grad(False)
        try:
            learner._actor_loss(batch, policy, inputs).backward()
        finally:
            learner._set_critic_grad(True)
        for key in ("hidden_states", "actions"):
            self.assertIsNotNone(policy[key].grad)
            self.assertGreater(policy[key].grad.abs().sum().item(), 0)
        self.assertGreater(learner.mac.agent.rnn.weight_hh.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in learner.critic_params))
        torch.testing.assert_close(critic_before, learner.critic.state_dict(), rtol=0, atol=0)

    def test_target_deepcopy_tau_and_load_state_never_rescales(self):
        _, learner = make_learner()
        torch.testing.assert_close(learner.critic.state_dict(),
                                   learner.target_critic.state_dict(), rtol=0, atol=0)
        before = {key: value.clone() for key, value in learner.target_critic.state_dict().items()}
        with torch.no_grad():
            for parameter in learner.critic.parameters():
                parameter.add_(0.1)
        learner._update_targets_soft(0.002)
        expected = {key: value.lerp(learner.critic.state_dict()[key], 0.002)
                    for key, value in before.items()}
        torch.testing.assert_close(expected, learner.target_critic.state_dict(), rtol=0, atol=0)
        saved = copy.deepcopy(learner.critic.state_dict())
        learner.critic.load_state_dict(saved)
        learner.critic.load_state_dict(saved)
        torch.testing.assert_close(saved, learner.critic.state_dict(), rtol=0, atol=0)

    def test_diagnostics_actual_hooks_checkpoint_and_invalid_finite(self):
        batch, learner = make_learner()
        learner.train(batch, 1, 1)
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            learner.save_models(directory)
            record = json.loads((Path(directory) / "h08_diagnostics.json").read_text())
            self.assertEqual(record, learner.diagnostics())
            self.assertFalse(record["exact_resume_supported"])
            _, loaded = make_learner()
            critic_before = copy.deepcopy(loaded.critic.state_dict())
            loaded.load_models(directory)
            self.assertEqual(loaded.diagnostics(), record)
            torch.testing.assert_close(critic_before, loaded.critic.state_dict(), rtol=0, atol=0)
        batch["reward"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            learner.train(batch, 2, 2)
        self.assertEqual(learner.finite_updates, 1)
        with self.assertRaisesRegex(ValueError, "CPU float32"):
            make_learner_for_double()

    def test_worker_explicit_class_and_state_load(self):
        for flag in (None, False, True):
            batch, mac, args, scheme = fixture(learner_workers=2)
            if flag is not None:
                args.critic_small_output_init = flag
            with patch.object(torch, "set_num_interop_threads"), \
                 patch.multiple(gradient_pool, _learner=None, _scheme=None,
                                _groups=None, _transport=None):
                gradient_pool._initialize(args, scheme, Mock())
                worker = gradient_pool._learner
                self.assertIs(type(worker), SmallInitGAVDACLearner if flag else GAVDACLearner)
                self.assertEqual(args.learner_workers, 0)
                state = copy.deepcopy(worker.critic.state_dict())
                worker.critic.load_state_dict(state)
                torch.testing.assert_close(state, worker.critic.state_dict(), rtol=0, atol=0)
        args.critic_small_output_init = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            gradient_pool._initialize(args, scheme, Mock())

    def test_two_worker_varying_uneven_full_updates_and_shared_transport(self):
        _, serial = make_learner(use_cuda=False)
        _, parallel = make_learner(use_cuda=False, learner_workers=2)
        previous = None
        try:
            for step, (size, length) in enumerate(((5, 7), (2, 4), (2, 4), (7, 8))):
                batch, _, _, _ = fixture(batch_size=size, length=length)
                if step == 0:
                    batch["filled"][:2] = 0
                serial.train(batch, step + 1, step)
                parallel.train(batch, step + 1, step)
                self.assert_learners_close(serial, parallel, rtol=1e-4, atol=1e-5)
                pool = parallel.gradient_pool
                self.assertEqual(len(pool.worker_pids), 2)
                self.assertEqual(pool.last_phase_pids["critic"], pool.last_phase_pids["actor"])
                self.assertEqual(len(pool.last_phase_pids["critic"]), 2)
                self.assertIsNone(pool.pending_batch)
                self.assertTrue(all(value.is_shared() for value in pool.batch_buffers.values()))
                pointers = {key: value.data_ptr() for key, value in pool.batch_buffers.items()}
                if step == 2:
                    self.assertEqual(previous, pointers)
                previous = pointers
                saved = {key: value.clone() for key, value in pool.batch_buffers.items()}
                batch["obs"].add_(10)
                torch.testing.assert_close(saved, pool.batch_buffers, rtol=0, atol=0)
                self.assertEqual(parallel.diagnostics(), serial.diagnostics())
        finally:
            serial.close()
            parallel.close()

    def test_strict_saved_source_validation_731_732_physical_and_hashes(self):
        config = entry("gavdac_c2_h08")
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            config["env_args"].update(result_dir=directory, run_id="h08_strict")
            runner, mac = make_evaluation_runner(config)
            try:
                evaluation = evaluate_fixed(runner, mac, [731, 732], 0.35, "validation", "rl", 0)
                store = BestPolicyStore(Path(directory) / "best", config)
                checkpoint = store.consider(mac, "rl", 0, evaluation, runner.env)
                output = Path(directory) / "strict"
                result = subprocess.run(
                    [sys.executable, "-I", str(ROOT / "tools/evaluate_microgrid_comparison.py"),
                     "--checkpoint", str(checkpoint), "--split", "validation",
                     "--mode", "continuous", "--rows", "731", "732", "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                summary = json.loads((output / "summary.json").read_text())
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                self.assertTrue(summary["physical_checks_passed"])
                self.assertEqual(summary["rows"], [731, 732])
                self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
                self.assertEqual(summary["data_sha256"], metadata["data_sha256"])
                for name, expected in summary["source_sha256"].items():
                    actual = (Path(metadata["source_snapshot"]) / name).read_bytes()
                    self.assertEqual(hashlib.sha256(actual).hexdigest(), expected)
                for name, expected in metadata["source_sha256"].items():
                    self.assertEqual(summary["source_sha256"][name], expected)
                self.assertEqual(config, direct_main.load_yaml(checkpoint / "config.yaml"))
                self.assertEqual(runner.t_env, 0)
            finally:
                runner.close_env()


def make_learner_for_double():
    batch, mac, args, scheme = fixture(critic_small_output_init=True)
    mac.agent.double()
    return SmallInitGAVDACLearner(mac, scheme, LossLog(), args)


if __name__ == "__main__":
    unittest.main()
