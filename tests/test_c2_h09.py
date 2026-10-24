"""H09 Actor-objective ablation, frozen-baseline, worker and saved-source gates."""

import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "tools")]

import direct_main
import run
from learners import gradient_pool
from learners.gavdac_learner import GAVDACLearner
from learners.no_action_l2_learner import NoActionL2GAVDACLearner, no_action_l2_enabled
from test_gavdac_sequence import LossLog, fixture
from utils.comparison_evaluation import make_evaluation_runner
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed
from experiment_slot import validate_job, validate_worker_options

PRE_H09 = "5761821ecf97221274538ceda6a8c9952bdf6e96"


def entry(name):
    with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
         patch.object(direct_main, "run") as execute:
        direct_main.main()
    return execute.call_args.args[0]


def make_learner(candidate=True, cls=None, **options):
    options.setdefault("rnn_hidden_dim", 48)
    if candidate:
        options["actor_no_action_l2"] = True
    batch, mac, args, scheme = fixture(**options)
    cls = cls or (NoActionL2GAVDACLearner if candidate else GAVDACLearner)
    return batch, cls(mac, scheme, LossLog(), args)


def modules(learner):
    return gradient_pool._modules(learner)


class H09Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        source = subprocess.check_output(
            ["git", "show", PRE_H09 + ":src/learners/gavdac_learner.py"], cwd=ROOT, text=True)
        namespace = {"__name__": "_h09_frozen_baseline"}
        exec(compile(source, "<pre-H09 committed baseline>", "exec"), namespace)
        cls.original = namespace["GAVDACLearner"]

    def assert_learners_close(self, left, right, rtol=0, atol=0):
        for name in modules(left):
            torch.testing.assert_close(
                modules(left)[name].state_dict(), modules(right)[name].state_dict(),
                rtol=rtol, atol=atol)
        for name in ("agent_optimiser", "critic_optimiser"):
            torch.testing.assert_close(
                getattr(left, name).state_dict(), getattr(right, name).state_dict(),
                rtol=rtol, atol=atol)
        torch.testing.assert_close(
            [p.grad for p in left.agent_params + left.critic_params],
            [p.grad for p in right.agent_params + right.critic_params], rtol=rtol, atol=atol)

    def test_initialization_all_networks_parameters_and_rng_exact(self):
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        _, original = make_learner(False, cls=self.original)
        expected_rng = torch.get_rng_state().clone()
        _, candidate = make_learner()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        for left, right in zip(numpy_rng, np.random.get_state(), strict=True):
            np.testing.assert_equal(left, right)
        self.assert_learners_close(original, candidate)
        for name, module in modules(candidate).items():
            self.assertEqual(module.state_dict().keys(), modules(original)[name].state_dict().keys())
            self.assertEqual(sum(p.numel() for p in module.parameters()),
                             sum(p.numel() for p in modules(original)[name].parameters()))
            for value in module.state_dict().values():
                self.assertEqual(value.device.type, "cpu")
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(candidate.diagnostics()["finite_updates"], 0)

    def test_loss_gradient_oracle_and_invalid_shape(self):
        for size, length in ((1, 2), (5, 7)):
            batch, candidate = make_learner(batch_size=size, length=length)
            policy, inputs, _ = candidate._critic_loss(batch)
            candidate._set_critic_grad(False)
            actual = candidate._actor_loss(batch, policy, inputs)
            baseline = self.original._actor_loss(candidate, batch, policy, inputs)
            penalty = policy["raw_actions"].square().mean() * 1e-3
            torch.testing.assert_close(baseline, actual + penalty, rtol=0, atol=0)
            parameters = list(candidate.mac.parameters())
            base_grad = torch.autograd.grad(baseline, parameters, retain_graph=True)
            actual_grad = torch.autograd.grad(actual, parameters, retain_graph=True)
            penalty_grad = torch.autograd.grad(penalty, parameters, retain_graph=True)
            for a, b, c in zip(base_grad, actual_grad, penalty_grad, strict=True):
                torch.testing.assert_close(a, b+c, rtol=1e-4, atol=2e-7)
            self.assertGreater(sum(value.abs().sum().item() for value in penalty_grad), 0)
            raw_gradient = torch.autograd.grad(penalty, policy["raw_actions"])[0]
            torch.testing.assert_close(
                raw_gradient, policy["raw_actions"] * (2e-3 / policy["raw_actions"].numel()),
                rtol=1e-6, atol=1e-10)
            broken = {**policy, "actions": policy["actions"][..., :1]}
            with self.assertRaises(RuntimeError):
                candidate._actor_loss(batch, broken, inputs)

    def test_missing_false_complete_updates_against_committed_baseline_and_rng(self):
        for flag in (None, False):
            batch, original = make_learner(False, cls=self.original)
            options = {} if flag is None else {"actor_no_action_l2": False}
            _, baseline = make_learner(False, **options)
            self.assertFalse(no_action_l2_enabled(baseline.args))
            self.assert_learners_close(original, baseline)
            for step in range(3):
                before = torch.get_rng_state().clone()
                original.train(batch, step, step)
                expected = torch.get_rng_state().clone()
                torch.set_rng_state(before)
                baseline.train(batch, step, step)
                self.assertTrue(torch.equal(expected, torch.get_rng_state()))
                self.assert_learners_close(original, baseline)
                self.assertEqual(original.logger.values, baseline.logger.values)

    def test_entry_flag_validation_and_supervisor(self):
        parent, candidate = entry("gavdac_ea_main"), entry("gavdac_c2_h09")
        self.assertEqual(candidate, {**parent, "name": "gavdac_c2_h09", "actor_no_action_l2": True})
        validate_job("gavdac_c2_h09", 2, "h09fixture", 0, 1)
        validate_worker_options("gavdac_c2_h09", 1, 2, 2)
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                run.args_sanity_check({**candidate, "actor_no_action_l2": value}, Mock())
        batch, mac, args, scheme = fixture()
        with self.assertRaisesRegex(ValueError, "requires"):
            NoActionL2GAVDACLearner(mac, scheme, LossLog(), args)

    def test_runner_explicit_dispatch_without_environment_steps(self):
        class Selected(Exception):
            pass
        for flag, ea, selected in ((None, True, "EA_GAVDACLearner"),
                                   (False, False, "GAVDACLearner"),
                                   (False, True, "EA_GAVDACLearner"),
                                   (True, True, "NoActionL2GAVDACLearner")):
            config = entry("gavdac_ea_main")
            config.update(EA=ea, save_model=False, buffer_size=2)
            if flag is not None:
                config["actor_no_action_l2"] = flag
            with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
                config["env_args"].update(result_dir=directory, run_id="h09_dispatch")
                runner, _ = make_evaluation_runner(config)
                try:
                    with ExitStack() as resources:
                        mocks = {name: resources.enter_context(patch.object(run, name, side_effect=Selected))
                                 for name in ("GAVDACLearner", "EA_GAVDACLearner", "NoActionL2GAVDACLearner")}
                        with self.assertRaises(Selected):
                            run._run_with_runner(runner.args, LossLog(), runner, resources)
                        self.assertEqual(mocks[selected].call_count, 1)
                        self.assertEqual(sum(m.call_count for m in mocks.values()), 1)
                        self.assertEqual(runner.t_env, 0)
                finally:
                    runner.close_env()

    def test_h09_reference_full_updates_masks_targets_and_order(self):
        class Reference(self.original):
            def _actor_loss(self, batch, policy, inputs):
                self.critic.init_hidden(batch.batch_size)
                q, self.critic.hidden_states = self.critic(inputs, policy["actions"], self.critic.hidden_states)
                if self.mixer is not None:
                    q = self._mix_q_values(
                        self.mixer, q, batch["state"][:, :-1],
                        agent_obs=batch["obs"][:, :-1], hidden_states=policy["hidden_states"])
                else:
                    q = q.view(batch.batch_size, batch.max_seq_length-1, self.n_agents)
                return -q.mean()
        for agents, optimizer, mode in ((3, "adam", "soft"), (1, "rmsprop", "hard")):
            batch, candidate = make_learner(n_agents=agents, optimizer=optimizer, target_update_mode=mode)
            _, reference = make_learner(cls=Reference, n_agents=agents, optimizer=optimizer, target_update_mode=mode)
            events = []
            handles = [
                candidate.critic_optimiser.register_step_post_hook(lambda *unused: events.append("critic")),
                candidate.agent_optimiser.register_step_post_hook(lambda *unused: events.append("actor"))]
            try:
                for step in range(3):
                    reference.train(batch, step, step)
                    candidate.train(batch, step, step)
                    self.assert_learners_close(candidate, reference)
                    self.assertEqual(candidate.logger.values, reference.logger.values)
                    self.assertEqual(events, ["critic", "actor"] * (step + 1))
                    self.assertEqual([candidate.critic_steps, candidate.actor_steps,
                                      candidate.target_steps, candidate.finite_updates], [step+1]*4)
                    for module in modules(candidate):
                        if module.startswith("target_"):
                            self.assertTrue(all(p.grad is None for p in modules(candidate)[module].parameters()))
            finally:
                for handle in handles:
                    handle.remove()

    def test_actor_live_context_action_recurrence_critic_immutable(self):
        batch, learner = make_learner()
        policy, inputs, _ = learner._critic_loss(batch)
        for key in ("hidden_states", "actions"):
            policy[key].retain_grad()
        before = {name: copy.deepcopy(module.state_dict()) for name, module in modules(learner).items()}
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
        for name, module in modules(learner).items():
            torch.testing.assert_close(before[name], module.state_dict(), rtol=0, atol=0)

    def test_target_deepcopy_every_step_tau_and_state_load_unchanged(self):
        _, learner = make_learner()
        torch.testing.assert_close(learner.critic.state_dict(), learner.target_critic.state_dict(), rtol=0, atol=0)
        before = {key: value.clone() for key, value in learner.target_critic.state_dict().items()}
        with torch.no_grad():
            for parameter in learner.critic.parameters():
                parameter.add_(0.1)
        learner._update_targets_soft(.002)
        expected = {key: value.lerp(learner.critic.state_dict()[key], .002) for key, value in before.items()}
        torch.testing.assert_close(expected, learner.target_critic.state_dict(), rtol=0, atol=0)
        saved = copy.deepcopy(learner.critic.state_dict())
        learner.critic.load_state_dict(saved)
        learner.critic.load_state_dict(saved)
        torch.testing.assert_close(saved, learner.critic.state_dict(), rtol=0, atol=0)

    def test_actual_hooks_diagnostics_checkpoint_finite_and_dtype(self):
        batch, learner = make_learner()
        learner.train(batch, 1, 1)
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            learner.save_models(directory)
            record = json.loads((Path(directory) / "h09_diagnostics.json").read_text())
            self.assertEqual(record, learner.diagnostics())
            self.assertEqual(record["actor_action_l2_coefficient"], 0.0)
            self.assertFalse(record["exact_resume_supported"])
            _, loaded = make_learner()
            before = copy.deepcopy(loaded.critic.state_dict())
            loaded.load_models(directory)
            self.assertEqual(loaded.diagnostics(), record)
            torch.testing.assert_close(before, loaded.critic.state_dict(), rtol=0, atol=0)
        batch["reward"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            learner.train(batch, 2, 2)
        self.assertEqual(learner.finite_updates, 1)
        batch, mac, args, scheme = fixture(actor_no_action_l2=True)
        mac.agent.double()
        with self.assertRaisesRegex(ValueError, "CPU float32"):
            NoActionL2GAVDACLearner(mac, scheme, LossLog(), args)

    def test_worker_explicit_class_missing_false_and_loaded_parameters(self):
        for flag in (None, False, True):
            batch, mac, args, scheme = fixture(learner_workers=2)
            if flag is not None:
                args.actor_no_action_l2 = flag
            with patch.object(torch, "set_num_interop_threads"), \
                 patch.multiple(gradient_pool, _learner=None, _scheme=None, _groups=None, _transport=None):
                gradient_pool._initialize(args, scheme, Mock())
                worker = gradient_pool._learner
                self.assertIs(type(worker), NoActionL2GAVDACLearner if flag else GAVDACLearner)
                self.assertEqual(args.learner_workers, 0)
                saved = copy.deepcopy(worker.critic.state_dict())
                worker.critic.load_state_dict(saved)
                torch.testing.assert_close(saved, worker.critic.state_dict(), rtol=0, atol=0)
        for invalid in (1, None, "false"):
            args.actor_no_action_l2 = invalid
            with self.assertRaisesRegex(ValueError, "boolean"):
                gradient_pool._initialize(args, scheme, Mock())

    def test_two_worker_uneven_varying_full_update_and_shared_transport(self):
        _, serial = make_learner(use_cuda=False)
        _, parallel = make_learner(use_cuda=False, learner_workers=2)
        previous = None
        try:
            for step, (size, length) in enumerate(((5, 7), (2, 4), (2, 4), (7, 8))):
                batch, _, _, _ = fixture(batch_size=size, length=length)
                if step == 0:
                    batch["filled"][:2] = 0
                serial.train(batch, step+1, step)
                parallel.train(batch, step+1, step)
                self.assert_learners_close(serial, parallel, rtol=1e-4, atol=1e-5)
                pool = parallel.gradient_pool
                self.assertEqual(len(pool.worker_pids), 2)
                self.assertEqual(pool.last_phase_pids["critic"], pool.last_phase_pids["actor"])
                self.assertEqual(len(pool.last_phase_pids["actor"]), 2)
                self.assertIsNone(pool.pending_batch)
                self.assertTrue(all(value.is_shared() for value in pool.batch_buffers.values()))
                pointers = {key: value.data_ptr() for key, value in pool.batch_buffers.items()}
                if step == 2:
                    self.assertEqual(previous, pointers)
                previous = pointers
                saved = {key: value.clone() for key, value in pool.batch_buffers.items()}
                batch["obs"].add_(10)
                torch.testing.assert_close(saved, pool.batch_buffers, rtol=0, atol=0)
                self.assertEqual(serial.diagnostics(), parallel.diagnostics())
        finally:
            serial.close()
            parallel.close()

    def test_strict_saved_source_validation_731_732_physical_and_hashes(self):
        config = entry("gavdac_c2_h09")
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            config["env_args"].update(result_dir=directory, run_id="h09_strict")
            runner, mac = make_evaluation_runner(config)
            try:
                evaluation = evaluate_fixed(runner, mac, [731, 732], .35, "validation", "rl", 0)
                store = BestPolicyStore(Path(directory) / "best", config)
                checkpoint = store.consider(mac, "rl", 0, evaluation, runner.env)
                output = Path(directory) / "strict"
                result = subprocess.run(
                    ["uv", "run", "--frozen", "python", str(ROOT / "tools/evaluate_microgrid_comparison.py"),
                     "--checkpoint", str(checkpoint), "--split", "validation",
                     "--mode", "continuous", "--rows", "731", "732", "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                summary = json.loads((output / "summary.json").read_text())
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                self.assertTrue(summary["physical_checks_passed"])
                self.assertEqual(summary["rows"], [731, 732])
                self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
                self.assertEqual(summary["data_sha256"], metadata["data_sha256"])
                for name, expected in summary["source_sha256"].items():
                    self.assertEqual(hashlib.sha256((Path(metadata["source_snapshot"]) / name).read_bytes()).hexdigest(), expected)
                for name, expected in metadata["source_sha256"].items():
                    self.assertEqual(summary["source_sha256"][name], expected)
                self.assertEqual(config, direct_main.load_yaml(checkpoint / "config.yaml"))
                self.assertEqual(runner.t_env, 0)
            finally:
                runner.close_env()


if __name__ == "__main__":
    unittest.main()
