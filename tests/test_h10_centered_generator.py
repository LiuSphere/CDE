"""H10 coordinate, gradient, committed-baseline and worker/full-update gates."""

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "tools")]

from controllers.cqmix_controller import CQMixMAC
from learners.gavdac_learner import GAVDACLearner
from learners.centered_generator_learner import (
    CenteredGeneratorGAVDACLearner, validate_center_generator_flag,
)
from modules.critic import GAVDACCritic
from modules.centered_generator_critic import CenteredGeneratorGAVDACCritic
from learners import gradient_pool
from test_gavdac_sequence import fixture, LossLog
from test_main_algorithm import load_config
import run as training
from experiment_slot import validate_job, validate_worker_options

PRE_H10 = "381e54f51ea9aff266b929b07b077a49f18b4a3a"


def fraction_fixture(**options):
    options.setdefault("action_spaces", [
        Box(np.array([0., -1.], dtype=np.float32), np.ones(2, dtype=np.float32))
        for _ in range(3)
    ])
    options.setdefault("use_cuda", False)
    options.setdefault("critic_center_generator_action", True)
    batch, mac, args, scheme = fixture(**options)
    batch["actions"][..., 0].sigmoid_()
    batch["actions"][..., 1].clamp_(-1, 1)
    return batch, mac, args, scheme


def centered(actions):
    return torch.stack((2 * actions[..., 0] - 1, actions[..., 1]), dim=-1)


def oracle_pre_hook(module, arguments):
    return (arguments[0], centered(arguments[1]), *arguments[2:])


def worker_observation():
    learner = gradient_pool._learner
    return {
        "learner": type(learner).__name__, "critic": type(learner.critic).__name__,
        "target": type(learner.target_critic).__name__,
        "critic_guards": getattr(learner, "critic_phase_guards", None),
        "actor_guards": getattr(learner, "actor_phase_guards", None),
        "critic_steps": getattr(learner, "critic_steps", None),
        "actor_steps": getattr(learner, "actor_steps", None),
        "shared": all(value.is_shared() for buffers in gradient_pool._transport["weights"].values()
                      for value in buffers.values()),
    }


class H10Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        source = subprocess.check_output(
            ["git", "show", f"{PRE_H10}:src/learners/gavdac_learner.py"],
            cwd=ROOT, text=True,
        )
        module = types.ModuleType("pre_h10_baseline")
        exec(compile(source, f"{PRE_H10}:gavdac_learner.py", "exec"), module.__dict__)
        cls.committed_baseline = module.GAVDACLearner

    def assert_learners(self, left, right, *, rtol=0, atol=0):
        for name in ("critic", "mixer", "target_critic", "target_mixer",
                     "agent_optimiser", "critic_optimiser"):
            torch.testing.assert_close(
                getattr(left, name).state_dict(), getattr(right, name).state_dict(),
                rtol=rtol, atol=atol,
            )
        for name in ("mac", "target_mac"):
            torch.testing.assert_close(
                getattr(left, name).agent.state_dict(),
                getattr(right, name).agent.state_dict(), rtol=rtol, atol=atol,
            )
        torch.testing.assert_close(
            [p.grad for p in left.agent_params + left.critic_params],
            [p.grad for p in right.agent_params + right.critic_params], rtol=rtol, atol=atol,
        )

    def test_entry_only_name_and_bool_and_supervisor_support(self):
        baseline = load_config(["--config", "gavdac_ea_main"])
        variant = load_config(["--config", "gavdac_c2_h10"])
        self.assertEqual(variant, {
            **baseline, "name": "gavdac_c2_h10", "critic_center_generator_action": True})
        self.assertEqual((variant["rollout_workers"], variant["learner_workers"]), (0, 0))
        validate_job("gavdac_c2_h10", 2, "h10ls0a", 0, 1)
        validate_worker_options("gavdac_c2_h10", 1, 2, 2)
        with self.assertRaises(ValueError):
            validate_worker_options("gavdac_c2_h10", 1, 2, 2, "evaluation.json")
        for value in (None, 0, 1, "true", np.bool_(True)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                validate_center_generator_flag(value)
            with self.assertRaisesRegex(ValueError, "boolean"):
                training.args_sanity_check(
                    {**baseline, "critic_center_generator_action": value}, None)
        self.assertFalse(validate_center_generator_flag())
        self.assertFalse(validate_center_generator_flag(False))

    def test_critic_endpoints_layout_oracle_gradient_and_no_mutation(self):
        batch, mac, args, scheme = fraction_fixture(batch_size=2, length=7)
        candidate = CenteredGeneratorGAVDACCritic(scheme, args)
        reference = GAVDACCritic(scheme, args)
        reference.load_state_dict(candidate.state_dict())
        generator = torch.Generator().manual_seed(83)
        for shape in ((3, 2), (2, 3, 2), (2, 7, 3, 2)):
            actions = torch.rand(shape, generator=generator).requires_grad_()
            obs = torch.randn(actions.numel() // 2, 20, generator=generator)
            before = actions.detach().clone()
            actual, _ = candidate(obs, actions)
            z = centered(actions.detach()).requires_grad_()
            expected, _ = reference(obs, z)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            da = torch.autograd.grad(actual.sum(), actions)[0]
            dz = torch.autograd.grad(expected.sum(), z)[0]
            torch.testing.assert_close(da, dz * torch.tensor([2., 1.]), rtol=0, atol=0)
            self.assertTrue(torch.equal(actions, before))
        actions = batch["actions"].transpose(0, 1)
        obs = batch["obs"].transpose(0, 1).reshape(-1, 20)
        self.assertFalse(actions.is_contiguous())
        torch.testing.assert_close(
            candidate(obs, actions)[0], reference(obs, centered(actions))[0], rtol=0, atol=0)
        actions = torch.tensor([[[0., -1.], [.5, 0.], [1., 1.]]])
        captured = []
        hook = candidate.fc1.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        candidate(torch.zeros(3, 20), actions)
        hook.remove()
        self.assertTrue(torch.equal(captured[0][:, -2:],
                                    torch.tensor([[-1., -1.], [0., 0.], [1., 1.]])))
        raw = (batch["actions"] * .5).detach().requires_grad_()
        environment_actions = mac._scale_actions_to_space(raw)
        z = centered(environment_actions)
        torch.testing.assert_close(z, raw, rtol=0, atol=1e-7)
        dqraw = torch.autograd.grad(
            candidate(batch["obs"].reshape(-1, 20), environment_actions)[0].sum(), raw)[0]
        independent = z.detach().requires_grad_()
        dqz = torch.autograd.grad(reference(
            batch["obs"].reshape(-1, 20), independent)[0].sum(), independent)[0]
        torch.testing.assert_close(dqraw, dqz, rtol=0, atol=0)
        self.assertTrue(torch.equal(environment_actions[..., 1], z[..., 1]))

    def test_unsupported_box_shape_dtype_and_none_fail_explicitly(self):
        batch, _, args, scheme = fraction_fixture()
        critic = CenteredGeneratorGAVDACCritic(scheme, args)
        obs = batch["obs"].reshape(-1, 20)
        for observations, actions in (
            (obs, None), (obs, batch["actions"].reshape(-1, 2)),
            (obs[:-1], batch["actions"]), (batch["obs"], batch["actions"]),
            (obs.double(), batch["actions"]),
            (obs, batch["actions"].double()),
            (obs, torch.full_like(batch["actions"], float("nan"))),
        ):
            with self.subTest(shape=None if actions is None else actions.shape), \
                 self.assertRaises(ValueError):
                critic(observations, actions)
        for field, value in (
            ("n_agents", 1), ("n_actions", 1),
            ("action_spaces", [Box(-np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32))] * 3),
            ("action_spaces", args.action_spaces[:2]),
        ):
            invalid = copy.deepcopy(args)
            setattr(invalid, field, value)
            with self.assertRaises(ValueError):
                CenteredGeneratorGAVDACCritic(scheme, invalid)

    def test_original_draws_state_dict_load_deepcopy_and_counts(self):
        batch, mac, args, scheme = fraction_fixture(
            rnn_hidden_dim=48, mixing_embed_dim=32, hypernet_embed=64)
        rng = torch.get_rng_state()
        expected = self.committed_baseline(copy.deepcopy(mac), scheme, LossLog(), args)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual = CenteredGeneratorGAVDACLearner(mac, scheme, LossLog(), args)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        self.assert_learners(expected, actual)
        self.assertEqual(sum(p.numel() for p in actual.agent_params), 15458)
        self.assertEqual(sum(p.numel() for p in actual.critic_params), 65915)
        self.assertEqual(list(actual.critic.buffers()), [])
        clone = copy.deepcopy(actual)
        clone.critic.load_state_dict(expected.critic.state_dict(), strict=True)
        self.assertIs(type(clone.target_critic), CenteredGeneratorGAVDACCritic)
        x, a = batch["obs"].reshape(-1, 20), batch["actions"]
        torch.testing.assert_close(actual.critic(x, a)[0], clone.critic(x, a)[0], rtol=0, atol=0)
        self.assertGreater(float((expected.critic(x, a)[0] - actual.critic(x, a)[0]).abs().max().detach()), 1e-5)

    def test_missing_false_multistep_committed_baseline_update_optimizer_target_rng_exact(self):
        for flag in ("missing", False):
            batch, mac, args, scheme = fraction_fixture()
            if flag == "missing":
                del args.critic_center_generator_action
            else:
                args.critic_center_generator_action = False
            rng = torch.get_rng_state()
            original = self.committed_baseline(copy.deepcopy(mac), scheme, LossLog(), args)
            torch.set_rng_state(rng)
            candidate = GAVDACLearner(mac, scheme, LossLog(), args)
            self.assert_learners(original, candidate)
            for step in range(4):
                before = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
                original.train(batch, step, step)
                expected_rng = torch.get_rng_state()
                torch.set_rng_state(before[0])
                candidate.train(batch, step, step)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                np.testing.assert_equal(before[1], np.random.get_state())
                self.assertEqual(before[2], random.getstate())
                self.assert_learners(original, candidate)
                self.assertEqual(original.logger.values, candidate.logger.values)

    def test_live_actor_action_context_recurrence_l2_and_critic_immutability(self):
        batch, mac, args, scheme = fraction_fixture()
        learner = CenteredGeneratorGAVDACLearner(mac, scheme, LossLog(), args)
        policy = mac.forward_sequence(batch, end=-1)
        policy["actions"].retain_grad()
        policy["hidden_states"].retain_grad()
        inputs = learner._build_inputs_sequence(batch, end=-1)
        before = copy.deepcopy(learner.critic.state_dict())
        learner._set_critic_grad(False)
        q, _ = learner.critic(inputs, policy["actions"])
        qmix = learner._mix_q_values(
            learner.mixer, q, batch["state"][:, :-1],
            agent_obs=batch["obs"][:, :-1], hidden_states=policy["hidden_states"])
        expected = -qmix.mean() + policy["raw_actions"].square().mean() * 1e-3
        actual = learner._actor_loss(batch, policy, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        self.assertGreater(policy["actions"].grad.norm().item(), 0)
        self.assertGreater(policy["hidden_states"].grad.norm().item(), 0)
        self.assertGreater(mac.agent.rnn.weight_hh.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None for p in learner.critic_params))
        torch.testing.assert_close(before, learner.critic.state_dict(), rtol=0, atol=0)
        learner._finite_tensors(p.grad for p in learner.agent_params if p.grad is not None)

    def test_one_step_mse_padding_global_denominator_target_detach(self):
        batch, mac, args, scheme = fraction_fixture(batch_size=5, length=7)
        learner = CenteredGeneratorGAVDACLearner(mac, scheme, LossLog(), args)
        outputs, target_outputs = [], []
        hooks = [
            learner.critic.register_forward_hook(lambda m, a, out: outputs.append(out[0])),
            learner.target_critic.register_forward_hook(lambda m, a, out: target_outputs.append(out[0])),
        ]
        policy, inputs, actual = learner._critic_loss(batch)
        with torch.no_grad():
            target_policy = learner.target_mac.forward_sequence(batch)
            target_q = learner._mix_q_values(
                learner.target_mixer, target_outputs[-1], batch["state"][:, 1:],
                agent_obs=batch["obs"][:, 1:], hidden_states=target_policy["hidden_states"][:, 1:])
        q = learner._mix_q_values(
            learner.mixer, outputs[-1], batch["state"][:, :-1],
            agent_obs=batch["obs"][:, :-1], hidden_states=policy["hidden_states"].detach())
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float().clone()
        mask[:, 1:] *= 1 - terminated[:, :-1]
        target = batch["reward"][:, :-1].expand_as(target_q) + args.gamma * (
            1 - terminated.expand_as(target_q)) * target_q
        expected = (((target - q) * mask.expand_as(q)) ** 2).sum() / mask.expand_as(q).sum()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _, _, global_loss = learner._critic_loss(batch, denominator=float(mask.sum()) * 2)
        torch.testing.assert_close(global_loss, actual / 2, rtol=0, atol=0)
        actual.backward()
        self.assertTrue(all(p.grad is None for p in learner.target_critic.parameters()))
        self.assertTrue(all(p.grad is None for p in learner.target_mac.parameters()))
        self.assertTrue(all(p.grad is None for p in learner.mac.parameters()))
        for hook in hooks:
            hook.remove()

    def test_full_update_matches_independent_input_oracle_and_original_target_order(self):
        for mode, optimizer in (("soft", "adam"), ("hard", "rmsprop")):
            batch, mac, args, scheme = fraction_fixture(
                target_update_mode=mode, optimizer=optimizer)
            rng = torch.get_rng_state()
            reference = self.committed_baseline(copy.deepcopy(mac), scheme, LossLog(), args)
            torch.set_rng_state(rng)
            actual = CenteredGeneratorGAVDACLearner(mac, scheme, LossLog(), args)
            reference.critic.register_forward_pre_hook(oracle_pre_hook)
            reference.target_critic.register_forward_pre_hook(oracle_pre_hook)
            for step in range(3):
                events = []
                hooks = [
                    actual.critic_optimiser.register_step_post_hook(lambda *args: events.append("critic")),
                    actual.agent_optimiser.register_step_post_hook(lambda *args: events.append("actor")),
                ]
                reference.train(batch, step, step)
                actual.train(batch, step, step)
                for hook in hooks:
                    hook.remove()
                self.assertEqual(events, ["critic", "actor"])
                self.assert_learners(reference, actual)
                self.assertEqual(reference.logger.values, actual.logger.values)
                self.assertEqual(actual.target_steps, step + 1)
                self.assertEqual(actual.critic_phase_guards, step + 1)
                self.assertEqual(actual.actor_phase_guards, step + 1)

    def test_two_worker_uneven_varying_fullupdate_and_actual_worker_class(self):
        batch, mac, args, scheme = fraction_fixture(batch_size=5, length=7)
        rng = torch.get_rng_state()
        other_mac = copy.deepcopy(mac)
        serial = CenteredGeneratorGAVDACLearner(mac, scheme, LossLog(), args)
        torch.set_rng_state(rng)
        parallel_args = copy.deepcopy(args)
        parallel_args.learner_workers = 2
        parallel = CenteredGeneratorGAVDACLearner(other_mac, scheme, LossLog(), parallel_args)
        try:
            pointers = None
            for step, (size, length) in enumerate(((5, 7), (3, 5), (3, 5), (1, 6))):
                batch, _, _, _ = fraction_fixture(batch_size=size, length=length)
                if size > 1:
                    batch["filled"][0] = 0
                serial.train(batch, step, step)
                parallel.train(batch, step, step)
                self.assert_learners(serial, parallel, rtol=1e-4, atol=1e-5)
                pool = parallel.gradient_pool
                self.assertEqual(len(pool.pids), 2)
                self.assertEqual(pool.last_phase_pids["critic"], pool.last_phase_pids["actor"])
                self.assertEqual(len(pool.last_phase_pids["actor"]), min(size, 2))
                current = {key: value.data_ptr() for key, value in pool.batch_buffers.items()}
                if step == 2:
                    self.assertEqual(current, pointers)
                pointers = current
                self.assertTrue(all(v.is_shared() for v in pool.batch_buffers.values()))
                self.assertEqual(parallel.h10_diagnostics()["parent_critic_phase_guards"], 0)
            observed = [executor.submit(worker_observation).result() for executor in pool.executors]
            for index, row in enumerate(observed):
                self.assertEqual(row["learner"], "CenteredGeneratorGAVDACLearner")
                self.assertEqual(row["critic"], "CenteredGeneratorGAVDACCritic")
                self.assertEqual(row["target"], "CenteredGeneratorGAVDACCritic")
                self.assertEqual(row["critic_guards"], 4 if index == 0 else 3)
                self.assertEqual(row["actor_guards"], row["critic_guards"])
                self.assertEqual(row["critic_steps"], 0)
                self.assertEqual(row["actor_steps"], 0)
                self.assertTrue(row["shared"])
            self.assertEqual(parallel.finite_updates, 4)
        finally:
            serial.close()
            parallel.close()

    def test_false_worker_is_original_class_and_matches_original_fullupdate(self):
        batch, mac, args, scheme = fraction_fixture(critic_center_generator_action=False)
        rng = torch.get_rng_state()
        expected = self.committed_baseline(copy.deepcopy(mac), scheme, LossLog(), args)
        torch.set_rng_state(rng)
        actual = GAVDACLearner(mac, scheme, LossLog(), args)
        actual.args.learner_workers = 2
        try:
            for step in range(2):
                expected.train(batch, step, step)
                actual.train(batch, step, step)
                self.assert_learners(expected, actual, rtol=1e-4, atol=1e-5)
            for executor in actual.gradient_pool.executors:
                observed = executor.submit(worker_observation).result()
                self.assertEqual(observed["learner"], "GAVDACLearner")
                self.assertEqual(observed["critic"], "GAVDACCritic")
        finally:
            actual.close()

    def test_actual_finite_hooks_checkpoint_diagnostics_and_actor_load(self):
        batch, mac, args, scheme = fraction_fixture()
        learner = CenteredGeneratorGAVDACLearner(mac, scheme, LossLog(), args)
        learner.train(batch, 1, 1)
        diagnostics = learner.h10_diagnostics()
        for key in ("critic_steps", "actor_steps", "target_steps", "finite_updates", "train_updates"):
            self.assertEqual(diagnostics[key], 1)
        self.assertFalse(diagnostics["exact_resume_supported"])
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            learner.save_models(directory)
            stored = json.loads((Path(directory) / "h10_diagnostics.json").read_text())
            self.assertEqual(stored, diagnostics)
            clone_mac = CQMixMAC(scheme, batch.groups, args)
            clone = CenteredGeneratorGAVDACLearner(clone_mac, scheme, LossLog(), args)
            clone.load_models(directory)
            self.assertIs(type(clone.critic), CenteredGeneratorGAVDACCritic)
            torch.testing.assert_close(clone.mac.agent.state_dict(), mac.agent.state_dict(), rtol=0, atol=0)
            self.assertEqual({p.name for p in Path(directory).iterdir()},
                             {"agent.th", "mixer.th", "opt.th", "h10_diagnostics.json"})
        with torch.no_grad():
            next(learner.mac.parameters()).fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "finite CPU float32"):
            learner.h10_diagnostics()

    def test_strict_saved_source_two_validation_days_and_actual_hashes(self):
        from utils.comparison_evaluation import make_evaluation_runner
        from utils.policy_evaluation import BestPolicyStore, evaluate_fixed
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            config = load_config(["--config", "gavdac_c2_h10"])
            config["env_args"].update(result_dir=directory, run_id="h10_strict_fixture", seed=17)
            runner, mac = make_evaluation_runner(config)
            try:
                evaluation = evaluate_fixed(runner, mac, [731, 732], .35, "validation", "rl", 123)
                store = BestPolicyStore(Path(directory) / "best", config)
                checkpoint = store.consider(mac, "rl", 123, evaluation, runner.env)
                output = Path(directory) / "strict"
                command = [
                    "uv", "run", "python", "tools/evaluate_microgrid_comparison.py",
                    "--checkpoint", str(checkpoint), "--split", "validation",
                    "--mode", "continuous", "--rows", "731", "732", "--output", str(output)]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
                self.assertIn('"physical_checks_passed": true', result.stdout)
                summary = json.loads((output / "summary.json").read_text())
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                self.assertTrue(summary["physical_checks_passed"])
                self.assertEqual(summary["policy_sha256"], hashlib.sha256(
                    (checkpoint / "agent.th").read_bytes()).hexdigest())
                for name, expected in metadata["source_sha256"].items():
                    path = Path(metadata["source_snapshot"]) / name
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
                for name, expected in summary["source_sha256"].items():
                    self.assertEqual(hashlib.sha256((output / "runtime" / name).read_bytes()).hexdigest(), expected)
                for name, expected in summary["data_sha256"].items():
                    self.assertEqual(hashlib.sha256((ROOT / "src/envs/data" / name).read_bytes()).hexdigest(), expected)
            finally:
                runner.close_env()


if __name__ == "__main__":
    unittest.main()
