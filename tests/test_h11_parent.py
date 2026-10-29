"""NEW H11 parent qualification against actual committed H03, not old mains."""

import copy
import importlib.util
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from learners import gradient_pool
from learners.delayed_actor_learner import (
    DelayedActorGAVDACLearner, validate_update_intervals,
)
from learners.gavdac_learner import GAVDACLearner
from test_gavdac_sequence import LossLog, fixture

H03 = "e9975bf8c7af85a30b20d41f95c6436698463fb0"
PRE = "f18155fec81ab51539f0409a047dbe6a42e11ab2"


def git_source(commit, path):
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT,
    )


def without_registered_h12_hooks(source):
    if b"def _td_error_loss(" not in source:
        return source
    replacements = (
        (
            b'        huber = getattr(args, "scaled_huber_critic", False)\n'
            b"        if type(huber) is not bool:\n"
            b'            raise ValueError("scaled_huber_critic must be boolean")\n'
            b'        if huber and not getattr(self, "scaled_huber_critic", False):\n'
            b'            raise ValueError("scaled_huber_critic requires its explicit H12 learner")\n',
            b"",
        ),
        (
            b"        loss = self._td_error_loss(masked_td_error,\n",
            b"        loss = (masked_td_error**2).sum() / (\n",
        ),
        (
            b"    def _td_error_loss(self, masked_td_error, denominator):\n"
            b"        return (masked_td_error**2).sum() / denominator\n\n",
            b"",
        ),
    )
    # Accept only the registered extraction, then check all other parent bytes.
    for current, original in replacements:
        if source.count(current) != 1:
            raise AssertionError("Not the exact registered H12 neutral hook")
        source = source.replace(current, original, 1)
    return source


def without_registered_h14_hooks(source, path):
    replacements = {
        "src/learners/gavdac_learner.py": [(
            b'        signed = getattr(args, "signed_mixer_hyperweights", False)\n'
            b"        if type(signed) is not bool:\n"
            b'            raise ValueError("signed_mixer_hyperweights must be boolean")\n'
            b'        if signed and not getattr(self, "signed_mixer_hyperweights", False):\n'
            b'            raise ValueError("signed_mixer_hyperweights requires its explicit H14 learner")\n',
            b"", 1)],
        "src/modules/graph_mixer.py": [
            (b'        signed = getattr(args, "signed_mixer_hyperweights", False)\n'
             b"        if type(signed) is not bool:\n"
             b'            raise ValueError("signed_mixer_hyperweights must be boolean")\n', b"", 1),
            (b'weights_operation="signed_sigmoid" if signed else "sigmoid",',
             b'weights_operation="sigmoid",', 1)],
        "src/modules/graph/gnn.py": [
            (b'        elif self.weights_operation == "signed_sigmoid":\n'
             b"            wout = 2 * torch.sigmoid(wout) - 1\n", b"", 1),
            (b'if self.weights_operation in ("sigmoid", "signed_sigmoid"):',
             b'if self.weights_operation == "sigmoid":', 1)],
        "src/modules/graph/gin_layer.py": [
            (b'        elif self.weights_operation == "signed_sigmoid":\n'
             b"            weight1 = 2 * torch.sigmoid(weight1) - 1\n"
             b"            weight2 = 2 * torch.sigmoid(weight2) - 1\n", b"", 1),
            (b'if self.weights_operation in ("sigmoid", "signed_sigmoid"):',
             b'if self.weights_operation == "sigmoid":', 2)],
    }
    for current, original, count in replacements.get(path, []):
        if source.count(current) != count:
            raise AssertionError("Not the exact registered H14 neutral hook")
        source = source.replace(current, original)
    return source


def import_source(name, source, directory):
    path = Path(directory) / f"{name}.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def states(learner):
    result = {
        name: copy.deepcopy(module.state_dict())
        for name, module in gradient_pool._modules(learner).items()
    }
    result.update({
        name: copy.deepcopy(getattr(learner, name).state_dict())
        for name in ("agent_optimiser", "critic_optimiser")
    })
    result["gradients"] = [
        None if p.grad is None else p.grad.clone()
        for p in learner.agent_params + learner.critic_params
    ]
    result["losses"] = copy.deepcopy(learner.logger.values)
    return result


class H11ParentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        cls.directory = tempfile.TemporaryDirectory(prefix="h11parent_")
        sys.path.insert(0, cls.directory.name)
        cls.original = import_source(
            "h11_actual_committed_h03",
            git_source(H03, "src/learners/delayed_actor_learner.py"),
            cls.directory.name,
        )
        cls.old_pool = import_source(
            "h11_actual_committed_h03_pool",
            git_source(H03, "src/learners/gradient_pool.py"),
            cls.directory.name,
        )
        cls.root = import_source(
            "h11_actual_root",
            git_source(PRE, "src/learners/gavdac_learner.py"),
            cls.directory.name,
        )

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.directory.name)
        for name in ("h11_actual_committed_h03", "h11_actual_committed_h03_pool",
                     "h11_actual_root"):
            sys.modules.pop(name, None)
        cls.directory.cleanup()

    def pair(self, reference_class, **options):
        batch, mac, args, scheme = fixture(
            use_cuda=False, actor_update_interval=2, target_update_interval=1,
            **options,
        )
        rng = torch.get_rng_state()
        reference = reference_class(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
        after_reference = torch.get_rng_state()
        torch.set_rng_state(rng)
        candidate = DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
        self.assertTrue(torch.equal(after_reference, torch.get_rng_state()))
        torch.testing.assert_close(states(candidate), states(reference), rtol=0, atol=0)
        return batch, reference, candidate

    def compare(self, reference, candidate, rtol=0, atol=0):
        torch.testing.assert_close(states(candidate), states(reference), rtol=rtol, atol=atol)
        for name in ("critic_steps", "actor_steps", "target_updates", "finite_checks"):
            self.assertEqual(getattr(candidate, name), getattr(reference, name))
        self.assertEqual(candidate.optimizer_finite_checks, {
            "critic": candidate.critic_steps, "actor": candidate.actor_steps,
        })
        for name, module in gradient_pool._modules(candidate).items():
            for p in module.parameters():
                self.assertEqual((p.device.type, p.dtype), ("cpu", torch.float32))
                self.assertTrue(torch.isfinite(p).all())
                if name.startswith("target"):
                    self.assertIsNone(p.grad)

    def test_actual_root_and_parent_loss_sources_are_unchanged(self):
        for path in ("src/learners/gavdac_learner.py",
                     "src/controllers/cqmix_controller.py", "src/modules/policy.py",
                     "src/modules/critic.py", "src/modules/graph_mixer.py",
                     "src/modules/graph/gnn.py", "src/modules/graph/gin_layer.py"):
            actual = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
            actual = without_registered_h14_hooks(actual, path)
            if path == "src/learners/gavdac_learner.py":
                actual = without_registered_h12_hooks(actual)
            self.assertEqual(actual, git_source(PRE, path).replace(b"\r\n", b"\n"))
            self.assertEqual(actual, git_source(H03, path).replace(b"\r\n", b"\n"))

    def test_registered_h12_hook_guard_rejects_unrelated_mutations(self):
        path = "src/learners/gavdac_learner.py"
        actual = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        actual = without_registered_h14_hooks(actual, path)
        original = git_source(PRE, path).replace(b"\r\n", b"\n")
        for before, after in (
            (b"masked_td_error**2", b"masked_td_error**3"),
            (b"mask.sum() if denominator", b"mask.mean() if denominator"),
            (b"1e-3", b"2e-3"),
            (b"self.args = args", b"self.args = copy.copy(args)"),
        ):
            with self.subTest(mutation=before):
                self.assertIn(before, actual)
                with self.assertRaises(AssertionError):
                    self.assertEqual(
                        without_registered_h12_hooks(actual.replace(before, after)),
                        original,
                    )
        self.assertEqual(without_registered_h12_hooks(original), original)

    def test_registered_h14_hooks_reject_missing_selection_and_unrelated_source_mutations(self):
        changes = {
            "src/learners/gavdac_learner.py": [
                (b"if signed and not getattr", b"if not signed and not getattr"),
                (b"mask.sum() if denominator", b"mask.mean() if denominator"),
                (b"1e-3", b"2e-3")],
            "src/modules/graph_mixer.py": [
                (b'if signed else "sigmoid"', b'if signed else "abs"'),
                (b"F.softmax(attention", b"F.relu(attention"),
                (b"hidden_states).view", b"hidden_states.detach()).view")],
            "src/modules/graph/gnn.py": [
                (b"2 * torch.sigmoid(wout) - 1", b"torch.sigmoid(wout) - 1"),
                (b"scalar_out * self.weight_scale", b"scalar_out * 0.2"),
                (b"torch.sum(features * alive", b"torch.sum(features + alive")],
            "src/modules/graph/gin_layer.py": [
                (b"weight2 = 2 * torch.sigmoid(weight2) - 1", b"weight2 = torch.sigmoid(weight2)"),
                (b"torch.matmul(adjacency, input_features)", b"torch.matmul(adjacency, input_features.detach())"),
                (b"self.b1(states)", b"self.b2(states)")],
        }
        for path, mutations in changes.items():
            actual = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
            expected = git_source("8883790a9508cc6144c578bb3f6cf11571844961", path).replace(b"\r\n", b"\n")
            self.assertEqual(without_registered_h14_hooks(actual, path), expected)
            for before, after in mutations:
                with self.subTest(path=path, mutation=before):
                    self.assertIn(before, actual)
                    with self.assertRaises(AssertionError):
                        self.assertEqual(without_registered_h14_hooks(actual.replace(before, after), path), expected)

    def test_serial_full_update_exact_against_committed_H03_across_bursts_transfer(self):
        for optimizer, mode in (("adam", "soft"), ("rmsprop", "hard")):
            with self.subTest(optimizer=optimizer, mode=mode):
                batch, reference, candidate = self.pair(
                    self.original.DelayedActorGAVDACLearner,
                    batch_size=5, length=7, optimizer=optimizer,
                    target_update_mode=mode,
                )
                for step in range(1, 8):
                    # Mimic an external EA Actor transfer without touching the cadence.
                    if step == 4:
                        with torch.no_grad():
                            for learner in (reference, candidate):
                                for p in learner.mac.parameters():
                                    p.add_(0.0007)
                    rng = torch.get_rng_state()
                    reference.train(batch, step, step)
                    after_reference = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    candidate.train(batch, step, step)
                    self.assertTrue(torch.equal(after_reference, torch.get_rng_state()))
                    self.compare(reference, candidate)
                    self.assertEqual(candidate.actor_steps, step // 2)
                    self.assertEqual(candidate.target_updates, step)
                    if step % 2:
                        self.assertFalse(candidate.mac.hidden_states.requires_grad)

    def test_root_missing_and_explicit_default_complete_updates_exact(self):
        from run import args_sanity_check
        for explicit in (False, True):
            batch, mac, args, scheme = fixture(use_cuda=False, batch_size=5, length=7)
            if explicit:
                args.actor_update_interval = 1
                args.target_update_interval = 1
            self.assertEqual(validate_update_intervals(args), (1, 1))
            config = {**vars(args), "test_nepisode": 8, "batch_size_run": 1}
            args_sanity_check(config, SimpleNamespace(warning=lambda *_: None))
            rng = torch.get_rng_state()
            reference = self.root.GAVDACLearner(
                copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args),
            )
            after = torch.get_rng_state()
            torch.set_rng_state(rng)
            candidate = GAVDACLearner(mac, scheme, LossLog(), args)
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            for step in range(4):
                rng = torch.get_rng_state()
                reference.train(batch, step, step)
                after = torch.get_rng_state()
                torch.set_rng_state(rng)
                candidate.train(batch, step, step)
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                torch.testing.assert_close(states(candidate), states(reference), rtol=0, atol=0)

    def test_root_real_workers_missing_default_exact_to_original_pool(self):
        for explicit in (False, True):
            batch, mac, args, scheme = fixture(
                use_cuda=False, batch_size=5, length=7, learner_workers=2,
            )
            if explicit:
                args.actor_update_interval = args.target_update_interval = 1
            rng = torch.get_rng_state()
            reference = self.root.GAVDACLearner(
                copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args),
            )
            torch.set_rng_state(rng)
            candidate = GAVDACLearner(mac, scheme, LossLog(), args)
            reference.gradient_pool = self.old_pool.GradientPool(reference, scheme)
            try:
                for step in range(3):
                    rng = torch.get_rng_state()
                    reference.train(batch, step, step)
                    after = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    candidate.train(batch, step, step)
                    self.assertTrue(torch.equal(after, torch.get_rng_state()))
                    torch.testing.assert_close(states(candidate), states(reference), rtol=0, atol=0)
            finally:
                reference.close()
                candidate.close()

    def test_actual_direct_entry_only_adds_parent_intervals_and_name(self):
        import direct_main
        configs = {}
        for name in ("gavdac_ea_main", "gavdac_c2_h11_parent"):
            with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
                 patch.object(direct_main, "run") as run:
                direct_main.main()
                configs[name] = run.call_args.args[0]
        root = configs["gavdac_ea_main"]
        parent = configs["gavdac_c2_h11_parent"]
        self.assertEqual(parent, {**root, "name": "gavdac_c2_h11_parent",
                                  "actor_update_interval": 2, "target_update_interval": 1})

    def test_real_workers_exact_against_original_H03_pool_and_varying_participation(self):
        batch, reference, candidate = self.pair(
            self.original.DelayedActorGAVDACLearner, learner_workers=2,
            batch_size=5, length=7,
        )
        reference.gradient_pool = self.old_pool.GradientPool(reference, reference.scheme)
        try:
            for step, (size, length) in enumerate(((5, 7), (1, 4), (3, 6),
                                                   (5, 7), (1, 4), (3, 6)), 1):
                batch, _, _, _ = fixture(
                    use_cuda=False, batch_size=size, length=length,
                )
                reference.train(batch, step, step)
                candidate.train(batch, step, step)
                self.compare(reference, candidate)
                for learner in (reference, candidate):
                    self.assertIsNone(learner.gradient_pool.pending_batch)
                pool = candidate.gradient_pool
                phase = "actor" if step % 2 == 0 else "discard"
                self.assertEqual(pool.last_phase_pids[phase],
                                 pool.last_phase_pids["critic"])
                self.assertEqual(len(pool.last_phase_pids[phase]), min(size, 2))
                self.assertTrue(all(value.is_shared() for value in pool.batch_buffers.values()))
        finally:
            reference.close()
            candidate.close()

    def test_serial_two_workers_uneven_full_grad_optimizer_target_tolerance(self):
        batch, serial, parallel = self.pair(DelayedActorGAVDACLearner)
        parallel.args.learner_workers = 2
        try:
            for step, (size, length) in enumerate(((5, 7), (1, 4), (3, 6),
                                                   (5, 7), (2, 5), (1, 4)), 1):
                batch, _, _, _ = fixture(use_cuda=False, batch_size=size, length=length)
                serial.train(batch, step, step)
                parallel.train(batch, step, step)
                self.compare(serial, parallel, rtol=1e-4, atol=1e-5)
                self.assertIsNone(parallel.gradient_pool.pending_batch)
        finally:
            serial.close()
            parallel.close()

    def test_discard_round_batch_and_graph_lifecycle_and_explicit_worker_class(self):
        batch, _, learner = self.pair(DelayedActorGAVDACLearner)
        class Barrier:
            def wait(self, timeout):
                pass
        with patch.multiple(gradient_pool, _learner=None, _pending=None,
                            _scheme=None, _groups=None, _transport=None), \
             patch.object(torch, "set_num_interop_threads"):
            gradient_pool._initialize(copy.deepcopy(learner.args), learner.scheme, Barrier())
            self.assertIsInstance(gradient_pool._learner, DelayedActorGAVDACLearner)
            weights = {
                name: {key: value.numpy().copy() for key, value in module.state_dict().items()}
                for name, module in gradient_pool._modules(learner).items()
            }
            data = {key: value.numpy().copy()
                    for key, value in batch.data.transition_data.items()}
            mask = batch["filled"][:, :-1].float().clone()
            mask[:, 1:] *= 1 - batch["terminated"][:, :-2].float()
            gradient_pool._gradients("critic", weights, data, float(mask.sum()), 1, 1)
            self.assertTrue(gradient_pool._learner.mac.hidden_states.requires_grad)
            with self.assertRaisesRegex(RuntimeError, "matching cached"):
                gradient_pool._discard_actor(2)
            gradient_pool._discard_actor(1)
            self.assertIsNone(gradient_pool._pending)
            self.assertFalse(gradient_pool._learner.mac.hidden_states.requires_grad)
            with self.assertRaisesRegex(RuntimeError, "matching cached"):
                gradient_pool._discard_actor(1)
            gradient_pool._gradients("critic", weights, data, float(mask.sum()), 1, 2)
            gradient_pool._discard_actor(2)
        learner.args.learner_workers = 2
        try:
            learner.train(batch, 1, 1)
            with self.assertRaisesRegex(RuntimeError, "same critic batch"):
                learner.gradient_pool.discard_actor(copy.deepcopy(batch))
        finally:
            learner.close()

    def test_validation_diagnostics_load_and_actual_post_step_finite_guard(self):
        for name, value in (("actor_update_interval", True), ("actor_update_interval", 2.0),
                            ("actor_update_interval", 3), ("target_update_interval", False),
                            ("target_update_interval", 1.0), ("target_update_interval", 2)):
            args = SimpleNamespace(actor_update_interval=2, target_update_interval=1)
            setattr(args, name, value)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_update_intervals(args)
        batch, reference, candidate = self.pair(DelayedActorGAVDACLearner)
        for step in range(1, 3):
            reference.train(batch, step, step)
            candidate.train(batch, step, step)
        with tempfile.TemporaryDirectory(prefix="h11_parent_load_") as directory:
            reference.save_models(directory)
            diagnostics = json.loads((Path(directory) / "h11_parent_diagnostics.json").read_text())
            self.assertEqual(diagnostics["optimizer_post_step_finite_checks"],
                             {"critic": 2, "actor": 1})
            self.assertEqual(diagnostics["target_updates"], 2)
            self.assertFalse(diagnostics["exact_resume_supported"])
            candidate.load_models(directory)
            torch.testing.assert_close(reference.mac.agent.state_dict(),
                                       candidate.mac.agent.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(reference.agent_optimiser.state_dict(),
                                       candidate.agent_optimiser.state_dict(), rtol=0, atol=0)
        network = copy.deepcopy(reference.critic)
        torch.testing.assert_close(network.state_dict(), reference.critic.state_dict(), rtol=0, atol=0)
        with torch.no_grad():
            candidate.critic_params[0].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            candidate.critic_optimiser.step()


if __name__ == "__main__":
    unittest.main()
