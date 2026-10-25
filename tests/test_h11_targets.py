"""H11 target cadence, inherited gradients and NEW saved-source qualification."""

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

import numpy as np
import torch
from gymnasium.spaces import Box

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from controllers.cqmix_controller import CQMixMAC
from direct_main import load_yaml, recursive_dict_update
from learners import gradient_pool
from learners.delayed_actor_learner import DelayedActorGAVDACLearner, validate_update_intervals
from learners.delayed_targets_learner import DelayedTargetsGAVDACLearner
from learners.gavdac_learner import GAVDACLearner
from runners.episode_runner import EpisodeRunner
from test_gavdac_sequence import LossLog, fixture
from test_h11_parent import git_source, import_source, states
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed

PARENT = "4e3aa00eacf814e3d328be781e4af8632ec0488f"


def targets(learner):
    return {name: copy.deepcopy(module.state_dict())
            for name, module in gradient_pool._modules(learner).items()
            if name.startswith("target")}


def online(learner):
    return {key: value for key, value in states(learner).items()
            if not key.startswith("target")}


class H11TargetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"

    def make(self, cls=DelayedTargetsGAVDACLearner, **options):
        target = 1 if cls is DelayedActorGAVDACLearner else 2
        batch, mac, args, scheme = fixture(
            use_cuda=False, actor_update_interval=2, target_update_interval=target,
            **options,
        )
        return batch, cls(mac, scheme, LossLog(), args)

    def test_same_initial_draws_counts_and_first_online_update_as_parent(self):
        batch, mac, args, scheme = fixture(
            use_cuda=False, actor_update_interval=2, target_update_interval=1,
            rnn_hidden_dim=48, mixing_embed_dim=32, hypernet_embed=64,
            action_spaces=[Box(np.array([0, -1], dtype=np.float32),
                               np.array([1, 1], dtype=np.float32)) for _ in range(3)],
        )
        rng = torch.get_rng_state()
        parent = DelayedActorGAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        args.target_update_interval = 2
        child = DelayedTargetsGAVDACLearner(mac, scheme, LossLog(), args)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        torch.testing.assert_close(states(parent), states(child), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in child.agent_params), 15458)
        self.assertEqual(sum(p.numel() for p in child.critic_params), 65915)
        before = targets(child)
        for learner in (parent, child):
            rng = torch.get_rng_state()
            learner.train(batch, 1, 1)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.testing.assert_close(online(parent), online(child), rtol=0, atol=0)
        torch.testing.assert_close(targets(child), before, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, before[name][key])
                            for name, state in targets(parent).items()
                            for key, value in state.items()))
        # Parent targets already moved; equality after this point is not required.
        for learner in (parent, child):
            learner.train(batch, 2, 2)
        self.assertEqual((child.critic_steps, child.actor_steps, child.target_updates), (2, 1, 1))

    def test_exact_odd_even_target_oracle_post_actor_order_bursts_and_transfer(self):
        for mode in ("soft", "hard", "exponential_moving_average"):
            with self.subTest(mode=mode):
                batch, learner = self.make(target_update_mode=mode)
                order = []
                hooks = [
                    learner.critic_optimiser.register_step_post_hook(
                        lambda *_: order.append("critic")),
                    learner.agent_optimiser.register_step_post_hook(
                        lambda *_: order.append("actor")),
                ]
                original = learner._target_phase
                def phase():
                    order.append("target_phase")
                    original()
                try:
                    with patch.object(learner, "_target_phase", side_effect=phase):
                        for step in range(1, 8):
                            if step == 4:
                                with torch.no_grad():
                                    for parameter in learner.mac.parameters():
                                        parameter.add_(.0007)
                            before = targets(learner)
                            order.clear()
                            rng = torch.get_rng_state()
                            learner.train(batch, step, step)
                            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                            self.assertEqual(order, ["critic", "actor", "target_phase"]
                                             if step % 2 == 0 else ["critic", "target_phase"])
                            expected = copy.deepcopy(before)
                            if step % 2 == 0:
                                modules = gradient_pool._modules(learner)
                                for name, state in expected.items():
                                    live = modules[name.removeprefix("target_")].state_dict()
                                    for key, value in state.items():
                                        if mode == "hard":
                                            value.copy_(live[key])
                                        elif value.is_floating_point():
                                            value.lerp_(live[key], .002)
                            torch.testing.assert_close(targets(learner), expected, rtol=0, atol=0)
                            self.assertEqual(learner.actor_steps, step // 2)
                            self.assertEqual(learner.target_updates, step // 2)
                            self.assertEqual(learner.finite_checks, step)
                            self.assertEqual(learner.optimizer_finite_checks,
                                             {"critic": step, "actor": step // 2})
                            for name, module in gradient_pool._modules(learner).items():
                                if name.startswith("target"):
                                    self.assertTrue(all(p.grad is None for p in module.parameters()))
                finally:
                    for hook in hooks:
                        hook.remove()

    def test_real_two_worker_uneven_varying_updates_and_graph_release(self):
        batch, serial = self.make()
        batch, parallel = self.make(learner_workers=2)
        # fixture resets Actor draws; Critic/mixer construction follows the same RNG.
        torch.testing.assert_close(states(serial), states(parallel), rtol=0, atol=0)
        try:
            for step, (size, length) in enumerate(((5, 7), (1, 4), (3, 6), (5, 7),
                                                   (1, 4), (3, 6), (2, 5)), 1):
                batch, _, _, _ = fixture(use_cuda=False, batch_size=size, length=length)
                if step == 4:
                    with torch.no_grad():
                        for learner in (serial, parallel):
                            for p in learner.mac.parameters():
                                p.add_(.0007)
                serial.train(batch, step, step)
                parallel.train(batch, step, step)
                torch.testing.assert_close(states(serial), states(parallel), rtol=1e-4, atol=1e-5)
                self.assertEqual((parallel.critic_steps, parallel.actor_steps,
                                  parallel.target_updates, parallel.finite_checks),
                                 (step, step // 2, step // 2, step))
                pool = parallel.gradient_pool
                self.assertIsNone(pool.pending_batch)
                phase = "actor" if step % 2 == 0 else "discard"
                self.assertEqual(pool.last_phase_pids[phase], pool.last_phase_pids["critic"])
                self.assertEqual(len(pool.last_phase_pids[phase]), min(size, 2))
                self.assertTrue(all(value.is_shared() for value in pool.batch_buffers.values()))
        finally:
            serial.close()
            parallel.close()

    def test_explicit_worker_class_and_only_parent_owns_update_counters(self):
        batch, learner = self.make()
        class Barrier:
            def wait(self, timeout):
                pass
        with patch.multiple(gradient_pool, _learner=None, _pending=None,
                            _scheme=None, _groups=None, _transport=None), \
             patch.object(torch, "set_num_interop_threads"):
            gradient_pool._initialize(copy.deepcopy(learner.args), learner.scheme, Barrier())
            worker = gradient_pool._learner
            self.assertIs(type(worker), DelayedTargetsGAVDACLearner)
            weights = {
                name: {key: value.numpy().copy() for key, value in module.state_dict().items()}
                for name, module in gradient_pool._modules(learner).items()
            }
            data = {key: value.numpy().copy()
                    for key, value in batch.data.transition_data.items()}
            mask = batch["filled"][:, :-1].float().clone()
            mask[:, 1:] *= 1 - batch["terminated"][:, :-2].float()
            gradient_pool._gradients("critic", weights, data, float(mask.sum()), 1, 1)
            gradient_pool._discard_actor(1)
            self.assertIsNone(gradient_pool._pending)
            self.assertFalse(worker.mac.hidden_states.requires_grad)
            self.assertEqual((worker.critic_steps, worker.actor_steps, worker.target_updates,
                              worker.finite_checks), (0, 0, 0, 0))
            self.assertEqual(worker.optimizer_finite_checks, {"critic": 0, "actor": 0})

    def test_original_live_action_context_recurrence_and_critic_immutability(self):
        batch, learner = self.make()
        policy, inputs, loss = learner._critic_loss(batch)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in learner.agent_params))
        learner.critic_optimiser.step()
        before = {name: copy.deepcopy(module.state_dict())
                  for name, module in (("critic", learner.critic), ("mixer", learner.mixer))}
        grads = [p.grad.clone() if p.grad is not None else None for p in learner.critic_params]
        for name in ("actions", "hidden_states", "raw_actions"):
            policy[name].retain_grad()
        learner._set_critic_grad(False)
        seen = []
        hook = learner.mixer.register_forward_hook(lambda module, args, result: seen.append(result[0]))
        try:
            actor_loss = learner._actor_loss(batch, policy, inputs)
            expected = -seen[0].mean() + .001 * policy["raw_actions"].square().mean()
            torch.testing.assert_close(actor_loss, expected, rtol=0, atol=0)
            actor_loss.backward()
        finally:
            hook.remove()
            learner._set_critic_grad(True)
        for name in ("actions", "hidden_states", "raw_actions"):
            self.assertGreater(float(policy[name].grad.abs().sum()), 0)
        self.assertGreater(float(learner.mac.agent.rnn.weight_hh.grad.abs().sum()), 0)
        torch.testing.assert_close(
            [p.grad for p in learner.critic_params], grads, rtol=0, atol=0)
        for name, module in (("critic", learner.critic), ("mixer", learner.mixer)):
            torch.testing.assert_close(module.state_dict(), before[name], rtol=0, atol=0)

    def test_original_one_step_MSE_terminal_padding_global_denominator_and_detach(self):
        batch, learner = self.make(batch_size=5, length=7)
        batch["filled"][-1, 3:] = 0
        seen = []
        original = learner._mix_q_values
        def mixing(*args, **kwargs):
            value = original(*args, **kwargs)
            seen.append(value)
            return value
        with patch.object(learner, "_mix_q_values", side_effect=mixing):
            _, _, loss = learner._critic_loss(batch)
        current, target = seen
        terminal = batch["terminated"][:, :-1].float()
        expected_target = (batch["reward"][:, :-1].expand_as(target)
                           + learner.args.gamma * (1 - terminal.expand_as(target)) * target)
        mask = batch["filled"][:, :-1].float().clone()
        mask[:, 1:] *= 1 - terminal[:, :-1]
        mask = mask.expand_as(current)
        expected = ((expected_target.detach() - current) * mask).square().sum() / mask.sum()
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        self.assertFalse(target.requires_grad)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in learner.agent_params))
        for name, module in gradient_pool._modules(learner).items():
            if name.startswith("target"):
                self.assertTrue(all(p.grad is None for p in module.parameters()))
        learner.critic_optimiser.zero_grad(set_to_none=True)
        _, _, shard_loss = learner._critic_loss(batch, denominator=float(mask.sum()) * 2)
        torch.testing.assert_close(shard_loss, loss.detach() / 2, rtol=0, atol=0)

    def test_registered_interval_combinations_and_explicit_class_rejection(self):
        for actor, target in ((1, 1), (2, 1), (2, 2)):
            self.assertEqual(validate_update_intervals(SimpleNamespace(
                actor_update_interval=actor, target_update_interval=target)), (actor, target))
        for actor, target in ((True, 1), (2.0, 1), (0, 1), (3, 1),
                              (1, 2), (2, False), (2, 2.0), (2, 0), (2, 3)):
            with self.subTest(actor=actor, target=target), self.assertRaises(ValueError):
                validate_update_intervals(SimpleNamespace(
                    actor_update_interval=actor, target_update_interval=target))
        batch, mac, args, scheme = fixture(
            use_cuda=False, actor_update_interval=2, target_update_interval=2,
        )
        with self.assertRaisesRegex(ValueError, "class does not match"):
            DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
        args.target_update_interval = 1
        with self.assertRaisesRegex(ValueError, "class does not match"):
            DelayedTargetsGAVDACLearner(mac, scheme, LossLog(), args)

    def test_invalid_target_mode_visible_on_odd_and_even_calls(self):
        for step in (1, 2):
            _, learner = self.make(target_update_mode="unknown")
            learner.critic_steps = step
            with self.assertRaisesRegex(ValueError, "unknown target update mode"):
                learner._target_phase()

    def test_targets_checked_finite_even_on_skipped_odd_synchronization(self):
        batch, learner = self.make()
        original = learner._target_phase
        def injected_phase():
            original()
            with torch.no_grad():
                next(iter(learner.target_mac.parameters())).fill_(float("nan"))
        with patch.object(learner, "_target_phase", side_effect=injected_phase):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                learner.train(batch, 1, 1)
        self.assertEqual((learner.critic_steps, learner.actor_steps, learner.target_updates,
                          learner.finite_checks), (1, 0, 0, 0))
        self.assertEqual(learner.optimizer_finite_checks, {"critic": 1, "actor": 0})

    def test_state_load_deepcopy_and_diagnostics_do_not_claim_exact_resume(self):
        batch, learner = self.make()
        for step in range(1, 4):
            learner.train(batch, step, step)
        with tempfile.TemporaryDirectory(prefix="h11_child_load_") as directory:
            learner.save_models(directory)
            payload = json.loads((Path(directory) / "h11_diagnostics.json").read_text())
            self.assertEqual(payload["lineage_role"], "H11_child_of_reconstructed_H03")
            self.assertEqual((payload["critic_optimizer_steps"], payload["actor_optimizer_steps"],
                              payload["target_updates"], payload["finite_update_checks"]),
                             (3, 1, 1, 3))
            self.assertEqual((payload["actor_update_interval"], payload["target_update_interval"],
                              payload["target_update_tau"]), (2, 2, .002))
            self.assertFalse(payload["exact_resume_supported"])
            _, restored = self.make()
            restored.load_models(directory)
            torch.testing.assert_close(restored.mac.agent.state_dict(),
                                       learner.mac.agent.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(restored.agent_optimiser.state_dict(),
                                       learner.agent_optimiser.state_dict(), rtol=0, atol=0)
            self.assertEqual(restored.critic_steps, 0)
            self.assertFalse((Path(directory) / "critic.th").exists())
        network = copy.deepcopy(learner.critic)
        torch.testing.assert_close(network.state_dict(), learner.critic.state_dict(), rtol=0, atol=0)

    def test_parent_unchanged_against_separately_committed_reconstruction(self):
        with tempfile.TemporaryDirectory(prefix="h11_actual_parent_") as directory:
            module = import_source("h11_parent_commit_reference",
                                   git_source(PARENT, "src/learners/delayed_actor_learner.py"),
                                   directory)
            try:
                batch, mac, args, scheme = fixture(
                    use_cuda=False, actor_update_interval=2, target_update_interval=1,
                )
                rng = torch.get_rng_state()
                reference = module.DelayedActorGAVDACLearner(
                    copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
                after = torch.get_rng_state()
                torch.set_rng_state(rng)
                parent = DelayedActorGAVDACLearner(mac, scheme, LossLog(), args)
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                for step in range(1, 8):
                    reference.train(batch, step, step)
                    parent.train(batch, step, step)
                    torch.testing.assert_close(states(parent), states(reference), rtol=0, atol=0)
                    self.assertEqual((parent.critic_steps, parent.actor_steps,
                                      parent.target_updates, parent.finite_checks),
                                     (step, step // 2, step, step))
            finally:
                sys.modules.pop("h11_parent_commit_reference", None)

    def test_actual_entry_config_and_H11_strict_saved_source_physical_inference(self):
        import direct_main
        configs = {}
        for name in ("gavdac_c2_h11_parent", "gavdac_c2_h11"):
            with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
                 patch.object(direct_main, "run") as run:
                direct_main.main()
                configs[name] = run.call_args.args[0]
        self.assertEqual(configs["gavdac_c2_h11"], {
            **configs["gavdac_c2_h11_parent"], "name": "gavdac_c2_h11",
            "target_update_interval": 2,
        })
        with tempfile.TemporaryDirectory(dir=ROOT / "results", prefix="h11_strict_") as directory:
            config = configs["gavdac_c2_h11"]
            config.update(device="cpu", rollout_workers=2, learner_workers=2)
            config["env_args"].update(result_dir=directory, run_id="h11_strict_fixture")
            args = SimpleNamespace(**copy.deepcopy(config))
            runner = EpisodeRunner(args, LossLog())
            info = runner.get_env_info()
            for key in ("n_agents", "n_actions", "state_shape", "obs_shape", "action_spaces"):
                setattr(args, key, info[key])
            batch, _, _, scheme = fixture(use_cuda=False, length=7)
            mac = CQMixMAC(scheme, {"agents": 3}, args)
            runner.setup(scheme, {"agents": 3}, {}, mac)
            learner = DelayedTargetsGAVDACLearner(mac, scheme, LossLog(), args)
            try:
                for step in range(1, 3):
                    learner.train(batch, step, step)
                self.assertEqual((learner.critic_steps, learner.actor_steps, learner.target_updates),
                                 (2, 1, 1))
                evaluation = evaluate_fixed(
                    runner, mac, [731, 732], .35, "validation", "rl", 24)
                store = BestPolicyStore(Path(directory) / "best", config)
                checkpoint = store.consider(mac, "rl", 24, evaluation, runner.env)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                for name in ("learners/delayed_actor_learner.py",
                             "learners/delayed_targets_learner.py", "learners/gradient_pool.py"):
                    self.assertIn(name, metadata["source_sha256"])
            finally:
                learner.close()
                runner.close_env()
            output = Path(directory) / "strict_inference"
            process = subprocess.run(
                ["uv", "run", "--frozen", "python", "tools/evaluate_microgrid_comparison.py",
                 "--checkpoint", str(checkpoint), "--split", "validation",
                 "--mode", "continuous", "--rows", "731", "732", "--output", str(output)],
                cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                   OPENBLAS_NUM_THREADS="1"), capture_output=True, timeout=90,
            )
            self.assertEqual(process.returncode, 0, process.stderr.decode("utf-8", errors="replace"))
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["physical_checks_passed"])
            self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
            self.assertEqual(summary["data_sha256"], metadata["data_sha256"])
            for name, digest in metadata["source_sha256"].items():
                self.assertEqual(hashlib.sha256((output / "runtime" / name).read_bytes()).hexdigest(),
                                 digest)
            for name in ("gavdac_c2_h11_parent.yaml", "gavdac_c2_h11.yaml"):
                saved = Path(metadata["source_snapshot"]) / "config/algs" / name
                runtime = output / "runtime/config/algs" / name
                self.assertEqual(saved.read_bytes(), runtime.read_bytes())


if __name__ == "__main__":
    unittest.main()
