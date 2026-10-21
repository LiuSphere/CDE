"""One intervention: freeze mixer context only in the Actor gradient."""

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import direct_main
from ea.ssne_base import SSNEBase
from learners.gavdac_learner import GAVDACLearner
from test_gavdac_sequence import LossLog, fixture
from test_paper_experiments import load_config
from tools.experiment_slot import validate_job


class ContextGradientTest(unittest.TestCase):
    def test_entry_config_diff_is_exactly_one_intervention(self):
        base = load_config("gavdac_ea_main")
        expected = {**base, "name": "gavdac_c2_h01", "actor_mixer_context_grad": False}
        self.assertEqual(load_config("gavdac_c2_h01"), expected)
        validate_job("gavdac_c2_h01", 0, "h01s0a", 0, 2)
        with self.assertRaisesRegex(ValueError, "registered hypothesis"):
            validate_job("unregistered_variant", 0, "bad", 0, 2)

    def test_default_complete_update_remains_identical_to_explicit_true(self):
        batch, mac, args, scheme = fixture()
        original = GAVDACLearner(mac, scheme, LossLog(), args)
        explicit = copy.deepcopy(original)
        explicit.args.actor_mixer_context_grad = True
        for step in range(2):
            original.train(batch, step, step)
            explicit.train(batch, step, step)
            for name in ("critic", "mixer", "target_critic", "target_mixer"):
                th.testing.assert_close(
                    getattr(original, name).state_dict(),
                    getattr(explicit, name).state_dict(), rtol=0, atol=0)
            th.testing.assert_close(original.mac.agent.state_dict(),
                                    explicit.mac.agent.state_dict(), rtol=0, atol=0)
            th.testing.assert_close(original.agent_optimiser.state_dict(),
                                    explicit.agent_optimiser.state_dict(), rtol=0, atol=0)

    def test_context_detach_preserves_forward_q_and_action_gradient(self):
        batch, mac, args, scheme = fixture()
        learner = GAVDACLearner(mac, scheme, LossLog(), args)
        policy = mac.forward_sequence(batch, end=-1)
        inputs = learner._build_inputs_sequence(batch, end=-1)
        q, _ = learner.critic(inputs, policy["actions"])
        mixed = learner._mix_q_values(
            learner.mixer, q, batch["state"][:, :-1],
            hidden_states=policy["hidden_states"])
        fixed = learner._mix_q_values(
            learner.mixer, q, batch["state"][:, :-1],
            hidden_states=policy["hidden_states"].detach())
        th.testing.assert_close(fixed, mixed, rtol=0, atol=0)
        grad_full = th.autograd.grad(
            mixed.sum(), policy["hidden_states"], retain_graph=True)[0]
        grad_fixed = th.autograd.grad(
            fixed.sum(), policy["hidden_states"], retain_graph=True, allow_unused=True)[0]
        grad_actions = th.autograd.grad(fixed.sum(), policy["actions"])[0]
        self.assertGreater(grad_full.norm().item(), 0)
        self.assertIsNone(grad_fixed)
        self.assertGreater(grad_actions.norm().item(), 0)

    def test_complete_update_blocks_context_only_and_keeps_critic_update(self):
        batch, mac, args, scheme = fixture()
        base = GAVDACLearner(mac, scheme, LossLog(), args)
        candidate = copy.deepcopy(base)
        candidate.args.actor_mixer_context_grad = False
        captured, contexts = {}, []
        forward = candidate.mac.forward_sequence

        def capture(*args, **kwargs):
            result = forward(*args, **kwargs)
            for name in ("actions", "hidden_states"):
                result[name].retain_grad()
                captured[name] = result[name]
            return result

        hook = candidate.mixer.register_forward_pre_hook(
            lambda module, args, kwargs: contexts.append(kwargs["hidden_states"].requires_grad),
            with_kwargs=True)
        try:
            base.train(batch, 1, 1)
            with patch.object(candidate.mac, "forward_sequence", capture):
                candidate.train(batch, 1, 1)
        finally:
            hook.remove()
        self.assertEqual(contexts, [False, False])
        self.assertIsNone(captured["hidden_states"].grad)
        self.assertGreater(captured["actions"].grad.norm().item(), 0)
        self.assertGreater(candidate.mac.agent.rnn.weight_ih.grad.norm().item(), 0)
        self.assertTrue(all(math.isfinite(v) for v in candidate.logger.values.values()))
        self.assertTrue(all(p.dtype == th.float32 and p.device.type == "cpu"
                            for p in candidate.mac.parameters()))
        for name in ("critic", "mixer"):
            th.testing.assert_close(getattr(base, name).state_dict(),
                                    getattr(candidate, name).state_dict(), rtol=0, atol=0)
        self.assertTrue(any(not th.equal(a, b) for a, b in zip(
            base.mac.parameters(), candidate.mac.parameters(), strict=True)))
        self.assertTrue(all(p.requires_grad for p in candidate.critic_params))
        self.assertTrue(all(p.grad is None for p in candidate.target_mac.parameters()))

    def test_real_h01_crosses_warmup_and_completes_ea_cycle(self):
        counts = {"learner": 0, "ea": 0}
        learn, evolve = GAVDACLearner.train, SSNEBase.epoch

        def train(learner, *args, **kwargs):
            self.assertFalse(learner.args.actor_mixer_context_grad)
            counts["learner"] += 1
            return learn(learner, *args, **kwargs)

        def epoch(evolver, *args, **kwargs):
            counts["ea"] += 1
            return evolve(evolver, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            argv = ["direct_main.py", "--config", "gavdac_c2_h01"]
            overrides = {
                "t_max": 12500, "wandb_mode": "disabled", "save_model": False,
                "showcase_enabled": False, "local_results_path": directory,
                "env_args.result_dir": directory, "wandb_dir": f"{directory}/wandb",
                "unique_token": "h01_test",
            }
            for key, value in overrides.items():
                argv += ["--override", f"{key}={value}"]
            with patch.object(sys, "argv", argv), \
                 patch.object(GAVDACLearner, "train", train), \
                 patch.object(SSNEBase, "epoch", epoch):
                direct_main.main()
        self.assertGreater(counts["learner"], 0)
        self.assertGreater(counts["ea"], 0)


if __name__ == "__main__":
    unittest.main()
