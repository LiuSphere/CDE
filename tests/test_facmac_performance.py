"""Guard inference and replay optimizations without wall-clock assertions."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch as th
from gymnasium.spaces import Box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from components.episode_buffer import EpisodeBatch, ReplayBuffer
from controllers.cqmix_controller import CQMixMAC


def make_batch(batch_size=4, max_seq_length=5):
    scheme = {
        "obs": {"vshape": 20, "group": "agents"},
        "actions": {"vshape": (2,), "group": "agents"},
        "state": {"vshape": 60},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "episode_id": {"vshape": (1,), "episode_const": True, "dtype": th.long},
    }
    batch = EpisodeBatch(scheme, {"agents": 3}, batch_size, max_seq_length)
    generator = th.Generator().manual_seed(17)
    batch.update({
        "obs": th.randn(batch_size, max_seq_length, 3, 20, generator=generator),
        "actions": th.randn(batch_size, max_seq_length, 3, 2, generator=generator),
        "state": th.randn(batch_size, max_seq_length, 60, generator=generator),
        "reward": th.randn(batch_size, max_seq_length, 1, generator=generator),
        "terminated": th.zeros(batch_size, max_seq_length, 1, dtype=th.uint8),
        "episode_id": th.arange(batch_size).view(-1, 1),
    })
    return batch


def make_mac(batch):
    args = SimpleNamespace(
        n_agents=3, n_actions=2, rnn_hidden_dim=16,
        obs_last_action=True, obs_agent_id=True,
        action_spaces=[
            Box(np.array([-2.0, 0.0], dtype=np.float32),
                np.array([1.0, 4.0], dtype=np.float32))
            for _ in range(3)
        ],
        exploration_mode="gaussian", start_steps=0, act_noise=0.15,
    )
    th.manual_seed(23)
    return CQMixMAC(batch.scheme, batch.groups, args)


def make_buffer(batch, size):
    scheme = {key: value for key, value in batch.scheme.items() if key != "filled"}
    return ReplayBuffer(scheme, batch.groups, size, batch.max_seq_length)


class PolicyInferenceTest(unittest.TestCase):
    def test_selection_does_not_retain_recurrent_autograd_graph(self):
        batch = make_batch()
        mac = make_mac(batch)
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            actions = mac.select_actions(batch, t, t_env=100, test_mode=True)
            self.assertFalse(actions.requires_grad)
            self.assertFalse(mac.hidden_states.requires_grad)
            self.assertIsNone(mac.hidden_states.grad_fn)

    def test_default_selection_does_not_slice_episode_batch(self):
        batch = make_batch()
        mac = make_mac(batch)
        mac.init_hidden(batch.batch_size)
        original = EpisodeBatch.__getitem__

        def field_access_only(instance, item):
            self.assertIsInstance(item, str, "Full-batch inference must not slice.")
            return original(instance, item)

        with patch.object(EpisodeBatch, "__getitem__", field_access_only):
            mac.select_actions(batch, 0, t_env=100, test_mode=True)
            mac.args.start_steps = 1000
            mac.select_actions(batch, 1, t_env=0, test_mode=False)

    def test_actions_and_hidden_states_match_forward_and_actor_keeps_gradients(self):
        batch = make_batch()
        for selection in (slice(None), [0, 2], slice(1, 3)):
            with self.subTest(selection=selection):
                selected = batch[selection]
                reference = make_mac(batch)
                candidate = make_mac(batch)
                reference.init_hidden(selected.batch_size)
                candidate.init_hidden(selected.batch_size)
                for t in range(batch.max_seq_length):
                    with th.no_grad():
                        expected = reference.forward(selected, t, select_actions=True)
                        expected_actions = reference._clamp_actions_to_space(
                            expected["actions"]
                        )
                    actual = candidate.select_actions(
                        batch, t, t_env=100, bs=selection, test_mode=True
                    )
                    th.testing.assert_close(actual, expected_actions, rtol=0, atol=0)
                    th.testing.assert_close(
                        candidate.hidden_states, reference.hidden_states, rtol=0, atol=0
                    )

        candidate.init_hidden(batch.batch_size)
        outputs = candidate.forward(batch, 0, select_actions=True)
        outputs["actions"].sum().backward()
        self.assertTrue(all(p.grad is not None for p in candidate.parameters()))
        self.assertTrue(all(bool(p.grad.isfinite().all()) for p in candidate.parameters()))

    def test_gaussian_exploration_remains_bounded(self):
        batch = make_batch()
        mac = make_mac(batch)
        mac.init_hidden(batch.batch_size)
        th.manual_seed(51)
        with th.no_grad():
            expected = mac.forward(batch, 0, select_actions=True)["actions"].clone()
            expected += mac.args.act_noise * th.randn_like(expected)
            expected = mac._clamp_actions_to_space(expected)
        mac.init_hidden(batch.batch_size)
        th.manual_seed(51)
        actual = mac.select_actions(batch, 0, t_env=100, test_mode=False)
        th.testing.assert_close(actual, expected, rtol=0, atol=0)


class ReplaySamplingTest(unittest.TestCase):
    def test_samples_are_independent_and_copy_only_once(self):
        batch = make_batch(batch_size=6)
        buffer = make_buffer(batch, 8)
        buffer.insert_episode_batch(batch)
        for size in (6, 3):
            with self.subTest(size=size):
                np.random.seed(31)
                ids = (np.arange(6) if size == 6
                       else np.random.choice(6, size, replace=False))
                np.random.seed(31)
                clone_calls = []
                original_clone = EpisodeBatch.clone

                def count_clone(instance):
                    clone_calls.append(instance)
                    return original_clone(instance)

                with patch.object(EpisodeBatch, "clone", count_clone), \
                     patch.object(EpisodeBatch, "share", side_effect=AssertionError(
                         "Single-process sampling must not allocate shared storage."
                     )):
                    sample = buffer.sample(size)
                self.assertEqual(len(clone_calls), 1 if size == 6 else 0)
                self.assertEqual(sample.batch_size, size)
                self.assertEqual(sample.device, "cpu")
                self.assertEqual(sample.max_seq_length, batch.max_seq_length)
                for key in batch.scheme:
                    expected = batch[key][ids]
                    th.testing.assert_close(sample[key], expected, rtol=0, atol=0)
                    self.assertEqual(sample[key].dtype, expected.dtype)
                    self.assertNotEqual(sample[key].data_ptr(), buffer[key].data_ptr())
                sample["obs"].zero_()
                sample["episode_id"].fill_(-1)
                th.testing.assert_close(buffer["obs"][:6], batch["obs"])
                th.testing.assert_close(buffer["episode_id"][:6], batch["episode_id"])

    def test_sample_survives_buffer_wraparound(self):
        batch = make_batch(batch_size=4)
        buffer = make_buffer(batch, 5)
        buffer.insert_episode_batch(batch)
        sample = buffer.sample(4)
        expected = {key: sample[key].clone() for key in sample.scheme}
        buffer.insert_episode_batch(batch)
        buffer.insert_episode_batch(batch)
        for key in sample.scheme:
            th.testing.assert_close(sample[key], expected[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
