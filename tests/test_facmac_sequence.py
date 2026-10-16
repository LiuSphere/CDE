"""Differential checks for sequence execution and complete FACMAC updates."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th
from gymnasium.spaces import Box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from components.episode_buffer import EpisodeBatch
from controllers.cqmix_controller import CQMixMAC
from learners.facmac_learner import FACMACLearner
from modules.graph_mixer import GraphDec
from modules.graph.gin_layer import GINGraphConvolution


class LossLog:
    def __init__(self):
        self.values = {}

    def log_stat(self, name, value, step):
        self.values[name] = value


def fixture(n_agents=3, batch_size=4, length=5, **options):
    scheme = {
        "obs": {"vshape": 20, "group": "agents"},
        "actions": {"vshape": (2,), "group": "agents"},
        "state": {"vshape": 20 * n_agents},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    batch = EpisodeBatch(scheme, {"agents": n_agents}, batch_size, length)
    generator = th.Generator().manual_seed(43)
    for key in ("obs", "state", "actions", "reward"):
        batch.update({key: th.randn(batch[key].shape, generator=generator) * 0.25})
    if length > 2:
        batch["terminated"][0, -2] = 1
        batch["filled"][-1, -1] = 0
    args = SimpleNamespace(
        n_agents=n_agents, n_actions=2, rnn_hidden_dim=16,
        state_shape=20 * n_agents, obs_last_action=True, obs_agent_id=True,
        mixing_embed_dim=16, hypernet_embed=16, mixer_weight_scale=0.1,
        action_spaces=[Box(np.array([-2.0, 0.0], dtype=np.float32),
                           np.array([1.0, 4.0], dtype=np.float32))
                       for _ in range(n_agents)],
        recurrent_critic=False, optimizer="adam", lr=3e-4, critic_lr=1e-4,
        grad_norm_clip=5, gamma=0.99, learner_log_interval=1,
        target_update_mode="soft", target_update_tau=0.002,
        optim_alpha=0.99, optim_eps=1e-5,
    )
    vars(args).update(options)
    th.manual_seed(47)
    mac = CQMixMAC(scheme, batch.groups, args)
    return batch, mac, args, scheme


def stepwise_sequence(mac, batch, end=None):
    actions, raw, hidden = [], [], []
    mac.init_hidden(batch.batch_size)
    length = batch["obs"][:, :end].shape[1]
    for t in range(length):
        result = mac.forward(batch, t, select_actions=True)
        actions.append(result["actions"])
        raw.append(result["raw_actions"])
        hidden.append(mac.hidden_states.reshape(batch.batch_size, mac.n_agents, -1))
    return {
        "actions": th.stack(actions, dim=1),
        "raw_actions": th.stack(raw, dim=1),
        "hidden_states": th.stack(hidden, dim=1),
    }


def reference_update(learner, batch):
    """Keep the original separate critic-feature and actor-policy passes."""
    bs, length = batch.batch_size, batch.max_seq_length
    rewards, actions = batch["reward"][:, :-1], batch["actions"][:, :-1]
    terminated = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    mask[:, 1:] *= 1 - terminated[:, :-1]
    with th.no_grad():
        target = stepwise_sequence(learner.target_mac, batch)
        target_actions = learner.target_mac._clamp_actions_to_space(target["actions"])
        current = stepwise_sequence(learner.mac, batch)

    q_taken, _ = learner.critic(
        learner._build_inputs_sequence(batch, end=-1), actions.detach()
    )
    if learner.mixer is not None:
        q_taken = learner._mix_q_values(
            learner.mixer, q_taken, batch["state"][:, :-1],
            hidden_states=current["hidden_states"][:, :-1],
        )
    else:
        q_taken = q_taken.view(bs, length - 1, learner.n_agents)

    with th.no_grad():
        target_q, _ = learner.target_critic(
            learner._build_inputs_sequence(batch, start=1), target_actions[:, 1:]
        )
        if learner.mixer is not None:
            target_q = learner._mix_q_values(
                learner.target_mixer, target_q, batch["state"][:, 1:],
                hidden_states=target["hidden_states"][:, 1:],
            )
        else:
            target_q = target_q.view(bs, length - 1, learner.n_agents)
        targets = rewards.expand_as(target_q) + learner.args.gamma * (
            1 - terminated.expand_as(target_q)
        ) * target_q
    td_error = targets - q_taken
    mask = mask.expand_as(td_error)
    critic_loss = ((td_error * mask) ** 2).sum() / mask.sum()
    learner.critic_optimiser.zero_grad(set_to_none=True)
    critic_loss.backward()
    th.nn.utils.clip_grad_norm_(learner.critic_params, learner.args.grad_norm_clip)
    learner.critic_optimiser.step()

    actor = stepwise_sequence(learner.mac, batch, end=-1)
    learner._set_critic_grad(False)
    try:
        policy_q, _ = learner.critic(
            learner._build_inputs_sequence(batch, end=-1), actor["actions"]
        )
        if learner.mixer is not None:
            policy_q = learner._mix_q_values(
                learner.mixer, policy_q, batch["state"][:, :-1],
                hidden_states=actor["hidden_states"],
            )
        policy_loss = -policy_q.mean() + actor["raw_actions"].square().mean() * 1e-3
        learner.agent_optimiser.zero_grad(set_to_none=True)
        policy_loss.backward()
    finally:
        learner._set_critic_grad(True)
    th.nn.utils.clip_grad_norm_(learner.agent_params, learner.args.grad_norm_clip)
    learner.agent_optimiser.step()
    if learner.args.target_update_mode == "hard":
        learner._update_targets()
    else:
        learner._update_targets_soft(learner.args.target_update_tau)
    return critic_loss.item(), policy_loss.item()


class PolicySequenceTest(unittest.TestCase):
    def test_sequence_matches_steps_and_gradients(self):
        for length, last_action, agent_id in (
            (1, True, True), (5, True, True), (25, True, True),
            (5, False, True), (5, True, False), (5, False, False),
        ):
            with self.subTest(length=length, last_action=last_action, agent_id=agent_id):
                batch, reference, _, _ = fixture(
                    length=length, obs_last_action=last_action, obs_agent_id=agent_id
                )
                candidate = copy.deepcopy(reference)
                expected = stepwise_sequence(reference, batch)
                actual = candidate.forward_sequence(batch)
                th.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
                for outputs in (expected, actual):
                    loss = (outputs["actions"].square().mean()
                            + outputs["hidden_states"].square().mean() * 0.1)
                    loss.backward()
                for expected_param, actual_param in zip(
                    reference.parameters(), candidate.parameters(), strict=True
                ):
                    th.testing.assert_close(
                        actual_param.grad, expected_param.grad, rtol=3e-5, atol=1e-6
                    )
                th.testing.assert_close(
                    candidate.hidden_states, reference.hidden_states, rtol=2e-5, atol=1e-6
                )

    def test_prefix_and_noncontiguous_batch(self):
        batch, reference, _, _ = fixture(batch_size=8)
        batch = batch[::2]
        candidate = copy.deepcopy(reference)
        expected = stepwise_sequence(reference, batch, end=-1)
        actual = candidate.forward_sequence(batch, end=-1)
        th.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
        self.assertEqual(actual["actions"].shape, (4, 4, 3, 2))
        self.assertEqual(reference.agent.state_dict().keys(), candidate.agent.state_dict().keys())
        candidate.agent.load_state_dict(reference.agent.state_dict(), strict=True)

    def test_empty_prefix_is_rejected(self):
        batch, mac, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "at least one"):
            mac.forward_sequence(batch, end=0)


class LearnerSequenceTest(unittest.TestCase):
    def test_policy_recurrence_is_run_once_per_update(self):
        batch, mac, args, scheme = fixture()
        learner = FACMACLearner(mac, scheme, LossLog(), args)
        calls = []
        hook = mac.agent.rnn.register_forward_hook(lambda *unused: calls.append(1))
        try:
            learner.train(batch, 0, 0)
        finally:
            hook.remove()
        self.assertEqual(len(calls), batch.max_seq_length - 1)
        self.assertFalse(learner.target_mac.hidden_states.requires_grad)
        self.assertTrue(all(p.grad is None for p in learner.target_mac.parameters()))

    def test_complete_updates_match_reference(self):
        for n_agents, optimizer, target_mode in (
            (3, "adam", "soft"), (3, "rmsprop", "hard"), (1, "adam", "soft"),
        ):
            with self.subTest(n_agents=n_agents, optimizer=optimizer, target=target_mode):
                batch, mac, args, scheme = fixture(
                    n_agents=n_agents, optimizer=optimizer, target_update_mode=target_mode
                )
                logger = LossLog()
                candidate = FACMACLearner(mac, scheme, logger, args)
                reference = copy.deepcopy(candidate)
                for update in range(3):
                    expected_losses = reference_update(reference, batch)
                    candidate.train(batch, update, update)
                    actual_losses = (
                        logger.values["loss/critic_loss"],
                        logger.values["loss/policy_loss"],
                    )
                    np.testing.assert_allclose(actual_losses, expected_losses, rtol=3e-5, atol=1e-6)
                    names = ["mac.agent", "target_mac.agent", "critic", "target_critic"]
                    if n_agents > 1:
                        names.extend(["mixer", "target_mixer"])
                    for name in names:
                        expected_module, actual_module = reference, candidate
                        for attribute in name.split("."):
                            expected_module = getattr(expected_module, attribute)
                            actual_module = getattr(actual_module, attribute)
                        th.testing.assert_close(
                            actual_module.state_dict(), expected_module.state_dict(),
                            rtol=3e-5, atol=2e-6,
                        )
                        for expected_param, actual_param in zip(
                            expected_module.parameters(), actual_module.parameters(), strict=True
                        ):
                            th.testing.assert_close(
                                actual_param.grad, expected_param.grad, rtol=5e-5, atol=2e-6
                            )
                    for name in ("agent_optimiser", "critic_optimiser"):
                        th.testing.assert_close(
                            getattr(candidate, name).state_dict(),
                            getattr(reference, name).state_dict(), rtol=5e-5, atol=2e-6,
                        )


class MixerReadoutTest(unittest.TestCase):
    def test_graph_contraction_matches_scaled_weights_and_gradients(self):
        th.manual_seed(71)
        for operation in ("sigmoid", "abs", "clamp", None):
            with self.subTest(operation=operation):
                reference = GINGraphConvolution(
                    1, 16, 60, 16, weights_operation=operation, weight_scale=0.1
                )
                candidate = copy.deepcopy(reference)
                inputs = [
                    th.randn(20, 3, 1, requires_grad=True),
                    th.randn(20, 3, 3, requires_grad=True),
                    th.randn(20, 60, requires_grad=True),
                ]
                other = [value.detach().clone().requires_grad_() for value in inputs]
                features, adjacency, states = inputs
                weight1 = reference.w1(states).view(-1, 1, reference.hidden_features)
                weight2 = reference.w2(states).view(-1, reference.hidden_features, 16)
                if operation == "sigmoid":
                    weight1, weight2 = weight1.sigmoid() * 0.1, weight2.sigmoid() * 0.1
                elif operation == "abs":
                    weight1, weight2 = weight1.abs(), weight2.abs()
                elif operation == "clamp":
                    weight1, weight2 = weight1.relu(), weight2.relu()
                hidden = th.nn.functional.leaky_relu(
                    th.matmul(th.matmul(adjacency, features), weight1)
                    + reference.b1(states).view(20, 1, -1)
                )
                expected = th.matmul(hidden, weight2) + reference.b2(states).view(20, 1, -1)
                actual = candidate(*other)
                th.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-6)
                expected.square().mean().backward()
                actual.square().mean().backward()
                for left, right in zip(inputs, other, strict=True):
                    th.testing.assert_close(right.grad, left.grad, rtol=5e-5, atol=1e-6)
                for left, right in zip(reference.parameters(), candidate.parameters(), strict=True):
                    th.testing.assert_close(right.grad, left.grad, rtol=5e-5, atol=1e-6)

    def test_total_q_skips_unused_local_reward_heads(self):
        batch, _, args, _ = fixture()
        mixer = GraphDec(args)
        calls = []
        hooks = [
            head.register_forward_hook(lambda *unused: calls.append(1))
            for head in (mixer.mixing_GNN.wout_per_node, mixer.mixing_GNN.bout_per_node)
        ]
        try:
            mixer(
                th.randn(4 * 5 * 3, 1), batch["state"],
                hidden_states=th.randn(4, 5, 3, 16),
            )
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(calls, [])

    def test_optional_local_rewards_preserve_total_q_and_gradients(self):
        batch, _, args, _ = fixture()
        reference = GraphDec(args)
        candidate = copy.deepcopy(reference)
        inputs = {
            "agent_qs": th.randn(4, 5, 3, 1, requires_grad=True),
            "states": batch["state"].clone().requires_grad_(),
            "hidden_states": th.randn(4, 5, 3, 16, requires_grad=True),
        }
        other = {key: value.detach().clone().requires_grad_() for key, value in inputs.items()}
        reward = th.randn(4, 5, 1)
        q_reference, local_rewards, active_reference = reference(**inputs, team_rewards=reward)
        q_actual, local_actual, active_actual = candidate(**other)
        th.testing.assert_close(q_actual, q_reference, rtol=0, atol=0)
        th.testing.assert_close(local_rewards.sum(dim=-1, keepdim=True), reward)
        self.assertIsNone(local_actual)
        self.assertTrue(th.equal(active_actual, active_reference))
        q_reference.sum().backward()
        q_actual.sum().backward()
        for key in inputs:
            th.testing.assert_close(inputs[key].grad, other[key].grad, rtol=0, atol=0)
        for expected, actual in zip(reference.parameters(), candidate.parameters(), strict=True):
            th.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
