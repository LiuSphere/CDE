"""H14 independent signed-mixer oracle and actual-ancestor complete updates."""

import copy
import hashlib
import importlib
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
import yaml
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from controllers.cqmix_controller import CQMixMAC
from ea.mod_neuro_evo import SSNE
from learners import gradient_pool
from learners import gavdac_learner as base_module
from learners.gavdac_learner import GAVDACLearner
from learners.delayed_actor_learner import DelayedActorGAVDACLearner
from learners.delayed_actor_huber_learner import HuberDelayedActorGAVDACLearner
from learners.signed_mixer_huber_learner import SignedMixerHuberGAVDACLearner, signed_mixer_enabled
from modules.graph_mixer import GraphDec
from modules.graph.gnn import GNN
from modules.graph.gin_layer import GINGraphConvolution
from runners.episode_runner import EpisodeRunner
from test_gavdac_sequence import fixture, LossLog
from test_h11_parent import git_source, import_source, states
from test_policy_evaluation import QuietLogger
from tools.freeze_h11_protocol import entry_config
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed

PARENT = "8883790a9508cc6144c578bb3f6cf11571844961"
H03 = "e9975bf8c7af85a30b20d41f95c6436698463fb0"
PRE = "f18155fec81ab51539f0409a047dbe6a42e11ab2"
PARENT_OPTIMIZER = entry_config("gavdac_c2_h12")["optimizer"]


def linear(layer, inputs):
    return F.linear(inputs, layer.weight, layer.bias)


def mlp(module, inputs):
    return linear(module[2], F.relu(linear(module[0], inputs)))


def graph_oracle(mixer, utilities, states_input, hidden, signed=True):
    """Explicit matrices and scalar placement, independent of production forwards."""
    batch = states_input.shape[0]
    state = states_input.reshape(-1, mixer.state_dim)
    norm = mixer.state_norm
    state = ((state - state.mean(-1, keepdim=True)) *
             torch.rsqrt(state.var(-1, unbiased=False, keepdim=True) + norm.eps))
    state = state * norm.weight + norm.bias
    embedded = F.relu(linear(mixer.obs_encoder[0], hidden)).reshape(-1, mixer.n_agents, 16)
    scores = (linear(mixer.W_attn_query, embedded) @
              linear(mixer.W_attn_key, embedded).transpose(1, 2)) / 4
    adjacency = torch.softmax(scores, dim=2)
    features = utilities.reshape(-1, mixer.n_agents, 1)
    raw = []
    for layer in mixer.mixing_GNN.layers:
        a, b = mlp(layer.w1, state), mlp(layer.w2, state)
        raw.extend((a, b))
        w1 = torch.sigmoid(a).reshape(-1, layer.in_features, layer.hidden_features)
        w2 = torch.sigmoid(b).reshape(-1, layer.hidden_features, layer.out_features)
        if signed:
            w1, w2 = 2 * w1 - 1, 2 * w2 - 1
        first = ((adjacency @ features) @ w1) * .1 + linear(layer.b1, state).unsqueeze(1)
        middle = torch.where(first >= 0, first, .01 * first)
        output = (middle @ w2) * .1 + linear(layer.b2, state).unsqueeze(1)
        features = torch.where(output > 0, output, torch.expm1(output.clamp_max(0)))
    readout = features.sum(1) / (mixer.n_agents + 1e-10)
    out = mlp(mixer.mixing_GNN.wout, state)
    raw.append(out)
    weight = torch.sigmoid(out)
    if signed:
        weight = 2 * weight - 1
    total = (readout.unsqueeze(1) @ weight.unsqueeze(2)) * .1
    total = total + mlp(mixer.V, state).reshape(-1, 1, 1)
    return total.reshape(batch, -1, 1), raw


class H14SignedMixerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        cls.directory = tempfile.TemporaryDirectory(prefix="h14_ancestors_")
        base = Path(cls.directory.name)
        package = base / "h14_actual_graph"
        (package / "graph").mkdir(parents=True)
        for name in ("graph_mixer.py", "graph/gnn.py", "graph/gin_layer.py"):
            (package / name).write_bytes(git_source(PARENT, "src/modules/" + name))
        sys.path.insert(0, str(base))
        cls.old_graph = importlib.import_module("h14_actual_graph.graph_mixer")
        cls.old_gnn = importlib.import_module("h14_actual_graph.graph.gnn")
        cls.old_gin = importlib.import_module("h14_actual_graph.graph.gin_layer")
        cls.old_root = import_source("h14_actual_root", git_source(PRE, "src/learners/gavdac_learner.py"), base)
        cls.old_h03 = import_source("h14_actual_h03", git_source(H03, "src/learners/delayed_actor_learner.py"), base)
        cls.old_parent_core = import_source("h14_actual_h12_core", git_source(PARENT, "src/learners/gavdac_learner.py"), base)
        cls.old_parent = import_source("h14_actual_h12", git_source(PARENT, "src/learners/delayed_actor_huber_learner.py"), base)
        cls.old_root.GraphDec = cls.old_parent_core.GraphDec = cls.old_graph.GraphDec

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.directory.name)
        for name in list(sys.modules):
            if name.startswith(("h14_actual_",)):
                del sys.modules[name]
        cls.directory.cleanup()

    def child(self, **options):
        arguments = dict(use_cuda=False, actor_update_interval=2, target_update_interval=1,
                         scaled_huber_critic=True, signed_mixer_hyperweights=True,
                         rnn_hidden_dim=48, mixing_embed_dim=32, hypernet_embed=64)
        arguments.update(options)
        batch, mac, args, scheme = fixture(**arguments)
        return batch, SignedMixerHuberGAVDACLearner(mac, scheme, LossLog(), args)

    def test_full_graph_forward_input_state_hidden_hyperoutput_parameter_gradient_oracle(self):
        for signed, magnitude in ((True, 1.), (False, 1.), (True, 1000.), (True, 0.)):
            with self.subTest(signed=signed, magnitude=magnitude):
                _, learner = self.child()
                actual = learner.mixer
                if not signed:
                    actual.mixing_GNN.weights_operation = "sigmoid"
                    for layer in actual.mixing_GNN.layers:
                        layer.weights_operation = "sigmoid"
                reference = copy.deepcopy(actual)
                q = (torch.randn(2, 3, 3, 1) * magnitude).requires_grad_()
                state = (torch.randn(2, 3, 60) * magnitude).requires_grad_()
                hidden = (torch.randn(2, 3, 3, 48) * magnitude).requires_grad_()
                rq, rs, rh = [v.detach().clone().requires_grad_() for v in (q, state, hidden)]
                captured, handles = [], []
                for module in (actual.mixing_GNN.layers[0].w1, actual.mixing_GNN.layers[0].w2,
                               actual.mixing_GNN.wout):
                    handles.append(module.register_forward_hook(lambda m, a, out: captured.append(out)))
                try:
                    output, local, active = actual(q, state, hidden_states=hidden)
                finally:
                    for handle in handles:
                        handle.remove()
                expected, raw = graph_oracle(reference, rq, rs, rh, signed)
                self.assertIsNone(local)
                self.assertTrue(active.all())
                torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
                gradients = torch.autograd.grad(output.sum(), [q, state, hidden, *actual.parameters(), *captured],
                                                allow_unused=True)
                oracle = torch.autograd.grad(expected.sum(), [rq, rs, rh, *reference.parameters(), *raw],
                                             allow_unused=True)
                for found, desired in zip(gradients, oracle, strict=True):
                    if desired is None:
                        self.assertIsNone(found)
                    else:
                        self.assertEqual((found.device.type, found.dtype), ("cpu", torch.float32))
                        self.assertTrue(torch.isfinite(found).all())
                        torch.testing.assert_close(found, desired, rtol=1e-4, atol=1e-5)

    def test_constructive_negative_utility_partial_with_fixed_state_and_attention(self):
        _, learner = self.child()
        mixer = learner.mixer
        with torch.no_grad():
            for p in mixer.parameters():
                p.zero_()
            mixer.mixing_GNN.layers[0].w1[-1].bias.fill_(1)
            mixer.mixing_GNN.layers[0].w2[-1].bias.fill_(1)
            mixer.mixing_GNN.wout[-1].bias.fill_(-1)
        positive = copy.deepcopy(mixer)
        positive.mixing_GNN.weights_operation = "sigmoid"
        for layer in positive.mixing_GNN.layers:
            layer.weights_operation = "sigmoid"
        q = torch.full((1, 1, 3, 1), .2, requires_grad=True)
        s, h = torch.zeros(1, 1, 60), torch.zeros(1, 1, 3, 48)
        for graph, negative in ((mixer, True), (positive, False)):
            gradient, = torch.autograd.grad(graph(q, s, hidden_states=h)[0].sum(), q)
            self.assertTrue((gradient < 0).all() if negative else (gradient > 0).all())

    def test_original_all_weight_operations_values_and_gradients_equal_actual_Git(self):
        for operation in ("sigmoid", "abs", "clamp", None):
            for original, candidate, arguments in (
                (self.old_gin.GINGraphConvolution, GINGraphConvolution, (1, 16, 60, 16)),
                (self.old_gnn.GNN, GNN, (1, [16], 60, 16)),
            ):
                with self.subTest(operation=operation, klass=candidate):
                    rng = torch.get_rng_state()
                    ref = original(*arguments, weights_operation=operation, weight_scale=.1)
                    after = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    new = candidate(*arguments, weights_operation=operation, weight_scale=.1)
                    self.assertTrue(torch.equal(after, torch.get_rng_state()))
                    features = torch.randn(2, 3, 1, requires_grad=True)
                    adjacency = torch.softmax(torch.randn(2, 3, 3), -1)
                    state = torch.randn(2, 60, requires_grad=True)
                    inputs = (features, adjacency, state)
                    a = ref(*inputs) if candidate is GINGraphConvolution else ref(*inputs, 3)[1]
                    b = new(*inputs) if candidate is GINGraphConvolution else new(*inputs, 3)[1]
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    ga = torch.autograd.grad(a.sum(), [features, state, *ref.parameters()], allow_unused=True)
                    gb = torch.autograd.grad(b.sum(), [features, state, *new.parameters()], allow_unused=True)
                    torch.testing.assert_close(ga, gb, rtol=0, atol=0)

    def test_missing_false_actual_root_H03_H12_complete_multistep_optimizer_target_RNG(self):
        for node in ("root", "H03", "H12"):
            for explicit in (False, True):
                for optimizer, mode in ((PARENT_OPTIMIZER, "soft"), ("rmsprop", "hard")):
                    with self.subTest(node=node, explicit=explicit, optimizer=optimizer):
                        options = dict(use_cuda=False, optimizer=optimizer, target_update_mode=mode,
                                       batch_size=3, length=5)
                        if node != "root":
                            options.update(actor_update_interval=2, target_update_interval=1)
                        if node == "H12":
                            options["scaled_huber_critic"] = True
                        if explicit:
                            options["signed_mixer_hyperweights"] = False
                        batch, mac, args, scheme = fixture(**options)
                        klass = {"root": self.old_root.GAVDACLearner,
                                 "H03": self.old_h03.DelayedActorGAVDACLearner,
                                 "H12": self.old_parent.HuberDelayedActorGAVDACLearner}[node]
                        current = {"root": GAVDACLearner, "H03": DelayedActorGAVDACLearner,
                                   "H12": HuberDelayedActorGAVDACLearner}[node]
                        rng = torch.get_rng_state()
                        with patch.object(base_module, "GraphDec", self.old_graph.GraphDec):
                            reference = klass(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
                        core = self.old_parent_core if node == "H12" else self.old_root
                        reference._critic_loss = MethodType(core.GAVDACLearner._critic_loss, reference)
                        reference._actor_loss = MethodType(core.GAVDACLearner._actor_loss, reference)
                        after = torch.get_rng_state()
                        torch.set_rng_state(rng)
                        candidate = current(mac, scheme, LossLog(), args)
                        self.assertTrue(torch.equal(after, torch.get_rng_state()))
                        for step in range(1, 5):
                            if step == 3:
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

    def test_true_initial_tensors_rng_counts_actor_equal_parent_not_mixer_function(self):
        batch, mac, args, scheme = fixture(use_cuda=False, rnn_hidden_dim=48, mixing_embed_dim=32,
                                          hypernet_embed=64, actor_update_interval=2,
                                          target_update_interval=1, scaled_huber_critic=True)
        rng = torch.get_rng_state()
        parent = HuberDelayedActorGAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
        after = torch.get_rng_state()
        args.signed_mixer_hyperweights = True
        torch.set_rng_state(rng)
        child = SignedMixerHuberGAVDACLearner(mac, scheme, LossLog(), args)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        for name, module in gradient_pool._modules(child).items():
            torch.testing.assert_close(module.state_dict(), gradient_pool._modules(parent)[name].state_dict(),
                                       rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in child.agent_params), 15458)
        self.assertEqual(sum(p.numel() for p in child.critic_params), 65915)
        torch.testing.assert_close(child.mac.forward_sequence(batch), parent.mac.forward_sequence(batch), rtol=0, atol=0)
        h = child.mac.forward_sequence(batch)["hidden_states"]
        q = torch.ones(batch.batch_size, batch.max_seq_length, 3, 1)
        self.assertFalse(torch.equal(child.mixer(q, batch["state"], hidden_states=h)[0],
                                     parent.mixer(q, batch["state"], hidden_states=h)[0]))
        self.assertIs(SignedMixerHuberGAVDACLearner.train, HuberDelayedActorGAVDACLearner.train)
        self.assertIs(SignedMixerHuberGAVDACLearner._td_error_loss, HuberDelayedActorGAVDACLearner._td_error_loss)

    def test_live_action_context_history_and_frozen_critic_in_actor_phase(self):
        batch, learner = self.child()
        batch["obs"].requires_grad_()
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
        self.assertGreater(float(batch["obs"].grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(learner.mac.agent.rnn.weight_hh.grad.abs().sum()), 0)
        for p, old in zip(learner.critic_params, before, strict=True):
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p, old, rtol=0, atol=0)

    def test_inherited_mask_target_Huber_global_denominator_and_no_reward_decomposition(self):
        batch, learner = self.child()
        batch["reward"].mul_(20)
        values = []
        original = learner._mix_q_values
        def capture(*args, **kwargs):
            self.assertIsNone(kwargs.get("team_rewards"))
            out = original(*args, **kwargs)
            values.append(out)
            return out
        with patch.object(learner, "_mix_q_values", side_effect=capture), \
             patch.object(learner, "_td_error_loss", wraps=learner._td_error_loss) as reduce:
            _, _, loss = learner._critic_loss(batch, denominator=111.)
        q, target = values
        self.assertFalse(target.requires_grad)
        terminal = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float().clone()
        mask[:, 1:] *= 1 - terminal[:, :-1]
        error = (batch["reward"][:, :-1] + .99 * (1 - terminal) * target - q) * mask
        torch.testing.assert_close(reduce.call_args.args[0], error, rtol=0, atol=0)
        self.assertEqual(reduce.call_args.args[1], 111.)
        expected = torch.where(error.abs() <= 1, error.square(), 2 * error.abs() - 1).sum() / 111.
        torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)

    def test_actual_EA_shared_actor_clone_transfer_then_inherited_update_target_cadence(self):
        batch, learner = self.child()
        settings = entry_config("gavdac_c2_h14")
        for key in ("pop_size", "elite_fraction", "mutation_prob", "crossover_prob",
                    "prob_reset_and_sup", "frac", "SAME"):
            setattr(learner.args, key, settings[key])
        population = [copy.deepcopy(learner.mac) for _ in range(5)]
        evolver = SSNE(learner.args)
        with patch.object(evolver, "crossover_inplace", side_effect=AssertionError("crossover disabled")):
            elite = evolver.epoch(population, [5, 4, 3, 2, 1], 0, agent_level=True)
        self.assertIn(elite, range(5))
        from run import rl_to_evo
        rl_to_evo(learner.mac, population[1])
        torch.testing.assert_close(learner.mac.agent.state_dict(), population[1].agent.state_dict(), rtol=0, atol=0)
        for step in range(1, 8):
            if step == 4:
                learner.mac.load_state(population[elite])
            modules = gradient_pool._modules(learner)
            before = {n: [p.detach().clone() for p in modules["target_" + n].parameters()]
                      for n in ("actor", "critic", "mixer")}
            learner.train(batch, step, step)
            for name in before:
                for old, current, target in zip(before[name], modules[name].parameters(),
                                                modules["target_" + name].parameters(), strict=True):
                    torch.testing.assert_close(target, old.lerp(current.detach(), .002), rtol=0, atol=0)
                    self.assertIsNone(target.grad)
            self.assertEqual((learner.critic_steps, learner.actor_steps, learner.target_updates,
                              learner.finite_checks), (step, step // 2, step, step))
            self.assertEqual(learner.optimizer_finite_checks, {"critic": step, "actor": step // 2})
            if step % 2:
                self.assertFalse(learner.mac.hidden_states.requires_grad)

    def test_real_two_workers_varying_participation_complete_grad_optimizer_targets(self):
        for optimizer in (PARENT_OPTIMIZER, "rmsprop"):
            batch, mac, args, scheme = fixture(
                use_cuda=False, actor_update_interval=2, target_update_interval=1,
                scaled_huber_critic=True, signed_mixer_hyperweights=True,
                rnn_hidden_dim=48, mixing_embed_dim=32, hypernet_embed=64, optimizer=optimizer)
            rng = torch.get_rng_state()
            serial = SignedMixerHuberGAVDACLearner(copy.deepcopy(mac), scheme, LossLog(), copy.deepcopy(args))
            torch.set_rng_state(rng)
            args.learner_workers = 2
            parallel = SignedMixerHuberGAVDACLearner(mac, scheme, LossLog(), args)
            try:
                for step, (size, length) in enumerate(((5, 7), (1, 4), (3, 6), (2, 5), (5, 7), (1, 4)), 1):
                    batch, _, _, _ = fixture(use_cuda=False, batch_size=size, length=length)
                    serial.train(batch, step, step)
                    parallel.train(batch, step, step)
                    torch.testing.assert_close(states(serial), states(parallel), rtol=1e-4, atol=1e-5)
                    self.assertEqual(parallel.optimizer_finite_checks, {"critic": step, "actor": step // 2})
                    pool = parallel.gradient_pool
                    self.assertIsNone(pool.pending_batch)
                    self.assertEqual(pool.last_phase_pids["actor" if step % 2 == 0 else "discard"],
                                     pool.last_phase_pids["critic"])
                    self.assertEqual(len(pool.last_phase_pids["critic"]), min(2, size))
                    self.assertEqual(len(pool.worker_pids), 2)
            finally:
                serial.close()
                parallel.close()

    def test_explicit_worker_class_and_round_guard(self):
        batch, learner = self.child()
        class Barrier:
            def wait(self, timeout):
                pass
        with patch.multiple(gradient_pool, _learner=None, _pending=None, _scheme=None, _groups=None, _transport=None), \
             patch.object(torch, "set_num_interop_threads"):
            gradient_pool._initialize(copy.deepcopy(learner.args), learner.scheme, Barrier())
            self.assertIs(type(gradient_pool._learner), SignedMixerHuberGAVDACLearner)
            weights = {n: {k: v.numpy().copy() for k, v in m.state_dict().items()}
                       for n, m in gradient_pool._modules(learner).items()}
            data = {k: v.numpy().copy() for k, v in batch.data.transition_data.items()}
            gradient_pool._gradients("critic", weights, data, 100., 1., 1)
            with self.assertRaisesRegex(RuntimeError, "matching cached"):
                gradient_pool._discard_actor(2)
            gradient_pool._discard_actor(1)
            self.assertIsNone(gradient_pool._pending)
            self.assertFalse(gradient_pool._learner.mac.hidden_states.requires_grad)

    def test_hard_target_copies_and_posthook_rejects_nonfinite_gradient(self):
        batch, learner = self.child(target_update_mode="hard")
        for step in (1, 2):
            learner.train(batch, step, step)
            modules = gradient_pool._modules(learner)
            for name in ("actor", "critic", "mixer"):
                torch.testing.assert_close(modules[name].state_dict(), modules["target_" + name].state_dict(),
                                           rtol=0, atol=0)
        learner.agent_params[0].grad = torch.full_like(learner.agent_params[0], float("inf"))
        self.assertTrue(torch.isfinite(learner.agent_params[0]).all())
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            learner._actor_post_step(learner.agent_optimiser, (), {})

    def test_saved_diagnostics_model_actor_optimizer_and_actual_posthook_finite_guards(self):
        batch, learner = self.child()
        learner.train(batch, 1, 1)
        learner.train(batch, 2, 2)
        with tempfile.TemporaryDirectory(prefix="h14_load_") as directory:
            learner.save_models(directory)
            path = Path(directory) / "h14_diagnostics.json"
            value = json.loads(path.read_text())
            self.assertEqual(value["optimizer_post_step_finite_checks"], {"critic": 2, "actor": 1})
            self.assertEqual(value["target_updates"], 2)
            self.assertEqual(value["weights_operation"], "signed_sigmoid")
            _, loaded = self.child()
            loaded.load_models(directory)
            torch.testing.assert_close(learner.mac.agent.state_dict(), loaded.mac.agent.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(learner.agent_optimiser.state_dict(),
                                       loaded.agent_optimiser.state_dict(), rtol=0, atol=0)
            value["signed_mixer_hyperweights"] = 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "configuration"):
                loaded.load_models(directory)
        for phase in ("actor", "critic"):
            _, damaged = self.child()
            optimizer = getattr(damaged, "agent_optimiser" if phase == "actor" else "critic_optimiser")
            params = damaged.agent_params if phase == "actor" else damaged.critic_params
            with torch.no_grad():
                params[0].fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                optimizer.step()
        _, damaged = self.child()
        target_phase = damaged._target_phase
        def corrupt_after_target_phase():
            target_phase()
            with torch.no_grad():
                next(damaged.target_mixer.parameters()).fill_(float("inf"))
        with patch.object(damaged, "_target_phase", side_effect=corrupt_after_target_phase), \
             self.assertRaisesRegex(ValueError, "Non-finite"):
            damaged.train(batch, 1, 1)

    def test_true_guard_nonboolean_unsupported_and_retired_flags_and_wrong_classes(self):
        _, learner = self.child()
        for value in (None, 0, 1, "true", 1.):
            args = copy.deepcopy(learner.args)
            args.signed_mixer_hyperweights = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                signed_mixer_enabled(args)
            with self.assertRaises(ValueError):
                GraphDec(args)
        for key, value in (("scaled_huber_critic", False), ("actor_update_interval", 1),
                           ("target_update_interval", 2), ("target_update_tau", .004),
                           ("rnn_hidden_dim", 16), ("mixer_weight_scale", .2),
                           ("use_cuda", True), ("actor_feature_layer_norm", False),
                           ("target_policy_smoothing", False), ("clipped_double_critic", False)):
            args = copy.deepcopy(learner.args)
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                signed_mixer_enabled(args)
        for klass in (GAVDACLearner, DelayedActorGAVDACLearner, HuberDelayedActorGAVDACLearner):
            with self.subTest(klass=klass), self.assertRaisesRegex(ValueError, "explicit H14"):
                klass(copy.deepcopy(learner.mac), learner.scheme, LossLog(), copy.deepcopy(learner.args))

    def test_strict_saved_source_validation_physical_Actor_config_data_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results", prefix="h14strict_") as directory:
            config = entry_config("gavdac_c2_h14", {"rollout_workers": 2, "learner_workers": 2})
            config.update(device="cpu", local_results_path=directory)
            config["env_args"].update(seed=config["seed"], result_dir=directory, run_id="h14strict")
            args = SimpleNamespace(**copy.deepcopy(config))
            runner = EpisodeRunner(args, QuietLogger())
            info = runner.get_env_info()
            for key in ("n_agents", "n_actions", "state_shape", "obs_shape", "action_spaces"):
                setattr(args, key, info[key])
            scheme = {
                "state": {"vshape": info["state_shape"]}, "obs": {"vshape": info["obs_shape"], "group": "agents"},
                "actions": {"vshape": (info["n_actions"],), "group": "agents", "dtype": torch.float},
                "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8}}
            mac = CQMixMAC(scheme, {"agents": 3}, args)
            runner.setup(scheme, {"agents": 3}, {}, mac)
            learner = SignedMixerHuberGAVDACLearner(mac, scheme, LossLog(), args)
            try:
                result = evaluate_fixed(runner, mac, [731, 732], .35, "validation", "rl", 24)
                store = BestPolicyStore(Path(directory) / "best", config)
                checkpoint = store.consider(mac, "rl", 24, result, runner.env)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                saved_config = checkpoint / "config.yaml"
                config_raw_sha = hashlib.sha256(saved_config.read_bytes()).hexdigest()
                self.assertEqual(yaml.safe_load(saved_config.read_text()), config)
            finally:
                runner.close_env()
                learner.close()
            output = Path(directory) / "inference"
            completed = subprocess.run(
                ["uv", "run", "--frozen", "python", "tools/evaluate_microgrid_comparison.py",
                 "--checkpoint", str(checkpoint), "--split", "validation", "--mode", "continuous",
                 "--rows", "731", "732", "--output", str(output)], cwd=ROOT, capture_output=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", errors="replace"))
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["physical_checks_passed"])
            self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
            self.assertEqual(summary["data_sha256"], metadata["data_sha256"])
            self.assertEqual(hashlib.sha256(saved_config.read_bytes()).hexdigest(), config_raw_sha)
            self.assertTrue(yaml.safe_load(saved_config.read_text())["signed_mixer_hyperweights"])
            for name, raw in metadata["source_sha256"].items():
                self.assertEqual(hashlib.sha256((output / "runtime" / name).read_bytes()).hexdigest(), raw)
            for name, raw in summary["source_sha256"].items():
                self.assertEqual(hashlib.sha256((output / "runtime" / name).read_bytes()).hexdigest(), raw)


if __name__ == "__main__":
    unittest.main()
