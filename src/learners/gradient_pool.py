"""Synchronous CPU batch sharding; only the parent performs optimizer steps."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from components.episode_buffer import EpisodeBatch
from controllers.cqmix_controller import CQMixMAC


_learner = None
_scheme = None
_groups = None
_pending = None
_transport = None


class _NoLog:
    def log_stat(self, *args):
        pass


def _initialize(args, scheme, barrier, transport=None):
    global _learner, _scheme, _groups, _transport
    from learners.gavdac_learner import GAVDACLearner
    from learners.centered_generator_learner import (
        CenteredGeneratorGAVDACLearner, validate_center_generator_flag,
    )
    centered = validate_center_generator_flag(
        getattr(args, "critic_center_generator_action", False)
    )
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.learner_workers = 0
    _scheme, _groups = scheme, {"agents": args.n_agents}
    mac = CQMixMAC(scheme, _groups, args)
    if centered:
        _learner = CenteredGeneratorGAVDACLearner(mac, scheme, _NoLog(), args)
    else:
        _learner = GAVDACLearner(mac, scheme, _NoLog(), args)
    _transport = transport
    barrier.wait(timeout=120)


def _ready():
    return os.getpid()


def _modules(learner):
    modules = {
        "actor": learner.mac.agent, "target_actor": learner.target_mac.agent,
        "critic": learner.critic, "target_critic": learner.target_critic,
    }
    if learner.mixer is not None:
        modules.update(mixer=learner.mixer, target_mixer=learner.target_mixer)
    return modules


def _gradients(phase, weights, data, denominator, actor_weight, round_id,
               gradient_buffers=None):
    global _pending
    learner = _learner
    modules = _modules(learner)
    expected = (set(modules) if phase == "critic"
                else {"critic", "mixer"} if learner.mixer is not None else {"critic"})
    if set(weights) != expected:
        raise ValueError("Unexpected weights for gradient phase")
    if phase == "critic" and _pending is not None:
        raise RuntimeError("Previous Actor phase is still pending")
    if phase == "actor" and (_pending is None or _pending[0] != round_id):
        raise RuntimeError("Actor phase needs the matching cached critic round")
    for name, state in weights.items():
        modules[name].load_state_dict({
            key: torch.from_numpy(value) for key, value in state.items()
        })
    if phase == "critic":
        transition = {key: torch.from_numpy(value) for key, value in data.items()}
        first = next(iter(transition.values()))
        batch = EpisodeBatch(
            _scheme, _groups, first.shape[0], first.shape[1],
            data=SimpleNamespace(transition_data=transition, episode_data={}),
        )
        learner.critic_optimiser.zero_grad(set_to_none=True)
        policy, inputs, loss = learner._critic_loss(batch, denominator=denominator)
        loss.backward()
        _pending = (round_id, batch, policy, inputs, actor_weight)
        params = learner.critic_params
    else:
        _, batch, policy, inputs, actor_weight = _pending
        _pending = None
        learner.agent_optimiser.zero_grad(set_to_none=True)
        learner._set_critic_grad(False)
        try:
            loss = learner._actor_loss(batch, policy, inputs) * actor_weight
            loss.backward()
        finally:
            learner._set_critic_grad(True)
        params = learner.agent_params
    if gradient_buffers is None:
        gradients = [None if p.grad is None else p.grad.numpy().copy() for p in params]
    else:
        gradients = []
        for buffer, param in zip(gradient_buffers, params, strict=True):
            gradients.append(param.grad is not None)
            if param.grad is not None:
                buffer.copy_(param.grad)
    return float(loss.detach()), gradients, os.getpid()


def _shared_gradients(phase, denominator, actor_weight, round_id, start, stop, data):
    if data is not None:
        _transport["batch"] = data
    names = (_transport["weights"] if phase == "critic"
             else ("critic", "mixer") if _learner.mixer is not None else ("critic",))
    weights = {
        name: {key: value.numpy() for key, value in _transport["weights"][name].items()}
        for name in names
    }
    transition = ({key: value[start:stop].numpy()
                   for key, value in _transport["batch"].items()}
                  if phase == "critic" else None)
    return _gradients(phase, weights, transition, denominator, actor_weight, round_id,
                      _transport["gradients"][phase])


def _shared_grad_buffers(params):
    flat = torch.empty(sum(param.numel() for param in params), dtype=torch.float32).share_memory_()
    views = []
    offset = 0
    for param in params:
        views.append(flat[offset:offset + param.numel()].view_as(param))
        offset += param.numel()
    return views


class GradientPool:
    def __init__(self, learner, scheme):
        workers = learner.args.learner_workers
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("learner_workers must be a nonnegative integer")
        if learner.args.use_cuda:
            raise ValueError("Parallel learner supports CPU execution only")
        if any(p.dtype != torch.float32 or p.device.type != "cpu"
               for module in _modules(learner).values() for p in module.parameters()):
            raise ValueError("Parallel learner networks must be CPU float32")
        context = multiprocessing.get_context("spawn")
        self.weight_buffers = {
            name: {key: value.detach().clone().share_memory_()
                   for key, value in module.state_dict().items()}
            for name, module in _modules(learner).items()
        }
        self.gradient_buffers = [{
            "critic": _shared_grad_buffers(learner.critic_params),
            "actor": _shared_grad_buffers(learner.agent_params),
        } for _ in range(workers)]
        self.batch_buffers = {}
        self.batch_generation = 0
        self.worker_batch_generations = [0] * workers
        self.executors = []
        initialized = False
        try:
            barrier = context.Barrier(workers)
            for index in range(workers):
                self.executors.append(ProcessPoolExecutor(
                    max_workers=1, mp_context=context, initializer=_initialize,
                    initargs=(deepcopy(learner.args), scheme, barrier, {
                        "weights": self.weight_buffers,
                        "gradients": self.gradient_buffers[index],
                    }),
                ))
            ready = [executor.submit(_ready) for executor in self.executors]
            self.pids = tuple(future.result() for future in ready)
            initialized = True
        finally:
            if not initialized:
                self.close()
        self.workers = workers
        self.worker_pids = set(self.pids)
        self.last_phase_pids = {}
        self.pending_batch = None
        self.round_id = 0

    def backward(self, learner, batch, phase):
        if phase not in {"critic", "actor"}:
            raise ValueError("Unknown gradient phase")
        if phase == "critic" and self.pending_batch is not None:
            raise RuntimeError("Complete the pending Actor phase before another critic")
        if phase == "actor" and self.pending_batch is not batch:
            raise RuntimeError("Actor phase must reuse the same critic batch")
        modules = _modules(learner)
        if phase == "actor":
            # Reloading Actor parameters would invalidate its retained autograd graph.
            modules = {name: modules[name] for name in ("critic", "mixer") if name in modules}
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                self.weight_buffers[name][key].copy_(value.detach())
        denominator = None
        if phase == "critic":
            mask = batch["filled"][:, :-1].float().clone()
            mask[:, 1:] *= 1 - batch["terminated"][:, :-2].float()
            denominator = float(mask.sum())
            if denominator <= 0:
                raise ValueError("Learner batch has no valid transitions")
            if any(tensor.device.type != "cpu"
                   for tensor in batch.data.transition_data.values()):
                raise ValueError("Parallel learner batches must stay on CPU")
            transition = batch.data.transition_data
            if (self.batch_buffers.keys() != transition.keys() or any(
                value.shape != self.batch_buffers[key].shape
                or value.dtype != self.batch_buffers[key].dtype
                for key, value in transition.items()
            )):
                self.batch_buffers = {
                    key: torch.empty_like(value).share_memory_()
                    for key, value in transition.items()
                }
                self.batch_generation += 1
            for key, value in transition.items():
                self.batch_buffers[key].copy_(value.detach())
            self.round_id += 1
            self.pending_batch = batch
        futures = []
        for index, indices in enumerate(np.array_split(
            np.arange(batch.batch_size), min(batch.batch_size, self.workers),
        )):
            start, stop = int(indices[0]), int(indices[-1]) + 1
            data = (self.batch_buffers if phase == "critic"
                    and self.worker_batch_generations[index] != self.batch_generation else None)
            futures.append(self.executors[index].submit(
                _shared_gradients, phase, denominator,
                (stop - start) / batch.batch_size if phase == "critic" else None,
                self.round_id, start, stop, data,
            ))
        results = [future.result() for future in futures]
        self.last_phase_pids[phase] = tuple(pid for _, _, pid in results)
        if self.last_phase_pids[phase] != self.pids[:len(results)]:
            raise RuntimeError("Gradient shard changed worker identity")
        if phase == "critic":
            self.worker_batch_generations[:len(results)] = [self.batch_generation] * len(results)
        params = learner.critic_params if phase == "critic" else learner.agent_params
        for index, param in enumerate(params):
            gradients = [gradients[index] for _, gradients, _ in results]
            if not any(gradients):
                param.grad = None
            elif not all(gradients):
                raise ValueError("Inconsistent shard gradient participation")
            else:
                gradient = self.gradient_buffers[0][phase][index].clone()
                for worker in range(1, len(results)):
                    gradient.add_(self.gradient_buffers[worker][phase][index])
                if not torch.isfinite(gradient).all():
                    raise ValueError("Non-finite sharded gradient")
                param.grad = gradient
        self.worker_pids.update(pid for _, _, pid in results)
        loss = torch.tensor(sum(loss for loss, _, _ in results), dtype=torch.float32)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite sharded loss")
        if phase == "actor":
            self.pending_batch = None
        return loss

    def close(self):
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=True)
