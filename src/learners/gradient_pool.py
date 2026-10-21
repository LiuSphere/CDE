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


class _NoLog:
    def log_stat(self, *args):
        pass


def _initialize(args, scheme, barrier):
    global _learner, _scheme, _groups
    from learners.gavdac_learner import GAVDACLearner
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.learner_workers = 0
    _scheme, _groups = scheme, {"agents": args.n_agents}
    mac = CQMixMAC(scheme, _groups, args)
    _learner = GAVDACLearner(mac, scheme, _NoLog(), args)
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


def _gradients(phase, weights, data, denominator, actor_weight, round_id):
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
    return (
        float(loss.detach()),
        [None if param.grad is None else param.grad.numpy().copy() for param in params],
        os.getpid(),
    )


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
        self.executors = []
        initialized = False
        try:
            barrier = context.Barrier(workers)
            for _ in range(workers):
                self.executors.append(ProcessPoolExecutor(
                    max_workers=1, mp_context=context, initializer=_initialize,
                    initargs=(deepcopy(learner.args), scheme, barrier),
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
        weights = {
            name: {key: value.detach().numpy().copy()
                   for key, value in module.state_dict().items()}
            for name, module in modules.items()
        }
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
            self.round_id += 1
            self.pending_batch = batch
        futures = []
        for index, indices in enumerate(np.array_split(
            np.arange(batch.batch_size), min(batch.batch_size, self.workers),
        )):
            start, stop = int(indices[0]), int(indices[-1]) + 1
            data = ({key: tensor[start:stop].detach().numpy().copy()
                     for key, tensor in batch.data.transition_data.items()}
                    if phase == "critic" else None)
            futures.append(self.executors[index].submit(
                _gradients, phase, weights, data, denominator,
                (stop - start) / batch.batch_size if phase == "critic" else None,
                self.round_id,
            ))
        results = [future.result() for future in futures]
        self.last_phase_pids[phase] = tuple(pid for _, _, pid in results)
        if self.last_phase_pids[phase] != self.pids[:len(results)]:
            raise RuntimeError("Gradient shard changed worker identity")
        params = learner.critic_params if phase == "critic" else learner.agent_params
        for index, param in enumerate(params):
            gradients = [gradients[index] for _, gradients, _ in results]
            if all(value is None for value in gradients):
                param.grad = None
            elif any(value is None for value in gradients):
                raise ValueError("Inconsistent shard gradient participation")
            else:
                gradient = torch.from_numpy(gradients[0]).clone()
                for value in gradients[1:]:
                    gradient.add_(torch.from_numpy(value))
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
