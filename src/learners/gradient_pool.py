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


def _gradients(phase, weights, data, denominator, actor_weight):
    learner = _learner
    for name, module in _modules(learner).items():
        module.load_state_dict({
            key: torch.from_numpy(value) for key, value in weights[name].items()
        })
    transition = {key: torch.from_numpy(value) for key, value in data.items()}
    first = next(iter(transition.values()))
    batch = EpisodeBatch(
        _scheme, _groups, first.shape[0], first.shape[1],
        data=SimpleNamespace(transition_data=transition, episode_data={}),
    )
    if phase == "critic":
        learner.critic_optimiser.zero_grad(set_to_none=True)
        _, _, loss = learner._critic_loss(batch, denominator=denominator)
        loss.backward()
        params = learner.critic_params
    else:
        learner.agent_optimiser.zero_grad(set_to_none=True)
        learner._set_critic_grad(False)
        try:
            policy = learner.mac.forward_sequence(batch, end=-1)
            inputs = learner._build_inputs_sequence(batch, end=-1)
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
        self.executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_initialize,
            initargs=(deepcopy(learner.args), scheme, context.Barrier(workers)),
        )
        initialized = False
        try:
            ready = [self.executor.submit(_ready) for _ in range(workers)]
            for future in ready:
                future.result()
            initialized = True
        finally:
            if not initialized:
                self.executor.shutdown(wait=True, cancel_futures=True)
        self.workers = workers
        self.worker_pids = set()

    def backward(self, learner, batch, phase):
        weights = {
            name: {key: value.detach().numpy().copy()
                   for key, value in module.state_dict().items()}
            for name, module in _modules(learner).items()
        }
        mask = batch["filled"][:, :-1].float().clone()
        mask[:, 1:] *= 1 - batch["terminated"][:, :-2].float()
        denominator = float(mask.sum())
        if denominator <= 0:
            raise ValueError("Learner batch has no valid transitions")
        if any(tensor.device.type != "cpu"
               for tensor in batch.data.transition_data.values()):
            raise ValueError("Parallel learner batches must stay on CPU")
        futures = []
        for indices in np.array_split(np.arange(batch.batch_size),
                                      min(batch.batch_size, self.workers)):
            start, stop = int(indices[0]), int(indices[-1]) + 1
            data = {key: tensor[start:stop].detach().numpy().copy()
                    for key, tensor in batch.data.transition_data.items()}
            futures.append(self.executor.submit(
                _gradients, phase, weights, data, denominator,
                (stop - start) / batch.batch_size,
            ))
        results = [future.result() for future in futures]
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
        return loss

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
