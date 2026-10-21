"""Persistent spawned workers for independent, non-exploring CPU episodes."""

import csv
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import torch

from controllers.cqmix_controller import CQMixMAC
from runners.episode_runner import EpisodeRunner


_worker = None
_ENV_FIELDS = (
    "idx", "_training_state", "is_testing", "test_source",
    "trajectory_context", "ep_counter", "ep_buffer", "_ep_steps", "_stats", "state",
)


class _NoLog:
    def log_stat(self, *args):
        pass


def _snapshot(env):
    state = {key: deepcopy(getattr(env, key)) for key in _ENV_FIELDS}
    state["socs"] = env.params["socs"].copy()
    return state


def _restore(env, state):
    for key in _ENV_FIELDS:
        setattr(env, key, deepcopy(state[key]))
    env.params["socs"] = state["socs"].copy()


def _initialize(args, scheme, groups, preprocess, barrier):
    global _worker
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.rollout_workers = 0
    args.env_args = deepcopy(args.env_args)
    args.env_args["log_trajectories"] = False
    runner = EpisodeRunner(args, _NoLog())
    mac = CQMixMAC(scheme, groups, args)
    runner.setup(scheme, groups, preprocess, mac)
    _worker = runner
    barrier.wait(timeout=120)


def _ready():
    return os.getpid()


def _episode(weights, state, test_mode, test_options, training_options, t_env):
    runner = _worker
    _restore(runner.env, state)
    runner.t_env = t_env
    runner.mac.agent.load_state_dict(
        {key: torch.from_numpy(value) for key, value in weights.items()},
    )
    batch, value = runner.run(
        runner.mac, test_mode=test_mode, test_options=test_options,
        training_options=training_options, explore=False,
    )
    return _result(runner, batch, value)


def _result(runner, batch, value):
    return {
        "transition": {key: tensor.numpy() for key, tensor in
                       batch.data.transition_data.items()},
        "episode": {key: tensor.numpy() for key, tensor in batch.data.episode_data.items()},
        "return": value,
        "env": _snapshot(runner.env),
        "hidden": runner.mac.hidden_states.detach().numpy(),
        "pid": os.getpid(),
    }


class RolloutPool:
    def __init__(self, runner, workers):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("rollout_workers must be a nonnegative integer")
        if runner.args.use_cuda:
            raise ValueError("Parallel rollouts support CPU execution only")
        self.tail_runner = None
        context = multiprocessing.get_context("spawn")
        self.executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=context,
            initializer=_initialize,
            initargs=(deepcopy(runner.args), runner.scheme, runner.groups,
                      runner.preprocess, context.Barrier(workers)),
        )
        # Start every worker before the first short group can drain on one process.
        initialized = False
        try:
            ready = [self.executor.submit(_ready) for _ in range(workers)]
            for future in ready:
                future.result()
            initialized = True
        finally:
            if not initialized:
                self.executor.shutdown(wait=True, cancel_futures=True)
        self.worker_pids = set()

    def run(self, runner, policies, test_mode, options, training_options, tail_policy=None):
        if test_mode:
            if training_options is not None or any(
                option is None or option.get("row_idx") is None
                or option.get("initial_soc") is None for option in options
            ):
                raise ValueError("Parallel evaluation needs explicit row and initial SOC")
        elif training_options is None:
            raise ValueError("Parallel training needs a common explicit scenario")
        initial = _snapshot(runner.env)
        # Validation preserves the training inventory exactly as the first serial reset.
        if test_mode and not initial["is_testing"]:
            initial["_training_state"] = (
                initial["idx"].copy(), initial["socs"].copy(),
            )
        futures = []
        for index, (mac, option) in enumerate(zip(policies, options, strict=True)):
            if any(p.device.type != "cpu" or p.dtype != torch.float32
                   for p in mac.parameters()):
                raise ValueError("Parallel policies must be CPU float32")
            weights = {key: value.detach().numpy().copy()
                       for key, value in mac.agent.state_dict().items()}
            state = deepcopy(initial)
            state["ep_counter"] += index
            futures.append(self.executor.submit(
                _episode, weights, state, test_mode, option, training_options,
                runner.t_env + (0 if test_mode else index * runner.episode_limit),
            ))
        tail_result = None
        if tail_policy is not None:
            if self.tail_runner is None:
                args = deepcopy(runner.args)
                args.rollout_workers = 0
                args.env_args["log_trajectories"] = False
                self.tail_runner = EpisodeRunner(args, _NoLog())
                self.tail_runner.setup(runner.scheme, runner.groups,
                                       runner.preprocess, tail_policy)
            tail = self.tail_runner
            state = deepcopy(initial)
            state["ep_counter"] += len(policies)
            _restore(tail.env, state)
            tail.t_env = runner.t_env + len(policies) * runner.episode_limit
            batch, value = tail.run(tail_policy, training_options=training_options)
            tail_result = _result(tail, batch, value)
        # Resolve the whole group before mutating parent state; worker errors propagate.
        results = [future.result() for future in futures]
        if tail_result is not None:
            results.append(tail_result)
            policies = [*policies, tail_policy]
        episodes = []
        for mac, result in zip(policies, results, strict=True):
            if result["pid"] != os.getpid():
                self.worker_pids.add(result["pid"])
            _restore(runner.env, result["env"])
            runner.t = runner.env._ep_steps
            runner.batch = runner.new_batch()
            for category in ("transition", "episode"):
                target = (runner.batch.data.transition_data if category == "transition"
                          else runner.batch.data.episode_data)
                for key, value in result[category].items():
                    target[key].copy_(torch.from_numpy(value))
            mac.hidden_states = torch.from_numpy(result["hidden"]).clone()
            runner.mac = mac
            env = runner.env
            write = env.log_trajectories and (
                env.log_trajectories_mode == "all"
                or env.log_trajectories_mode == ("test" if test_mode else "train")
            )
            if write:
                path = env._current_csv_path()
                exists = path.exists()
                with path.open("a" if exists else "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=list(env.ep_buffer[0]))
                    if not exists:
                        writer.writeheader()
                    writer.writerows(env.ep_buffer)
            info = dict(env._stats)
            info["mean_soc"] /= env._ep_steps
            base = info["baseline_grid_purchase_cost"]
            info["grid_purchase_saving_ratio"] = (
                info["grid_purchase_saving"] / base if base > 1e-9 else 0
            )
            info["episode_limit"] = True
            runner._record_episode(result["return"], info, test_mode)
            episodes.append((runner.batch, result["return"]))
        return episodes

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.tail_runner is not None:
            self.tail_runner.close_env()
