"""Profile one fixed-length GAVDAC run without changing production code."""

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict
from copy import deepcopy
from os.path import abspath, dirname

import fastrand
import numpy as np
import torch as th

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from components.episode_buffer import EpisodeBatch, ReplayBuffer
from controllers.cqmix_controller import CQMixMAC
from direct_main import config_copy, load_yaml, normalize_scalars, recursive_dict_update
from ea.ssne_base import SSNEBase
from envs.maenv import MGEnv
from learners.gavdac_learner import GAVDACLearner
from run import run
from runners.episode_runner import EpisodeRunner
from utils.logging import get_logger


class Timer:
    def __init__(self):
        self.calls = defaultdict(int)
        self.seconds = defaultdict(float)

    def record(self, name, elapsed):
        self.calls[name] += 1
        self.seconds[name] += elapsed

    def snapshot(self):
        return dict(self.calls), dict(self.seconds)

    def delta(self, snapshot, name):
        calls, seconds = snapshot
        return self.calls[name] - calls.get(name, 0), self.seconds[name] - seconds.get(
            name, 0.0
        )


class CsvTimingLog:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def write(self, kind, seconds, t_env="", episode="", step=""):
        self.rows.append(
            {
                "kind": kind,
                "t_env": t_env,
                "episode": episode,
                "step": step,
                "seconds": f"{seconds:.9f}",
            }
        )

    def close(self):
        with open(self.path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("kind", "t_env", "episode", "step", "seconds"),
            )
            writer.writeheader()
            writer.writerows(self.rows)


def timed_method(cls, method_name, timer, metric_name, after=None):
    original = getattr(cls, method_name)

    def wrapped(self, *args, **kwargs):
        started = time.perf_counter()
        result = original(self, *args, **kwargs)
        elapsed = time.perf_counter() - started
        timer.record(metric_name, elapsed)
        if after is not None:
            after(self, args, kwargs, elapsed)
        return result

    setattr(cls, method_name, wrapped)


def load_config(config_name):
    config_dir = os.path.join(PROJECT_ROOT, "src", "config")
    config = load_yaml(os.path.join(config_dir, "default.yaml"))
    config = recursive_dict_update(
        config, load_yaml(os.path.join(config_dir, "envs", "microgrid.yaml"))
    )
    config = recursive_dict_update(
        config, load_yaml(os.path.join(config_dir, "algs", f"{config_name}.yaml"))
    )
    return normalize_scalars(config_copy(config))


def main():
    parser = argparse.ArgumentParser(description="Profile GAVDAC training bottlenecks.")
    parser.add_argument("--config", default="gavdac_ea")
    parser.add_argument("--timesteps", type=int, default=12144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/training_timing.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    fastrand.pcg32_seed(args.seed)
    original_env_init = MGEnv.__init__

    def seeded_env_init(env, *method_args, **method_kwargs):
        original_env_init(env, *method_args, **method_kwargs)
        for index, space in enumerate(env.get_env_info()["action_spaces"]):
            space.seed(args.seed + index)

    MGEnv.__init__ = seeded_env_init
    config = load_config(args.config)
    episode_limit = MGEnv(deepcopy(config["env_args"]), config["name"]).episode_limit
    if args.timesteps % episode_limit:
        raise ValueError(
            f"--timesteps must be divisible by the {episode_limit}-step episode limit."
        )

    # run.py uses <=, so stop one episode early to end at exactly --timesteps.
    config.update(
        {
            "seed": args.seed,
            "use_cuda": False,
            "wandb_mode": "disabled",
            "save_model": False,
            "test_nepisode": 1,
            "t_max": args.timesteps - episode_limit,
            # Keep the profiler output limited to its CSV and final summary.
            "test_interval": args.timesteps + episode_limit,
            "log_interval": args.timesteps + episode_limit,
            "runner_log_interval": args.timesteps + episode_limit,
            "learner_log_interval": args.timesteps + episode_limit,
        }
    )
    config["env_args"]["seed"] = args.seed
    config["env_args"]["run_id"] = "timing_profile"

    timer = Timer()
    csv_log = CsvTimingLog(args.output)
    training_steps = 0
    training_started = None
    first_update_started = None
    first_update_t_env = None
    phase_times = {}
    run_started = time.perf_counter()

    def report_episode(runner, method_args, method_kwargs, elapsed):
        nonlocal training_steps, training_started
        test_mode = method_kwargs.get("test_mode", False)
        if not test_mode:
            if training_started is None:
                training_started = time.perf_counter() - elapsed
            training_steps += runner.t
            csv_log.write("episode_rollout", elapsed, runner.t_env, step=runner.t)

    def report_update(learner, method_args, method_kwargs, elapsed):
        nonlocal first_update_started, first_update_t_env
        if first_update_started is None:
            first_update_started = time.perf_counter() - elapsed
            first_update_t_env = method_args[1]
        csv_log.write("learner_update", elapsed, method_args[1], method_args[2])

    def report_training_end(runner, method_args, method_kwargs, elapsed):
        finished = time.perf_counter()
        if training_started is not None:
            phase_times["training"] = (training_steps, finished - training_started)
        if first_update_started is not None:
            phase_times["post_warmup"] = (
                training_steps - first_update_t_env, finished - first_update_started
            )
        for name, (steps, seconds) in phase_times.items():
            csv_log.write(f"{name}_wall_clock", seconds, t_env=steps)

    def report_environment_step(env, method_args, method_kwargs, elapsed):
        csv_log.write("environment_step", elapsed, step=env.idx[1])

    def report_evolution(evolver, method_args, method_kwargs, elapsed):
        csv_log.write("evolution", elapsed)

    def report_action_selection(mac, method_args, method_kwargs, elapsed):
        csv_log.write("action_selection", elapsed)

    def report_batch_update(batch, method_args, method_kwargs, elapsed):
        csv_log.write("batch_update", elapsed)

    def report_replay_sample(buffer, method_args, method_kwargs, elapsed):
        csv_log.write("replay_sample", elapsed)

    timed_method(
        MGEnv, "step", timer, "environment_step", report_environment_step
    )
    timed_method(
        CQMixMAC,
        "select_actions",
        timer,
        "action_selection",
        report_action_selection,
    )
    timed_method(EpisodeBatch, "update", timer, "batch_update", report_batch_update)
    timed_method(ReplayBuffer, "sample", timer, "replay_sample", report_replay_sample)
    timed_method(EpisodeRunner, "run", timer, "episode_rollout", report_episode)
    timed_method(EpisodeRunner, "close_env", timer, "environment_close", report_training_end)
    timed_method(GAVDACLearner, "train", timer, "learner_update", report_update)
    timed_method(SSNEBase, "epoch", timer, "evolution", report_evolution)

    logger = get_logger()
    logger.setLevel("WARNING")
    try:
        run(config, logger)
    finally:
        total = time.perf_counter() - run_started
        csv_log.write("wall_clock", total, t_env=training_steps)
        csv_log.close()

    print("\nTiming summary")
    print(f"wall_clock: {total:.3f}s")
    print(f"training_steps: {training_steps}")
    print(f"training_steps_per_second: {training_steps / total:.3f}")
    print(f"torch_threads: {th.get_num_threads()}")
    for name, (steps, seconds) in phase_times.items():
        print(f"{name}_phase_wall_clock: {seconds:.3f}s")
        print(f"{name}_phase_steps_per_second: {steps / seconds:.3f}")
    print(f"detail_csv: {args.output}")
    for name in (
        "episode_rollout",
        "learner_update",
        "evolution",
        "replay_sample",
        "action_selection",
        "environment_step",
        "batch_update",
    ):
        calls = timer.calls[name]
        seconds = timer.seconds[name]
        share = 100 * seconds / total if total else 0.0
        average = seconds / calls if calls else 0.0
        print(
            f"{name:20} {seconds:8.3f}s {share:5.1f}% "
            f"calls={calls:5d} avg={average:.5f}s"
        )


if __name__ == "__main__":
    main()
