"""Profile parent and persistent learner workers in a bounded real training run."""

import argparse
import cProfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from learners import gradient_pool
from benchmark_parallel_rollouts import input_hashes
import test_training_timing

_worker_profile = None


def profiled_gradients(function, *args):
    global _worker_profile
    if _worker_profile is None:
        _worker_profile = cProfile.Profile()
    return _worker_profile.runcall(function, *args)


def profile_summary(profile):
    if profile is None:
        raise RuntimeError("Worker did not execute gradient phases")
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream)
    stats.sort_stats("cumulative").print_stats(45)
    cumulative = stream.getvalue()
    stream.seek(0)
    stream.truncate()
    stats.sort_stats("tottime").print_stats(35)
    return {"cumulative": cumulative, "self_time": stream.getvalue()}


def worker_summary():
    return {"pid": os.getpid(), **profile_summary(_worker_profile)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=28800)
    parser.add_argument("--rollout-workers", type=int, default=2)
    parser.add_argument("--learner-workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    workers = []
    phases = defaultdict(list)
    original_close = gradient_pool.GradientPool.close
    original_backward = gradient_pool.GradientPool.backward
    original_submit = ProcessPoolExecutor.submit

    def submit(executor, function, *items, **kwargs):
        if (function.__module__ == gradient_pool.__name__
                and function.__name__ in {"_gradients", "_shared_gradients"}):
            return original_submit(executor, profiled_gradients, function, *items, **kwargs)
        return original_submit(executor, function, *items, **kwargs)

    def backward(pool, learner, batch, phase):
        started = time.perf_counter()
        result = original_backward(pool, learner, batch, phase)
        phases[phase].append(time.perf_counter() - started)
        return result

    def close(pool):
        try:
            if getattr(pool, "round_id", 0):
                workers.extend(executor.submit(worker_summary).result()
                               for executor in pool.executors)
        finally:
            original_close(pool)

    command = [
        "test_training_timing.py", "--config", "gavdac_ea_main",
        "--timesteps", str(args.timesteps), "--threads", "1",
        "--rollout-workers", str(args.rollout_workers),
        "--learner-workers", str(args.learner_workers),
        "--output", str(args.output / "timing.csv"),
    ]
    frozen = input_hashes()
    with patch.object(sys, "argv", command), \
         patch.object(ProcessPoolExecutor, "submit", submit), \
         patch.object(gradient_pool.GradientPool, "backward", backward), \
         patch.object(gradient_pool.GradientPool, "close", close):
        test_training_timing.main()
    if input_hashes() != frozen:
        raise ValueError("Inputs changed during profiling")
    with (args.output / "profile.json").open("x", encoding="utf-8") as output:
        json.dump({
            "input_sha256": frozen, "command": command,
            "parent_phase_seconds": dict(phases), "workers": workers,
            "note": "Instrumented timings are diagnostic, not an acceptance benchmark.",
        }, output, indent=2)


if __name__ == "__main__":
    main()
