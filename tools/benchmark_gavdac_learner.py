"""Benchmark fixed-shape CPU learner updates and optionally save a reference."""

import argparse
import importlib
import json
import random
import sys
import time
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch

import fastrand
import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]


class LossLog:
    def __init__(self):
        self.values = {}

    def log_stat(self, name, value, step):
        self.values.setdefault(name, []).append(float(value))


def load_reference(source_root):
    exports = (
        ("modules.policy", "RNNAgent"),
        ("controllers.cqmix_controller", "CQMixMAC"),
        ("modules.graph.gin_layer", "GINGraphConvolution"),
        ("modules.graph.gnn", "GNN"),
        ("modules.graph_mixer", "GraphDec"),
        ("learners.gavdac_learner", "GAVDACLearner"),
    )
    classes = {}
    with ExitStack() as stack:
        for name, symbol in exports:
            module = importlib.import_module(name)
            path = source_root.joinpath(*name.split(".")).with_suffix(".py")
            namespace = {"__name__": name, "__package__": name.rpartition(".")[0]}
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
            classes[symbol] = namespace[symbol]
            stack.enter_context(patch.object(module, symbol, namespace[symbol]))
    return classes["CQMixMAC"], classes["GAVDACLearner"]


def snapshot(learner, logger):
    modules = {
        "actor": learner.mac.agent, "target_actor": learner.target_mac.agent,
        "critic": learner.critic, "target_critic": learner.target_critic,
    }
    if learner.mixer is not None:
        modules.update(mixer=learner.mixer, target_mixer=learner.target_mixer)
    return {
        "networks": {name: module.state_dict() for name, module in modules.items()},
        "actor_optimizer": learner.agent_optimiser.state_dict(),
        "critic_optimizer": learner.critic_optimiser.state_dict(),
        "gradients": {
            name: {key: parameter.grad for key, parameter in module.named_parameters()}
            for name, module in modules.items()
        },
        "losses": deepcopy(logger.values),
    }


def timing_summary(elapsed):
    return {
        "seconds": elapsed, "median_seconds": median(elapsed),
        "mean_seconds": sum(elapsed) / len(elapsed),
        "updates_per_second": len(elapsed) / sum(elapsed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--compare-source-root", type=Path)
    parser.add_argument("--config", default="gavdac_ea")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.batch_size < 1:
        parser.error("iterations/batch-size must be positive and warmup nonnegative")
    for path in (args.output, args.state_output):
        if path is not None and path.exists():
            raise FileExistsError(path)
    sys.path.insert(0, str(ROOT / "src"))
    if args.source_root is not None:
        if not args.source_root.is_dir():
            raise FileNotFoundError(args.source_root)
        sys.path.insert(0, str(args.source_root.resolve()))

    from components.episode_buffer import EpisodeBatch
    from controllers.cqmix_controller import CQMixMAC
    from direct_main import load_yaml, normalize_scalars, recursive_dict_update
    from envs.maenv import MGEnv
    from learners.gavdac_learner import GAVDACLearner

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    fastrand.pcg32_seed(args.seed)
    if args.threads is not None:
        th.set_num_threads(args.threads)

    config_dir = ROOT / "src" / "config"
    config = load_yaml(config_dir / "default.yaml")
    recursive_dict_update(config, load_yaml(config_dir / "envs" / "microgrid.yaml"))
    recursive_dict_update(config, load_yaml(config_dir / "algs" / f"{args.config}.yaml"))
    config = normalize_scalars(config)
    env = MGEnv(deepcopy(config["env_args"]), config["name"])
    info = env.get_env_info()
    env.close()
    config.update(
        n_agents=info["n_agents"], n_actions=info["n_actions"],
        state_shape=info["state_shape"], obs_shape=info["obs_shape"],
        action_spaces=info["action_spaces"], learner_log_interval=1,
        device="cpu",
    )
    learner_args = SimpleNamespace(**config)
    scheme = {
        "obs": {"vshape": info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (info["n_actions"],), "group": "agents"},
        "state": {"vshape": info["state_shape"]},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": info["n_agents"]}
    batch = EpisodeBatch(scheme, groups, args.batch_size, info["episode_limit"] + 1)
    generator = th.Generator().manual_seed(args.seed + 1000)
    data = {}
    for name, tensor in batch.data.transition_data.items():
        if name in ("terminated", "filled"):
            continue
        data[name] = th.randn(tensor.shape, generator=generator) * 0.25
    low = th.as_tensor(np.stack([space.low for space in info["action_spaces"]]))
    high = th.as_tensor(np.stack([space.high for space in info["action_spaces"]]))
    data["actions"] = low + (data["actions"].tanh() + 1) * 0.5 * (high - low)
    batch.update(data)
    mac = CQMixMAC(scheme, groups, learner_args)
    logger = LossLog()
    learner = GAVDACLearner(mac, scheme, logger, learner_args)
    reference = None
    if args.compare_source_root is not None:
        reference_mac, reference_learner = load_reference(args.compare_source_root)
        th.manual_seed(args.seed)
        reference_logger = LossLog()
        reference = reference_learner(
            reference_mac(scheme, groups, learner_args), scheme, reference_logger, learner_args
        )
        for name in ("mac.agent", "target_mac.agent", "critic", "target_critic",
                     "mixer", "target_mixer"):
            current_module, reference_module = learner, reference
            for attribute in name.split("."):
                current_module = getattr(current_module, attribute, None)
                reference_module = getattr(reference_module, attribute, None)
            if current_module is not None:
                reference_module.load_state_dict(current_module.state_dict(), strict=True)
    elapsed = []
    reference_elapsed = []
    for index in range(args.warmup + args.iterations):
        order = [(learner, elapsed)]
        if reference is not None:
            order.append((reference, reference_elapsed))
            if index % 2:
                order.reverse()
        for implementation, times in order:
            started = time.perf_counter()
            implementation.train(batch, index, index)
            duration = time.perf_counter() - started
            if index >= args.warmup:
                times.append(duration)

    state = snapshot(learner, logger)
    if args.state_output is not None:
        with args.state_output.open("xb") as output:
            th.save(state, output)

    result = {
        "source_root": str(args.source_root or ROOT / "src"),
        "seed": args.seed, "threads": th.get_num_threads(),
        "batch_size": args.batch_size, "sequence_length": batch.max_seq_length,
        "warmup": args.warmup, "iterations": args.iterations,
        **timing_summary(elapsed),
        "losses": deepcopy(logger.values),
    }
    if reference is not None:
        reference_state = snapshot(reference, reference_logger)
        th.testing.assert_close(state, reference_state, rtol=1e-4, atol=1e-5)
        result["reference"] = timing_summary(reference_elapsed)
        result["speedup"] = (
            result["updates_per_second"] / result["reference"]["updates_per_second"]
        )
        result["max_parameter_difference"] = max(
            float((value - reference_state["networks"][name][key]).abs().max())
            for name, values in state["networks"].items() for key, value in values.items()
        )
        result["equivalence_tolerance"] = {"rtol": 1e-4, "atol": 1e-5}
    if args.profile:
        with th.profiler.profile(activities=[th.profiler.ProfilerActivity.CPU]) as profile:
            learner.train(batch, args.warmup + args.iterations, args.warmup + args.iterations)
        result["profile"] = profile.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=20
        )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("seconds", "losses")}, indent=2))


if __name__ == "__main__":
    main()
