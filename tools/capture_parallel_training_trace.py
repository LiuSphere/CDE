"""Capture a seeded short training trace from a selected immutable source root."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from benchmark_parallel_rollouts import input_hashes

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--rollout-workers", type=int, default=2)
    parser.add_argument("--learner-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source_root.resolve()
    sys.path[:0] = [str(source / "src"), str(source / "tests")]
    import torch
    import test_parallel_rollouts as trace_module

    torch.set_num_threads(1)
    original_config = trace_module.load_config

    def config(name):
        values = original_config(name)
        values["learner_workers"] = args.learner_workers
        return values

    frozen = input_hashes(source)
    critic_gradients = []
    original_train = trace_module.GAVDACLearner.train

    def train(learner, *items, **kwargs):
        result = original_train(learner, *items, **kwargs)
        critic_gradients.append(trace_module.tensor_digest(
            [param.grad for param in learner.critic_params],
        ))
        return result

    (ROOT / "results").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory, \
         patch.object(trace_module, "load_config", config), \
         patch.object(trace_module.GAVDACLearner, "train", train):
        trace = trace_module.training_trace(directory, args.rollout_workers)
    trace["critic_gradients"] = critic_gradients
    if input_hashes(source) != frozen:
        raise ValueError("Trace source changed during execution")
    result = {
        "source_root": str(source), "input_sha256": frozen,
        "rollout_workers": args.rollout_workers, "learner_workers": args.learner_workers,
        "trace": trace,
    }
    if args.compare is not None:
        reference = json.loads(args.compare.read_text(encoding="utf-8"))
        # JSON normalizes the recorded tuples to arrays.
        if json.loads(json.dumps(trace)) != reference["trace"]:
            raise AssertionError("Full training trace differs from reference")
        result["exact_trace_match"] = str(args.compare.resolve())
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({
        "replay_episodes": len(trace["replay"]), "updates": len(trace["updates"]),
        "exact_trace_match": result.get("exact_trace_match"),
    }))


if __name__ == "__main__":
    main()
