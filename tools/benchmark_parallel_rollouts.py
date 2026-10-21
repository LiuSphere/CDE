"""Compare identical real training budgets with serial/spawned execution."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def input_hashes(root=ROOT):
    root = root.resolve()
    paths = sorted(path for path in (root / "src").rglob("*")
                   if path.suffix in {".py", ".yaml", ".csv"})
    paths += [root / "uv.lock", root / "pyproject.toml",
              root / "tests/test_training_timing.py",
              root / "tools/benchmark_parallel_rollouts.py"]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def check_shared_inputs(candidate, reference, config):
    paths = [
        "src/envs/maenv.py", "src/config/default.yaml",
        "src/config/envs/microgrid.yaml", f"src/config/algs/{config}.yaml",
        "uv.lock", "pyproject.toml",
        *(f"src/envs/data/{key}.csv" for key in ("load", "pv", "price")),
    ]
    for path in paths:
        if ((candidate / path).read_bytes().replace(b"\r\n", b"\n")
                != (reference / path).read_bytes().replace(b"\r\n", b"\n")):
            raise ValueError(f"Reference shared input differs: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2, 4, 8])
    parser.add_argument("--learner-workers", nargs="+", type=int, default=[0])
    parser.add_argument("--setting", nargs=2, type=int, action="append",
                        metavar=("ROLLOUT", "LEARNER"),
                        help="Explicit worker pair; repeat to avoid a full grid")
    parser.add_argument("--timesteps", type=int, default=28800)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--config", default="gavdac_ea_main")
    parser.add_argument("--notes", default="")
    parser.add_argument("--reference-root", type=Path,
                        help="Pair each setting against an immutable source snapshot")
    args = parser.parse_args()
    if args.repeats < 1 or args.timesteps < 24 or args.timesteps % 24:
        parser.error("Use positive repeats and a positive whole-day interaction budget")
    for values in (args.workers, args.learner_workers):
        if any(value < 0 for value in values) or len(set(values)) != len(values):
            parser.error("Worker counts must be unique nonnegative integers")
    settings = ([tuple(pair) for pair in args.setting] if args.setting
                else [(r, l) for r in args.workers for l in args.learner_workers])
    if len(set(settings)) != len(settings) or any(min(pair) < 0 for pair in settings):
        parser.error("Settings must be unique nonnegative worker pairs")
    roots = {"candidate": ROOT.resolve()}
    if args.reference_root:
        roots["reference"] = args.reference_root.resolve()
        if roots["reference"] == roots["candidate"]:
            parser.error("Reference and candidate roots must differ")
        check_shared_inputs(roots["candidate"], roots["reference"], args.config)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen_inputs = {label: input_hashes(root) for label, root in roots.items()}
    (args.output / "manifest.json").write_text(json.dumps({
        "input_sha256": frozen_inputs, "platform": platform.platform(),
        "roots": {label: str(root) for label, root in roots.items()},
        "python": sys.version, "logical_cpus": os.cpu_count(),
        "threads_per_process": 1, "notes": args.notes,
        "command": sys.argv,
    }, indent=2) + "\n", encoding="utf-8")
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    results = []
    labels = ["reference", "candidate"] if args.reference_root else ["candidate"]
    cases = [(label, workers, learner_workers) for workers, learner_workers in settings
             for label in labels]
    for repeat in range(args.repeats):
        order = cases if repeat % 2 == 0 else list(reversed(cases))
        for label, workers, learner_workers in order:
            root = roots[label]
            stem = args.output / f"r{repeat}_{label}_w{workers}_l{learner_workers}"
            command = [
                sys.executable, str(root / "tests/test_training_timing.py"),
                "--config", args.config, "--timesteps", str(args.timesteps),
                "--seed", "0", "--threads", "1",
                "--rollout-workers", str(workers), "--output", str(stem.with_suffix(".csv")),
                "--learner-workers", str(learner_workers),
            ]
            with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=root, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            if any(input_hashes(root) != frozen_inputs[name] for name, root in roots.items()):
                raise ValueError("Benchmark source/config/data changed during execution")
            with stem.with_suffix(".csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            summary = {row["kind"]: float(row["seconds"]) for row in rows
                       if row["kind"] in {
                           "wall_clock", "training_wall_clock", "post_warmup_wall_clock",
                       }}
            summary.update(
                repeat=repeat, workers=workers,
                source=label,
                learner_workers=learner_workers,
                joint_steps=int(next(row["t_env"] for row in rows
                                     if row["kind"] == "wall_clock")),
                learner_updates=sum(row["kind"] == "learner_update" for row in rows),
                command=command,
            )
            updates = [float(row["seconds"]) for row in rows
                       if row["kind"] == "learner_update"]
            if len(updates) < 2:
                raise ValueError("Benchmark must cross warmup and complete multiple updates")
            summary["first_update_seconds"] = updates[0]
            summary["steady_mean_update_seconds"] = statistics.mean(updates[1:])
            if summary["joint_steps"] != args.timesteps:
                raise ValueError("Benchmark interaction budget mismatch")
            if results and summary["learner_updates"] != results[0]["learner_updates"]:
                raise ValueError("Benchmark learner-update budget mismatch")
            results.append(summary)
            (args.output / "summary.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8",
            )
            print(json.dumps(summary), flush=True)
    comparisons = []
    for label, *setting in cases:
        group = [row for row in results
                 if (row["workers"], row["learner_workers"]) == tuple(setting)
                 and row["source"] == label]
        comparisons.append({
            "source": label,
            "rollout_workers": setting[0], "learner_workers": setting[1],
            "repeats": len(group),
            **{key: statistics.mean(row[key] for row in group)
               for key in ("wall_clock", "training_wall_clock", "post_warmup_wall_clock",
                           "first_update_seconds", "steady_mean_update_seconds")},
        })
    for row in comparisons:
        baseline = next((baseline for baseline in comparisons
                         if baseline["source"] == row["source"]
                         and baseline["rollout_workers"] == baseline["learner_workers"] == 0),
                        None)
        if baseline is not None:
            row["post_warmup_throughput_gain"] = (
                baseline["post_warmup_wall_clock"] / row["post_warmup_wall_clock"] - 1
            )
            row["startup_inclusive_throughput_gain"] = (
                baseline["wall_clock"] / row["wall_clock"] - 1
            )
            row["steady_update_throughput_gain"] = (
                baseline["steady_mean_update_seconds"] / row["steady_mean_update_seconds"] - 1
            )
    (args.output / "comparison.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8",
    )
    if args.reference_root:
        paired = []
        for workers, learner_workers in settings:
            reference = [row for row in results if row["source"] == "reference"
                         and row["workers"] == workers and row["learner_workers"] == learner_workers]
            candidate = [row for row in results if row["source"] == "candidate"
                         and row["workers"] == workers and row["learner_workers"] == learner_workers]
            paired.append({
                "rollout_workers": workers, "learner_workers": learner_workers,
                "each_pair_total_throughput_gain": [
                    a["wall_clock"] / b["wall_clock"] - 1
                    for a, b in zip(reference, candidate, strict=True)
                ],
                **{f"{key}_throughput_gain": (
                    sum(row[key] for row in reference) / sum(row[key] for row in candidate) - 1
                ) for key in ("wall_clock", "post_warmup_wall_clock", "steady_mean_update_seconds")},
            })
        (args.output / "paired.json").write_text(
            json.dumps(paired, indent=2) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
