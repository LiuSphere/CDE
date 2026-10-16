"""Run a frozen-source, fully seeded v2/v3 comparison without changing training code."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def validate_run(path, source, seed, algorithm, timesteps):
    run = json.loads(path.read_text(encoding="utf-8"))
    if (run["seed"] != seed or run["algorithm"] != algorithm
            or run["requested_steps"] != timesteps or run["actual_steps"] < timesteps
            or not run.get("pre_rollout_reseed") or run["rng_protocol"] !=
            "entrypoint numpy/torch; benchmark python/fastrand/gym action spaces"):
        raise ValueError(f"Run protocol mismatch: {path}")
    expected = {"rl", "ea"} if algorithm == "facmac_ea" else {"rl"}
    final = [row for row in run["history"] if row.get("phase") == "final"]
    if {row["policy"] for row in final} != expected or len(final) != len(expected):
        raise ValueError(f"Final policies missing: {path}")
    if any(row["step"] != run["actual_steps"] for row in final):
        raise ValueError(f"Final evaluation step mismatch: {path}")
    for name, digest in run["source_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source mismatch: {name}")
    return run


def run_job(experiment, version, algorithm, seed, timesteps):
    source = experiment / "sources" / version
    name = f"{version}_{algorithm}_seed{seed}"
    destination = experiment / "runs" / name
    result = destination / "learning_check.json"
    if result.exists():
        validate_run(result, source, seed, algorithm, timesteps)
        return {"name": name, "result": str(result), "reused_valid_run": True}
    if destination.exists():
        raise FileExistsError(f"Incomplete run retained at {destination}; use a new experiment")
    relative = destination.relative_to(ROOT / "results/facmac_environment")
    command = [
        "uv", "run", "--no-sync", sys.executable, "-B",
        str(ROOT / "tools/benchmark_facmac_environment.py"),
        "--source-root", str(source), "--output-id", str(relative),
        "--seed", str(seed), "--config", algorithm, "--timesteps", str(timesteps),
        "--cpu-threads", "1",
    ]
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED=str(seed),
                       UV_NO_PROGRESS="1")
    attempts = []
    for attempt in range(1, 6):
        log = experiment / "logs" / f"{name}_attempt{attempt}_{uuid4().hex[:6]}.log"
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        attempts.append({"returncode": process.returncode, "log": str(log)})
        if process.returncode == 0:
            validate_run(result, source, seed, algorithm, timesteps)
            return {"name": name, "result": str(result), "attempts": attempts}
        output = log.read_text(encoding="utf-8", errors="replace")
        launcher_failure = (
            not destination.exists() and "error:" in output
            and ("permission denied" in output.lower() or "os error 5" in output.lower())
            and ("uv trampoline" in output or "Failed to spawn" in output)
        )
        if not launcher_failure or attempt == 5:
            raise RuntimeError(f"Run failed (not hidden or skipped): {log}")
        print(f"Recorded uv launch failure for {name}; retrying before Python started", flush=True)
        time.sleep(2)
    raise AssertionError("Unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    if not experiment.is_relative_to(ROOT / "results/facmac_environment"):
        raise ValueError("Experiment must remain in the project results directory")
    if len(set(args.seeds)) != len(args.seeds) or args.workers not in (1, 2):
        raise ValueError("Use distinct seeds and one or two CPU workers")
    manifest = {
        "seeds": args.seeds, "timesteps_per_run": args.timesteps,
        "workers": args.workers, "versions": ["v2", "v3"],
        "algorithms": ["facmac_ea", "facmac_no_ea"],
        "cpu_threads_per_run": 1,
        "source_hashes": {
            str(path.relative_to(experiment)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((experiment / "sources").rglob("*"))
            if path.is_file() and path.suffix in {".py", ".yaml", ".csv"}
        },
        "benchmark_sha256": hashlib.sha256(
            (ROOT / "tools/benchmark_facmac_environment.py").read_bytes()
        ).hexdigest(),
    }
    manifest_path = experiment / "experiment.json"
    if args.resume:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != manifest or (experiment / "completed.json").exists():
            raise ValueError("Resume requires identical sources/protocol and an unfinished matrix")
        with (experiment / f"resume_{uuid4().hex[:8]}.json").open("x", encoding="utf-8") as file:
            json.dump({"launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "reason": "retry retained pre-Python uv launch failures only"}, file, indent=2)
    else:
        with manifest_path.open("x", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
        (experiment / "logs").mkdir(exist_ok=False)
    jobs = [
        (version, algorithm, seed)
        for seed in args.seeds
        for algorithm in manifest["algorithms"]
        for version in manifest["versions"]
    ]
    completed = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_job, experiment, version, algorithm, seed, args.timesteps)
            for version, algorithm, seed in jobs
        ]
        for future in as_completed(futures):
            completed.append(future.result())
            print(f"Completed {len(completed)}/{len(jobs)}: {completed[-1]['name']}", flush=True)
    for name, digest in manifest["source_hashes"].items():
        if hashlib.sha256((experiment / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen experiment source changed: {name}")
    if hashlib.sha256((ROOT / "tools/benchmark_facmac_environment.py").read_bytes()).hexdigest() != manifest["benchmark_sha256"]:
        raise RuntimeError("Benchmark changed during the experiment")
    with (experiment / "completed.json").open("x", encoding="utf-8") as file:
        json.dump({"runs": completed, "seconds": time.perf_counter() - start}, file, indent=2)
    print(f"COMPLETED={experiment}", flush=True)


if __name__ == "__main__":
    main()
