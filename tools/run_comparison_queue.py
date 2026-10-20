"""Execute a frozen per-host evaluation queue in isolated source runtimes."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_manifest(manifest):
    if manifest["rows"] != list(range(913, 1095)):
        raise ValueError("Formal comparison requires the entire frozen182-day holdout")
    for name, expected in manifest["implementation_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen evaluation implementation changed: {name}")
    keys, outputs = [], []
    for job in manifest["jobs"]:
        if job["kind"] not in {"policy", "rule", "milp"}:
            raise ValueError("Unknown evaluation job kind")
        if job["mode"] not in {"continuous", "daily_reset"}:
            raise ValueError("Unknown inventory mode")
        output = Path(job["output"]).resolve()
        if not output.is_relative_to((ROOT / "results").resolve()):
            raise ValueError("Evaluation outputs must be below this host's results/")
        keys.append(job["id"])
        outputs.append(str(output))
        if job["kind"] == "policy":
            checkpoint = Path(job["checkpoint"])
            if hashlib.sha256((checkpoint / "metadata.json").read_bytes()).hexdigest() != job["metadata_sha256"]:
                raise ValueError("Selected policy metadata changed after freeze")
            if hashlib.sha256((checkpoint / "agent.th").read_bytes()).hexdigest() != job["policy_sha256"]:
                raise ValueError("Selected policy changed after freeze")
            if hashlib.sha256((checkpoint / "config.yaml").read_bytes()).hexdigest() != job["config_sha256"]:
                raise ValueError("Selected policy configuration changed after freeze")
    if len(keys) != len(set(keys)) or len(outputs) != len(set(outputs)):
        raise ValueError("Duplicate frozen evaluation job/output")


def job_command(job, rows):
    if job["kind"] == "milp":
        return [
            "uv", "run", "--frozen", "python", "tools/run_continuous_microgrid_milp.py",
            "--split", "test", "--retrospective-holdout", "--mode", job["mode"],
            "--rows", *map(str, rows), "--segments", str(job["segments"]),
            "--time-limit", str(job["time_limit"]), "--mip-gap", str(job["mip_gap"]),
            "--output", job["output"],
        ]
    command = [
        "uv", "run", "--frozen", "python", "tools/evaluate_microgrid_comparison.py",
        "--split", "test", "--retrospective-holdout", "--mode", job["mode"],
        "--all-days", "--output", job["output"],
    ]
    return (command + ["--rule"] if job["kind"] == "rule"
            else command + ["--checkpoint", job["checkpoint"]])


def execute(manifest_path, seed, token):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    directory = ROOT / "results" / token
    results = []
    # The supervisor owns the outer directory and logs. Job outputs are new,
    # immutable children; a failure never retries or overwrites an earlier job.
    for job in manifest["jobs"]:
        if job["seed"] is not None and job["seed"] != seed:
            continue
        started = time.perf_counter()
        stdout = directory / f"{job['id']}.stdout.log"
        stderr = directory / f"{job['id']}.stderr.log"
        command = job_command(job, manifest["rows"])
        with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
            result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
        record = {
            "id": job["id"], "command": command, "returncode": result.returncode,
            "wall_seconds": time.perf_counter() - started, "output": job["output"],
        }
        with (directory / f"{job['id']}.exit.json").open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
        if result.returncode:
            raise RuntimeError(f"Evaluation failed: {job['id']}; preserve per-job logs")
        summary = json.loads((Path(job["output"]) / "summary.json").read_text())
        if (summary["rows"] != manifest["rows"] or summary["inventory_mode"] != job["mode"]
                or not summary["physical_checks_passed"]
                or (job["kind"] == "milp" and not summary["acceptance_passed"])):
            raise ValueError(f"Evaluation acceptance failed: {job['id']}")
        if job["kind"] == "policy" and summary["policy_sha256"] != job["policy_sha256"]:
            raise ValueError("Evaluated actor differs from frozen selection")
        results.append(record)
        print(json.dumps({"job": job["id"], "completed_jobs": len(results),
                          "wall_seconds": record["wall_seconds"]}), flush=True)
    if not results:
        raise ValueError("No jobs selected for this host/seed")
    with (directory / "completion_checks.json").open("x", encoding="utf-8") as file:
        json.dump({
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "seed": seed, "jobs": results, "all_jobs_accepted": True,
        }, file, indent=2)
    print("Finished Evaluation\nExiting script", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    execute(args.manifest.resolve(), args.seed, args.token)


if __name__ == "__main__":
    main()
