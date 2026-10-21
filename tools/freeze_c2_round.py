"""Freeze H01 and the immutable original-host C3 comparator identities."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import direct_main
from tools.report_comparison_results import load_evaluation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(name):
    loaded = []
    with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
         patch.object(direct_main, "run", lambda conf, logger: loaded.append(conf)):
        direct_main.main()
    return loaded[0]


def freeze():
    parent, candidate = config("gavdac_ea_main"), config("gavdac_c2_h01")
    if candidate != {**parent, "name": "gavdac_c2_h01", "actor_mixer_context_grad": False}:
        raise ValueError("H01 contains changes beyond the registered intervention")
    previous = ROOT / "results/cmp20"
    index = json.loads((previous / "report_index_three_a.json").read_text())
    frozen = json.loads((previous / "evaluation_seed2_a.json").read_text())
    protected = [
        "src/run_mappo.py", "src/modules/mappo.py",
        "src/config/algs/gavdac_mappo.yaml", "src/config/default.yaml",
        "src/config/envs/microgrid.yaml", "src/envs/maenv.py", "src/direct_main.py",
        "src/envs/data/load.csv", "src/envs/data/price.csv", "src/envs/data/pv.csv",
        "uv.lock", "pyproject.toml",
    ]
    for name in protected:
        if digest(ROOT / name) != frozen["implementation_sha256"][name]:
            raise ValueError(f"Frozen shared condition or C3 implementation changed: {name}")
    baseline = {}
    for seed in (0, 1, 2):
        item = next(i for i in index["evaluations"] if i["group"] == "C3"
                    and i["seed"] == seed and i["mode"] == "continuous")
        summary, _ = load_evaluation(item)
        manifest_path = previous / f"evaluation_seed{seed}_a.json"
        manifest = json.loads(manifest_path.read_text())
        selected = next(i for i in manifest["selected"] if i["group"] == "C3")
        if summary["policy_sha256"] != selected["policy_sha256"]:
            raise ValueError("C3 baseline does not match its frozen original Actor")
        baseline[str(seed)] = {
            **item, "selected": selected,
            "original_manifest_sha256": digest(manifest_path),
            "economic_cost": summary["totals"]["economic_cost"],
            "candidate_maximum_cost": .97 * summary["totals"]["economic_cost"],
        }
    return {
        "campaign": "Exploratory single-hypothesis C2 optimization",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "H01_verified_ready_to_launch",
        "latest_decision": "docs/c2_optimization.md",
        "original_campaign": "results/cmp20/state.json; preserved, P4 solver decision still paused",
        "active_hypothesis": "H01",
        "hypothesis": "Fixing the attention context during Actor optimization removes a "
                      "non-action policy-gradient route and improves physical dispatch",
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "rollback": "If any of three seeds or the mean misses3%, git revert ONLY "
                    "implementation_commit after all runs/evaluations finish; normal push",
        "config": candidate,
        "baseline_C3": baseline,
        "protected_sha256": {name: digest(ROOT / name) for name in protected},
        "source_config_sha256": {
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in (ROOT / "src").rglob("*") if path.suffix in {".py", ".yaml"}},
        "data_sha256": {path.name: digest(path) for path in (ROOT / "src/envs/data").glob("*.csv")},
        "run_plan": [
            {"seed": seed, "host": "local" if seed == 2 else "707",
             "root": str(ROOT) if seed == 2 else "/home/arch707/data/CDE",
             "slot": 0 if seed == 2 else seed, "capacity": 1 if seed == 2 else 2,
             "token": f"h01s{seed}a", "status": "planned",
             "config_name": "gavdac_c2_h01", "joint_budget": 6000000}
            for seed in (0, 1, 2)],
        "active_runs": [], "completed_runs": [],
        "evaluation": {
            "rows": list(range(913, 1095)), "mode": "continuous",
            "actor": "validation-selected rl", "initial_soc": .35,
            "terminal_soc": "free within device bounds", "minimum_saving_fraction": .03,
            "requirement": "Each seed-label saving AND ratio-of-means saving >=0.03",
            "limitation": "Exploratory repeated retained-data development, not independent test evidence",
        },
        "checks": {
            "focused": "5H01tests passed16.258s, including real warmup/learner/EA smoke",
            "regression_log": "results/c2opt21/h01_regression_a.log",
            "cross_host_preflight": "required before launch",
        },
        "scheduling": {"earliest_eta_utc": None, "next_wake_target_utc": None},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--register-runs", action="store_true")
    args = parser.parse_args()
    path = args.output.resolve()
    if not path.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Round freeze must stay under results/")
    if args.register_runs:
        register_runs(path)
        return
    payload = freeze()
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    print(json.dumps({"output": str(path), "hypothesis": "H01",
                      "targets": {seed: item["candidate_maximum_cost"]
                                  for seed, item in payload["baseline_C3"].items()}}))


def register_runs(state_path):
    state = json.loads(state_path.read_text())
    if state["status"] != "H01_verified_ready_to_launch" or state["active_runs"]:
        raise ValueError("Register the initial frozen wave exactly once")
    active = []
    for plan in state["run_plan"]:
        relative = f"results/{plan['token']}/launch.json"
        manager = json.loads((state_path.parent / f"{plan['token']}_detach.json").read_text())
        if plan["host"] == "local":
            launch = json.loads((ROOT / relative).read_text())
            from tools.experiment_slot import process_identity
            identities = {str(pid): process_identity(pid) for pid in (
                manager["manager_pid"], launch["supervisor_pid"], launch["child_pid"])}
        else:
            launch = json.loads(subprocess.check_output(
                ["ssh", "-o", "BatchMode=yes", "707",
                 shlex.join(["cat", f"{plan['root']}/{relative}"])], text=True))
            probe = (
                "import json; from tools.experiment_slot import process_identity; "
                f"print(json.dumps({{str(p):process_identity(p) for p in "
                f"{[manager['manager_pid'], launch['supervisor_pid'], launch['child_pid']]!r}}}))")
            identities = json.loads(subprocess.check_output(
                ["ssh", "-o", "BatchMode=yes", "707", shlex.join([
                    "uv", "--directory", plan["root"], "run", "--frozen", "python", "-c", probe])],
                text=True))
        if (launch["token"] != plan["token"] or launch["seed"] != plan["seed"]
                or launch["config"] != "gavdac_c2_h01" or launch["cpu_threads"] != 1
                or launch["git_commit"] != state["implementation_commit"]):
            raise ValueError("Actual run differs from the registered hypothesis")
        for pid_key, identity_key, record in (
                ("manager_pid", "manager_identity", manager),
                ("supervisor_pid", "supervisor_identity", launch),
                ("child_pid", "child_identity", launch)):
            if identities[str(record[pid_key])] != record[identity_key]:
                raise ValueError("Run process creation identity mismatch")
        # The first estimate uses the accepted C2 timing on the same hardware.
        duration = 14649.672 if plan["host"] == "local" else 19305.711
        eta = datetime.fromisoformat(launch["created_at"]) + timedelta(seconds=duration)
        active.append({
            **plan, "status": "running", "manager": manager, "launch": launch,
            "eta_utc": eta.isoformat(),
            "eta_basis": "Initial same-host accepted C2 timing; reestimate from actual progress",
        })
        plan["status"] = "running"
    earliest = min(datetime.fromisoformat(run["eta_utc"]) for run in active)
    state.update(status="H01_three_seeds_running", active_runs=active)
    state["scheduling"].update(
        earliest_eta_utc=earliest.isoformat(),
        next_wake_target_utc=(earliest + timedelta(minutes=2)).isoformat())
    with state_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, allow_nan=False)
    print(json.dumps({"active": [run["token"] for run in active],
                      "earliest_eta_utc": earliest.isoformat()}))


if __name__ == "__main__":
    main()
