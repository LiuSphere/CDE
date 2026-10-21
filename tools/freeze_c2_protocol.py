"""Freeze the registered H02 round without launching or resuming any process."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import direct_main
from tools.report_comparison_results import load_evaluation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entry_config(name, overrides=None):
    argv = ["direct_main.py", "--config", name, "--env-config", "microgrid"]
    for key, value in (overrides or {}).items():
        argv += ["--override", f"{key}={json.dumps(value)}"]
    values = []
    with patch.object(sys, "argv", argv), \
         patch.object(direct_main, "run", lambda value, logger: values.append(value)):
        direct_main.main()
    return values[0]


def validate_candidate(parent, candidate):
    expected = {**parent, "name": "gavdac_c2_h02", "clipped_double_critic": True}
    if candidate != expected:
        raise ValueError("H02 differs beyond its name and registered critic option")
    if (candidate["t_max"] != 6000000 or not candidate["EA"]
            or candidate["crossover_prob"] != 0 or candidate["use_cuda"]
            or candidate["checkpoint_path"] or candidate["policy_checkpoint"]):
        raise ValueError("H02 requires fresh CPU weights and the frozen budget/EA conditions")


def compare_host(previous, current):
    for name, expected in previous["files"].items():
        if current["files"].get(name, {}).get("raw") != expected["raw"]:
            raise ValueError(f"Qualified host file changed: {name}")
    if (current["slots"] != [{"slot": slot, "state": "free"}
                            for slot in range(len(current["slots"]))]
            or current["dirty_scope"] or current["existing_tokens"]):
        raise ValueError("Host is dirty, occupied, or a proposed token already exists")
    if current["platform"] != previous["platform"]:
        raise ValueError("Qualified host OS changed")


def validate_evaluation(evaluation):
    required = {
        "rows": list(range(913, 1095)), "mode": "continuous",
        "actor": "validation-selected rl", "initial_soc": .35,
        "terminal_soc": "free within device bounds", "minimum_saving_fraction": .03,
    }
    if any(evaluation.get(name) != value for name, value in required.items()):
        raise ValueError("Registered evaluation or economic threshold changed")


HOST_PROBE = r'''
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"
os.environ["CUDA_VISIBLE_DEVICES"]=""
root=Path.cwd()
sys.path.insert(0,str(root))
from tools.experiment_slot import slot_status
payload=json.load(sys.stdin)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
files={}
for base in ("src","tools","tests"):
    for path in (root/base).rglob("*"):
        if path.suffix in {".py",".yaml",".csv"}:
            raw=path.read_bytes()
            files[path.relative_to(root).as_posix()]={
                "raw":hashlib.sha256(raw).hexdigest(),
                "lf":hashlib.sha256(raw.replace(b"\r\n",b"\n")).hexdigest()}
for name in ("pyproject.toml","uv.lock"):
    files[name]={"raw":sha(root/name),"lf":hashlib.sha256((root/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()}
existing=[]
for token in payload["tokens"]:
    if (root/"results"/token).exists() or any(
        (root/"results/cmp20/supervisors"/(token+suffix)).exists()
        for suffix in (".stdout.log",".stderr.log")):
        existing.append(token)
actors=[]
if payload["actors"]:
    import torch
    torch.set_num_threads(1)
    for actor in payload["actors"]:
        checkpoint=Path(actor["checkpoint"])
        for name,key in (("metadata.json","metadata_sha256"),("agent.th","policy_sha256"),
                         ("config.yaml","config_sha256")):
            if sha(checkpoint/name)!=actor[key]:
                raise ValueError("Frozen C3checkpoint changed: "+str(checkpoint/name))
        metadata=json.loads((checkpoint/"metadata.json").read_text())
        if metadata["validation"]["rows"]!=list(range(731,739)) or metadata["policy_source"]!="rl":
            raise ValueError("Frozen C3selection changed")
        source=Path(metadata["source_snapshot"]).resolve()
        for name,expected in metadata["source_sha256"].items():
            path=(source/name).resolve()
            if not path.is_relative_to(source) or sha(path)!=expected:
                raise ValueError("Frozen C3saved source changed: "+name)
        for name,expected in metadata["data_sha256"].items():
            if sha(root/"src/envs/data"/name)!=expected:
                raise ValueError("Frozen C3input bytes changed: "+name)
        weights=torch.load(checkpoint/"agent.th",map_location="cpu",weights_only=True)
        if not all(not value.is_floating_point() or
                   (value.dtype==torch.float32 and torch.isfinite(value).all())
                   for value in weights.values()):
            raise ValueError("Frozen C3Actor weights nonfinite or not float32")
        actors.append({"seed":actor["seed"],"checkpoint":str(checkpoint),
                       "raw_identity_and_source_data_checks":True})
print(json.dumps({
    "root":str(root),"platform":platform.platform(),"logical_cpu_count":os.cpu_count(),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "dirty_scope":subprocess.check_output(["git","status","--porcelain","--",
        "src","tools","tests","pyproject.toml","uv.lock"],text=True),
    "slots":slot_status(payload["slots"]),"files":files,"existing_tokens":existing,
    "original_host_C3_actor_checks":actors,
}))
'''


def probe(host, root, tokens, actors, slots):
    child = ["uv", "run", "--frozen", "python", "-c", HOST_PROBE]
    command = (child if host == "local" else [
        "ssh", "-o", "ConnectTimeout=15", host, shlex.join([
            "uv", "--directory", root, "run", "--frozen", "python", "-c", HOST_PROBE])])
    result = subprocess.run(
        command, cwd=ROOT, input=json.dumps({"tokens": tokens, "actors": actors, "slots": slots}),
        text=True, encoding="utf-8", capture_output=True, check=True)
    return json.loads(result.stdout)


def freeze(state_path):
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (state["active_hypothesis"] != "H02" or state["active_runs"]
            or state["status"] != "H02_engineering_passed_pending_formal_freeze"):
        raise ValueError("Freeze requires the idle, engineering-accepted H02 state")
    parent, candidate = entry_config("gavdac_ea_main"), entry_config("gavdac_c2_h02")
    validate_candidate(parent, candidate)
    validate_evaluation(state["evaluation"])
    subprocess.run(["git", "diff", "--exit-code", state["implementation_commit"], "--", "src"],
                   cwd=ROOT, check=True, capture_output=True)
    for name, expected in state["protected_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Protected comparison condition changed: {name}")
    base = state_path.parent
    original_baseline = base / "h01_protocol_b.json"
    if state["baseline_C3"] != json.loads(original_baseline.read_text())["baseline_C3"]:
        raise ValueError("C3identities differ from the immutable comparator freeze")
    engineering_path = base / "h02_engineering_acceptance_a.json"
    engineering = json.loads(engineering_path.read_text())
    if engineering["status"] != "engineering_passed_not_economic_acceptance":
        raise ValueError("Engineering gate is incomplete")
    for name, evidence in engineering["logs"].items():
        if digest(base / name) != evidence["sha256"]:
            raise ValueError("Engineering log identity changed")
    smokes_path = base / "h02_exclusive_smokes_acceptance_a.json"
    smokes = json.loads(smokes_path.read_text())
    if not smokes["accepted"] or len(smokes["smokes"]) != 4:
        raise ValueError("Exclusive smoke gate is incomplete")
    for seed, item in state["baseline_C3"].items():
        summary, _ = load_evaluation(item)
        manifest_path = ROOT / f"results/cmp20/evaluation_seed{seed}_a.json"
        if digest(manifest_path) != item["original_manifest_sha256"]:
            raise ValueError("Original C3manifest changed")
        selected = next(value for value in json.loads(manifest_path.read_text())["selected"]
                        if value["group"] == "C3")
        if (selected != item["selected"] or summary["policy_sha256"] != selected["policy_sha256"]
                or summary["totals"]["economic_cost"] != item["economic_cost"]
                or item["candidate_maximum_cost"] != .97 * item["economic_cost"]):
            raise ValueError("Frozen C3selection, cost, or threshold changed")
    plans = [
        {"seed": seed, "host": "local" if seed == 2 else "707",
         "root": str(ROOT) if seed == 2 else "/home/arch707/data/CDE",
         "slot": 0 if seed == 2 else seed, "capacity": 1 if seed == 2 else 2,
         "token": f"h02s{seed}a", "config_name": "gavdac_c2_h02",
         "joint_budget": 6000000, "status": "frozen_not_launched"}
        for seed in (0, 1, 2)]
    tokens = [plan["token"] for plan in plans]
    hosts = {}
    for host, root, capacity in (
        ("local", str(ROOT), 1), ("707", "/home/arch707/data/CDE", 2),
        ("715", "/home/bless/projects/CDE_c2opt21", 1),
    ):
        actors = [item["selected"] for seed, item in state["baseline_C3"].items()
                  if ("local" if seed == "2" else "707") == host]
        hosts[host] = probe(host, root, tokens, actors, capacity)
        previous = json.loads((base / f"h02_host_{host}_a.json").read_text())
        compare_host(previous, hosts[host])
        if hosts[host]["git_commit"] != hosts["local"]["git_commit"]:
            raise ValueError("Synchronize idle hosts to the frozen runtime commit first")
        for smoke in smokes["smokes"]:
            if smoke["host"] == host:
                for name, expected in smoke["launch"]["source_sha256"].items():
                    if hosts[host]["files"][name]["raw"] != expected:
                        raise ValueError("Smoke-tested training source changed")
    for host in ("707", "715"):
        if set(hosts[host]["files"]) != set(hosts["local"]["files"]):
            raise ValueError("Host source/tool/test file set differs")
        for name, hashes in hosts["local"]["files"].items():
            if hosts[host]["files"][name]["lf"] != hashes["lf"]:
                raise ValueError(f"Host source differs beyond EOL: {host}/{name}")
    for plan in plans:
        directory = (Path(plan["root"]) / "results" / plan["token"]).as_posix()
        # Remote roots are POSIX paths even when this freeze runs on Windows.
        if plan["host"] != "local":
            directory = f"{plan['root']}/results/{plan['token']}"
        plan["entry_config"] = entry_config("gavdac_c2_h02", {
            "seed": plan["seed"], "unique_token": plan["token"],
            "local_results_path": directory, "env_args.result_dir": directory,
            "wandb_dir": directory + "/wandb", "use_cuda": False, "wandb_mode": "online",
        })
    return {
        "hypothesis": "H02", "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_before_freeze_sha256": digest(state_path),
        "algorithm_implementation_commit": state["implementation_commit"],
        "rollback_commits_chronological": state["rollback_commits"],
        "tooling_commit_not_an_algorithm_intervention": hosts["local"]["git_commit"],
        "engineering_acceptance_sha256": digest(engineering_path),
        "immutable_C3_original_freeze_sha256": digest(original_baseline),
        "exclusive_smokes_acceptance_sha256": digest(smokes_path),
        "hosts": hosts, "candidate_config": candidate, "baseline_C3": state["baseline_C3"],
        "protected_sha256": state["protected_sha256"], "run_plan": plans,
        "selection": {"rows": list(range(731, 739)), "interval_joint_steps": 10000, "actor": "rl"},
        "evaluation": state["evaluation"],
        "runtime_note": "Actual launch config also adds device=cpu and changes env_args.run_id to the token",
        "limitation": "Exploratory repeated retained-data development, not independent superiority evidence",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()) or output.exists():
        raise ValueError("Freeze output must be a new file below results/")
    payload = freeze(args.state.resolve())
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "tokens": [plan["token"] for plan in payload["run_plan"]]}))


if __name__ == "__main__":
    main()
