"""Freeze the accepted H03 round, without starting training or editing state."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import direct_main
from tools.report_comparison_results import load_evaluation

IMPLEMENTATION = "e9975bf8c7af85a30b20d41f95c6436698463fb0"
ADDITIONS = {"tools/freeze_h03_protocol.py", "tests/test_h03_protocol.py"}
HOSTS = {
    "local": (str(ROOT), 1),
    "707": ("/home/arch707/data/CDE", 2),
    "715": ("/home/bless/projects/CDE_c2opt21", 1),
}
C3_COSTS = (344913.0888879933, 344550.706793143, 354488.7203334839)


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
    if candidate != {**parent, "name": "gavdac_c2_h03", "actor_update_interval": 2}:
        raise ValueError("H03 differs beyond its registered name and Actor cadence")
    expected = {
        "t_max": 6000000, "EA": True, "crossover_prob": 0.0, "use_cuda": False,
        "checkpoint_path": "", "policy_checkpoint": "", "evaluate": False,
        "rollout_workers": 2, "learner_workers": 2, "test_interval": 10000,
        "fixed_eval_days": 8, "evaluation_initial_soc": .35,
    }
    if any(candidate.get(key) != value for key, value in expected.items()):
        raise ValueError("Fresh weights, frozen budget/selection and explicit 2/2 CPU execution required")
    if any(type(candidate[key]) is not int for key in ("t_max", "rollout_workers", "learner_workers")):
        raise ValueError("Budget and worker counts must be integers, not booleans/floats")
    if candidate.get("validation_rows") not in ([], list(range(731, 739))):
        raise ValueError("Validation rows changed")


def evaluation_protocol():
    return {
        "rows": list(range(913, 1095)), "mode": "continuous",
        "actor": "validation-selected rl", "initial_soc": .35,
        "terminal_soc": "free within device bounds",
        "hidden_and_previous_action_reset": "daily", "primary_metric": "economic_cost",
        "supplementary_metric": "full_cost", "minimum_saving_fraction": .03,
        "require_each_seed_and_ratio_of_means": True, "expected_agent_hours": 13104,
        "retrospective_retained_development": True,
    }


def validate_evaluation(value):
    if value != evaluation_protocol():
        raise ValueError("Registered evaluation or economic gate changed")


def make_plans(tokens):
    if len(tokens) != 3 or len(set(tokens)) != 3 or any(
        not re.fullmatch(r"h03s[a-z0-9_]{1,25}", token) for token in tokens
    ):
        raise ValueError("Three unique new H03 tokens required")
    return [
        {
            "seed": seed, "host": "local" if seed == 2 else "707",
            "root": HOSTS["local" if seed == 2 else "707"][0],
            "slot": 0 if seed == 2 else seed, "capacity": 1 if seed == 2 else 2,
            "token": tokens[seed], "config_name": "gavdac_c2_h03", "joint_budget": 6000000,
            "rollout_workers": 2, "learner_workers": 2, "threads": 1,
            "status": "frozen_not_launched",
        }
        for seed in (0, 1, 2)
    ]


def validate_plans(plans):
    if any(type(plan[key]) is not int for plan in plans
           for key in ("seed", "slot", "capacity", "joint_budget", "rollout_workers", "learner_workers", "threads")):
        raise ValueError("Seed, slot, budget and process counts must be integers")
    if plans != make_plans([plan["token"] for plan in plans]):
        raise ValueError("Seed/host/slot/budget/worker assignment changed")


def validate_rollback(implementation, commits):
    if implementation != IMPLEMENTATION or commits != [IMPLEMENTATION]:
        raise ValueError("Unreviewed algorithm or rollback identity")


def compare_host(previous, current, capacity, host):
    for name, expected in previous["files"].items():
        if current["files"].get(name) != expected:
            raise ValueError(f"Original-host qualified bytes changed: {host}/{name}")
    if set(current["files"]) - set(previous["files"]) != ADDITIONS:
        raise ValueError("Unexpected support-tool/test file additions")
    if current["slots"] != [{"slot": slot, "state": "free"} for slot in range(capacity)]:
        raise ValueError("Host is occupied or has a stale lock")
    if current["dirty_scope"] or current["existing_tokens"]:
        raise ValueError("Dirty execution scope or an existing proposed token")
    if current["platform"] != previous["platform"] or current["root"] != previous["root"]:
        raise ValueError("Qualified host/root changed")
    if current["logical_cpus"] != previous["logical_cpus"]:
        raise ValueError("Qualified CPU allocation changed")
    minimum_gib = 3 if host == "local" else 8
    if current["memory"]["available_bytes"] < minimum_gib * 1024**3:
        raise ValueError("Insufficient currently available RAM for accepted worker budget")


HOST_PROBE = r'''
import csv,ctypes,hashlib,io,json,os,platform,subprocess,sys
from pathlib import Path
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
root=Path.cwd()
sys.path[:0]=[str(root),str(root/"src")]
from tools.experiment_slot import slot_status
payload=json.load(sys.stdin)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
files={}
for base in ("src","tools","tests"):
    for path in sorted((root/base).rglob("*")):
        if path.suffix in {".py",".yaml",".csv"}:
            raw=path.read_bytes()
            files[path.relative_to(root).as_posix()]={
                "raw":hashlib.sha256(raw).hexdigest(),
                "lf":hashlib.sha256(raw.replace(b"\r\n",b"\n")).hexdigest()}
for name in ("pyproject.toml","uv.lock"):
    raw=(root/name).read_bytes()
    files[name]={"raw":sha(root/name),"lf":hashlib.sha256(raw.replace(b"\r\n",b"\n")).hexdigest()}
existing=[token for token in payload["tokens"] if (root/"results"/token).exists() or any(
    (root/"results/cmp20/supervisors"/(token+suffix)).exists()
    for suffix in (".stdout.log",".stderr.log"))]
actors=[]
if payload["actors"]:
    import torch
    torch.set_num_threads(1)
    for actor in payload["actors"]:
        checkpoint=Path(actor["checkpoint"])
        for name,key in (("metadata.json","metadata_sha256"),("agent.th","policy_sha256"),
                         ("config.yaml","config_sha256")):
            if sha(checkpoint/name)!=actor[key]:raise ValueError("Frozen C3checkpoint changed: "+name)
        if sha(checkpoint.parent.parent/"best_rl.json")!=actor["selection_pointer_sha256"]:
            raise ValueError("Frozen C3selection pointer changed")
        metadata=json.loads((checkpoint/"metadata.json").read_text())
        if (metadata["validation"]["rows"]!=list(range(731,739)) or metadata["policy_source"]!="rl"
            or metadata["checkpoint_step"]!=actor["checkpoint_step"]):
            raise ValueError("Frozen C3Actor selection changed")
        source=Path(metadata["source_snapshot"]).resolve()
        for name,expected in metadata["source_sha256"].items():
            path=(source/name).resolve()
            if not path.is_relative_to(source) or sha(path)!=expected:
                raise ValueError("Frozen C3saved source changed: "+name)
        for name,expected in metadata["data_sha256"].items():
            if sha(root/"src/envs/data"/name)!=expected:raise ValueError("Frozen C3data changed: "+name)
        weights=torch.load(checkpoint/"agent.th",map_location="cpu",weights_only=True)
        if not all(not value.is_floating_point() or
                   (value.dtype==torch.float32 and torch.isfinite(value).all()) for value in weights.values()):
            raise ValueError("Frozen C3Actor nonfinite or not CPUfloat32")
        actors.append({"seed":actor["seed"],"checkpoint":str(checkpoint),
                       "metadata_policy_config_pointer_source_data_rawhash_passed":True})
if os.name=="nt":
    class Memory(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[
            (n,ctypes.c_ulonglong) for n in ("total","available","total_page","available_page",
                                          "total_virtual","available_virtual","extended")]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):raise ctypes.WinError()
    ram={"total_bytes":memory.total,"available_bytes":memory.available}
    rows=csv.reader(io.StringIO(subprocess.check_output(["tasklist","/FO","CSV","/NH"]).decode("oem")))
    processes=[{"pid":int(row[1]),"name":row[0]} for row in rows
               if row and any(word in row[0].lower() for word in ("python","uv.exe","wandb"))]
    load=None
else:
    info={r.split()[0].rstrip(":"):int(r.split()[1])*1024
          for r in Path("/proc/meminfo").read_text().splitlines()
          if r.startswith(("MemTotal:","MemAvailable:"))}
    ram={"total_bytes":info["MemTotal"],"available_bytes":info["MemAvailable"]}
    rows=[line.split(maxsplit=1) for line in subprocess.check_output(["ps","-eo","pid=,comm="],text=True).splitlines()]
    processes=[{"pid":int(row[0]),"name":row[1]} for row in rows
               if len(row)==2 and any(word in row[1].lower() for word in ("python","uv","wandb"))]
    load=list(os.getloadavg())
print(json.dumps({
    "root":str(root),"platform":platform.platform(),"logical_cpus":os.cpu_count(),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "dirty_scope":subprocess.check_output(["git","status","--porcelain","--","src","tools","tests",
                                         "pyproject.toml","uv.lock"],text=True),
    "slots":slot_status(payload["slots"]),"files":files,"existing_tokens":existing,
    "original_host_C3_actor_checks":actors,"memory":ram,"named_processes":processes,"load_average":load,
    "process_inventory_scope":"Python/uv/W&B names only; no credentials or process argument values",
}))
'''


def probe(host, root, tokens, actors, slots):
    child = ["uv", "run", "--frozen", "python", "-c", HOST_PROBE]
    command = child if host == "local" else ["ssh", "-o", "ConnectTimeout=15", host, shlex.join(
        ["uv", "--directory", root, *child[1:]],
    )]
    result = subprocess.run(
        command, cwd=ROOT, input=json.dumps({"tokens": tokens, "actors": actors, "slots": slots}),
        capture_output=True, text=True, encoding="utf-8", timeout=120, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{host} freeze probe failed: {result.stderr}")
    return json.loads(result.stdout)


def validate_baseline(baseline, original):
    if baseline != original or set(baseline) != {"0", "1", "2"}:
        raise ValueError("C3 identities differ from the immutable original freeze")
    for seed, cost in enumerate(C3_COSTS):
        item = baseline[str(seed)]
        if item["economic_cost"] != cost or item["candidate_maximum_cost"] != .97 * cost:
            raise ValueError("C3 cost or the fixed 3% threshold changed")


def freeze(state_path, tokens):
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (state["active_hypothesis"] != "H03" or state["active_runs"] or state["formal_protocol"]
            or state["status"] != "h03_all_host_engineering_accepted_pending_formal_protocol_freeze"):
        raise ValueError("Freeze requires idle engineering-accepted, not yet frozen H03")
    validate_rollback(state["implementation_commit"], state["rollback_commits"])
    workers = {"rollout_workers": 2, "learner_workers": 2}
    parent = entry_config("gavdac_ea_main", workers)
    candidate = entry_config("gavdac_c2_h03", workers)
    validate_candidate(parent, candidate)
    evaluation = evaluation_protocol()
    validate_evaluation(evaluation)
    plans = make_plans(tokens)
    validate_plans(plans)
    subprocess.run(["git", "diff", "--exit-code", IMPLEMENTATION, "--", "src"],
                   cwd=ROOT, capture_output=True, check=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT) != subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=ROOT,
    ):
        raise ValueError("Push the reviewed support tool before freezing")
    base = state_path.parent
    tool_acceptance_path = base / "h03_protocol_tool_acceptance_a.json"
    tool_acceptance = json.loads(tool_acceptance_path.read_text())
    if not tool_acceptance["accepted"]:
        raise ValueError("Support-tool validation is incomplete")
    adoption = json.loads((base / "mp_adoption_protocol_a.json").read_text())
    protected = dict(state["protected_sha256"])
    protected["src/config/default.yaml"] = adoption["protected_hash_exception"]["new_raw"]
    if any(digest(ROOT / name) != value for name, value in protected.items()):
        raise ValueError("Protected shared comparison condition changed")
    original = base / "h01_protocol_b.json"
    validate_baseline(state["baseline_C3"], json.loads(original.read_text())["baseline_C3"])
    engineering_path = base / "h03_engineering_acceptance_a.json"
    if digest(engineering_path) != state["H03_engineering"]["acceptance_sha256"]:
        raise ValueError("Engineering acceptance identity changed")
    engineering = json.loads(engineering_path.read_text())
    if not engineering["accepted"] or len(engineering["exclusive_smokes"]) != 4:
        raise ValueError("Engineering acceptance is incomplete")
    for name, value in engineering["evidence_sha256"].items():
        if digest(base / name) != value:
            raise ValueError("Engineering input changed: " + name)
    for seed, item in state["baseline_C3"].items():
        summary, _ = load_evaluation(item)
        manifest = ROOT / f"results/cmp20/evaluation_seed{seed}_a.json"
        if digest(manifest) != item["original_manifest_sha256"]:
            raise ValueError("Original C3manifest changed")
        selected = next(value for value in json.loads(manifest.read_text())["selected"]
                        if value["group"] == "C3")
        if (selected != item["selected"] or summary["policy_sha256"] != selected["policy_sha256"]
                or summary["totals"]["economic_cost"] != item["economic_cost"]):
            raise ValueError("Frozen C3selection or independently checked cost changed")
    hosts = {}
    for host, (root, capacity) in HOSTS.items():
        actors = [item["selected"] for seed, item in state["baseline_C3"].items()
                  if ("local" if seed == "2" else "707") == host]
        hosts[host] = probe(host, root, tokens, actors, capacity)
        previous = json.loads((base / f"h03_host_{host}_a.json").read_text())
        compare_host(previous, hosts[host], capacity, host)
        checked = tool_acceptance["hosts"][host]
        if checked["focused_tests"] != 5 or any(
            hosts[host]["files"][name]["raw"] != checked["raw_sha256"][name] for name in ADDITIONS
        ):
            raise ValueError("Support tool/test bytes differ from their validated host versions")
        if hosts[host]["git_commit"] != hosts["local"]["git_commit"]:
            raise ValueError("Synchronize idle support-tool commits first")
        for smoke in engineering["exclusive_smokes"]:
            if smoke["host"] == host and smoke["launch"]["source_sha256"] != {
                name: value["raw"] for name, value in hosts[host]["files"].items()
                if name.startswith("src/")
            }:
                raise ValueError("Smoke-tested original-host runtime bytes changed")
    for host in ("707", "715"):
        if hosts[host]["files"].keys() != hosts["local"]["files"].keys() or any(
            value["lf"] != hosts[host]["files"][name]["lf"]
            for name, value in hosts["local"]["files"].items()
        ):
            raise ValueError("Host source differs beyond EOL; raw hashes are never substituted")
    for plan in plans:
        directory = (ROOT / "results" / plan["token"]).as_posix() if plan["host"] == "local" else (
            f"{plan['root']}/results/{plan['token']}"
        )
        plan["entry_config"] = entry_config("gavdac_c2_h03", {
            **workers, "seed": plan["seed"], "unique_token": plan["token"],
            "local_results_path": directory, "env_args.result_dir": directory,
            "wandb_dir": directory + "/wandb", "use_cuda": False, "wandb_mode": "online",
        })
    return {
        "hypothesis": "H03", "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_before_freeze_sha256": digest(state_path),
        "algorithm_implementation_commit": IMPLEMENTATION,
        "rollback_commits_reverse_order": state["rollback_commits"],
        "support_tooling_commit_not_an_algorithm_intervention": hosts["local"]["git_commit"],
        "engineering_acceptance_sha256": digest(engineering_path),
        "support_tool_acceptance_sha256": digest(tool_acceptance_path),
        "immutable_original_C3_freeze_sha256": digest(original),
        "hosts": hosts, "candidate_config": candidate, "run_plan": plans,
        "baseline_C3": state["baseline_C3"], "original_protected_sha256": state["protected_sha256"],
        "future_execution_protected_default_exception": adoption["protected_hash_exception"],
        "selection": {"rows": list(range(731, 739)), "interval_joint_steps": 10000, "actor": "rl"},
        "evaluation": evaluation, "cpu_process_budget_per_slot": 5,
        "707_two_slot_torch_process_budget": 10, "compute_threads_per_process": 1,
        "resource_limitation": "Available memory/process snapshots are not peak-memory or throughput guarantees",
        "runtime_note": "Training adds device=cpu and changes env_args.run_id to the token",
        "limitation": "Exploratory retained-data reuse; not independent superiority or statistical significance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs=3, default=["h03s0a", "h03s1a", "h03s2a"])
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()) or output.exists():
        raise ValueError("Freeze requires a NEW output below results/")
    payload = freeze(args.state.resolve(), args.tokens)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "tokens": [plan["token"] for plan in payload["run_plan"]]}))


if __name__ == "__main__":
    main()
