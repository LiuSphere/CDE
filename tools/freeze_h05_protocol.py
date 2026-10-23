"""Freeze H05 after accepted engineering and support guards; never launch jobs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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

IMPLEMENTATION = "28170d8aeb700203268ec3aa1e872e578ecdc7b5"
ORIGINAL_C3_FREEZE = "f402b9d41f5486e7fd4d676d9ad234c835e0ae094031b2534ed1cb6cdd666875"
ENGINEERING = "cea4b93c01b8435c6042ea5ab6eac82eb9df7928b7c500d9cbb97bd1a0cb1019"
MP_ADOPTION = "b19e211bd612fb4c5e604617ffc6b6f8e1dc453ba6c864c35561fdb31cba4edd"
ADDITIONS = {"tools/freeze_h05_protocol.py", "tests/test_h05_protocol.py"}
FOCUSED_TESTS = 10
HOSTS = {
    "local": (str(ROOT), 1),
    "707": ("/home/arch707/data/CDE", 2),
    "715": ("/home/bless/projects/CDE_c2opt21", 1),
}
C3_COSTS = (344913.0888879933, 344550.706793143, 354488.7203334839)
MANUSCRIPTS = {"paper/manuscript/sections/method.tex", "paper/manuscript/sections/modeling.tex"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entry_config(name, overrides=None):
    argv = ["direct_main.py", "--config", name, "--env-config", "microgrid"]
    for key, value in (overrides or {}).items():
        argv += ["--override", f"{key}={json.dumps(value)}"]
    values = []
    with patch.object(sys, "argv", argv), patch.object(
        direct_main, "run", lambda value, logger: values.append(value),
    ):
        direct_main.main()
    return values[0]


def validate_candidate(parent, candidate):
    if candidate != {**parent, "name": "gavdac_c2_h05", "five_step_critic": True}:
        raise ValueError("H05 differs beyond name and registered five-step flag")
    expected = {
        "t_max": 6000000, "EA": True, "crossover_prob": 0.0, "use_cuda": False,
        "checkpoint_path": "", "policy_checkpoint": "", "evaluate": False,
        "rollout_workers": 2, "learner_workers": 2, "test_interval": 10000,
        "fixed_eval_days": 8, "evaluation_initial_soc": .35, "five_step_critic": True,
        "save_model": True, "save_best_policy": True, "wandb_mode": "online",
    }
    if any(candidate.get(key) != value or type(candidate[key]) is not type(value)
           for key, value in expected.items()):
        raise ValueError("Fresh CPU weights, fixed budget/selection and explicit 2/2 required")
    if candidate.get("validation_rows") not in ([], list(range(731, 739))):
        raise ValueError("Validation rows changed")
    if any(key in candidate for key in (
        "actor_mixer_context_grad", "clipped_double_critic", "actor_update_interval", "scaled_huber_critic",
    )):
        raise ValueError("Retired intervention cannot be combined with H05")


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
    if len(tokens) != 3 or any(not isinstance(token, str) for token in tokens) or len(set(tokens)) != 3 or any(
        not re.fullmatch(r"h05s[a-z0-9_]{1,25}", token) for token in tokens
    ):
        raise ValueError("Three unique short H05 tokens required")
    return [{
        "seed": seed, "host": "local" if seed == 2 else "707",
        "root": HOSTS["local" if seed == 2 else "707"][0],
        "slot": 0 if seed == 2 else seed, "capacity": 1 if seed == 2 else 2,
        "token": tokens[seed], "config_name": "gavdac_c2_h05", "joint_budget": 6000000,
        "rollout_workers": 2, "learner_workers": 2, "threads": 1, "status": "frozen_not_launched",
    } for seed in (0, 1, 2)]


def validate_plans(plans):
    if len(plans) != 3 or any(type(plan[key]) is not int for plan in plans for key in (
        "seed", "slot", "capacity", "joint_budget", "rollout_workers", "learner_workers", "threads",
    )):
        raise ValueError("Exactly three integer seed/slot/budget/process assignments required")
    if plans != make_plans([plan["token"] for plan in plans]):
        raise ValueError("Seed/host/slot/budget/worker assignment changed")


def validate_rollback(implementation, commits):
    if implementation != IMPLEMENTATION or commits != [IMPLEMENTATION]:
        raise ValueError("Unreviewed implementation or economic rollback list")


def validate_baseline(baseline, original):
    if baseline != original or set(baseline) != {"0", "1", "2"}:
        raise ValueError("C3 identities differ from original freeze")
    for seed, cost in enumerate(C3_COSTS):
        item = baseline[str(seed)]
        if item["economic_cost"] != cost or item["candidate_maximum_cost"] != .97 * cost:
            raise ValueError("C3 costs or 3% gate changed")


def compare_host(previous, current, capacity, host):
    for name, expected in previous["files"].items():
        if current["files"].get(name) != expected:
            raise ValueError(f"Original-host qualified raw bytes changed: {host}/{name}")
    if set(current["files"]) - set(previous["files"]) != ADDITIONS:
        raise ValueError("Unexpected support tool/test additions")
    if current["slots"] != [{"slot": slot, "state": "free"} for slot in range(capacity)]:
        raise ValueError("Occupied slot or stale lock")
    allowed_dirty = MANUSCRIPTS if host == "local" else set()
    if any(row["status"] != " M" or row["path"] not in allowed_dirty for row in current["git_status"]):
        raise ValueError("Unknown dirty scope; only existing local manuscript edits allowed")
    if current["existing_tokens"]:
        raise ValueError("Proposed formal token already exists")
    for key in ("platform", "root", "logical_cpus", "python", "torch"):
        if current[key] != previous[key]:
            raise ValueError("Qualified allocation or runtime changed: " + key)
    if current["memory"]["available_bytes"] < (3 if host == "local" else 8) * 1024**3:
        raise ValueError("Insufficient currently available RAM for accepted process budget")


# Read-only original-host audit; neither legacy freeze nor training entry is called.
HOST_PROBE = r'''
import csv,ctypes,hashlib,io,json,os,platform,subprocess,sys
from pathlib import Path
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
root=Path.cwd()
sys.path[:0]=[str(root),str(root/"src")]
from tools.experiment_slot import slot_status
import torch
torch.set_num_threads(1)
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
    (root/"results/cmp20/supervisors").glob(token+"*"))]
actors=[]
for actor in payload["actors"]:
    checkpoint=Path(actor["checkpoint"])
    for name,key in (("metadata.json","metadata_sha256"),("agent.th","policy_sha256"),
                     ("config.yaml","config_sha256")):
        if sha(checkpoint/name)!=actor[key]:raise ValueError("Frozen C3 checkpoint changed: "+name)
    if sha(checkpoint.parent.parent/"best_rl.json")!=actor["selection_pointer_sha256"]:
        raise ValueError("Frozen C3 selection pointer changed")
    metadata=json.loads((checkpoint/"metadata.json").read_text())
    if (metadata["validation"]["rows"]!=list(range(731,739)) or metadata["policy_source"]!="rl"
        or metadata["checkpoint_step"]!=actor["checkpoint_step"]
        or metadata["policy_sha256"]!=actor["policy_sha256"]):
        raise ValueError("Frozen C3 Actor selection changed")
    source=Path(metadata["source_snapshot"]).resolve()
    for manifest in (metadata["source_sha256"],actor["evaluation_source_sha256"]):
        for name,expected in manifest.items():
            path=(source/name).resolve()
            if not path.is_relative_to(source) or sha(path)!=expected:
                raise ValueError("Frozen C3 saved source changed: "+name)
    if metadata["data_sha256"]!=actor["evaluation_data_sha256"]:
        raise ValueError("Frozen C3 evaluation and Actor data differ")
    for name,expected in metadata["data_sha256"].items():
        if sha(root/"src/envs/data"/name)!=expected:raise ValueError("Frozen C3 data changed: "+name)
    weights=torch.load(checkpoint/"agent.th",map_location="cpu",weights_only=True)
    if not weights or not all(value.device.type=="cpu" and (not value.is_floating_point() or
               (value.dtype==torch.float32 and torch.isfinite(value).all())) for value in weights.values()):
        raise ValueError("Frozen C3 Actor not finite CPU float32")
    actors.append({"seed":actor["seed"],"checkpoint":str(checkpoint),
                   "metadata_policy_config_pointer_source_data_rawhash_passed":True,
                   "metadata_and_evaluation_source_keys_checked_individually":True})
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
status=subprocess.check_output(["git","status","--porcelain=v1","-z","--untracked-files=all"])
git_status=[{"status":row[:2].decode(),"path":row[3:].decode("utf-8")}
            for row in status.split(b"\0") if row]
print(json.dumps({
    "root":str(root),"platform":platform.platform(),"logical_cpus":os.cpu_count(),
    "python":platform.python_version(),"torch":torch.__version__,
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "git_status":git_status,"slots":slot_status(payload["slots"]),"files":files,"existing_tokens":existing,
    "original_host_C3_actor_checks":actors,"memory":ram,"named_processes":processes,"load_average":load,
    "process_inventory_scope":"Python/uv/W&B names only, no credentials/arguments; not peak-memory proof",
}))
'''


def probe(host, root, tokens, actors, slots):
    child = ["uv", "run", "--frozen", "python", "-c", HOST_PROBE]
    command = child if host == "local" else [
        "ssh", "-o", "ConnectTimeout=15", host, shlex.join(["uv", "--directory", root, *child[1:]]),
    ]
    result = subprocess.run(
        command, cwd=ROOT, input=json.dumps({"tokens": tokens, "actors": actors, "slots": slots}),
        capture_output=True, text=True, encoding="utf-8", timeout=120, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{host} H05 read-only probe failed: {result.stderr}")
    return json.loads(result.stdout)


def revalidate_C3(state, base):
    original = base / "h01_protocol_b.json"
    if digest(original) != ORIGINAL_C3_FREEZE:
        raise ValueError("Immutable original C3 freeze bytes changed")
    original_state = json.loads(original.read_text())
    validate_baseline(state["baseline_C3"], original_state["baseline_C3"])
    if state["protected_sha256"] != original_state["protected_sha256"]:
        raise ValueError("Original protected-file manifest changed")
    costs = {}
    for seed, item in state["baseline_C3"].items():
        summary, records = load_evaluation(item)
        manifest = ROOT / f"results/cmp20/evaluation_seed{seed}_a.json"
        if digest(manifest) != item["original_manifest_sha256"]:
            raise ValueError("Original C3 manifest changed")
        selected = next(row for row in json.loads(manifest.read_text())["selected"] if row["group"] == "C3")
        if (selected != item["selected"] or summary["policy_sha256"] != selected["policy_sha256"]
                or summary["totals"]["economic_cost"] != item["economic_cost"]):
            raise ValueError("Original C3 Actor or independently summed trace cost changed")
        for record in records:
            components = sum(float(record[name]) for name in ("buy_cost", "gen_cost", "bess_cost", "exchange_cost"))
            if not math.isclose(components, float(record["economic_cost"]), rel_tol=0, abs_tol=1e-7):
                raise ValueError("Original C3 record economic cost formula differs")
        costs[seed] = {
            "economic_cost": item["economic_cost"], "agent_hours": len(records),
            "summary_sha256": item["summary_sha256"], "trajectory_sha256": item["trajectory_sha256"],
            "independent_record_formula_daily_and_period_totals_checked": True,
        }
    return costs


def actors_for_host(state, host):
    actors = []
    for seed, item in state["baseline_C3"].items():
        if ("local" if seed == "2" else "707") == host:
            summary = json.loads((Path(item["directory"]) / "summary.json").read_text())
            actors.append({**item["selected"], "evaluation_source_sha256": summary["source_sha256"],
                           "evaluation_data_sha256": summary["data_sha256"]})
    return actors


def validate_tooling(current, checked):
    if not checked["accepted"] or checked["focused_tests"] != FOCUSED_TESTS or any(
        current["files"][name]["raw"] != checked["raw_sha256"][name] for name in ADDITIONS
    ):
        raise ValueError("Support tool/test differs from accepted original-host focused checks")


def freeze(state_path, tokens):
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (state["active_hypothesis"] != "H05" or state["active_runs"] or state["run_plan"]
            or state["formal_protocol"] or state["status"] !=
            "h05_all_host_engineering_accepted_pending_formal_protocol_freeze"):
        raise ValueError("Freeze requires idle engineering-accepted, not yet frozen H05")
    validate_rollback(state["implementation_commit"], state["rollback_commits"])
    workers = {"rollout_workers": 2, "learner_workers": 2}
    parent, candidate = (entry_config(name, workers) for name in ("gavdac_ea_main", "gavdac_c2_h05"))
    validate_candidate(parent, candidate)
    if state["config"] != candidate:
        raise ValueError("State differs from actual H05 entry config")
    plans = make_plans(tokens)
    validate_plans(plans)
    evaluation = evaluation_protocol()
    validate_evaluation(evaluation)
    subprocess.run(["git", "diff", "--exit-code", IMPLEMENTATION, "--", "src"],
                   cwd=ROOT, capture_output=True, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip():
        raise ValueError("Push reviewed support tool before freezing")
    base = state_path.parent
    tool_path = base / "h05_protocol_tool_acceptance_a.json"
    tooling = json.loads(tool_path.read_text())
    if not tooling["accepted"] or tooling["support_commit"] != head:
        raise ValueError("Support-tool acceptance does not identify current pushed support commit")
    engineering_path = base / "h05_engineering_acceptance_a.json"
    if digest(engineering_path) != ENGINEERING or state["H05_engineering"]["sha256"] != ENGINEERING:
        raise ValueError("Accepted H05 engineering bytes changed")
    engineering = json.loads(engineering_path.read_text())
    if (not engineering["accepted"] or len(engineering["exclusive_smokes"]) != 4
            or engineering["implementation_commit"] != IMPLEMENTATION):
        raise ValueError("H05 engineering incomplete")
    for name, expected in engineering["evidence_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Engineering evidence changed: " + name)
    adoption_path = base / "mp_adoption_protocol_a.json"
    if digest(adoption_path) != MP_ADOPTION:
        raise ValueError("Accepted MP/default exception manifest changed")
    adoption = json.loads(adoption_path.read_text())
    protected = dict(state["protected_sha256"])
    protected["src/config/default.yaml"] = adoption["protected_hash_exception"]["new_raw"]
    if any(digest(ROOT / name) != expected for name, expected in protected.items()):
        raise ValueError("Protected comparison condition changed")
    costs = revalidate_C3(state, base)
    hosts = {}
    for host, (root, capacity) in HOSTS.items():
        current = hosts[host] = probe(host, root, tokens, actors_for_host(state, host), capacity)
        previous = json.loads((base / f"h05_host_{host}_a.json").read_text())
        compare_host(previous, current, capacity, host)
        validate_tooling(current, tooling["hosts"][host])
        if current["git_commit"] != head:
            raise ValueError("Synchronize idle hosts to reviewed support commit first")
        for smoke in engineering["exclusive_smokes"]:
            if smoke["host"] == host and smoke["launch"]["source_sha256"] != {
                name: value["raw"] for name, value in current["files"].items() if name.startswith("src/")
            }:
                raise ValueError("Smoke-tested original-host training source changed")
    for host in ("707", "715"):
        if hosts[host]["files"].keys() != hosts["local"]["files"].keys() or any(
            value["lf"] != hosts[host]["files"][name]["lf"]
            for name, value in hosts["local"]["files"].items()
        ):
            raise ValueError("Host bytes differ beyond accepted EOL; raw remains authoritative")
    for plan in plans:
        directory = (ROOT / "results" / plan["token"]).as_posix() if plan["host"] == "local" else (
            f"{plan['root']}/results/{plan['token']}"
        )
        plan["entry_config"] = entry_config("gavdac_c2_h05", {
            **workers, "seed": plan["seed"], "unique_token": plan["token"],
            "local_results_path": directory, "env_args.result_dir": directory,
            "wandb_dir": directory + "/wandb", "use_cuda": False, "wandb_mode": "online",
        })
    return {
        "hypothesis": "H05", "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_before_freeze_sha256": digest(state_path),
        "algorithm_implementation_commit": IMPLEMENTATION,
        "rollback_commits_reverse_order": state["rollback_commits"],
        "support_tooling_commit_not_an_algorithm_intervention": head,
        "engineering_acceptance_sha256": ENGINEERING, "support_tool_acceptance_sha256": digest(tool_path),
        "immutable_original_C3_freeze_sha256": ORIGINAL_C3_FREEZE,
        "original_C3_trace_revalidation": costs, "hosts": hosts, "candidate_config": candidate,
        "run_plan": plans, "baseline_C3": state["baseline_C3"],
        "original_protected_sha256": state["protected_sha256"],
        "future_execution_protected_default_exception": adoption["protected_hash_exception"],
        "selection": {"rows": list(range(731, 739)), "interval_joint_steps": 10000, "actor": "rl"},
        "evaluation": evaluation, "cpu_process_budget_per_slot": 5,
        "707_two_slot_torch_process_budget": 10, "compute_threads_per_process": 1,
        "resource_limitation": "Current RAM/process snapshots, not peak-memory or throughput guarantees",
        "runtime_note": "Training adds device=cpu and changes env_args.run_id to token",
        "limitation": "Exploratory retained reuse, not independent superiority/statistical significance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs=3, default=["h05s0a", "h05s1a", "h05s2a"])
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()) or output.exists():
        raise ValueError("Freeze requires NEW output below results/")
    payload = freeze(args.state.resolve(), args.tokens)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "tokens": [plan["token"] for plan in payload["run_plan"]]}))


if __name__ == "__main__":
    main()
