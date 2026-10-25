"""Independently guard/freeze H10; never launch, resume or alter an experiment."""

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import direct_main
from tools.report_comparison_results import load_evaluation

IMPLEMENTATION = "bc95989f26227cb70d086bbcd114a8bab95d5017"
ENGINEERING = "e8ce167114a187b508ca3324d639d80683f9af8f39d1cae0c49b0f95e7068c27"
ORIGINAL_C3_FREEZE = "f402b9d41f5486e7fd4d676d9ad234c835e0ae094031b2534ed1cb6cdd666875"
MP_ADOPTION = "b19e211bd612fb4c5e604617ffc6b6f8e1dc453ba6c864c35561fdb31cba4edd"
ADDITIONS = {"tools/freeze_h10_protocol.py", "tests/test_h10_protocol.py"}
FOCUSED_TESTS = 18
HOSTS = {"local": (str(ROOT), 1), "707": ("/home/arch707/data/CDE", 2),
         "715": ("/home/bless/projects/CDE_c2opt21", 1)}
C3_COSTS = (344913.0888879933, 344550.706793143, 354488.7203334839)
MANUSCRIPTS = {"paper/manuscript/sections/method.tex", "paper/manuscript/sections/modeling.tex"}
RETIRED = ("actor_mixer_context_grad", "clipped_double_critic", "actor_update_interval",
           "scaled_huber_critic", "five_step_critic", "critic_layer_norm", "target_policy_smoothing",
           "critic_small_output_init", "actor_no_action_l2")
EVIDENCE = {
    "h10_host_local_a.json": "da26de0decde856f7473f1c1802b7a8e35b52c6e77d03729e35575e37c00aa39",
    "h10_host_707_a.json": "2e62b6c3dc8607a4d833e3c981cd93a0eabba3835982e1cbe9f706e7f7287d38",
    "h10_host_715_a.json": "23bdaee3e19075d1d4fea50191daf5b76305468bed21f9396e9e4a58d30032d9",
    "h10_host_comparison_a.json": "fdf30008dc3b8897c9671475e5929954fb24f190ed46ba200b906a23438761a6",
    "h10_implementation_record_a.json": "7dd748ee21baba61492f2a5a1590f0b62a81c8fa158507b75c12c774d1beaa5e",
    "h10_local_smoke_preflight_a.json": "05193de13ad9514c1c21968b13f6515ea518662d10a79dfd0da5f611d10ce357",
    "h10_local_focused_b.json": "5f41682ec1b3316c1d6e1a955199b07c5b3f79af1aa237381262d4e2b4de3242",
    "h10_local_full_summary_b.json": "68587e7777fddffe8dece07199294322b0fe7c8d980d19c1fcbe3e4f16f0f756",
    "h10_707_qualification_a.json": "55e6cc64beaa2a258e8781513d75f83ea42b1ea55dce4c61a0773d1e2d01d260",
    "h10_707_qualification_a_transport.json": "a4a57caaaef989bfdeb9be1fee2b570664481954ee05e6b2ddedf2199a165e4f",
    "h10_715_qualification_a.json": "55e6cc64beaa2a258e8781513d75f83ea42b1ea55dce4c61a0773d1e2d01d260",
    "h10_715_qualification_a_transport.json": "322f481bd36df1fea3bb69f053c00f1850b5e0d44f7c8ab968bf642cef044a57",
    "h10_707_smoke_live_overlap_a.json": "0c601f97e0ec4a2b2dcba9661af99903b7f9f3c8463c10fa8e49d5b72f0d2caa",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def same_typed(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def entry_config(name, overrides=None):
    argv = ["direct_main.py", "--config", name, "--env-config", "microgrid"]
    for key, value in (overrides or {}).items():
        argv += ["--override", f"{key}={json.dumps(value)}"]
    values = []
    with patch.object(sys, "argv", argv), patch.object(
        direct_main, "run", lambda config, logger: values.append(config),
    ):
        direct_main.main()
    return values[0]


def validate_candidate(parent, candidate):
    if not same_typed(candidate, {**parent, "name": "gavdac_c2_h10", "critic_center_generator_action": True}):
        raise ValueError("H10 differs beyond name and registered Critic generator-fraction centering")
    expected = {
        "t_max": 6000000, "EA": True, "crossover_prob": 0.0, "use_cuda": False,
        "checkpoint_path": "", "policy_checkpoint": "", "evaluate": False,
        "rollout_workers": 2, "learner_workers": 2, "test_interval": 10000,
        "fixed_eval_days": 8, "evaluation_initial_soc": .35, "critic_center_generator_action": True,
        "save_model": True, "save_best_policy": True, "wandb_mode": "online", "rnn_hidden_dim": 48,
    }
    if any(candidate.get(key) != value or type(candidate.get(key)) is not type(value)
           for key, value in expected.items()):
        raise ValueError("Fresh CPU weights, fixed budget/selection and explicit2/2 required")
    rows = candidate.get("validation_rows")
    if rows not in ([], list(range(731, 739))) or any(type(row) is not int for row in rows):
        raise ValueError("Validation rows changed")
    if any(key in candidate for key in RETIRED):
        raise ValueError("Retired intervention cannot be combined with H10")


def intervention():
    return {
        "critic_center_generator_action": True, "critic_action_map": "[2*a0-1,a1]",
        "scope": "Only local Critic input generator-fraction coordinate; battery unchanged",
        "action_Box_low": [0.0, -1.0], "action_Box_high": [1.0, 1.0],
        "same_dimensionless_fraction_Box_all_three_agents": True,
        "physical_kW_normalization": False,
        "critic": "CenteredGeneratorGAVDACCritic", "learner": "CenteredGeneratorGAVDACLearner",
        "worker": "Explicit same classes, original shared transport",
        "replay_actor_history_environment_coordinates_unchanged": True,
        "actor_action_l2_coefficient": .001,
        "live_mixer_context_action_recurrent_gradients_preserved": True,
        "critic_before_actor_every_step_actor_target_tau_preserved": True,
        "original_parameter_draws_and_count_preserved": True, "additional_RNG_draws": 0,
        "new_parameters_or_buffers": False, "weight_compensation": False,
        "single_critic": True, "exact_resume_supported": False,
    }


def evaluation_protocol():
    return {
        "rows": list(range(913, 1095)), "mode": "continuous",
        "actor": "validation-selected rl", "initial_soc": .35,
        "terminal_soc": "free within device bounds", "hidden_and_previous_action_reset": "daily",
        "primary_metric": "economic_cost", "supplementary_metric": "full_cost",
        "minimum_saving_fraction": .03, "require_each_seed_and_ratio_of_means": True,
        "expected_agent_hours": 13104, "retrospective_retained_development": True,
    }


def validate_evaluation(value):
    if not same_typed(value, evaluation_protocol()):
        raise ValueError("Registered evaluation or economic gate changed")


def make_plans(tokens):
    if len(tokens) != 3 or any(not isinstance(token, str) for token in tokens) or len(set(tokens)) != 3 or any(
        not re.fullmatch(r"h10s[a-z0-9_]{1,25}", token) for token in tokens
    ):
        raise ValueError("Three unique short H10 tokens required")
    return [{
        "seed": seed, "host": "local" if seed == 2 else "707",
        "root": HOSTS["local" if seed == 2 else "707"][0],
        "slot": 0 if seed == 2 else seed, "capacity": 1 if seed == 2 else 2,
        "token": tokens[seed], "config_name": "gavdac_c2_h10", "joint_budget": 6000000,
        "rollout_workers": 2, "learner_workers": 2, "threads": 1, "status": "frozen_not_launched",
    } for seed in (0, 1, 2)]


def validate_plans(plans):
    if len(plans) != 3 or any(type(plan.get(key)) is not int for plan in plans for key in (
        "seed", "slot", "capacity", "joint_budget", "rollout_workers", "learner_workers", "threads",
    )):
        raise ValueError("Exactly three integer seed/slot/budget/process assignments required")
    if plans != make_plans([plan["token"] for plan in plans]):
        raise ValueError("Seed/host/slot/budget/worker assignment changed")


def validate_rollback(implementation, commits):
    if implementation != IMPLEMENTATION or commits != [IMPLEMENTATION]:
        raise ValueError("Unreviewed implementation or economic rollback list")


def validate_baseline(baseline, original):
    if not same_typed(baseline, original) or set(baseline) != {"0", "1", "2"}:
        raise ValueError("C3 identities differ from original freeze")
    for seed, cost in enumerate(C3_COSTS):
        if (baseline[str(seed)]["economic_cost"] != cost or
                baseline[str(seed)]["candidate_maximum_cost"] != .97 * cost):
            raise ValueError("C3 costs or3% gate changed")


def compare_host(previous, current, capacity, host):
    if host not in HOSTS or type(capacity) is not int or capacity != HOSTS[host][1]:
        raise ValueError("Unknown host or wrong capacity")
    for name, expected in previous["files"].items():
        if current["files"].get(name) != expected:
            raise ValueError(f"Original-host qualified raw bytes changed: {host}/{name}")
    if set(current["files"]) - set(previous["files"]) != ADDITIONS:
        raise ValueError("Unexpected support tool/test additions")
    if current["slots"] != [{"slot": slot, "state": "free"} for slot in range(capacity)]:
        raise ValueError("Occupied slot or stale lock")
    if any(row["status"] != " M" or row["path"] not in (MANUSCRIPTS if host == "local" else set())
           for row in current["git_status"]):
        raise ValueError("Unknown dirty scope; only existing local manuscript edits allowed")
    if current["existing_tokens"]:
        raise ValueError("Proposed formal token already exists")
    for key in ("platform", "root", "logical_cpus", "python", "torch", "host"):
        if current[key] != previous[key]:
            raise ValueError("Qualified allocation or runtime changed: " + key)
    if current["torch_threads"] != 1 or type(current["torch_threads"]) is not int:
        raise ValueError("One compute thread per process required")
    if current["memory"]["available_bytes"] <= (3 if host == "local" else 8) * 1024**3:
        raise ValueError("Insufficient currently available RAM for accepted process budget")


def check_c3_actor(actor, root):
    import torch
    checkpoint = Path(actor["checkpoint"])
    for name, key in (("metadata.json", "metadata_sha256"), ("agent.th", "policy_sha256"),
                      ("config.yaml", "config_sha256")):
        if digest(checkpoint / name) != actor[key]:
            raise ValueError("Frozen C3 checkpoint changed: " + name)
    if digest(checkpoint.parent.parent / "best_rl.json") != actor["selection_pointer_sha256"]:
        raise ValueError("Frozen C3 selection pointer changed")
    metadata = read(checkpoint / "metadata.json")
    if (metadata["validation"]["rows"] != list(range(731, 739)) or metadata["policy_source"] != "rl"
            or metadata["checkpoint_step"] != actor["checkpoint_step"]
            or metadata["policy_sha256"] != actor["policy_sha256"]):
        raise ValueError("Frozen C3 Actor selection changed")
    source = Path(metadata["source_snapshot"]).resolve()
    # Metadata and evaluation have independent source-key sets, both raw-authoritative.
    for manifest in (metadata["source_sha256"], actor["evaluation_source_sha256"]):
        for name, expected in manifest.items():
            path = (source / name).resolve()
            if not path.is_relative_to(source) or digest(path) != expected:
                raise ValueError("Frozen C3 saved source changed: " + name)
    if metadata["data_sha256"] != actor["evaluation_data_sha256"]:
        raise ValueError("Frozen C3 evaluation and Actor data differ")
    data_root = (root / "src/envs/data").resolve()
    for name, expected in metadata["data_sha256"].items():
        path = (data_root / name).resolve()
        if not path.is_relative_to(data_root) or digest(path) != expected:
            raise ValueError("Frozen C3 data changed: " + name)
    weights = torch.load(checkpoint / "agent.th", map_location="cpu", weights_only=True)
    if not weights or not all(
        value.device.type == "cpu" and (not value.is_floating_point() or
        (value.dtype == torch.float32 and torch.isfinite(value).all())) for value in weights.values()
    ):
        raise ValueError("Frozen C3 Actor not finite CPUfloat32")
    return {"seed": actor["seed"], "checkpoint": str(checkpoint),
            "metadata_policy_config_pointer_source_data_rawhash_passed": True,
            "metadata_and_evaluation_source_keys_checked_individually": True}


def inspect_host(payload):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    import torch
    from tools.experiment_slot import slot_status
    torch.set_num_threads(1)
    root = Path.cwd()
    paths = [path for area in ("src", "tools", "tests") for path in sorted((root / area).rglob("*"))
             if path.suffix in {".py", ".yaml", ".csv"}]
    paths += [root / name for name in ("pyproject.toml", "uv.lock", "src/config/algs/README.md")]
    files = {}
    for path in paths:
        raw = path.read_bytes()
        files[path.relative_to(root).as_posix()] = {
            "raw": hashlib.sha256(raw).hexdigest(),
            "lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}
    existing = [token for token in payload["tokens"] if (root / "results" / token).exists() or any(
        (root / "results/cmp20/supervisors").glob(token + "*"))]
    actors = [check_c3_actor(actor, root) for actor in payload["actors"]]
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available",
                                                      "virtual_total", "virtual_available", "extended")]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise ctypes.WinError()
        ram = {"total_bytes": memory.total, "available_bytes": memory.available}
        rows = csv.reader(io.StringIO(subprocess.check_output(["tasklist", "/FO", "CSV", "/NH"]).decode("oem")))
        processes = [{"pid": int(row[1]), "name": row[0]} for row in rows
                     if row and any(word in row[0].lower() for word in ("python", "uv.exe", "wandb"))]
        cpu = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | "
            "Where-Object Name -eq '_Total' | Select-Object PercentProcessorTime | ConvertTo-Json"],
            text=True))["PercentProcessorTime"]
        load = None
    else:
        info = {row.split()[0].rstrip(":"): int(row.split()[1]) * 1024
                for row in Path("/proc/meminfo").read_text().splitlines()
                if row.startswith(("MemTotal:", "MemAvailable:"))}
        ram = {"total_bytes": info["MemTotal"], "available_bytes": info["MemAvailable"]}
        rows = [line.split(maxsplit=1) for line in subprocess.check_output(
            ["ps", "-eo", "pid=,comm="], text=True).splitlines()]
        processes = [{"pid": int(row[0]), "name": row[1]} for row in rows
                     if len(row) == 2 and any(word in row[1].lower() for word in ("python", "uv", "wandb"))]
        load = list(os.getloadavg())
        def cpu_ticks():
            return [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        before = cpu_ticks()
        time.sleep(.25)
        delta = [end - start for start, end in zip(before, cpu_ticks())]
        cpu = 100 * (1 - (delta[3] + delta[4]) / sum(delta))
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    return {
        "root": str(root), "host": platform.node(), "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "python": platform.python_version(), "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": [{"status": row[:2].decode(), "path": row[3:].decode("utf-8")}
                       for row in status.split(b"\0") if row],
        "slots": slot_status(payload["slots"]), "files": files, "existing_tokens": existing,
        "original_host_C3_actor_checks": actors, "memory": ram, "named_processes": processes,
        "load_average": load, "cpu_busy_percent_snapshot": cpu,
        "process_inventory_scope": "Python/uv/W&B names only, no credentials/arguments; not peak-memory proof",
    }


HOST_PROBE = (
    "import json,sys;from tools.freeze_h10_protocol import inspect_host;"
    "print(json.dumps(inspect_host(json.load(sys.stdin))))"
)


def probe(host, root, tokens, actors, slots):
    child = ["uv", "run", "--frozen", "python", "-c", HOST_PROBE]
    command = child if host == "local" else [
        "ssh", "-o", "ConnectTimeout=15", host, shlex.join(["uv", "--directory", root, *child[1:]])]
    result = subprocess.run(command, cwd=ROOT, input=json.dumps({"tokens": tokens, "actors": actors, "slots": slots}),
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    if result.returncode:
        raise RuntimeError(f"{host} H10 read-only probe failed: {result.stderr}")
    return json.loads(result.stdout)


def revalidate_C3(state, base):
    original = base / "h01_protocol_b.json"
    if digest(original) != ORIGINAL_C3_FREEZE:
        raise ValueError("Immutable original C3 freeze bytes changed")
    original_state = read(original)
    validate_baseline(state["baseline_C3"], original_state["baseline_C3"])
    if state["protected_sha256"] != original_state["protected_sha256"]:
        raise ValueError("Original protected-file manifest changed")
    costs = {}
    for seed, item in state["baseline_C3"].items():
        summary, records = load_evaluation(item)
        manifest = ROOT / f"results/cmp20/evaluation_seed{seed}_a.json"
        if digest(manifest) != item["original_manifest_sha256"]:
            raise ValueError("Original C3 manifest changed")
        selected = next(row for row in read(manifest)["selected"] if row["group"] == "C3")
        if (selected != item["selected"] or summary["policy_sha256"] != selected["policy_sha256"]
                or summary["totals"]["economic_cost"] != item["economic_cost"]):
            raise ValueError("Original C3 Actor or independently summed trace cost changed")
        for record in records:
            cost = sum(float(record[name]) for name in ("buy_cost", "gen_cost", "bess_cost", "exchange_cost"))
            if not math.isclose(cost, float(record["economic_cost"]), rel_tol=0, abs_tol=1e-7):
                raise ValueError("Original C3 record economic cost formula differs")
        costs[seed] = {
            "economic_cost": item["economic_cost"], "agent_hours": len(records),
            "summary_sha256": item["summary_sha256"], "trajectory_sha256": item["trajectory_sha256"],
            "independent_record_formula_daily_and_period_totals_checked": True}
    return costs


def actors_for_host(state, host):
    actors = []
    for seed, item in state["baseline_C3"].items():
        if ("local" if seed == "2" else "707") == host:
            summary = read(Path(item["directory"]) / "summary.json")
            actors.append({**item["selected"], "evaluation_source_sha256": summary["source_sha256"],
                           "evaluation_data_sha256": summary["data_sha256"]})
    return actors


def validate_tooling(current, checked):
    if (checked["accepted"] is not True or type(checked["focused_tests"]) is not int
            or checked["focused_tests"] != FOCUSED_TESTS or any(
        current["files"][name]["raw"] != checked["raw_sha256"][name] for name in ADDITIONS
    )):
        raise ValueError("Support tool/test differs from accepted original-host focused checks")


def validate_phase(state):
    if (state["active_hypothesis"] != "H10" or state["active_runs"] or state["run_plan"]
            or state["formal_protocol"] is not None or state["status"] !=
            "h10_all_host_engineering_accepted_pending_independent_protocol_freeze"
            or state["H10_development"]["all_host_engineering_accepted"] is not True):
        raise ValueError("Freeze requires idle engineering-accepted, not yet frozen H10")


def validate_engineering(base):
    path = base / "h10_engineering_acceptance_a.json"
    if digest(path) != ENGINEERING:
        raise ValueError("Accepted H10 engineering bytes changed")
    engineering = read(path)
    if engineering["accepted"] is not True or set(engineering["exclusive_smokes"]) != {
        "h10ls0a", "h10rs0a", "h10rs1a", "h10715a",
    }:
        raise ValueError("H10 engineering incomplete")
    for name, expected in EVIDENCE.items():
        if digest(base / name) != expected:
            raise ValueError("Engineering evidence changed: " + name)
    if engineering["sole_algorithm_rollback_commit"] != IMPLEMENTATION:
        raise ValueError("Engineering implementation changed")
    for token, item in engineering["exclusive_smokes"].items():
        path = ROOT / item["report"]
        if digest(path) != item["sha256"]:
            raise ValueError("Smoke acceptance changed: " + token)
        smoke = read(path)
        expected_runtime = IMPLEMENTATION if token == "h10ls0a" else "6b060c6350811677edc707fdeadd8e95748ac43e"
        if (smoke["accepted"] is not True or smoke["launch"]["git_commit"] != expected_runtime
                or smoke["completion"]["critic_center_generator_action"] is not True
                or smoke["completion"]["critic_action_map"] != "[2*a0-1,a1]"
                or smoke["completion"]["exact_resume_supported"] is not False
                or any(type(smoke["completion"][key]) is not int or smoke["completion"][key] != 8
                       for key in ("critic_steps", "actor_steps", "target_steps", "finite_updates", "train_updates"))):
            raise ValueError("H10 smoke identity/counters changed")
    return engineering


def freeze(state_path, tokens):
    state = read(state_path)
    validate_phase(state)
    validate_rollback(state["implementation_commit"], state["rollback_commits"])
    workers = {"rollout_workers": 2, "learner_workers": 2}
    parent, candidate = (entry_config(name, workers) for name in ("gavdac_ea_main", "gavdac_c2_h10"))
    validate_candidate(parent, candidate)
    if not same_typed(state["config"], candidate):
        raise ValueError("State differs from actual H10 entry config")
    plans = make_plans(tokens)
    validate_plans(plans)
    evaluation = evaluation_protocol()
    validate_evaluation(evaluation)
    subprocess.run(["git", "diff", "--exit-code", IMPLEMENTATION, "--", "src", "tools/experiment_slot.py"],
                   cwd=ROOT, capture_output=True, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote_head = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"],
                                          cwd=ROOT, text=True).split()[0]
    if head != remote_head:
        raise ValueError("Push reviewed support tool before freezing")
    base = state_path.parent
    tool_path = base / "h10_protocol_tool_acceptance_a.json"
    tooling = read(tool_path)
    if tooling["accepted"] is not True or tooling["support_commit"] != head or set(tooling["hosts"]) != set(HOSTS):
        raise ValueError("Support acceptance must identify current pushed support commit and all hosts")
    engineering = validate_engineering(base)
    if digest(base / "mp_adoption_protocol_a.json") != MP_ADOPTION:
        raise ValueError("Accepted MP/default exception manifest changed")
    adoption = read(base / "mp_adoption_protocol_a.json")
    protected = dict(state["protected_sha256"])
    protected["src/config/default.yaml"] = adoption["protected_hash_exception"]["new_raw"]
    if any(digest(ROOT / name) != expected for name, expected in protected.items()):
        raise ValueError("Protected comparison condition changed")
    costs = revalidate_C3(state, base)
    hosts = {}
    for host, (root, capacity) in HOSTS.items():
        current = hosts[host] = probe(host, root, tokens, actors_for_host(state, host), capacity)
        compare_host(read(base / f"h10_host_{host}_a.json"), current, capacity, host)
        validate_tooling(current, tooling["hosts"][host])
        if current["git_commit"] != head:
            raise ValueError("Synchronize idle hosts to reviewed support commit first")
        for item in engineering["exclusive_smokes"].values():
            smoke = read(ROOT / item["report"])
            if smoke["host"] == host and smoke["launch"]["source_sha256"] != {
                name: value["raw"] for name, value in current["files"].items()
                if name.startswith("src/") and Path(name).suffix in {".py", ".yaml", ".csv"}
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
            f"{plan['root']}/results/{plan['token']}")
        plan["entry_config"] = entry_config("gavdac_c2_h10", {
            **workers, "seed": plan["seed"], "unique_token": plan["token"],
            "local_results_path": directory, "env_args.result_dir": directory,
            "wandb_dir": directory + "/wandb", "use_cuda": False, "wandb_mode": "online"})
    return {
        "hypothesis": "H10", "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_before_freeze_sha256": digest(state_path), "algorithm_implementation_commit": IMPLEMENTATION,
        "rollback_commits_reverse_order": state["rollback_commits"],
        "support_tooling_commit_not_an_algorithm_intervention": head,
        "engineering_acceptance_sha256": ENGINEERING, "engineering_evidence_sha256": EVIDENCE,
        "support_tool_acceptance_sha256": digest(tool_path), "intervention": intervention(),
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
    parser.add_argument("--tokens", nargs=3, default=["h10s0a", "h10s1a", "h10s2a"])
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()) or output.exists():
        raise ValueError("Freeze requires NEW output below results/")
    payload = freeze(args.state.resolve(), args.tokens)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "tokens": [plan["token"] for plan in payload["run_plan"]]}))


if __name__ == "__main__":
    main()
