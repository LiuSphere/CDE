"""Independently freeze the registered H11 lineage; never launch an experiment."""

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

CHILD = "7ca9284d6a0c2076c22541c088f2d952fae6b230"
PARENT = "4e3aa00eacf814e3d328be781e4af8632ec0488f"
PREREGISTRATION = "5fdf17a4886b493ac338d434f0bfc4b04ed2449c"
ENGINEERING = "d7a6c533c2cd0d1f18e04e3fa85644468d9768ad9d14b384b915f7e04a2d8a1a"
C3_AUDIT = "eaf8d94c7d6b986a2533b66c73e6345767af1603437dd702b31fafb683e05546"
ROLLBACK_SCOPE = "de110706f3f157c2580d5a059474d19832a41ab0d841d24704eac4d9f603c261"
LINEAGE = "aee2fd5763066f770ba9e85386e7f2cccd181144bd81ab6bde70497dad7afb0c"
ORIGINAL_C3_FREEZE = "f402b9d41f5486e7fd4d676d9ad234c835e0ae094031b2534ed1cb6cdd666875"
MP_ADOPTION = "b19e211bd612fb4c5e604617ffc6b6f8e1dc453ba6c864c35561fdb31cba4edd"
ADDITIONS = {"tools/freeze_h11_protocol.py", "tests/test_h11_protocol.py"}
FOCUSED_TESTS = 24
HOSTS = {"local": (str(ROOT), 1), "707": ("/home/arch707/data/CDE", 2)}
MANUSCRIPTS = {"paper/manuscript/sections/method.tex", "paper/manuscript/sections/modeling.tex"}
C3_COSTS = (344913.0888879933, 344550.706793143, 354488.7203334839)
PARENT_MEAN = 343419.27471629175
RETIRED = (
    "actor_mixer_context_grad", "clipped_double_critic", "scaled_huber_critic",
    "five_step_critic", "critic_layer_norm", "target_policy_smoothing",
    "critic_small_output_init", "actor_no_action_l2", "critic_center_generator_action",
)
PINS = {
    "h11_implementation_record_a.json": "707bdb67ab712dbc2e7b0aa3fdc9621b10d7b840351c24008b0a3c3159fc7c8f",
    "h11_parent_record_a.json": "8f57916fb33a4e010da297dfcc58ed5d090a294f5985f4794effb97b7cfff23a",
    "h11_preregistration_a.json": "0390fba9db423b58bc89554a473b409257522b59609fa5daf9df04326364c8c8",
    "h11_host_local_a.json": "034736870f528bdc885baf7da5e413e3ee1bdd996fcc4fde54fc1c19544d4361",
    "h11_host_707_a.json": "2323ddf3d18dc880e34f1c7b0aac3716b2dc8ed9e789b662937036927ca0b152",
    "h11_child_rollback_scope_a.json": ROLLBACK_SCOPE,
    "lineage_registry_d.json": LINEAGE,
    "h11_original_C3_revalidation_a.json": C3_AUDIT,
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_typed(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def entry_config(name, overrides=None):
    argv = ["direct_main.py", "--config", name, "--env-config", "microgrid"]
    for name, value in (overrides or {}).items():
        argv += ["--override", f"{name}={json.dumps(value)}"]
    configs = []
    with patch.object(sys, "argv", argv), patch.object(
        direct_main, "run", lambda config, logger: configs.append(config)
    ):
        direct_main.main()
    return configs[0]


def validate_candidate(root, parent, candidate):
    require(same_typed(parent, {**root, "name": "gavdac_c2_h11_parent",
                               "actor_update_interval": 2, "target_update_interval": 1}),
            "H11 parent must inherit only the registered H03 Actor cadence")
    require(same_typed(candidate, {**parent, "name": "gavdac_c2_h11", "target_update_interval": 2}),
            "H11 differs beyond its name and all-target synchronization cadence")
    required = {
        "t_max": 6000000, "EA": True, "crossover_prob": 0.0, "use_cuda": False,
        "checkpoint_path": "", "policy_checkpoint": "", "evaluate": False,
        "rollout_workers": 2, "learner_workers": 2, "test_interval": 10000,
        "fixed_eval_days": 8, "evaluation_initial_soc": .35,
        "save_model": True, "save_best_policy": True, "wandb_mode": "online",
        "rnn_hidden_dim": 48, "target_update_tau": .002,
        "actor_update_interval": 2, "target_update_interval": 2,
    }
    require(all(same_typed(candidate.get(key), value) for key, value in required.items()),
            "H11 requires fresh CPU weights, original tau/budget/selection and explicit2/2")
    rows = candidate.get("validation_rows")
    require(rows in ([], list(range(731, 739))) and all(type(row) is int for row in rows),
            "Validation selection rows changed")
    require(not any(key in config for config in (root, parent, candidate) for key in RETIRED),
            "Unregistered retired intervention cannot be inherited or combined")


def intervention():
    return {
        "historical_parent": "H03", "working_parent": "H11_H03_RECONSTRUCTED", "root": "C2_ROOT",
        "inherited_actor_update_interval": 2, "parent_target_update_interval": 1,
        "child_target_update_interval": 2, "target_update_tau": .002,
        "target_scope": ["Actor", "Critic", "mixer"],
        "synchronization": "Even successful global Critic step,after that call's Actor",
        "odd_target_parameters_and_buffers": "bitwise unchanged",
        "counter_reset_after_EA_burst_or_Actor_transfer": False,
        "tau_compensation": False, "parent_learner": "DelayedActorGAVDACLearner",
        "child_learner": "DelayedTargetsGAVDACLearner",
        "worker": "Explicit intended learner class; original shared transport and graph discard",
        "actor_loss_action_l2_coefficient": .001, "single_critic": True,
        "live_action_context_recurrent_gradient_preserved": True,
        "one_step_MSE_padding_global_denominator_target_detach_preserved": True,
        "original_initialization_parameter_counts_and_optimizers_preserved": True,
        "new_parameters_or_buffers": False, "additional_initial_RNG_draws": 0,
        "environment_replay_rollout_EA_noise_unchanged": True,
        "exact_resume_supported": False,
    }


def evaluation_protocol():
    return {
        "rows": list(range(913, 1095)), "mode": "continuous",
        "actor": "validation-selected rl", "initial_soc": .35,
        "terminal_soc": "free within device bounds", "hidden_and_previous_action_reset": "daily",
        "primary_metric": "economic_cost", "supplementary_metric": "full_cost",
        "expected_agent_hours": 13104, "retrospective_retained_development": True,
        "final_gate_version": "each_seed_better_mean_saving_3pct_v2",
        "each_seed_strictly_below_corresponding_C3": True,
        "per_seed_minimum_saving_fraction": 0.0,
        "mean_rule": "1-mean(candidate_cost)/mean(C3_cost)>=0.03",
        "mean_minimum_saving_fraction": .03, "mean_of_percentage_savings": False,
    }


def retention_protocol():
    return {
        "all_seed_cost_strictly_below_C3": True,
        "mean_cost_strictly_below_historical_parent": PARENT_MEAN,
        "per_seed_parent_threshold": None, "all_parent_seed_regressions_disclosed": True,
        "not_final_success_or_independent_significance": True,
    }


def validate_evaluation(value):
    require(same_typed(value, evaluation_protocol()), "H11 metric/inventory/final gate changed")


def validate_retention(value):
    require(same_typed(value, retention_protocol()), "Preregistered parent-retention gate changed")


def economic_gates(costs):
    require(len(costs) == 3 and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in costs),
            "All three accepted finite seed costs required")
    each = all(candidate < baseline for candidate, baseline in zip(costs, C3_COSTS, strict=True))
    saving = 1 - sum(costs) / sum(C3_COSTS)
    return {"each_seed_better": each, "ratio_of_means_saving": saving,
            "final_target_passed": each and saving >= .03,
            "parent_retention_passed": each and sum(costs) / 3 < PARENT_MEAN}


def validate_state_gates(state):
    gate = state["current_economic_gate"]
    fixed = {
        "version": "each_seed_better_mean_saving_3pct_v2", "metric": "economic_cost",
        "seed_labels": [0, 1, 2], "per_seed_rule": "candidate_cost < corresponding_C3_cost",
        "per_seed_minimum_saving_fraction": 0.0, "per_seed_comparison": "strictly_greater",
        "mean_rule": "1-mean(candidate_cost)/mean(C3_cost) >=0.03",
        "mean_minimum_saving_fraction": .03, "arithmetic_mean_of_percentage_savings": False,
        "fixed_C3_costs": list(C3_COSTS),
    }
    require(all(same_typed(gate.get(key), value) for key, value in fixed.items()),
            "Current final C3 economic gate changed")
    require(same_typed(state["H11_registration"]["final_gate"], gate),
            "H11 preregistration and current final gate disagree")
    validate_retention(state["H11_registration"]["retention_gate"])


def make_plans(tokens):
    require(len(tokens) == 3 and all(type(token) is str for token in tokens)
            and len(set(tokens)) == 3
            and all(re.fullmatch(r"h11s[a-z0-9_]{1,25}", token) for token in tokens),
            "Three unique short NEW H11 formal tokens required")
    return [{
        "seed": seed, "host": "local" if seed == 2 else "707",
        "root": HOSTS["local" if seed == 2 else "707"][0],
        "slot": 0 if seed == 2 else seed, "capacity": 1 if seed == 2 else 2,
        "token": tokens[seed], "config_name": "gavdac_c2_h11", "joint_budget": 6000000,
        "rollout_workers": 2, "learner_workers": 2, "threads": 1, "status": "frozen_not_launched",
    } for seed in (0, 1, 2)]


def validate_plans(plans):
    require(len(plans) == 3 and all(type(plan.get(key)) is int for plan in plans for key in (
        "seed", "slot", "capacity", "joint_budget", "rollout_workers", "learner_workers", "threads")),
        "Exactly three integer seed/slot/budget/process assignments required")
    require(same_typed(plans, make_plans([plan["token"] for plan in plans])),
            "Original seed/host/slot/budget/process assignment changed")


def validate_rollback(implementation, commits, destination, scope, registered_scope):
    require(implementation == CHILD and commits == [CHILD] and destination == PARENT,
            "Only registered child delta may revert to exact qualified parent,not root/support")
    require(same_typed(scope, registered_scope)
            and scope["child_reverse_commit_order"] == [CHILD]
            and scope["exact_parent_code_destination"] == PARENT
            and set(scope["preserve_current_ledger_document_paths"]) == {
                "docs/c2_hypotheses/H11.md", "docs/c2_hypotheses/README.md"}
            and len(scope["registered_child_code_paths"]) == 11
            and scope["parent_reconstruction_is_not_child_delta_or_support"] is True,
            "Registered ledger-preserving rollback scope changed")


def validate_phase(state):
    require(state["active_hypothesis"] == "H11" and not state["active_runs"] and not state["run_plan"]
            and state["formal_protocol"] is None
            and state["status"] == "h11_enabled_engineering_accepted_pending_independent_freeze"
            and state["H11_development"]["enabled_host_engineering_accepted"] is True,
            "Freeze requires idle engineering-accepted and not-yet-frozen H11")
    require(state["authorized_slot_capacity"] == 3
            and state["execution_slots"]["arch715_0"]["enabled"] is False
            and state["execution_host_amendment"]["enabled_slot_keys"] == ["local0", "707_0", "707_1"],
            "User-disabled715 must remain excluded from the three enabled slots")


def compare_host(previous, current, capacity, host):
    require(host in HOSTS and type(capacity) is int and HOSTS[host][1] == capacity,
            "Unknown/disabled host or changed slot capacity")
    require(all(current["files"].get(name) == value for name, value in previous["files"].items()),
            "Original qualified host raw/LF source/config/data bytes changed")
    require(set(current["files"]) - set(previous["files"]) == ADDITIONS,
            "Only two independent support additions are authorized")
    require(current["slots"] == [{"slot": slot, "state": "free"} for slot in range(capacity)],
            "Occupied slot or stale lock")
    require(not current["existing_tokens"], "Proposed new formal token already exists")
    require(all(row["status"] == " M" and row["path"] in (MANUSCRIPTS if host == "local" else set())
                for row in current["git_status"]), "Unknown dirty/staged scope")
    require(all(same_typed(current[key], previous[key]) for key in
                ("root", "host", "platform", "logical_cpus", "python", "torch", "numpy", "pandas")),
            "Qualified host allocation or dependencies changed")
    require(type(current["torch_threads"]) is int and current["torch_threads"] == 1,
            "CPU compute threads must be1")
    require(current["memory"]["available_bytes"] > (3 if host == "local" else 8) * 1024**3,
            "Insufficient currently available RAM for the accepted process budget")
    busy = current["cpu_busy_percent_snapshot"]
    require(type(busy) in (int, float) and math.isfinite(busy) and 0 <= busy <= 100,
            "A finite read-only CPU competition snapshot is required")


def check_c3_actor(actor, root):
    import torch
    checkpoint = Path(actor["checkpoint"])
    for name, key in (("metadata.json", "metadata_sha256"), ("agent.th", "policy_sha256"),
                      ("config.yaml", "config_sha256")):
        require(digest(checkpoint / name) == actor[key], "Frozen C3 checkpoint raw changed: " + name)
    require(digest(checkpoint.parent.parent / "best_rl.json") == actor["selection_pointer_sha256"],
            "Frozen C3 selection pointer changed")
    metadata = read(checkpoint / "metadata.json")
    require(metadata["validation"]["rows"] == list(range(731, 739))
            and metadata["policy_source"] == "rl" and metadata["checkpoint_step"] == actor["checkpoint_step"]
            and metadata["policy_sha256"] == actor["policy_sha256"], "Frozen C3 Actor selection changed")
    saved = Path(metadata["source_snapshot"]).resolve()
    # Python metadata and YAML-inclusive evaluation summaries own independent key sets.
    for manifest in (metadata["source_sha256"], actor["evaluation_source_sha256"]):
        for name, expected in manifest.items():
            path = (saved / name).resolve()
            require(path.is_relative_to(saved) and digest(path) == expected,
                    "Frozen C3 saved source raw changed: " + name)
    require(metadata["data_sha256"] == actor["evaluation_data_sha256"], "Frozen C3 evaluation data changed")
    data_root = (root / "src/envs/data").resolve()
    for name, expected in metadata["data_sha256"].items():
        path = (data_root / name).resolve()
        require(path.is_relative_to(data_root) and digest(path) == expected, "Frozen C3 data raw changed")
    weights = torch.load(checkpoint / "agent.th", map_location="cpu", weights_only=True)
    require(bool(weights) and all(v.device.type == "cpu" and (not v.is_floating_point() or
            (v.dtype == torch.float32 and torch.isfinite(v).all())) for v in weights.values()),
            "Frozen C3 Actor must be finite CPUfloat32")
    return {"seed": actor["seed"], "checkpoint": str(checkpoint),
            "metadata_policy_config_pointer_source_data_rawhash_passed": True,
            "metadata_and_evaluation_source_keys_checked_individually": True}


def actors_for_host(state, host):
    require(host in HOSTS, "Disabled/unknown host")
    actors = []
    for seed, item in state["baseline_C3"].items():
        if ("local" if seed == "2" else "707") == host:
            summary = read(Path(item["directory"]) / "summary.json")
            actors.append({**item["selected"], "evaluation_source_sha256": summary["source_sha256"],
                           "evaluation_data_sha256": summary["data_sha256"]})
    return actors


def validate_baseline(baseline, original):
    require(same_typed(baseline, original) and set(baseline) == {"0", "1", "2"},
            "Immutable original C3 identities changed")
    require(all(same_typed(baseline[str(seed)]["economic_cost"], cost)
                for seed, cost in enumerate(C3_COSTS)), "Fixed C3 costs changed")
    # Original .97 maxima are history, not H11's strict per-seed threshold.


def revalidate_c3_trace(state, base):
    original = base / "h01_protocol_b.json"
    require(digest(original) == ORIGINAL_C3_FREEZE, "Immutable original C3 freeze changed")
    frozen = read(original)
    validate_baseline(state["baseline_C3"], frozen["baseline_C3"])
    require(same_typed(state["protected_sha256"], frozen["protected_sha256"]),
            "Original protected-file manifest changed")
    costs = {}
    for seed, item in state["baseline_C3"].items():
        summary, records = load_evaluation(item)
        manifest = ROOT / f"results/cmp20/evaluation_seed{seed}_a.json"
        require(digest(manifest) == item["original_manifest_sha256"], "Original C3 manifest changed")
        selected = next(row for row in read(manifest)["selected"] if row["group"] == "C3")
        require(same_typed(selected, item["selected"]) and summary["policy_sha256"] == selected["policy_sha256"]
                and summary["totals"]["economic_cost"] == item["economic_cost"],
                "Original C3 selected Actor or accepted total changed")
        require(all(math.isclose(sum(float(row[name]) for name in (
                    "buy_cost", "gen_cost", "bess_cost", "exchange_cost")),
                    float(row["economic_cost"]), rel_tol=0, abs_tol=1e-7) for row in records),
                "Independent original C3 per-record formula differs")
        costs[seed] = {"economic_cost": item["economic_cost"], "agent_hours": len(records),
                       "summary_sha256": item["summary_sha256"], "trajectory_sha256": item["trajectory_sha256"],
                       "independent_record_formula_daily_and_period_totals_checked": True}
    return costs


def inspect_host(payload):
    require(payload["host"] in HOSTS, "Disabled/unknown host")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy
    import pandas
    import torch
    from tools.experiment_slot import slot_status
    torch.set_num_threads(1)
    root = Path.cwd()
    paths = [p for area in ("src", "tools", "tests") for p in sorted((root / area).rglob("*"))
             if p.suffix in {".py", ".yaml", ".csv"}]
    paths += [root / name for name in ("pyproject.toml", "uv.lock", "src/config/algs/README.md")]
    files = {}
    for path in paths:
        raw = path.read_bytes()
        files[path.relative_to(root).as_posix()] = {
            "raw": hashlib.sha256(raw).hexdigest(), "lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}
    existing = [token for token in payload["tokens"] if (root / "results" / token).exists()
                or any((root / "results/cmp20/supervisors").glob(token + "*"))]
    actors = [check_c3_actor(actor, root) for actor in payload["actors"]]
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total", "available", "page_total", "page_available",
                    "virtual_total", "virtual_available", "extended")]
        ram = Memory()
        ram.length = ctypes.sizeof(ram)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ram)):
            raise ctypes.WinError()
        memory = {"total_bytes": ram.total, "available_bytes": ram.available}
        rows = csv.reader(io.StringIO(subprocess.check_output(["tasklist", "/FO", "CSV", "/NH"]).decode("oem")))
        processes = [{"pid": int(row[1]), "name": row[0]} for row in rows if row and any(
            word in row[0].lower() for word in ("python", "uv.exe", "wandb"))]
        busy = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | "
            "Where-Object Name -eq '_Total' | Select-Object PercentProcessorTime | ConvertTo-Json"],
            text=True))["PercentProcessorTime"]
        load = None
    else:
        info = {row.split()[0].rstrip(":"): int(row.split()[1]) * 1024
                for row in Path("/proc/meminfo").read_text().splitlines()
                if row.startswith(("MemTotal:", "MemAvailable:"))}
        memory = {"total_bytes": info["MemTotal"], "available_bytes": info["MemAvailable"]}
        rows = [line.split(maxsplit=1) for line in subprocess.check_output(
            ["ps", "-eo", "pid=,comm="], text=True).splitlines()]
        processes = [{"pid": int(row[0]), "name": row[1]} for row in rows if len(row) == 2 and any(
            word in row[1].lower() for word in ("python", "uv", "wandb"))]
        load = list(os.getloadavg())
        def ticks():
            return [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        before = ticks()
        time.sleep(.25)
        delta = [end - start for start, end in zip(before, ticks(), strict=True)]
        busy = 100 * (1 - (delta[3] + delta[4]) / sum(delta))
    raw_status = subprocess.check_output(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    return {
        "root": str(root), "host": platform.node(), "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": numpy.__version__, "pandas": pandas.__version__, "torch_threads": torch.get_num_threads(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": [{"status": row[:2].decode(), "path": row[3:].decode("utf-8")}
                       for row in raw_status.split(b"\0") if row],
        "slots": slot_status(payload["slots"]), "files": files, "existing_tokens": existing,
        "original_host_C3_actor_checks": actors, "memory": memory, "named_processes": processes,
        "load_average": load, "cpu_busy_percent_snapshot": busy,
        "resource_limitation": "Read-only snapshot, not peak RAM or6Mthroughput guarantee",
    }


HOST_PROBE = ("import json,sys;from tools.freeze_h11_protocol import inspect_host;"
              "print(json.dumps(inspect_host(json.load(sys.stdin))))")


def probe(host, root, tokens, actors, slots):
    require(host in HOSTS and (root, slots) == HOSTS[host], "Disabled/unknown host or wrong allocation")
    command = ["uv", "run", "--frozen", "python", "-c", HOST_PROBE]
    if host != "local":
        command = ["ssh", "-o", "ConnectTimeout=15", host,
                   shlex.join(["uv", "--directory", root, "run", "--frozen", "python", "-c", HOST_PROBE])]
    result = subprocess.run(command, cwd=ROOT,
        input=json.dumps({"host": host, "tokens": tokens, "actors": actors, "slots": slots}),
        capture_output=True, text=True, encoding="utf-8", timeout=120)
    require(result.returncode == 0, f"{host} H11 read-only probe failed: {result.stderr}")
    return json.loads(result.stdout)


def validate_tooling(current, checked):
    require(checked["accepted"] is True and type(checked["focused_tests"]) is int
            and checked["focused_tests"] == FOCUSED_TESTS and checked["returncode"] == 0
            and all(current["files"][name]["raw"] == checked["raw_sha256"][name] for name in ADDITIONS),
            "Original-host incremental tool/test qualification differs")


def validate_engineering(base):
    path = base / "h11_engineering_acceptance_a.json"
    require(digest(path) == ENGINEERING, "H11 engineering acceptance bytes changed")
    value = read(path)
    require(value["accepted"] is True and value["implementation_commit"] == CHILD
            and value["parent_reconstruction_commit"] == PARENT
            and value["enabled_hosts"] == ["local", "707"] and value["enabled_slot_count"] == 3
            and set(value["smokes"]) == {"h11ls0a", "h11rs0a", "h11rs1a"},
            "H11 enabled-host engineering incomplete")
    for name, expected in PINS.items():
        require(digest(base / name) == expected, "Pinned original H11 evidence changed: " + name)
    for name, expected in value["bound_original_artifacts"].items():
        require(digest(ROOT / name) == expected, "Engineering bound artifact changed: " + name)
    for token, item in value["smokes"].items():
        require(digest(ROOT / item["path"]) == item["sha256"], "H11 smoke acceptance changed: " + token)
        smoke = read(ROOT / item["path"])
        c = smoke["completion"]
        require(smoke["accepted"] is True and smoke["launch"]["git_commit"] == CHILD
                and smoke["launch"]["config"] == "gavdac_c2_h11", "H11 smoke runtime changed")
        expected = {
            "actor_update_interval": 2, "target_update_interval": 2, "target_update_tau": .002,
            "critic_optimizer_steps": 8, "actor_optimizer_steps": 4, "target_updates": 4,
            "finite_update_checks": 8, "train_updates": 8, "exact_resume_supported": False,
            "optimizer_post_step_finite_checks": {"critic": 8, "actor": 4},
            "joint_interactions": 12624, "agent_transitions": 37872, "ea_cycles": 11,
            "lineage_role": "H11_child_of_reconstructed_H03",
        }
        require(all(same_typed(c.get(k), v) for k, v in expected.items()), "H11 smoke cadence/counts changed")
    return value


def validate_lineage(state, base):
    registration = state["H11_registration"]
    require(registration["preregistration_commit"] == PREREGISTRATION
            and registration["parent_node_id"] == "H03" and registration["root_node_id"] == "C2_ROOT"
            and registration["parent_reconstruction_commits"] == [PARENT]
            and registration["child_delta_commits"] == [CHILD]
            and registration["rollback_destination_commit"] == PARENT,
            "Registered H11 parent/root/child identity changed")
    require(state["research_lineage_policy"]["version"] == "one_increment_per_recorded_parent_v1"
            and state["lineage"]["working_checkout_node_id"] == "H11"
            and state["lineage"]["selected_parent_node_id"] == "H03"
            and state["lineage"]["active_child_node_id"] == "H11"
            and state["lineage"]["new_working_parent_commit"] == PARENT
            and state["lineage"]["registry_sha256"] == LINEAGE
            and digest(ROOT / state["lineage"]["registry"]) == LINEAGE, "Lineage registry changed")
    scope = read(base / "h11_child_rollback_scope_a.json")
    require(digest(base / "h11_child_rollback_scope_a.json") == ROLLBACK_SCOPE,
            "Registered rollback scope bytes changed")
    validate_rollback(state["implementation_commit"], state["rollback_commits"],
                      state["rollback_destination_commit"], scope, registration["child_rollback_scope"])
    changed = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", CHILD], cwd=ROOT, text=True).splitlines()
    require(set(changed) == set(scope["registered_child_code_paths"])
            | set(scope["preserve_current_ledger_document_paths"]), "Child Git delta differs from scope")
    for before, after in ((PREREGISTRATION, PARENT), (PARENT, CHILD)):
        subprocess.run(["git", "merge-base", "--is-ancestor", before, after], cwd=ROOT, check=True)
    require(read(base / "h03_protocol_a.json") is not None, "Historical parent protocol unavailable")
    require(digest(base / "h03_protocol_a.json") == registration["parent_protocol_sha256"]
            and digest(base / "h03_result_a.json") == registration["parent_result_sha256"],
            "Historical parent protocol/result changed")
    validate_state_gates(state)
    return scope


def freeze(state_path, tokens):
    state = read(state_path)
    validate_phase(state)
    base = state_path.parent
    scope = validate_lineage(state, base)
    engineering = validate_engineering(base)
    workers = {"rollout_workers": 2, "learner_workers": 2}
    root, parent, candidate = [entry_config(name, workers) for name in (
        "gavdac_ea_main", "gavdac_c2_h11_parent", "gavdac_c2_h11")]
    validate_candidate(root, parent, candidate)
    require(same_typed(state["config"], candidate), "State differs from actual child entry")
    plans = make_plans(tokens)
    validate_plans(plans)
    subprocess.run(["git", "diff", "--exit-code", CHILD, "--", "src", "tools/experiment_slot.py"],
                   cwd=ROOT, check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"],
                                   cwd=ROOT, text=True).split()[0] == head,
            "Push independently qualified support tool before freeze")
    require(set(subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", head],
        cwd=ROOT, text=True).splitlines()) == ADDITIONS, "Current support commit must contain only two additions")
    tooling_path = base / "h11_protocol_tool_acceptance_a.json"
    tooling = read(tooling_path)
    require(tooling["accepted"] is True and tooling["support_commit"] == head
            and set(tooling["hosts"]) == set(HOSTS), "Incremental tool acceptance must bind both enabled hosts")
    adoption = base / "mp_adoption_protocol_a.json"
    require(digest(adoption) == MP_ADOPTION, "Original MP/default exception changed")
    protected = dict(state["protected_sha256"])
    exception = read(adoption)["protected_hash_exception"]
    protected["src/config/default.yaml"] = exception["new_raw"]
    protected.update(read(base / "h11_implementation_record_a.json")["protected"]["manuscript_actual_raw"])
    require(all(digest(ROOT / name) == value for name, value in protected.items()),
            "Protected comparison or user manuscript raw changed")
    c3_costs = revalidate_c3_trace(state, base)
    hosts = {}
    for host, (host_root, capacity) in HOSTS.items():
        current = hosts[host] = probe(host, host_root, tokens, actors_for_host(state, host), capacity)
        compare_host(read(base / f"h11_host_{host}_a.json"), current, capacity, host)
        validate_tooling(current, tooling["hosts"][host])
        require(current["git_commit"] == head, "Own idle-clean host must ff-only to exact new support code")
        for smoke in engineering["smokes"].values():
            launch = read(ROOT / smoke["path"])["launch"]
            smoke_host = "local" if launch["seed"] == 2 else "707"
            if smoke_host == host:
                require(launch["source_sha256"] == {
                    name: value["raw"] for name, value in current["files"].items()
                    if name.startswith("src/") and Path(name).suffix in {".py", ".yaml", ".csv"}},
                    "Original smoke-tested host training source changed")
    require(hosts["local"]["files"].keys() == hosts["707"]["files"].keys()
            and all(value["lf"] == hosts["707"]["files"][name]["lf"]
                    for name, value in hosts["local"]["files"].items()),
            "Host source differences exceed accepted EOL; actual raw remains authoritative")
    for plan in plans:
        directory = f"{plan['root']}/results/{plan['token']}".replace("\\", "/")
        plan["entry_config"] = entry_config("gavdac_c2_h11", {
            **workers, "seed": plan["seed"], "unique_token": plan["token"],
            "local_results_path": directory, "env_args.result_dir": directory,
            "wandb_dir": directory + "/wandb", "use_cuda": False, "wandb_mode": "online"})
    return {
        "hypothesis": "H11", "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_before_freeze_sha256": digest(state_path), "algorithm_implementation_commit": CHILD,
        "parent_reconstruction_commit": PARENT, "preregistration_commit": PREREGISTRATION,
        "historical_parent": "H03", "root": "C2_ROOT", "lineage_registry_sha256": LINEAGE,
        "rollback_commits_reverse_order": [CHILD], "exact_rollback_destination": PARENT,
        "ledger_preserving_rollback_scope": scope, "rollback_scope_sha256": ROLLBACK_SCOPE,
        "support_commit_not_economically_reverted": head,
        "engineering_acceptance_sha256": ENGINEERING, "original_C3_readonly_audit_sha256": C3_AUDIT,
        "support_tool_acceptance_sha256": digest(tooling_path), "intervention": intervention(),
        "inherited_parent_config": parent, "candidate_config": candidate,
        "run_plan": plans, "hosts": hosts, "disabled_hosts": ["715"],
        "baseline_C3": state["baseline_C3"], "immutable_original_C3_freeze_sha256": ORIGINAL_C3_FREEZE,
        "original_C3_trace_revalidation": c3_costs, "original_protected_sha256": state["protected_sha256"],
        "future_execution_protected_default_exception": exception,
        "selection": {"rows": list(range(731, 739)), "interval_joint_steps": 10000, "actor": "rl"},
        "evaluation": evaluation_protocol(), "retention": retention_protocol(),
        "current_final_gate": state["current_economic_gate"],
        "cpu_process_budget_per_slot": 5, "707_two_slot_torch_process_budget": 10,
        "additional_helpers": True, "compute_threads_per_process": 1,
        "resource_limitation": "Current read-only snapshots,not peak memory or6Mthroughput guarantees",
        "limitation": "Exploratory retained reuse/cumulative lineage,not root-isolated or independent significance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs=3, default=["h11s0a", "h11s1a", "h11s2a"])
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / "results").resolve()) and not output.exists(),
            "Freeze requires NEW output below results/")
    payload = freeze(args.state.resolve(), args.tokens)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "tokens": [plan["token"] for plan in payload["run_plan"]]}))


if __name__ == "__main__":
    main()
