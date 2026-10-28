"""Freeze qualified H12 provenance and three fresh runs; never launch training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
# Only generic entry loading, transport inventory and original-C3 readers are reused.
from tools.freeze_h11_protocol import (
    actors_for_host, check_c3_actor, digest, entry_config, inspect_host as inventory,
    read, require, revalidate_c3_trace, same_typed,
)

CHILD = "de78e9cf710663c010d41df7dfd0f29aa9217dc4"
PARENT = "4e3aa00eacf814e3d328be781e4af8632ec0488f"
PREREGISTRATION = "2448cf4da4c4c64959a85fea4d833b266391e43f"
AMENDMENT = "0b96c40ac225c29771effe17d239c7deac34e3e8"
ENGINEERING = "e03aff6e685572fd7b1c692c8f9b6c35a9f5e00c18d9b93f1fdd72cc36aabc0e"
C3_AUDIT = "82d3bab16bc13aee1f1656337635829c6633a68b746a6883a2a445f241dce5d2"
ROLLBACK_SCOPE = "f88f61b422902fd8227146a9ef42bd5e2b712526298155a7a110e30741e33b6b"
LINEAGE = "1e9ae90f01f62968caf690fc510211f2be41faca404b9adc0020f881300c945e"
MP_DEFAULT = "836837797231fd52c41881f9a7b002e842a1122b9cbf19e6f5e4570485bbcbab"
HOSTS = {"local": (str(ROOT), 1), "707": ("/home/arch707/data/CDE", 2)}
ADDITIONS = {"tools/freeze_h12_protocol.py", "tests/test_h12_protocol.py"}
MANUSCRIPTS = {"paper/manuscript/sections/method.tex", "paper/manuscript/sections/modeling.tex"}
FOCUSED_TESTS = 23
C3_COSTS = (344913.0888879933, 344550.706793143, 354488.7203334839)
PARENT_MEAN = 343419.27471629175
RETIRED = (
    "actor_mixer_context_grad", "clipped_double_critic", "five_step_critic",
    "critic_layer_norm", "target_policy_smoothing", "critic_small_output_init",
    "actor_no_action_l2", "critic_center_generator_action",
)
PINS = {
    "h12_preregistration_a.json": "5f239f63b811a48d254cf7e4ba25eacfe5b59ebb9db185aa0ea514bbbe83994d",
    "h12_parent_review_a.json": "4d3056515c5456f505647bdc64216f42993427093ad66681ab4750d50808c415",
    "h12_scope_amendment_a.json": "6d08b74a310623ae8fb2722ccc2c3238507b41e5f032d882338c9fdfbac7aa73",
    "h12_implementation_record_a.json": "c7e1b9e34164ddf9ceab6fcc09b2f487c30b3f40a5667d92e2ec796ab75360ab",
    "h12_child_rollback_scope_a.json": ROLLBACK_SCOPE,
    "h12_original_C3_revalidation_a.json": C3_AUDIT,
    "lineage_registry_k.json": LINEAGE,
}


def validate_candidate(root, parent, child):
    require(same_typed(parent, {**root, "name": "gavdac_c2_h11_parent",
                               "actor_update_interval": 2, "target_update_interval": 1}),
            "Qualified parent must inherit only H03 Actor2/targets1")
    require(same_typed(child, {**parent, "name": "gavdac_c2_h12", "scaled_huber_critic": True}),
            "H12 permits only its name and true boolean scaled Huber flag")
    required = {
        "t_max": 6000000, "EA": True, "crossover_prob": 0.0, "use_cuda": False,
        "checkpoint_path": "", "policy_checkpoint": "", "evaluate": False, "load_step": 0,
        "rollout_workers": 2, "learner_workers": 2, "test_interval": 10000,
        "fixed_eval_days": 8, "evaluation_initial_soc": .35, "save_model": True,
        "save_best_policy": True, "wandb_mode": "online", "rnn_hidden_dim": 48,
        "actor_update_interval": 2, "target_update_interval": 1, "target_update_tau": .002,
    }
    require(all(same_typed(child.get(k), v) for k, v in required.items()),
            "Fresh CPU budget, original selection/cadence/tau and integer2/2 required")
    rows = child.get("validation_rows")
    require(rows in ([], list(range(731, 739))) and all(type(v) is int for v in rows),
            "Validation rows must remain731-738")
    require(not any(k in c for c in (root, parent, child) for k in RETIRED),
            "Retired intervention cannot be combined, even with a false flag")
    require("scaled_huber_critic" not in root and "scaled_huber_critic" not in parent,
            "Huber must be child-only")


def intervention():
    return {
        "historical_parent": "H03", "working_parent": "H11_H03_RECONSTRUCTED", "root": "C2_ROOT",
        "inherited_actor_update_interval": 2, "target_update_interval": 1, "target_update_tau": .002,
        "parent_learner": "DelayedActorGAVDACLearner",
        "child_learner": "HuberDelayedActorGAVDACLearner",
        "worker_learner": "HuberDelayedActorGAVDACLearner",
        "sole_delta": "sum(2*Huber(maskedTD,0,delta=1))/original_global_valid_count",
        "mask_before_loss": True, "target_detached": True, "huber_delta": 1.,
        "huber_multiplier": 2., "single_critic": True, "actor_action_l2_coefficient": .001,
        "targets_after_critic_and_optional_actor_every_successful_call": True,
        "original_history_graph_discard_EA_rollout_replay_optimizer_rng_preserved": True,
        "reward_observation_target_or_physical_cost_clipping": False,
        "new_parameters_or_buffers": False, "exact_resume_supported": False,
    }


def evaluation_protocol():
    return {
        "rows": list(range(913, 1095)), "mode": "continuous", "actor": "validation-selected rl",
        "initial_soc": .35, "terminal_soc": "free within device bounds",
        "hidden_and_previous_action_reset": "daily", "primary_metric": "economic_cost",
        "supplementary_metric": "full_cost", "expected_agent_hours": 13104,
        "retrospective_retained_development": True,
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
    require(same_typed(value, evaluation_protocol()), "H12 metric/inventory/final gate changed")


def validate_retention(value):
    require(same_typed(value, retention_protocol()), "H12 preregistered retention changed")


def economic_gates(costs):
    require(len(costs) == 3 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in costs),
            "Three accepted finite seed costs required")
    each = all(v < c3 for v, c3 in zip(costs, C3_COSTS, strict=True))
    saving = 1 - sum(costs) / sum(C3_COSTS)
    return {"each_seed_better": each, "ratio_of_means_saving": saving,
            "final_target_passed": each and saving >= .03,
            "parent_retention_passed": each and sum(costs) / 3 < PARENT_MEAN}


def validate_state_gates(state):
    gate = state["current_economic_gate"]
    expected = {
        "version": "each_seed_better_mean_saving_3pct_v2", "metric": "economic_cost",
        "seed_labels": [0, 1, 2], "per_seed_rule": "candidate_cost < corresponding_C3_cost",
        "per_seed_minimum_saving_fraction": 0.0, "per_seed_comparison": "strictly_greater",
        "mean_rule": "1-mean(candidate_cost)/mean(C3_cost) >=0.03",
        "mean_minimum_saving_fraction": .03, "arithmetic_mean_of_percentage_savings": False,
        "fixed_C3_costs": list(C3_COSTS),
    }
    require(all(same_typed(gate.get(k), v) for k, v in expected.items())
            and same_typed(state["H12_registration"]["final_gate"], gate), "H12 final gate drift")
    validate_retention(state["H12_registration"]["retention_gate"])


def make_plans(tokens):
    import re
    require(len(tokens) == 3 and all(type(v) is str for v in tokens) and len(set(tokens)) == 3
            and all(re.fullmatch(r"h12s[a-z0-9_]{1,25}", v) for v in tokens),
            "Three unique safe NEW H12 formal tokens required")
    return [{
        "seed": seed, "host": "local" if seed == 2 else "707",
        "root": HOSTS["local" if seed == 2 else "707"][0], "slot": 0 if seed == 2 else seed,
        "capacity": 1 if seed == 2 else 2, "token": tokens[seed], "config_name": "gavdac_c2_h12",
        "joint_budget": 6000000, "rollout_workers": 2, "learner_workers": 2, "threads": 1,
        "status": "frozen_not_launched",
    } for seed in (0, 1, 2)]


def validate_plans(plans):
    require(len(plans) == 3 and all(type(p.get(k)) is int for p in plans for k in (
        "seed", "slot", "capacity", "joint_budget", "rollout_workers", "learner_workers", "threads")),
        "Exactly three integer host/slot/seed/budget/process assignments required")
    require(same_typed(plans, make_plans([p["token"] for p in plans])), "H12 allocation changed")


def validate_phase(state):
    require(state["active_hypothesis"] == "H12" and state["active_runs"] == []
            and state["run_plan"] == [] and state["formal_protocol"] is None
            and state["status"] == "h12_engineering_accepted_pending_independent_freeze"
            and state["H12_development"]["enabled_host_engineering_accepted"] is True,
            "Freeze requires idle engineering-accepted unfrozen H12")
    require(state["authorized_slot_capacity"] == 4
            and state["execution_slots"]["arch715_0"]["enabled"] is True
            and state["execution_slots"]["arch715_0"]["state"] == "enabled_idle_pending_requalification",
            "715 must remain enabled but idle and excluded from H12 formal allocation")
    checklist = state["H12_checklist"]
    require(all(checklist.get(k) is True for k in (
        "preregistered_and_normally_pushed", "parent_and_primary_review", "implementation",
        "local_engineering", "remote_engineering", "original_C3_readonly_audit"))
        and checklist["independent_freeze"] is False and checklist["formal_training_started"] is False,
        "H12 prerequisite checklist incomplete or already frozen/launched")


def validate_rollback(implementation, commits, destination, scope, registered):
    require(implementation == CHILD and commits == [CHILD] and destination == PARENT,
            "Only registered H12 child may revert to exact qualified parent")
    require(same_typed(scope, registered) and scope["child_reverse_commit_order"] == [CHILD]
            and scope["exact_parent_code_destination"] == PARENT
            and len(scope["registered_child_code_paths"]) == 11
            and scope["algorithm_commit_excludes_ledger_docs"] is True
            and scope["parent_reconstruction_is_not_child_delta_or_support"] is True
            and set(scope["preserve_current_ledger_document_paths"]) == {
                "docs/c2_hypotheses/H12.md", "docs/c2_hypotheses/README.md"},
            "Registered H12 child-only rollback scope changed")


def validate_lineage(state, base):
    reg = state["H12_registration"]
    expected = {
        "preregistration_commit": PREREGISTRATION, "engineering_scope_amendment_commit": AMENDMENT,
        "parent_node_id": "H03", "working_parent_node_id": "H11_H03_RECONSTRUCTED",
        "root_node_id": "C2_ROOT", "exact_parent_code_commit": PARENT,
        "child_delta_commits": [CHILD], "implementation_commit": CHILD,
        "rollback_destination_commit": PARENT, "formal_seed_labels": [0, 1, 2],
        "physical_joint_budget_including_EA": 6000000,
        "formal_host_slots": [["707", 0, 0], ["707", 1, 1], ["local", 0, 2]],
        "workers": {"rollout_workers": 2, "learner_workers": 2, "threads": 1},
        "auxiliary_715_required_for_H12": False,
    }
    require(all(same_typed(reg.get(k), v) for k, v in expected.items()), "H12 registration identity drift")
    lineage = state["lineage"]
    require(state["research_lineage_policy"]["version"] == "one_increment_per_recorded_parent_v1"
            and lineage["working_checkout_node_id"] == "H12"
            and lineage["selected_parent_node_id"] == "H03" and lineage["active_child_node_id"] == "H12"
            and lineage["new_working_parent_commit"] == PARENT
            and lineage["registry_sha256"] == LINEAGE and digest(ROOT / lineage["registry"]) == LINEAGE,
            "H12 parent/root/child lineage drift")
    scope_path = base / "h12_child_rollback_scope_a.json"
    require(digest(scope_path) == ROLLBACK_SCOPE, "H12 rollback scope raw changed")
    scope = read(scope_path)
    validate_rollback(state["implementation_commit"], state["rollback_commits"],
                      state["rollback_destination_commit"], scope, reg["child_rollback_scope"])
    changed = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", CHILD], cwd=ROOT, text=True).splitlines()
    require(set(changed) == set(scope["registered_child_code_paths"]), "H12 Git delta exceeds registered scope")
    for before, after in ((PARENT, PREREGISTRATION), (PREREGISTRATION, AMENDMENT), (AMENDMENT, CHILD)):
        subprocess.run(["git", "merge-base", "--is-ancestor", before, after], cwd=ROOT, check=True)
    for name, key in (("h03_protocol_a.json", "parent_protocol_sha256"), ("h03_result_a.json", "parent_result_sha256")):
        require(digest(base / name) == reg[key], "Historical qualified parent evidence changed")
    validate_state_gates(state)
    return scope


def validate_engineering(base):
    require(digest(base / "h12_engineering_acceptance_a.json") == ENGINEERING,
            "H12 engineering acceptance bytes changed")
    value = read(base / "h12_engineering_acceptance_a.json")
    require(value["accepted"] is True and value["implementation_commit"] == CHILD
            and value["exact_parent_code_destination"] == PARENT
            and value["formal_hosts"] == ["local", "707"] and value["formal_slots"] == 3
            and value["qualified_file_count"] == 144
            and set(value["smokes"]) == {"h12ls0a", "h12rs0a", "h12rs1a"},
            "H12 original-host engineering incomplete")
    for name, expected in PINS.items():
        require(digest(base / name) == expected, "Pinned H12 evidence changed: " + name)
    for name, expected in value["bound_original_artifacts"].items():
        require(digest(ROOT / name) == expected, "Bound original artifact changed: " + name)
    expected = {
        "lineage_role": "H12_child_of_reconstructed_H03", "actor_update_interval": 2,
        "target_update_interval": 1, "target_update_tau": .002, "scaled_huber_critic": True,
        "huber_delta": 1., "huber_multiplier": 2., "critic_optimizer_steps": 8,
        "actor_optimizer_steps": 4, "target_updates": 8, "finite_update_checks": 8,
        "train_updates": 8, "optimizer_post_step_finite_checks": {"critic": 8, "actor": 4},
        "exact_resume_supported": False, "joint_interactions": 12624,
        "agent_transitions": 37872, "ea_cycles": 11,
    }
    for item in value["smokes"].values():
        require(digest(ROOT / item["path"]) == item["sha256"], "Original H12 smoke raw changed")
        smoke = read(ROOT / item["path"])
        require(smoke["accepted"] is True and smoke["launch"]["git_commit"] == CHILD
                and smoke["launch"]["config"] == "gavdac_c2_h12"
                and same_typed(smoke["completion"], expected)
                and same_typed(item["completion"], expected), "H12 accepted smoke class/cadence/counts drift")
    return value


def inspect_host(payload):
    result = inventory(payload)
    import pandas
    result["numeric_data"] = {}
    for name in ("load", "pv", "price"):
        values = pandas.read_csv(ROOT / "src/envs/data" / (name + ".csv"), header=None).to_numpy(dtype="float64")
        result["numeric_data"][name] = {
            "shape": list(values.shape), "dtype": str(values.dtype),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    return result


def compare_host(previous, current, capacity, host):
    require(host in HOSTS and type(capacity) is int and HOSTS[host][1] == capacity,
            "Unknown formal host or changed capacity")
    require(all(current["files"].get(k) == v for k, v in previous["files"].items()),
            "Original144 qualified host raw/LF source/config/data/tools changed")
    require(set(current["files"]) - set(previous["files"]) == ADDITIONS,
            "Only two independent support additions permitted")
    require(same_typed(current["numeric_data"], previous["numeric_data"]), "Numeric float64 CSV data changed")
    require(current["slots"] == [{"slot": v, "state": "free"} for v in range(capacity)],
            "Occupied slot or stale lock")
    require(current["existing_tokens"] == [], "New formal token already exists")
    require(all(row["status"] == " M" and row["path"] in (MANUSCRIPTS if host == "local" else set())
                for row in current["git_status"]), "Unknown dirty or staged scope")
    require(all(same_typed(current[k], previous[k]) for k in (
        "root", "host", "platform", "logical_cpus", "python", "torch", "numpy", "pandas")),
        "Original host/dependencies changed")
    require(type(current["torch_threads"]) is int and current["torch_threads"] == 1, "Thread1 required")
    require(current["memory"]["available_bytes"] > (3 if host == "local" else 8) * 1024**3,
            "Insufficient free RAM for accepted process budget")
    busy = current["cpu_busy_percent_snapshot"]
    require(type(busy) in (int, float) and math.isfinite(busy) and 0 <= busy <= 100,
            "Finite read-only CPU competition snapshot required")


HOST_PROBE = ("import json,sys;from tools.freeze_h12_protocol import inspect_host;"
              "print(json.dumps(inspect_host(json.load(sys.stdin))))")


def probe(host, root, tokens, actors, slots):
    require(host in HOSTS and (root, slots) == HOSTS[host], "Unknown host/root/slot allocation")
    command = ["uv", "run", "--frozen", "python", "-c", HOST_PROBE]
    if host != "local":
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                   shlex.join(["uv", "--directory", root, "run", "--frozen", "python", "-c", HOST_PROBE])]
    result = subprocess.run(command, cwd=ROOT, input=json.dumps({
        "host": host, "tokens": tokens, "actors": actors, "slots": slots}).encode("utf-8"),
        capture_output=True, timeout=120)
    # Preserve original transport output before parsing, including failed probes.
    base = ROOT / "results/c2opt21"
    for suffix, data in (("stdout", result.stdout), ("stderr", result.stderr)):
        with (base / f"{tokens[0]}_freeze_{host}.{suffix}").open("xb") as stream:
            stream.write(data)
    require(result.returncode == 0, f"H12 {host} read-only probe failed; original transport logs retained")
    return json.loads(result.stdout)


def validate_tooling(current, checked):
    require(checked["accepted"] is True and type(checked["focused_tests"]) is int
            and checked["focused_tests"] == FOCUSED_TESTS and checked["returncode"] == 0
            and all(current["files"][k]["raw"] == checked["raw_sha256"][k] for k in ADDITIONS),
            "Original-host focused support qualification differs")


def validate_protection(state, record):
    protected = dict(state["protected_sha256"])
    previous = read(ROOT / "results/c2opt21/h11_implementation_record_a.json")["protected"]
    require(previous["MP_default_exception"]["new_raw"] == MP_DEFAULT,
            "Previously accepted MP exception changed")
    protected["src/config/default.yaml"] = MP_DEFAULT
    manuscripts = previous["manuscript_actual_raw"]
    require(set(manuscripts) == MANUSCRIPTS, "User manuscript scope changed")
    protected.update(manuscripts)
    require(record["qualified_local_raw_sha256"]["src/config/default.yaml"] == MP_DEFAULT
            and all(digest(ROOT / name) == expected for name, expected in protected.items()),
            "Protected file or user manuscript actual raw changed")
    return {"path": "src/config/default.yaml", "new_raw": MP_DEFAULT,
            "only_registered_change": "rollout_workers:0 and learner_workers:0",
            "original_manifest_preserved": True, "manuscript_actualraw": manuscripts}


def freeze(state_path, tokens):
    state = read(state_path)
    validate_phase(state)
    base = state_path.parent
    scope = validate_lineage(state, base)
    engineering = validate_engineering(base)
    workers = {"rollout_workers": 2, "learner_workers": 2}
    root, parent, child = [entry_config(name, workers) for name in (
        "gavdac_ea_main", "gavdac_c2_h11_parent", "gavdac_c2_h12")]
    validate_candidate(root, parent, child)
    record = read(base / "h12_implementation_record_a.json")
    require(same_typed(child, record["candidate_config"]) and same_typed(parent, record["parent_config"])
            and same_typed(state["config"], child), "Actual entry/config differs from qualified H12")
    plans = make_plans(tokens)
    validate_plans(plans)
    subprocess.run(["git", "diff", "--exit-code", CHILD, "--", "src", "tools/experiment_slot.py"],
                   cwd=ROOT, check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"],
                                   cwd=ROOT, text=True).split()[0] == head, "Push support before freeze")
    require(set(subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", head], cwd=ROOT, text=True).splitlines())
        == ADDITIONS, "Support commit must contain exactly the two independent additions")
    tooling_path = base / "h12_protocol_tool_acceptance_b.json"
    tooling = read(tooling_path)
    require(tooling["accepted"] is True and tooling["support_commit"] == head
            and set(tooling["hosts"]) == set(HOSTS), "Both formal-host focused qualifications required")
    exception = validate_protection(state, record)
    trace = revalidate_c3_trace(state, base)
    hosts = {}
    for host, (host_root, capacity) in HOSTS.items():
        current = hosts[host] = probe(host, host_root, tokens, actors_for_host(state, host), capacity)
        compare_host(read(base / f"h12_host_{host}_a.json"), current, capacity, host)
        validate_tooling(current, tooling["hosts"][host])
        require(current["git_commit"] == head, "Own idle-clean host must use exact support commit")
        for smoke in engineering["smokes"].values():
            launch = read(ROOT / smoke["path"])["launch"]
            if ("local" if launch["seed"] == 2 else "707") == host:
                require(same_typed(launch["source_sha256"], {
                    name: item["raw"] for name, item in current["files"].items()
                    if name.startswith("src/") and Path(name).suffix in {".py", ".yaml", ".csv"}}),
                    "Original smoke-qualified host training source raw changed")
    require(hosts["local"]["files"].keys() == hosts["707"]["files"].keys()
            and all(v["lf"] == hosts["707"]["files"][k]["lf"] for k, v in hosts["local"]["files"].items()),
            "Host files differ beyond accepted EOL; raw metadata remains original-host")
    for plan in plans:
        directory = f"{plan['root']}/results/{plan['token']}".replace("\\", "/")
        plan["entry_config"] = entry_config("gavdac_c2_h12", {
            **workers, "seed": plan["seed"], "unique_token": plan["token"],
            "local_results_path": directory, "env_args.result_dir": directory,
            "wandb_dir": directory + "/wandb", "use_cuda": False, "wandb_mode": "online"})
    return {
        "hypothesis": "H12", "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_before_freeze_sha256": digest(state_path), "algorithm_implementation_commit": CHILD,
        "parent_reconstruction_commit": PARENT, "preregistration_commit": PREREGISTRATION,
        "scope_amendment_commit": AMENDMENT, "historical_parent": "H03", "root": "C2_ROOT",
        "lineage_registry_sha256": LINEAGE, "rollback_commits_reverse_order": [CHILD],
        "exact_rollback_destination": PARENT, "ledger_preserving_rollback_scope": scope,
        "rollback_scope_sha256": ROLLBACK_SCOPE, "support_commit_not_economically_reverted": head,
        "engineering_acceptance_sha256": ENGINEERING, "original_C3_readonly_audit_sha256": C3_AUDIT,
        "support_tool_acceptance_sha256": digest(tooling_path), "intervention": intervention(),
        "inherited_parent_config": parent, "candidate_config": child, "run_plan": plans, "hosts": hosts,
        "auxiliary_715": "Enabled idle pending requalification;not needed or allocated",
        "baseline_C3": state["baseline_C3"], "original_C3_trace_revalidation": trace,
        "original_protected_sha256": state["protected_sha256"],
        "future_execution_protected_default_exception": exception,
        "selection": {"rows": list(range(731, 739)), "interval_joint_steps": 10000, "actor": "rl"},
        "evaluation": evaluation_protocol(), "retention": retention_protocol(),
        "current_final_gate": state["current_economic_gate"], "cpu_process_budget_per_slot": 5,
        "707_two_slot_torch_process_budget": 10, "additional_helpers": True,
        "compute_threads_per_process": 1,
        "expected_actual_counts_to_verify": {
            "joint_interactions": 6000144, "agent_transitions": 18000432, "ea_cycles": 24959,
            "critic_optimizer_steps": 124748, "actor_optimizer_steps": 62374,
            "target_updates": 124748, "finite_update_checks": 124748, "train_updates": 124748,
            "optimizer_post_step_finite_checks": {"critic": 124748, "actor": 62374}},
        "resource_limitation": "Read-only snapshot,not peak RAM or6Mthroughput guarantee",
        "limitation": "Exploratory retained reuse/cumulative lineage,not root-isolated or independent significance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs=3, default=["h12s0a", "h12s1a", "h12s2a"])
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / "results").resolve()) and not output.exists(),
            "Immutable freeze requires NEW output under results/")
    value = freeze(args.state.resolve(), args.tokens)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "tokens": [p["token"] for p in value["run_plan"]]}))


if __name__ == "__main__":
    main()
