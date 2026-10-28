"""Independently freeze H13 on retained H12. This tool never launches a job."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
# Reuse only reviewed generic entry, inventory, transport-data and C3 readers.
from tools.freeze_h11_protocol import (
    actors_for_host, check_c3_actor, digest, entry_config, inspect_host as inventory,
    read, require, revalidate_c3_trace, same_typed,
)

CHILD = "9d16922fd9565f05559977b266d9f2bf7bf6f39a"
PARENT = "8883790a9508cc6144c578bb3f6cf11571844961"
PARENT_ALGORITHM = "de78e9cf710663c010d41df7dfd0f29aa9217dc4"
ANCESTOR = "4e3aa00eacf814e3d328be781e4af8632ec0488f"
PREREGISTRATION = "8d69f790e36bee3dd6bd09a5accc0d471d65ca7f"
ENGINEERING = "334977d06fdae34f9506f4ff69c32dd030dbb6a3b8fb2f846c6b15798d6ad7d0"
C3_AUDIT = "9d3f4f819778256c5197b78e312e8e47cbecfcb4419ba3ef6fde22fe1f5913a3"
LINEAGE = "f33e30678bb226eb89a6cc22e5e9c345b5b8cf1bfa6f1d720cb72855d91113eb"
ROLLBACK = "5830ee47cb0d82908b7e226819d0ed51e715880935d0ba435652ea63ffec94ff"
MP_DEFAULT = "836837797231fd52c41881f9a7b002e842a1122b9cbf19e6f5e4570485bbcbab"
PARENT_PROTOCOL = "5da269b810ad7d503a6e3001ed4e74f9c208e79afc5d00427bdc5234e17e23db"
PARENT_RESULT = "7d9b0637da221c4e6612d2fcf8193541c06a858075a644e147a7a2e40c46e2b3"
HOSTS = {"local": (str(ROOT), 1), "707": ("/home/arch707/data/CDE", 2)}
ADDITIONS = {"tools/freeze_h13_protocol.py", "tests/test_h13_protocol.py"}
MANUSCRIPTS = {
    "paper/manuscript/sections/method.tex": "9b882e762b85407bf7e45dd8935508c299021c26bfe8377b194b88a2ee39798a",
    "paper/manuscript/sections/modeling.tex": "7fb773c1d715128a704d2c7af3f2316a06e799fd75ff237448f37a11a0c48e5f",
}
C3_COSTS = (344913.0888879933, 344550.706793143, 354488.7203334839)
PARENT_MEAN = 343315.2310756934
FOCUSED_TESTS = 26
RETIRED = (
    "actor_mixer_context_grad", "clipped_double_critic", "five_step_critic",
    "critic_layer_norm", "target_policy_smoothing", "critic_small_output_init",
    "actor_no_action_l2", "critic_center_generator_action",
)
INHERITED = [
    "Actor every2 successful Critic calls",
    "all targets every call tau.002",
    "scaled2*unitHuber maskedTD/original global valid count",
    "original singleCritic/mixer/Actorloss/history/EA/replay/rollout/physics/reward/data",
]
DELTA = "Actor fc1 -> per-example affine LayerNorm48 eps1e-5 -> ReLU"
PINS = {
    "h13_parent_review_a.json": "997eb5f061779cbf1333e976858858205d9775730a2029adb63a504a824c0487",
    "h13_preregistration_a.json": "15b51a8afbba1fb8969079c239c7100316f11c5e4b48b052fc19b4970fb8d350",
    "h13_implementation_record_a.json": "078a4d8b665dc5194e1cba852354c733e5466dfca8f32b493035a039e329ced1",
    "h13_child_rollback_scope_a.json": ROLLBACK,
    "h13_original_C3_revalidation_a.json": C3_AUDIT,
    "lineage_registry_r.json": LINEAGE,
    "h12_protocol_b.json": PARENT_PROTOCOL,
    "h12_result_a.json": PARENT_RESULT,
}


def validate_candidate(root, h03, parent, child):
    chain = (
        (h03, {**root, "name": "gavdac_c2_h11_parent",
               "actor_update_interval": 2, "target_update_interval": 1}),
        (parent, {**h03, "name": "gavdac_c2_h12", "scaled_huber_critic": True}),
        (child, {**parent, "name": "gavdac_c2_h13", "actor_feature_layer_norm": True}),
    )
    require(all(same_typed(actual, expected) for actual, expected in chain),
            "H13 requires exact root/H03/H12 inheritance and only the true boolean Norm delta")
    required = {
        "t_max": 6000000, "EA": True, "crossover_prob": 0.0, "use_cuda": False,
        "checkpoint_path": "", "policy_checkpoint": "", "evaluate": False, "load_step": 0,
        "rollout_workers": 2, "learner_workers": 2, "test_interval": 10000,
        "fixed_eval_days": 8, "evaluation_initial_soc": .35, "save_model": True,
        "save_best_policy": True, "wandb_mode": "online", "rnn_hidden_dim": 48,
        "actor_update_interval": 2, "target_update_interval": 1, "target_update_tau": .002,
    }
    require(all(same_typed(child.get(k), v) for k, v in required.items()),
            "H13 fresh CPU budget, selection, Actor2/target1/tau.002 and integer2/2 required")
    rows = child.get("validation_rows")
    require(type(rows) is list and rows in ([], list(range(731, 739)))
            and all(type(v) is int for v in rows), "Only validation731-738 is permitted")
    require(not any(k in config for config in (root, h03, parent, child) for k in RETIRED),
            "Retired flag forbidden even when false")
    require("scaled_huber_critic" not in root and "scaled_huber_critic" not in h03
            and not any("actor_feature_layer_norm" in c for c in (root, h03, parent)),
            "Huber belongs to H12; Actor Norm belongs only to H13")


def intervention():
    return {
        "root": "C2_ROOT", "parent": "H12", "ancestor": "H03",
        "inherited_mechanisms": INHERITED, "sole_delta": DELTA,
        "actor_class": "LayerNormRNNAgent", "learner_class": "ActorNormHuberGAVDACLearner",
        "worker_learner_class": "ActorNormHuberGAVDACLearner",
        "norm_width": 48, "norm_eps": 1e-5, "norm_affine": True,
        "actor_parameters": 15554, "parent_actor_parameters": 15458,
        "critic_and_mixer_parameters": 65915,
        "single_critic": True, "actor_action_l2_coefficient": .001,
        "actor_update_interval": 2, "target_update_interval": 1, "target_update_tau": .002,
        "huber_delta": 1., "huber_multiplier": 2.,
        "original_global_valid_denominator": True,
        "live_action_context_and_recurrent_gradients": True,
        "EA_complete_clone_transfer_norm_keys_mutation_skips_norm": True,
        "EA_RNG_consumption_may_change": True, "exact_resume_supported": False,
        "root_isolated_intervention_or_paired_RNG": False,
    }


def runtime_identity(config):
    """Initialization only: no optimizer update, rollout, worker or cloud run."""
    from types import SimpleNamespace
    import torch
    from learners.actor_norm_huber_learner import ActorNormHuberGAVDACLearner
    from learners.delayed_actor_huber_learner import HuberDelayedActorGAVDACLearner
    from learners.delayed_actor_learner import DelayedActorGAVDACLearner
    from modules.actor_feature_norm import LayerNormRNNAgent
    from modules.policy import RNNAgent
    import run
    with torch.random.fork_rng(devices=[]):
        actor = LayerNormRNNAgent(25, SimpleNamespace(**config, n_actions=2))
        original = RNNAgent(25, SimpleNamespace(**config, n_actions=2))
    norm = actor.lnorm
    return {
        "actor_class": type(actor).__name__,
        "learner_class": run.ActorNormHuberGAVDACLearner.__name__,
        "learner_binding_exact": run.ActorNormHuberGAVDACLearner is ActorNormHuberGAVDACLearner,
        "parent_class_exact": ActorNormHuberGAVDACLearner.__bases__ == (HuberDelayedActorGAVDACLearner,),
        "inherited_train_exact": ActorNormHuberGAVDACLearner.train is DelayedActorGAVDACLearner.train,
        "inherited_loss_exact": ActorNormHuberGAVDACLearner._td_error_loss
        is HuberDelayedActorGAVDACLearner._td_error_loss,
        "norm_class": type(norm).__name__, "norm_shape": list(norm.normalized_shape),
        "norm_eps": norm.eps, "norm_affine": norm.elementwise_affine,
        "actor_parameters": sum(p.numel() for p in actor.parameters()),
        "parent_actor_parameters": sum(p.numel() for p in original.parameters()),
        "finite_CPUfloat32": all(p.device.type == "cpu" and p.dtype == torch.float32
                                and bool(torch.isfinite(p).all()) for p in actor.parameters()),
        "norm_initial_one_zero": bool(torch.equal(norm.weight, torch.ones_like(norm.weight))
                                      and torch.equal(norm.bias, torch.zeros_like(norm.bias))),
    }


def validate_runtime(value):
    expected = {
        "actor_class": "LayerNormRNNAgent", "learner_class": "ActorNormHuberGAVDACLearner",
        "learner_binding_exact": True, "parent_class_exact": True,
        "inherited_train_exact": True, "inherited_loss_exact": True,
        "norm_class": "LayerNorm", "norm_shape": [48], "norm_eps": 1e-5, "norm_affine": True,
        "actor_parameters": 15554, "parent_actor_parameters": 15458,
        "finite_CPUfloat32": True, "norm_initial_one_zero": True,
    }
    require(same_typed(value, expected), "Actual H13 class/inheritance/Norm/CPUfloat32 mismatch")


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
    require(same_typed(value, evaluation_protocol()), "H13 final metric/selection/inventory gate drift")


def validate_retention(value):
    require(same_typed(value, retention_protocol()), "H13 independent parent retention drift")


def economic_gates(costs):
    require(len(costs) == 3 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in costs),
            "Exactly three accepted finite costs required")
    each = all(v < c for v, c in zip(costs, C3_COSTS, strict=True))
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
            and same_typed(state["H13_registration"]["final_gate"], gate), "H13 final state gate drift")
    validate_retention(state["H13_registration"]["retention_gate"])


def make_plans(tokens):
    require(len(tokens) == 3 and all(type(t) is str for t in tokens) and len(set(tokens)) == 3
            and all(re.fullmatch(r"h13s[a-z0-9_]{1,25}", t) for t in tokens),
            "Three unique short safe NEW H13 tokens required")
    return [{
        "seed": seed, "host": "local" if seed == 2 else "707",
        "root": HOSTS["local" if seed == 2 else "707"][0],
        "slot": 0 if seed == 2 else seed, "capacity": 1 if seed == 2 else 2,
        "token": tokens[seed], "config_name": "gavdac_c2_h13", "joint_budget": 6000000,
        "rollout_workers": 2, "learner_workers": 2, "threads": 1, "status": "frozen_not_launched",
    } for seed in (0, 1, 2)]


def validate_plans(plans):
    require(len(plans) == 3 and all(type(p.get(k)) is int for p in plans for k in (
        "seed", "slot", "capacity", "joint_budget", "rollout_workers", "learner_workers", "threads")),
        "Three exact integer original-host seed/slot/budget/worker assignments required")
    require(same_typed(plans, make_plans([p["token"] for p in plans])), "H13 allocation drift")


def validate_phase(state):
    require(state["active_hypothesis"] == "H13"
            and same_typed(state["active_runs"], []) and same_typed(state["run_plan"], [])
            and state["formal_protocol"] is None
            and state["status"] == "h13_engineering_accepted_pending_independent_freeze"
            and state["H13_development"]["enabled_host_engineering_accepted"] is True,
            "Only idle, engineering-accepted, unfrozen H13 may freeze")
    require(type(state["authorized_slot_capacity"]) is int and state["authorized_slot_capacity"] == 4
            and state["execution_slots"]["arch715_0"]["enabled"] is True
            and state["execution_slots"]["arch715_0"]["state"] == "enabled_idle_pending_requalification",
            "715 must stay enabled/unqualified/unallocated")
    checklist = state["H13_checklist"]
    require(all(checklist.get(k) is True for k in (
        "preregistered_and_normally_pushed", "parent_and_primary_review", "implementation",
        "local_engineering", "remote_engineering", "original_C3_readonly_audit"))
        and all(checklist.get(k) is False for k in (
            "independent_freeze", "formal_training_started", "formal_training_accepted",
            "primary_evaluation", "final_economic_gate", "parent_retention_gate")),
        "H13 prerequisite checklist incomplete or later phase already reached")


def validate_rollback(implementation, commits, destination, scope, registered):
    require(implementation == CHILD and same_typed(commits, [CHILD]) and destination == PARENT,
            "Only registered H13 child reverses to retained H12, never H03/root/support")
    require(same_typed(scope, registered) and scope["child_reverse_commit_order"] == [CHILD]
            and scope["exact_parent_code_destination"] == PARENT
            and len(scope["registered_child_code_paths"]) == 12
            and len(set(scope["registered_child_code_paths"])) == 12
            and scope["algorithm_commit_excludes_ledger_docs"] is True
            and scope["retained_H12_and_support_are_not_child_delta"] is True
            and set(scope["preserve_current_ledger_document_paths"]) == {
                "docs/c2_hypotheses/H13.md", "docs/c2_hypotheses/README.md"},
            "Exact H13 child-only rollback scope changed")


def validate_registration(state):
    reg = state["H13_registration"]
    expected = {
        "hypothesis": "H13", "preregistration_commit": PREREGISTRATION, "normal_push_verified": True,
        "parent_node_id": "H12", "root_node_id": "C2_ROOT", "exact_parent_code_commit": PARENT,
        "parent_algorithm_commit": PARENT_ALGORITHM, "parent_protocol_sha256": PARENT_PROTOCOL,
        "parent_result_sha256": PARENT_RESULT, "inherited_mechanisms": INHERITED, "sole_new_delta": DELTA,
        "implementation_commit": CHILD, "child_delta_commits": [CHILD],
        "rollback_destination_commit": PARENT, "never_rollback_retained_H12_or_support": True,
        "formal_seed_labels": [0, 1, 2], "physical_joint_budget_including_EA": 6000000,
        "formal_host_slots": [["707", 0, 0], ["707", 1, 1], ["local", 0, 2]],
        "workers": {"rollout_workers": 2, "learner_workers": 2, "threads": 1},
        "715_required": False, "formal_protocol": None,
    }
    require(all(same_typed(reg.get(k), v) for k, v in expected.items()), "H13 registration identity drift")
    lineage = state["lineage"]
    require(state["research_lineage_policy"]["version"] == "one_increment_per_recorded_parent_v1"
            and all(lineage.get(k) == v for k, v in {
                "working_checkout_node_id": "H13", "selected_parent_node_id": "H12",
                "active_child_node_id": "H13", "new_working_parent_commit": PARENT,
                "registry_sha256": LINEAGE, "registry": "results/c2opt21/lineage_registry_r.json",
            }.items()), "H13 exact parent/root/lineage drift")
    validate_state_gates(state)


def validate_lineage(state, base):
    validate_registration(state)
    for name, expected in PINS.items():
        require(digest(base / name) == expected, "Pinned H13 evidence raw changed: " + name)
    scope = read(base / "h13_child_rollback_scope_a.json")
    validate_rollback(state["implementation_commit"], state["rollback_commits"],
                      state["rollback_destination_commit"], scope, state["H13_registration"]["child_rollback_scope"])
    changed = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", CHILD], cwd=ROOT, text=True).splitlines()
    require(set(changed) == set(scope["registered_child_code_paths"]), "H13 Git delta outside registered scope")
    for before, after in ((ANCESTOR, PARENT_ALGORITHM), (PARENT_ALGORITHM, PARENT),
                          (PARENT, PREREGISTRATION), (PREREGISTRATION, CHILD)):
        subprocess.run(["git", "merge-base", "--is-ancestor", before, after], cwd=ROOT, check=True)
    registry = read(base / "lineage_registry_r.json")
    parent = next(n for n in registry["nodes"] if n["node_id"] == "H12")
    require(parent["status"] == "closed_retained_target_not_met", "H12 must stay retained, not rejected")
    return scope


def validate_engineering(base):
    path = base / "h13_engineering_acceptance_a.json"
    require(digest(path) == ENGINEERING, "H13 engineering acceptance raw changed")
    value = read(path)
    require(value["accepted"] is True and value["implementation_commit"] == CHILD
            and value["exact_parent_code_destination"] == PARENT
            and value["formal_hosts"] == ["local", "707"] and value["formal_slots"] == 3
            and value["qualified_file_count"] == 151
            and set(value["smokes"]) == {"h13ls0b", "h13rs0a", "h13rs1a"},
            "H13 original formal-host engineering incomplete")
    for name, expected in value["input_sha256"].items():
        require(digest(base / name) == expected, "Bound H13 engineering input changed: " + name)
    for token, item in value["smokes"].items():
        require(digest(ROOT / item["path"]) == item["sha256"], "Original H13 smoke bytes changed")
        smoke = read(ROOT / item["path"])
        completed = smoke["completion"]
        expected = {
            "lineage_role": "H13_child_of_retained_H12", "actor_update_interval": 2,
            "target_update_interval": 1, "target_update_tau": .002, "scaled_huber_critic": True,
            "huber_delta": 1., "huber_multiplier": 2., "critic_optimizer_steps": 8,
            "actor_optimizer_steps": 4, "target_updates": 8, "finite_update_checks": 8,
            "train_updates": 8, "optimizer_post_step_finite_checks": {"critic": 8, "actor": 4},
            "exact_resume_supported": False, "joint_interactions": 12624,
            "agent_transitions": 37872, "ea_cycles": 11, "actor_feature_layer_norm": True,
            "actor_feature_norm_width": 48, "actor_feature_norm_eps": 1e-5, "actor_feature_norm_affine": True,
        }
        require(smoke["accepted"] is True and smoke["launch"]["git_commit"] == CHILD
                and smoke["launch"]["config"] == "gavdac_c2_h13"
                and same_typed(completed, expected) and same_typed(item["completion"], completed),
                "H13 smoke class/Norm/inheritance/counters drift: " + token)
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
    result["configs"] = {name: entry_config(
        name, {} if name == "gavdac_mappo" else {"rollout_workers": 2, "learner_workers": 2})
                         for name in ("gavdac_ea_main", "gavdac_c2_h11_parent", "gavdac_c2_h12",
                                      "gavdac_c2_h13", "gavdac_mappo")}
    result["runtime_identity"] = runtime_identity(result["configs"]["gavdac_c2_h13"])
    return result


def compare_host(previous, current, capacity, host):
    require(host in HOSTS and type(capacity) is int and capacity == HOSTS[host][1],
            "Only original formal hosts/capacity permitted")
    require(len(previous["files"]) == 151 and len(current["files"]) == 153
            and all(same_typed(current["files"].get(k), v) for k, v in previous["files"].items())
            and set(current["files"]) - set(previous["files"]) == ADDITIONS,
            "Original151 actual raw/LF changed or additions exceed two independent support files")
    require(same_typed(current["numeric_data"], previous["numeric_data"])
            and same_typed(current["configs"], previous["configs"]), "Qualified numeric data/configs changed")
    require(all(same_typed(current[k], previous[k]) for k in (
        "root", "host", "platform", "logical_cpus", "python", "torch", "numpy", "pandas")),
        "Qualified host/dependencies changed")
    require(same_typed(current["slots"], [{"slot": i, "state": "free"} for i in range(capacity)])
            and same_typed(current["existing_tokens"], []), "Slot occupied/stale or old formal token exists")
    require(all(row["status"] == " M" and row["path"] in (MANUSCRIPTS if host == "local" else {})
                for row in current["git_status"]), "Unknown dirty/staged scope")
    require(type(current["logical_cpus"]) is int and current["logical_cpus"] >= 5 * capacity
            and type(current["torch_threads"]) is int and current["torch_threads"] == 1,
            "Insufficient CPU process budget or non-thread1")
    available = current["memory"]["available_bytes"]
    require(type(available) is int and available > (3 if host == "local" else 8) * 1024**3,
            "Insufficient available RAM")
    busy = current["cpu_busy_percent_snapshot"]
    require(type(busy) in (int, float) and math.isfinite(busy) and 0 <= busy <= 100,
            "Finite CPU competition snapshot required")
    validate_runtime(current["runtime_identity"])


HOST_PROBE = ("import json,sys;from tools.freeze_h13_protocol import inspect_host;"
              "print(json.dumps(inspect_host(json.load(sys.stdin))))")


def probe(host, root, tokens, actors, slots):
    require(host in HOSTS and same_typed((root, slots), HOSTS[host]), "Unknown formal host/root/capacity")
    command = ["uv", "run", "--frozen", "python", "-c", HOST_PROBE]
    if host != "local":
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                   shlex.join(["uv", "--directory", root, "run", "--frozen", "python", "-c", HOST_PROBE])]
    base = ROOT / "results/c2opt21"
    outputs = [base / f"{tokens[0]}_freeze_{host}.{suffix}" for suffix in ("stdout", "stderr")]
    require(not any(p.exists() for p in outputs), "Use NEW transport output paths")
    result = subprocess.run(command, cwd=ROOT, input=json.dumps({
        "host": host, "tokens": tokens, "actors": actors, "slots": slots,
    }).encode("utf-8"), capture_output=True, timeout=120)
    for path, data in zip(outputs, (result.stdout, result.stderr), strict=True):
        with path.open("xb") as stream:
            stream.write(data)
    require(result.returncode == 0, "H13 read-only probe failed; original transport bytes retained")
    return json.loads(result.stdout)


def validate_tooling(current, checked):
    require(checked["accepted"] is True and type(checked["focused_tests"]) is int
            and checked["focused_tests"] == FOCUSED_TESTS and type(checked["returncode"]) is int
            and checked["returncode"] == 0
            and set(checked["raw_sha256"]) == ADDITIONS
            and all(current["files"][p]["raw"] == checked["raw_sha256"][p] for p in ADDITIONS),
            "Exact original-host focused support qualification required")


def validate_protection(state, record):
    protected = {**state["protected_sha256"], "src/config/default.yaml": MP_DEFAULT, **MANUSCRIPTS}
    require(same_typed(record["manuscript_actualraw"], MANUSCRIPTS)
            and record["files"]["src/config/default.yaml"]["raw"] == MP_DEFAULT
            and all(digest(ROOT / p) == raw for p, raw in protected.items()),
            "Original protection/registered MP exception/manuscript actual raw drift")
    return {"original_manifest_preserved": True, "only_registered_MP_default_actualraw": MP_DEFAULT,
            "manuscript_actualraw": MANUSCRIPTS}


def freeze(state_path, tokens):
    state_path = Path(state_path)
    state = read(state_path)
    validate_phase(state)
    plans = make_plans(tokens)
    validate_plans(plans)
    base = state_path.parent
    scope = validate_lineage(state, base)
    engineering = validate_engineering(base)
    names = ("gavdac_ea_main", "gavdac_c2_h11_parent", "gavdac_c2_h12", "gavdac_c2_h13")
    configs = [entry_config(name, {"rollout_workers": 2, "learner_workers": 2}) for name in names]
    validate_candidate(*configs)
    require(same_typed(configs[-1], state["config"])
            and same_typed(configs[-1], state["H13_registration"]["proposed_entry_config"]),
            "Actual entry differs from registered H13")
    record = read(base / "h13_implementation_record_a.json")
    require(len(record["files"]) == 151 and all(
        digest(ROOT / p) == v["raw"] for p, v in record["files"].items()), "Local qualified151 raw changed")
    subprocess.run(["git", "diff", "--exit-code", CHILD, "--", "src", "tools/experiment_slot.py"],
                   cwd=ROOT, capture_output=True, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"],
                                   cwd=ROOT, text=True).split()[0] == head, "Push support normally before freeze")
    require(set(subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", head], cwd=ROOT, text=True).splitlines())
        == ADDITIONS, "Independent support commit must contain exactly the two additions")
    tooling_path = base / "h13_protocol_tool_acceptance_a.json"
    tooling = read(tooling_path)
    require(tooling["accepted"] is True and tooling["support_commit"] == head
            and set(tooling["hosts"]) == set(HOSTS), "Both original formal hosts need focused support acceptance")
    protection = validate_protection(state, record)
    trace = revalidate_c3_trace(state, base)
    hosts = {}
    for host, (host_root, capacity) in HOSTS.items():
        current = hosts[host] = probe(host, host_root, tokens, actors_for_host(state, host), capacity)
        compare_host(read(base / f"h13_host_{host}_a.json"), current, capacity, host)
        validate_tooling(current, tooling["hosts"][host])
        require(current["git_commit"] == head, "Original own-idle host must use exact qualified support")
        for item in engineering["smokes"].values():
            launch = read(ROOT / item["path"])["launch"]
            if ("local" if launch["seed"] == 2 else "707") == host:
                require(same_typed(launch["source_sha256"], {
                    p: v["raw"] for p, v in current["files"].items()
                    if p.startswith("src/") and Path(p).suffix in {".py", ".yaml", ".csv"}}),
                    "Original smoke-qualified training source raw changed")
    require(hosts["local"]["files"].keys() == hosts["707"]["files"].keys()
            and all(v["lf"] == hosts["707"]["files"][p]["lf"] for p, v in hosts["local"]["files"].items())
            and same_typed(hosts["local"]["numeric_data"], hosts["707"]["numeric_data"])
            and same_typed(hosts["local"]["configs"], hosts["707"]["configs"]),
            "Formal hosts differ beyond accepted raw EOL differences")
    for plan in plans:
        directory = f"{plan['root']}/results/{plan['token']}".replace("\\", "/")
        plan["entry_config"] = entry_config("gavdac_c2_h13", {
            "rollout_workers": 2, "learner_workers": 2, "seed": plan["seed"],
            "unique_token": plan["token"], "local_results_path": directory,
            "env_args.result_dir": directory, "wandb_dir": directory + "/wandb",
            "use_cuda": False, "wandb_mode": "online",
        })
    return {
        "hypothesis": "H13", "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_before_freeze_sha256": digest(state_path), "algorithm_implementation_commit": CHILD,
        "exact_parent_commit": PARENT, "parent_algorithm_commit": PARENT_ALGORITHM,
        "ancestral_H03_commit": ANCESTOR, "preregistration_commit": PREREGISTRATION,
        "root": "C2_ROOT", "lineage_registry_sha256": LINEAGE,
        "rollback_commits_reverse_order": [CHILD], "exact_rollback_destination": PARENT,
        "ledger_preserving_rollback_scope": scope, "rollback_scope_sha256": ROLLBACK,
        "support_commit_not_economically_reverted": head, "engineering_acceptance_sha256": ENGINEERING,
        "original_C3_readonly_audit_sha256": C3_AUDIT,
        "support_tool_acceptance_sha256": digest(tooling_path), "intervention": intervention(),
        "inherited_parent_config": configs[2], "candidate_config": configs[3],
        "run_plan": plans, "hosts": hosts, "baseline_C3": state["baseline_C3"],
        "original_C3_trace_revalidation": trace, "original_protected_sha256": state["protected_sha256"],
        "future_execution_protected_default_exception": protection,
        "selection": {"rows": list(range(731, 739)), "interval_joint_steps": 10000, "actor": "rl"},
        "evaluation": evaluation_protocol(), "retention": retention_protocol(),
        "current_final_gate": state["current_economic_gate"],
        "auxiliary_715": "Enabled idle pending requalification; not needed or allocated",
        "cpu_process_budget_per_slot": 5, "707_two_slot_torch_process_budget": 10,
        "additional_helpers": True, "compute_threads_per_process": 1,
        "expected_actual_counts_to_verify": {
            "joint_interactions": 6000144, "agent_transitions": 18000432, "ea_cycles": 24959,
            "critic_optimizer_steps": 124748, "actor_optimizer_steps": 62374, "target_updates": 124748,
            "finite_update_checks": 124748, "train_updates": 124748,
            "optimizer_post_step_finite_checks": {"critic": 124748, "actor": 62374},
        },
        "resource_limitation": "Snapshot, not peak RAM or6Mthroughput guarantee",
        "limitation": "Exploratory retained reuse/nonpairedRNG/cumulative lineage; not independent significance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs=3, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / "results").resolve()) and not output.exists(),
            "Immutable freeze output must be NEW under results/")
    value = freeze(args.state.resolve(), args.tokens)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "tokens": [p["token"] for p in value["run_plan"]]}))


if __name__ == "__main__":
    main()
