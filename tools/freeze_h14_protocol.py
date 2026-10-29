"""Freeze only the registered H14 signed-mixer child; never start a job."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.freeze_h11_protocol import (
    actors_for_host, check_c3_actor, digest, entry_config, inspect_host as inventory,
    read, require, revalidate_c3_trace, same_typed,
)

CHILD = "b9b6c7f04f60ada2df318c922e5adaa8e1677b0d"
PARENT = "8883790a9508cc6144c578bb3f6cf11571844961"
PARENT_ALGORITHM = "de78e9cf710663c010d41df7dfd0f29aa9217dc4"
ANCESTOR = "4e3aa00eacf814e3d328be781e4af8632ec0488f"
PREREGISTRATION = "b3583a9e86eec8906649a84fce3bf8c321948b4d"
ENGINEERING_HEAD = "4ccbbfb81208033dfa2b36dc1f9880037a5da9c3"
ENGINEERING = "8cf018fa51db34c8f9e043529e318aa295df63164adc4adcdd2af8212923345e"
C3_AUDIT = "25162cb6eb7dbf6d110e505eed8d97c7994acf35d3dfdd26afd11c7ca1a5e52c"
LINEAGE = "3478e41e29dceb89e06c8dca95b9e6db252a72a36d503d635e5990a306eb6859"
ROLLBACK = "35c972e4acf43ed713bb4e059a6d704ee630a45cc4c18c69a00821d7eb50953a"
PARENT_PROTOCOL = "5da269b810ad7d503a6e3001ed4e74f9c208e79afc5d00427bdc5234e17e23db"
PARENT_RESULT = "7d9b0637da221c4e6612d2fcf8193541c06a858075a644e147a7a2e40c46e2b3"
MP_DEFAULT = "836837797231fd52c41881f9a7b002e842a1122b9cbf19e6f5e4570485bbcbab"
HOSTS = {"707": ("/home/arch707/data/CDE", 2), "local": (str(ROOT), 1)}
ADDITIONS = {"tools/freeze_h14_protocol.py", "tests/test_h14_protocol.py"}
MANUSCRIPTS = {
    "paper/manuscript/sections/method.tex": "9b882e762b85407bf7e45dd8935508c299021c26bfe8377b194b88a2ee39798a",
    "paper/manuscript/sections/modeling.tex": "7fb773c1d715128a704d2c7af3f2316a06e799fd75ff237448f37a11a0c48e5f",
}
CONFIG_NAMES = ("gavdac_ea_main", "gavdac_c2_h11_parent", "gavdac_c2_h12", "gavdac_c2_h14", "gavdac_mappo")
RETIRED = (
    "actor_feature_layer_norm", "actor_mixer_context_grad", "clipped_double_critic",
    "five_step_critic", "critic_layer_norm", "target_policy_smoothing",
    "critic_small_output_init", "actor_no_action_l2", "critic_center_generator_action",
)
INHERITED = [
    "Actor every2 successful Critic calls",
    "all targets everycall tau.002",
    "scaled2*unitHuber maskedTD/original globalvalidcount",
    "singleCritic/mixer/originalActorloss/history/EA/replay/rollout/physics/reward/data",
]
DELTA = "Effective0.1*sigmoid ->0.1*(2*sigmoid-1) forGINw1/w2/GNNscalarwout"
C3_COSTS = (344913.0888879933, 344550.706793143, 354488.7203334839)
PARENT_MEAN = 343315.2310756934
FOCUSED_TESTS = 30
PINS = {
    "h14_parent_review_a.json": "ab6ac49cb237602a7d827c8ce0e1671c4192a509db2ae98b139bbcec0de00992",
    "h14_preregistration_a.json": "d48eae688f7468a49936269488d804b5d8a5e5fa3ce989768577e871cbc76097",
    "h14_implementation_record_a.json": "61da297c54b8d55a9042f725073d59f891f45c0101b9d8e7bb7628c37c8f2e11",
    "h14_child_rollback_scope_a.json": ROLLBACK,
    "h14_engineering_acceptance_a.json": ENGINEERING,
    "h14_original_C3_revalidation_a.json": C3_AUDIT,
    "lineage_registry_x.json": LINEAGE,
    "h12_protocol_b.json": PARENT_PROTOCOL, "h12_result_a.json": PARENT_RESULT,
}
CHILD_PATHS = {
    "src/config/algs/README.md", "src/config/algs/gavdac_c2_h14.yaml",
    "src/modules/graph_mixer.py", "src/modules/graph/gnn.py", "src/modules/graph/gin_layer.py",
    "src/learners/gavdac_learner.py", "src/learners/signed_mixer_huber_learner.py",
    "src/learners/gradient_pool.py", "src/run.py", "tools/experiment_slot.py",
    "tests/test_h11_parent.py", "tests/test_h14_signed_mixer.py",
    "tests/test_h14_launch.py", "tests/test_repository_scope.py",
}


def validate_candidate(root, h03, h12, h14):
    require(same_typed(h03, dict(root, name="gavdac_c2_h11_parent",
                                actor_update_interval=2, target_update_interval=1)),
            "Actual H03 must differ from root only by registered Actor2")
    require(same_typed(h12, dict(h03, name="gavdac_c2_h12", scaled_huber_critic=True)),
            "Actual H12 must inherit H03 and only scaled unit Huber")
    require(same_typed(h14, dict(h12, name="gavdac_c2_h14", signed_mixer_hyperweights=True)),
            "H14 permits only the true boolean signed hyperweight delta")
    fixed = {
        "t_max": 6000000, "use_cuda": False, "EA": True, "crossover_prob": 0.0,
        "rollout_workers": 2, "learner_workers": 2, "actor_update_interval": 2,
        "target_update_interval": 1, "target_update_tau": .002, "rnn_hidden_dim": 48,
        "mixer_weight_scale": .1, "checkpoint_path": "", "policy_checkpoint": "",
        "evaluate": False, "load_step": 0, "wandb_mode": "online", "save_model": True,
        "save_best_policy": True, "test_interval": 10000, "fixed_eval_days": 8,
        "evaluation_initial_soc": .35,
    }
    require(all(same_typed(h14.get(k), v) for k, v in fixed.items()),
            "H14 CPU budget, workers, inherited cadence, scale or selection drift")
    require(all(k not in c for c in (root, h03, h12, h14) for k in RETIRED),
            "Retired flags are forbidden even when false")
    require(all("signed_mixer_hyperweights" not in c for c in (root, h03, h12))
            and "scaled_huber_critic" not in root and "scaled_huber_critic" not in h03,
            "Mechanisms must first appear at their exact lineage node")
    rows = h14.get("validation_rows")
    require(type(rows) is list and rows in ([], list(range(731, 739)))
            and all(type(n) is int for n in rows), "Validation731-738 only")


def runtime_identity(config):
    """Construct networks only; preserve RNG and do not create workers or updates."""
    from copy import deepcopy
    from types import SimpleNamespace
    import numpy
    import torch
    from gymnasium.spaces import Box
    from controllers.cqmix_controller import CQMixMAC
    from learners import gradient_pool
    from learners.delayed_actor_huber_learner import HuberDelayedActorGAVDACLearner
    from learners.delayed_actor_learner import DelayedActorGAVDACLearner
    from learners.signed_mixer_huber_learner import SignedMixerHuberGAVDACLearner
    from modules.policy import RNNAgent
    import run

    args = SimpleNamespace(**dict(config, n_agents=3, n_actions=2, state_shape=60,
                                  action_spaces=[Box(-numpy.ones(2, dtype="float32"),
                                                     numpy.ones(2, dtype="float32")) for _ in range(3)]))
    scheme = {"obs": {"vshape": 20, "group": "agents"}, "actions": {"vshape": (2,), "group": "agents"}}
    with torch.random.fork_rng(devices=[]):
        before = torch.get_rng_state()
        mac = CQMixMAC(scheme, {"agents": 3}, args)
        child = SignedMixerHuberGAVDACLearner(mac, scheme, None, args)
        child_rng = torch.get_rng_state()
        torch.set_rng_state(before)
        parent_args = deepcopy(args)
        del parent_args.signed_mixer_hyperweights
        parent_mac = CQMixMAC(scheme, {"agents": 3}, parent_args)
        parent = HuberDelayedActorGAVDACLearner(parent_mac, scheme, None, parent_args)
        same_rng = torch.equal(child_rng, torch.get_rng_state())
        initial = all(torch.equal(a, b) for a, b in zip(
            child.agent_params + child.critic_params, parent.agent_params + parent.critic_params, strict=True))
    tree = ast.parse(inspect.getsource(gradient_pool._initialize))
    selected = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Name)
                and n.test.func.id == "signed_mixer_enabled"]
    worker_binding = len(selected) == 1 and len(selected[0].body) == 1 and isinstance(
        selected[0].body[0], ast.Assign) and isinstance(selected[0].body[0].value, ast.Call) and isinstance(
        selected[0].body[0].value.func, ast.Name) and (
        selected[0].body[0].value.func.id == "SignedMixerHuberGAVDACLearner")
    gnn = child.mixer.mixing_GNN
    return {
        "actor_class": type(mac.agent).__name__, "actor_binding_exact": type(mac.agent) is RNNAgent,
        "learner_class": type(child).__name__,
        "run_binding_exact": run.SignedMixerHuberGAVDACLearner is SignedMixerHuberGAVDACLearner,
        "worker_explicit_signed_branch": worker_binding,
        "parent_class_exact": SignedMixerHuberGAVDACLearner.__bases__ == (HuberDelayedActorGAVDACLearner,),
        "inherited_train_exact": SignedMixerHuberGAVDACLearner.train is DelayedActorGAVDACLearner.train,
        "inherited_TD_loss_exact": SignedMixerHuberGAVDACLearner._td_error_loss
        is HuberDelayedActorGAVDACLearner._td_error_loss,
        "mixer_operations": [gnn.weights_operation, *[layer.weights_operation for layer in gnn.layers]],
        "mixer_scales": [gnn.weight_scale, *[layer.weight_scale for layer in gnn.layers]],
        "actor_parameters": sum(p.numel() for p in child.agent_params),
        "critic_mixer_parameters": sum(p.numel() for p in child.critic_params),
        "initial_parameter_tensors_equal_parent": initial, "initial_rng_equal_parent": bool(same_rng),
        "all_parameters_CPUfloat32_finite": all(
            p.device.type == "cpu" and p.dtype == torch.float32 and bool(torch.isfinite(p).all())
            for p in child.agent_params + child.critic_params),
    }


def runtime_contract():
    return {
        "actor_class": "RNNAgent", "actor_binding_exact": True,
        "learner_class": "SignedMixerHuberGAVDACLearner", "run_binding_exact": True,
        "worker_explicit_signed_branch": True, "parent_class_exact": True,
        "inherited_train_exact": True, "inherited_TD_loss_exact": True,
        "mixer_operations": ["signed_sigmoid", "signed_sigmoid"], "mixer_scales": [.1, .1],
        "actor_parameters": 15458, "critic_mixer_parameters": 65915,
        "initial_parameter_tensors_equal_parent": True, "initial_rng_equal_parent": True,
        "all_parameters_CPUfloat32_finite": True,
    }


def validate_runtime(value):
    require(same_typed(value, runtime_contract()), "Actual H14 class, map, inheritance or initialization differs")


def evaluation_protocol():
    return {
        "rows": list(range(913, 1095)), "mode": "continuous", "initial_soc": .35,
        "terminal_soc": "free within device bounds", "actor": "validation-selected rl",
        "hidden_and_previous_action_reset": "daily", "expected_agent_hours": 13104,
        "primary_metric": "economic_cost", "supplementary_metric": "full_cost",
        "retrospective_retained_development": True,
        "final_gate_version": "each_seed_better_mean_saving_3pct_v2",
        "each_seed_strictly_below_corresponding_C3": True, "per_seed_minimum_saving_fraction": 0.0,
        "mean_rule": "1-mean(candidate_cost)/mean(C3_cost)>=0.03",
        "mean_minimum_saving_fraction": .03, "mean_of_percentage_savings": False,
    }


def retention_protocol():
    return {
        "all_seed_cost_strictly_below_C3": True, "mean_cost_strictly_below_historical_parent": PARENT_MEAN,
        "per_seed_parent_threshold": None, "all_parent_seed_regressions_disclosed": True,
        "not_final_success_or_independent_significance": True,
    }


def validate_evaluation(value):
    require(same_typed(value, evaluation_protocol()), "Frozen primary evaluation drift")


def validate_retention(value):
    require(same_typed(value, retention_protocol()), "Independent H14 retention drift")


def economic_gates(costs):
    require(len(costs) == 3 and all(type(c) in (int, float) and math.isfinite(c) and c >= 0 for c in costs),
            "Three accepted finite nonnegative seed costs required")
    each = all(c < fixed for c, fixed in zip(costs, C3_COSTS, strict=True))
    saving = 1 - sum(costs) / sum(C3_COSTS)
    return {"each_seed_better": each, "ratio_of_means_saving": saving,
            "final_target_passed": each and saving >= .03,
            "parent_retention_passed": each and sum(costs) / 3 < PARENT_MEAN}


def validate_state_gates(state):
    gate = state["current_economic_gate"]
    required = {
        "version": "each_seed_better_mean_saving_3pct_v2", "metric": "economic_cost",
        "seed_labels": [0, 1, 2], "per_seed_rule": "candidate_cost < corresponding_C3_cost",
        "per_seed_minimum_saving_fraction": 0.0, "per_seed_comparison": "strictly_greater",
        "mean_rule": "1-mean(candidate_cost)/mean(C3_cost) >=0.03",
        "mean_minimum_saving_fraction": .03, "arithmetic_mean_of_percentage_savings": False,
        "fixed_C3_costs": list(C3_COSTS),
    }
    require(all(same_typed(gate.get(k), v) for k, v in required.items())
            and same_typed(state["H14_registration"]["final_gate"], gate), "Final gate cannot be loosened")
    validate_retention(state["H14_registration"]["retention_gate"])


def make_plans(tokens):
    require(type(tokens) is list and len(tokens) == 3 and all(
        type(t) is str and re.fullmatch(r"h14s[a-z0-9_]{1,25}", t) for t in tokens)
        and len(set(tokens)) == 3, "Three unique safe NEW H14 formal tokens required")
    plans = []
    for seed, token in enumerate(tokens):
        host, slot = ("local", 0) if seed == 2 else ("707", seed)
        plans.append(dict(seed=seed, token=token, host=host, root=HOSTS[host][0], slot=slot,
                          capacity=HOSTS[host][1], config_name="gavdac_c2_h14", joint_budget=6000000,
                          rollout_workers=2, learner_workers=2, threads=1, status="frozen_not_launched"))
    return plans


def validate_plans(plans):
    require(type(plans) is list and len(plans) == 3 and same_typed(
        plans, make_plans([p["token"] for p in plans])), "Exact original seed-host-slot/process budget required")


def validate_phase(state):
    require(state["active_hypothesis"] == "H14"
            and state["status"] == "h14_engineering_accepted_pending_independent_freeze"
            and same_typed(state["active_runs"], []) and same_typed(state["run_plan"], [])
            and state["formal_protocol"] is None
            and state["H14_development"]["enabled_host_engineering_accepted"] is True,
            "Only idle engineering-accepted unfrozen H14 can freeze")
    checklist = state["H14_checklist"]
    before = ("preregistered_and_normally_pushed", "parent_and_primary_review", "implementation",
              "local_engineering", "remote_engineering", "original_C3_readonly_audit")
    after = ("independent_freeze", "formal_training_started", "formal_training_accepted",
             "primary_evaluation", "final_economic_gate", "parent_retention_gate")
    require(all(checklist.get(k) is True for k in before) and all(checklist.get(k) is False for k in after),
            "H14 prerequisite or later-phase checklist mismatch")
    require(same_typed(state["authorized_slot_capacity"], 4)
            and state["execution_slots"]["arch715_0"]["enabled"] is True
            and state["execution_slots"]["arch715_0"]["state"] == "enabled_idle_pending_requalification",
            "Auxiliary715 must stay unallocated and pending requalification")


def validate_rollback(implementation, commits, destination, scope, registered):
    require(implementation == CHILD and same_typed(commits, [CHILD]) and destination == PARENT,
            "Rollback only registered H14 child to exact retained H12")
    require(same_typed(scope, registered) and same_typed(scope["child_reverse_commit_order"], [CHILD])
            and scope["exact_parent_code_destination"] == PARENT
            and len(scope["registered_child_code_paths"]) == 14
            and set(scope["registered_child_code_paths"]) == CHILD_PATHS
            and scope["algorithm_commit_excludes_ledger_docs"] is True
            and scope["retained_H12_and_support_are_not_child_delta"] is True
            and set(scope["preserve_current_ledger_document_paths"]) == {
                "docs/c2_hypotheses/H14.md", "docs/c2_hypotheses/README.md"},
            "Registered child-only rollback paths or parent/support protection drift")


def validate_registration(state):
    reg = state["H14_registration"]
    identity = dict(
        hypothesis="H14", preregistration_commit=PREREGISTRATION, normal_push_verified=True,
        parent_node_id="H12", root_node_id="C2_ROOT", exact_parent_code_commit=PARENT,
        parent_algorithm_commit=PARENT_ALGORITHM, parent_protocol_sha256=PARENT_PROTOCOL,
        parent_result_sha256=PARENT_RESULT, inherited_mechanisms=INHERITED, sole_new_delta=DELTA,
        implementation_commit=CHILD, child_delta_commits=[CHILD], rollback_destination_commit=PARENT,
        never_rollback_retained_H12_or_support=True, formal_seed_labels=[0, 1, 2],
        physical_joint_budget_including_EA=6000000,
        formal_host_slots=[["707", 0, 0], ["707", 1, 1], ["local", 0, 2]],
        workers={"rollout_workers": 2, "learner_workers": 2, "threads": 1}, formal_protocol=None,
        config_name="gavdac_c2_h14", parent_config_name="gavdac_c2_h12", **{"715_required": False})
    require(all(same_typed(reg.get(k), v) for k, v in identity.items()), "H14 registration drift")
    require(state["research_lineage_policy"]["version"] == "one_increment_per_recorded_parent_v1"
            and all(state["lineage"].get(k) == v for k, v in dict(
                working_checkout_node_id="H14", selected_parent_node_id="H12", active_child_node_id="H14",
                new_working_parent_commit=PARENT, registry="results/c2opt21/lineage_registry_x.json",
                registry_sha256=LINEAGE).items()), "H14 exact parent/root/lineage drift")
    validate_state_gates(state)


def validate_lineage(state, base):
    validate_registration(state)
    for name, expected in PINS.items():
        require(digest(base / name) == expected, "Pinned H14 evidence changed: " + name)
    scope = read(base / "h14_child_rollback_scope_a.json")
    validate_rollback(state["implementation_commit"], state["rollback_commits"],
                      state["rollback_destination_commit"], scope, state["H14_registration"]["child_rollback_scope"])
    changed = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", CHILD], cwd=ROOT, text=True).splitlines()
    require(set(changed) == CHILD_PATHS, "Child Git commit must contain exactly registered14 paths")
    for before, after in ((ANCESTOR, PARENT_ALGORITHM), (PARENT_ALGORITHM, PARENT),
                          (PARENT, PREREGISTRATION), (PREREGISTRATION, CHILD)):
        subprocess.run(["git", "merge-base", "--is-ancestor", before, after], cwd=ROOT, check=True)
    parent = next(n for n in read(base / "lineage_registry_x.json")["nodes"] if n["node_id"] == "H12")
    require(parent["status"] == "closed_retained_target_not_met", "H12 must remain retained, not rejected")
    return scope


def completion_contract():
    return dict(lineage_role="H14_signed_mixer_child_of_retained_H12", signed_mixer_hyperweights=True,
                weights_operation="signed_sigmoid", mixer_weight_scale=.1, actor_update_interval=2,
                target_update_interval=1, target_update_tau=.002, scaled_huber_critic=True,
                huber_delta=1., huber_multiplier=2., critic_optimizer_steps=8, actor_optimizer_steps=4,
                target_updates=8, finite_update_checks=8, train_updates=8,
                optimizer_post_step_finite_checks={"critic": 8, "actor": 4}, exact_resume_supported=False,
                joint_interactions=12624, agent_transitions=37872, ea_cycles=11)


def validate_engineering(base):
    path = base / "h14_engineering_acceptance_a.json"
    require(digest(path) == ENGINEERING, "H14 engineering acceptance raw changed")
    value = read(path)
    require(value["accepted"] is True and value["implementation_commit"] == CHILD
            and value["exact_parent_code_destination"] == PARENT and value["formal_hosts"] == ["local", "707"]
            and same_typed(value["formal_slots"], 3) and same_typed(value["qualified_file_count"], 152)
            and set(value["smokes"]) == {"h14ls0a", "h14rs0a", "h14rs1a"}, "H14 engineering incomplete")
    for name, expected in value["input_sha256"].items():
        require(digest(base / name) == expected, "Original engineering input bytes changed: " + name)
    for token, item in value["smokes"].items():
        require(digest(ROOT / item["path"]) == item["sha256"], "Original smoke bytes changed: " + token)
        smoke = read(ROOT / item["path"])
        require(smoke["accepted"] is True and smoke["launch"]["git_commit"] == CHILD
                and smoke["launch"]["config"] == "gavdac_c2_h14"
                and same_typed(smoke["completion"], completion_contract())
                and same_typed(item["completion"], smoke["completion"]), "H14 smoke map/inheritance/count drift")
    return value


def inspect_host(payload):
    value = inventory(payload)
    import numpy
    import pandas
    import gymnasium
    value["gymnasium"] = gymnasium.__version__
    value["numeric_data"] = {}
    for name in ("load", "pv", "price"):
        array = pandas.read_csv(ROOT / "src/envs/data" / (name + ".csv"), header=None).to_numpy(dtype="float64")
        require(bool(numpy.isfinite(array).all()), "Nonfinite qualified input data")
        value["numeric_data"][name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                                          sha256=hashlib.sha256(array.tobytes()).hexdigest())
    value["configs"] = {n: entry_config(n, {} if n == "gavdac_mappo" else {
        "rollout_workers": 2, "learner_workers": 2}) for n in CONFIG_NAMES}
    validate_candidate(*(value["configs"][n] for n in CONFIG_NAMES[:4]))
    value["runtime_identity"] = runtime_identity(value["configs"]["gavdac_c2_h14"])
    return value


def compare_host(previous, current, capacity, host):
    require(host in HOSTS and same_typed(capacity, HOSTS[host][1]), "Original formal host/capacity only")
    require(len(previous["files"]) == 152 and len(current["files"]) == 154
            and set(current["files"]) - set(previous["files"]) == ADDITIONS
            and all(same_typed(current["files"].get(p), v) for p, v in previous["files"].items()),
            "Only two independent additions; all original152 actualraw/LF must remain exact")
    require(same_typed(current["configs"], previous["configs"])
            and same_typed(current["numeric_data"], previous["numeric_data"]),
            "Actual entry config or numeric data changed")
    require(all(same_typed(previous[k], current[k]) for k in (
        "root", "host", "platform", "logical_cpus", "python", "torch", "numpy", "pandas", "gymnasium")),
        "Qualified host/dependency identity drift")
    require(same_typed(current["slots"], [{"slot": i, "state": "free"} for i in range(capacity)])
            and same_typed(current["existing_tokens"], []), "Occupied/stale lock or token already exists")
    require(all(r["status"] == " M" and host == "local" and r["path"] in MANUSCRIPTS
                for r in current["git_status"]), "Unknown dirty or staged scope")
    require(type(current["logical_cpus"]) is int and current["logical_cpus"] >= 5 * capacity
            and same_typed(current["torch_threads"], 1), "CPU process/thread budget insufficient")
    available, busy = current["memory"]["available_bytes"], current["cpu_busy_percent_snapshot"]
    require(type(available) is int and available > (3 if host == "local" else 8) * 1024**3,
            "Insufficient currently available RAM")
    require(type(busy) in (int, float) and math.isfinite(busy) and 0 <= busy <= 100,
            "Finite CPU competition snapshot required")
    validate_runtime(current["runtime_identity"])


def probe(host, root, tokens, actors, capacity):
    require(host in HOSTS and same_typed((root, capacity), HOSTS[host]), "Unknown host/root/capacity")
    base = ROOT / "results/c2opt21"
    outputs = [base / f"{tokens[0]}_freeze_{host}.{s}" for s in ("stdout", "stderr")]
    require(not any(p.exists() for p in outputs), "Freeze transport outputs must be NEW")
    code = "import json,sys;from tools.freeze_h14_protocol import inspect_host;print(json.dumps(inspect_host(json.load(sys.stdin))))"
    command = ["uv", "run", "--frozen", "python", "-c", code]
    if host != "local":
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                   shlex.join(["uv", "--directory", root, *command[1:]])]
    result = subprocess.run(command, cwd=ROOT, input=json.dumps(
        dict(host=host, slots=capacity, tokens=tokens, actors=actors)).encode("utf-8"),
        capture_output=True, timeout=120)
    for p, raw in zip(outputs, (result.stdout, result.stderr), strict=True):
        with p.open("xb") as stream:
            stream.write(raw)
    require(result.returncode == 0, "Read-only H14 host probe failed; raw transport preserved, not free")
    return json.loads(result.stdout)


def validate_tooling(host, checked):
    require(checked["accepted"] is True and same_typed(checked["focused_tests"], FOCUSED_TESTS)
            and same_typed(checked["returncode"], 0) and set(checked["raw_sha256"]) == ADDITIONS
            and all(host["files"][p]["raw"] == checked["raw_sha256"][p] for p in ADDITIONS),
            "Original-host focused support count/exit/bytes not qualified")


def validate_protection(state, record):
    protected = {**state["protected_sha256"], "src/config/default.yaml": MP_DEFAULT, **MANUSCRIPTS}
    require(same_typed(record["manuscript_actualraw"], MANUSCRIPTS)
            and record["files"]["src/config/default.yaml"]["raw"] == MP_DEFAULT
            and all(digest(ROOT / p) == v for p, v in protected.items()), "Protected actualraw drift")
    return dict(original_manifest_preserved=True, only_registered_MP_default_actualraw=MP_DEFAULT,
                manuscript_actualraw=MANUSCRIPTS)


def intervention():
    return dict(root="C2_ROOT", parent="H12", ancestor="H03", inherited_mechanisms=INHERITED,
                sole_delta=DELTA, actor_class="RNNAgent", learner_class="SignedMixerHuberGAVDACLearner",
                worker_learner_class="SignedMixerHuberGAVDACLearner",
                actor_parameters=15458, critic_and_mixer_parameters=65915, single_critic=True,
                mixer_weight_scale=.1, actor_update_interval=2, target_update_interval=1,
                target_update_tau=.002, huber_delta=1., huber_multiplier=2.,
                original_global_valid_denominator=True, actor_action_l2_coefficient=.001,
                live_action_context_and_recurrent_gradients=True, complete_shared_Actor_EA_unchanged=True,
                initial_tensors_and_RNG_equal_parent=True, initial_function_and_map_derivative_change=True,
                pure_sign_causality_pairedRNG_or_root_isolated=False, exact_resume_supported=False)


def freeze(state_path, tokens):
    state_path = Path(state_path)
    state = read(state_path)
    validate_phase(state)
    plans = make_plans(tokens)
    validate_plans(plans)
    base = state_path.parent
    scope = validate_lineage(state, base)
    engineering = validate_engineering(base)
    configs = [entry_config(n, {"rollout_workers": 2, "learner_workers": 2}) for n in CONFIG_NAMES[:4]]
    validate_candidate(*configs)
    require(same_typed(configs[-1], state["config"]) and same_typed(
        configs[-1], state["H14_registration"]["proposed_entry_config"]), "Actual H14 differs from registration")
    record = read(base / "h14_implementation_record_a.json")
    require(len(record["files"]) == 152 and all(digest(ROOT / p) == v["raw"]
            for p, v in record["files"].items()), "Local qualified152 actualraw changed")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"],
                                   cwd=ROOT, text=True).split()[0] == head, "Normal support push required")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", ENGINEERING_HEAD, head], cwd=ROOT, text=True).splitlines()
    require(set(changed) == ADDITIONS, "Only independent two-file support since accepted engineering")
    tooling = read(base / "h14_protocol_tool_acceptance_a.json")
    require(tooling["accepted"] is True and tooling["support_commit"] == head
            and set(tooling["hosts"]) == set(HOSTS), "Both original formal hosts require support qualification")
    protection = validate_protection(state, record)
    traces = revalidate_c3_trace(state, base)
    hosts = {}
    for host, (root, capacity) in HOSTS.items():
        current = hosts[host] = probe(host, root, tokens, actors_for_host(state, host), capacity)
        compare_host(read(base / f"h14_host_{host}_a.json"), current, capacity, host)
        validate_tooling(current, tooling["hosts"][host])
        require(current["git_commit"] == head, "Own idle-clean formal host must use exact qualified support")
        for item in engineering["smokes"].values():
            launch = read(ROOT / item["path"])["launch"]
            if ("local" if launch["seed"] == 2 else "707") == host:
                require(same_typed(launch["source_sha256"], {
                    p: v["raw"] for p, v in current["files"].items()
                    if p.startswith("src/") and Path(p).suffix in {".py", ".yaml", ".csv"}}),
                    "Original smoke-qualified training actualraw changed")
    require(hosts["local"]["files"].keys() == hosts["707"]["files"].keys()
            and all(v["lf"] == hosts["707"]["files"][p]["lf"] for p, v in hosts["local"]["files"].items())
            and same_typed(hosts["local"]["configs"], hosts["707"]["configs"])
            and same_typed(hosts["local"]["numeric_data"], hosts["707"]["numeric_data"]),
            "Original formal hosts differ beyond preserved raw EOL differences")
    for plan in plans:
        directory = f"{plan['root']}/results/{plan['token']}".replace("\\", "/")
        plan["entry_config"] = entry_config("gavdac_c2_h14", dict(
            rollout_workers=2, learner_workers=2, seed=plan["seed"], unique_token=plan["token"],
            local_results_path=directory, wandb_dir=directory + "/wandb", use_cuda=False,
            wandb_mode="online", **{"env_args.result_dir": directory}))
    return dict(
        hypothesis="H14", created_utc=datetime.now(timezone.utc).isoformat(),
        state_before_freeze_sha256=digest(state_path), algorithm_implementation_commit=CHILD,
        exact_parent_commit=PARENT, parent_algorithm_commit=PARENT_ALGORITHM,
        ancestral_H03_commit=ANCESTOR, preregistration_commit=PREREGISTRATION,
        root="C2_ROOT", lineage_registry_sha256=LINEAGE, rollback_commits_reverse_order=[CHILD],
        exact_rollback_destination=PARENT, ledger_preserving_rollback_scope=scope, rollback_scope_sha256=ROLLBACK,
        support_commit_not_economically_reverted=head, engineering_acceptance_sha256=ENGINEERING,
        original_C3_readonly_audit_sha256=C3_AUDIT,
        support_tool_acceptance_sha256=digest(base / "h14_protocol_tool_acceptance_a.json"),
        intervention=intervention(), inherited_parent_config=configs[2], candidate_config=configs[3],
        run_plan=plans, hosts=hosts, baseline_C3=state["baseline_C3"], original_C3_trace_revalidation=traces,
        original_protected_sha256=state["protected_sha256"], future_execution_protected_default_exception=protection,
        selection=dict(rows=list(range(731, 739)), interval_joint_steps=10000, actor="rl"),
        evaluation=evaluation_protocol(), retention=retention_protocol(), current_final_gate=state["current_economic_gate"],
        auxiliary_715="Enabled idle pending requalification; not needed or contacted",
        cpu_process_budget_per_slot=5, **{"707_two_slot_torch_process_budget": 10},
        additional_helpers=True, compute_threads_per_process=1,
        expected_actual_counts_to_verify=dict(
            joint_interactions=6000144, agent_transitions=18000432, ea_cycles=24959,
            critic_optimizer_steps=124748, actor_optimizer_steps=62374, target_updates=124748,
            finite_update_checks=124748, train_updates=124748,
            optimizer_post_step_finite_checks={"critic": 124748, "actor": 62374}),
        resource_limitation="Snapshot, not peak RAM or6Mthroughput guarantee",
        limitation="Exploratory retained reuse/cumulativeActor2+Huber+signedmixer, not pairedRNG or independent significance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", nargs=3, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / "results").resolve()) and not output.exists(),
            "Immutable protocol requires a NEW path under results/")
    value = freeze(args.state.resolve(), args.tokens)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), sha256=digest(output), tokens=[p["token"] for p in value["run_plan"]])))


if __name__ == "__main__":
    main()
