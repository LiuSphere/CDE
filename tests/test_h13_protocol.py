"""H13 freeze guards survive child rollback: actual ancestors, synthetic child."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import freeze_h13_protocol as protocol


class H13ProtocolTest(unittest.TestCase):
    def entries(self):
        workers = {"rollout_workers": 2, "learner_workers": 2}
        configs = [protocol.entry_config(name, workers) for name in
                   ("gavdac_ea_main", "gavdac_c2_h11_parent", "gavdac_c2_h12")]
        return [*configs, {**configs[-1], "name": "gavdac_c2_h13", "actor_feature_layer_norm": True}]

    def runtime(self):
        return {
            "actor_class": "LayerNormRNNAgent", "learner_class": "ActorNormHuberGAVDACLearner",
            "learner_binding_exact": True, "parent_class_exact": True,
            "inherited_train_exact": True, "inherited_loss_exact": True,
            "norm_class": "LayerNorm", "norm_shape": [48], "norm_eps": 1e-5, "norm_affine": True,
            "actor_parameters": 15554, "parent_actor_parameters": 15458,
            "finite_CPUfloat32": True, "norm_initial_one_zero": True,
        }

    def phase(self):
        return {
            "active_hypothesis": "H13", "active_runs": [], "run_plan": [], "formal_protocol": None,
            "status": "h13_engineering_accepted_pending_independent_freeze",
            "H13_development": {"enabled_host_engineering_accepted": True},
            "authorized_slot_capacity": 4,
            "execution_slots": {"arch715_0": {"enabled": True, "state": "enabled_idle_pending_requalification"}},
            "H13_checklist": {
                **dict.fromkeys(("preregistered_and_normally_pushed", "parent_and_primary_review",
                                "implementation", "local_engineering", "remote_engineering",
                                "original_C3_readonly_audit"), True),
                **dict.fromkeys(("independent_freeze", "formal_training_started", "formal_training_accepted",
                                "primary_evaluation", "final_economic_gate", "parent_retention_gate"), False),
            },
        }

    def host(self, host="local"):
        capacity = protocol.HOSTS[host][1]
        previous = {
            "files": {f"qualified/{i}.py": {"raw": str(i), "lf": str(i)} for i in range(151)},
            "root": protocol.HOSTS[host][0], "host": host, "platform": "qualified", "logical_cpus": 12,
            "python": "qualified", "torch": "qualified", "numpy": "qualified", "pandas": "qualified",
            "numeric_data": {"load": {"shape": [1095, 24], "dtype": "float64", "sha256": "numeric"}},
            "configs": {"root": {"name": "root"}, "child": {"actor_feature_layer_norm": True}},
        }
        current = {
            **copy.deepcopy(previous), "slots": [{"slot": i, "state": "free"} for i in range(capacity)],
            "existing_tokens": [], "git_status": [], "torch_threads": 1,
            "memory": {"available_bytes": 10 * 1024**3}, "cpu_busy_percent_snapshot": 10.,
            "runtime_identity": self.runtime(),
        }
        current["files"].update({p: {"raw": p, "lf": p} for p in protocol.ADDITIONS})
        return previous, current

    def scope(self):
        return {
            "child_reverse_commit_order": [protocol.CHILD], "exact_parent_code_destination": protocol.PARENT,
            "registered_child_code_paths": [f"child/{i}.py" for i in range(12)],
            "algorithm_commit_excludes_ledger_docs": True, "retained_H12_and_support_are_not_child_delta": True,
            "preserve_current_ledger_document_paths": ["docs/c2_hypotheses/H13.md", "docs/c2_hypotheses/README.md"],
        }

    def registration(self):
        gate = {
            "version": "each_seed_better_mean_saving_3pct_v2", "metric": "economic_cost",
            "seed_labels": [0, 1, 2], "per_seed_rule": "candidate_cost < corresponding_C3_cost",
            "per_seed_minimum_saving_fraction": 0.0, "per_seed_comparison": "strictly_greater",
            "mean_rule": "1-mean(candidate_cost)/mean(C3_cost) >=0.03",
            "mean_minimum_saving_fraction": .03, "arithmetic_mean_of_percentage_savings": False,
            "fixed_C3_costs": list(protocol.C3_COSTS),
        }
        return {
            "current_economic_gate": gate,
            "H13_registration": {
                "hypothesis": "H13", "preregistration_commit": protocol.PREREGISTRATION,
                "normal_push_verified": True, "parent_node_id": "H12", "root_node_id": "C2_ROOT",
                "exact_parent_code_commit": protocol.PARENT, "parent_algorithm_commit": protocol.PARENT_ALGORITHM,
                "parent_protocol_sha256": protocol.PARENT_PROTOCOL, "parent_result_sha256": protocol.PARENT_RESULT,
                "inherited_mechanisms": copy.deepcopy(protocol.INHERITED), "sole_new_delta": protocol.DELTA,
                "implementation_commit": protocol.CHILD, "child_delta_commits": [protocol.CHILD],
                "rollback_destination_commit": protocol.PARENT, "never_rollback_retained_H12_or_support": True,
                "formal_seed_labels": [0, 1, 2], "physical_joint_budget_including_EA": 6000000,
                "formal_host_slots": [["707", 0, 0], ["707", 1, 1], ["local", 0, 2]],
                "workers": {"rollout_workers": 2, "learner_workers": 2, "threads": 1},
                "715_required": False, "formal_protocol": None,
                "final_gate": copy.deepcopy(gate), "retention_gate": protocol.retention_protocol(),
            },
            "research_lineage_policy": {"version": "one_increment_per_recorded_parent_v1"},
            "lineage": {
                "working_checkout_node_id": "H13", "selected_parent_node_id": "H12",
                "active_child_node_id": "H13", "new_working_parent_commit": protocol.PARENT,
                "registry_sha256": protocol.LINEAGE, "registry": "results/c2opt21/lineage_registry_r.json",
            },
        }

    def test_actual_ancestor_entries_and_only_synthetic_norm_delta(self):
        configs = self.entries()
        protocol.validate_candidate(*configs)
        for key, value in (("lr", .001), ("name", "gavdac_c2_h06"), ("learner", "GAVDACLearner")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_candidate(*configs[:3], {**configs[3], key: value})
        with self.assertRaises(ValueError):
            protocol.validate_candidate(configs[0], configs[1], {**configs[2], "lr": .001}, configs[3])

    def test_true_boolean_norm_and_huber_inheritance_only(self):
        configs = self.entries()
        for key in ("actor_feature_layer_norm", "scaled_huber_critic"):
            for value in (False, 1, 1., "true", None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    protocol.validate_candidate(*configs[:3], {**configs[3], key: value})
        for index in range(3):
            bad = copy.deepcopy(configs)
            for config in bad:
                config["actor_feature_layer_norm"] = True
            with self.assertRaises(ValueError):
                protocol.validate_candidate(*bad)

    def test_all_retired_flags_even_false_rejected(self):
        configs = self.entries()
        for flag in protocol.RETIRED:
            for value in (True, False, None):
                bad = [{**config, flag: value} for config in configs]
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    protocol.validate_candidate(*bad)

    def test_original_cadence_norm_width_budget_selection_and_workers(self):
        configs = self.entries()
        for key, value in (
            ("target_update_interval", 2), ("actor_update_interval", 1), ("target_update_tau", .004),
            ("rnn_hidden_dim", 64), ("t_max", 12500), ("t_max", 6000000.),
            ("learner_workers", True), ("learner_workers", 2.), ("rollout_workers", 0),
            ("use_cuda", True), ("checkpoint_path", "old"), ("policy_checkpoint", "old"),
            ("load_step", 100), ("evaluate", True), ("wandb_mode", "offline"),
            ("save_model", False), ("save_best_policy", False), ("test_interval", 1000),
            ("fixed_eval_days", 182), ("evaluation_initial_soc", .8),
            ("validation_rows", [913]), ("validation_rows", list(map(float, range(731, 739)))),
        ):
            bad = [{**c, key: value} for c in configs]
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                protocol.validate_candidate(*bad)

    def test_actual_class_contract_rejects_wrong_binding_types_and_norm(self):
        value = self.runtime()
        protocol.validate_runtime(value)
        for key, bad in (
            ("actor_class", "RNNAgent"), ("learner_class", "HuberDelayedActorGAVDACLearner"),
            ("learner_binding_exact", False), ("inherited_train_exact", False),
            ("inherited_loss_exact", False), ("parent_class_exact", False),
            ("norm_shape", [48.]), ("norm_eps", .001), ("norm_affine", 1),
            ("actor_parameters", 15458), ("parent_actor_parameters", 15554),
            ("finite_CPUfloat32", False), ("norm_initial_one_zero", False),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_runtime({**value, key: bad})

    def test_original_three_seed_host_allocation_and_typed_processes(self):
        plans = protocol.make_plans(["h13s0a", "h13s1a", "h13s2a"])
        protocol.validate_plans(plans)
        for key, value in (
            ("host", "715"), ("slot", 1), ("seed", 2), ("capacity", True),
            ("joint_budget", 6000000.), ("rollout_workers", 4), ("learner_workers", True),
            ("threads", 2), ("config_name", "gavdac_c2_h12"),
        ):
            bad = copy.deepcopy(plans)
            bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_plans(bad)

    def test_tokens_new_namespace_safe_unique_short(self):
        for tokens in (
            ["h13s0a"] * 3, ["h13s0a", "../bad", "h13s2a"],
            ["h12s0a", "h13s1a", "h13s2a"], ["h13s0a", "h13s1a"],
            ["h13s0a", None, "h13s2a"], ["h13s0a", "h13s1a", "h13s" + "x" * 26],
        ):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                protocol.make_plans(tokens)

    def test_evaluation_actor_metric_physical_inventory_and_strict_gate(self):
        protocol.validate_evaluation(protocol.evaluation_protocol())
        for key, value in (
            ("rows", [913]), ("mode", "daily_reset"), ("actor", "ea"), ("initial_soc", .8),
            ("terminal_soc", "cyclic"), ("hidden_and_previous_action_reset", "never"),
            ("primary_metric", "full_cost"), ("expected_agent_hours", 13104.),
            ("per_seed_minimum_saving_fraction", .03), ("mean_minimum_saving_fraction", .01),
            ("mean_of_percentage_savings", True), ("each_seed_strictly_below_corresponding_C3", 1),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_evaluation({**protocol.evaluation_protocol(), key: value})

    def test_independent_retention_no_per_seed_parent_gate(self):
        protocol.validate_retention(protocol.retention_protocol())
        for key, value in (
            ("mean_cost_strictly_below_historical_parent", 343419.27471629175),
            ("per_seed_parent_threshold", 0), ("all_parent_seed_regressions_disclosed", False),
            ("all_seed_cost_strictly_below_C3", False),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_retention({**protocol.retention_protocol(), key: value})
        costs = [344100., 342000., 343700.]
        self.assertTrue(protocol.economic_gates(costs)["parent_retention_passed"])
        self.assertFalse(protocol.economic_gates(costs)["final_target_passed"])
        self.assertFalse(protocol.economic_gates([protocol.PARENT_MEAN] * 3)["parent_retention_passed"])

    def test_final_strict_each_not_each_three_percent(self):
        costs = [c * r for c, r in zip(protocol.C3_COSTS, (.99, .95, .95))]
        self.assertTrue(protocol.economic_gates(costs)["final_target_passed"])
        for excess in (0, 1):
            costs[0] = protocol.C3_COSTS[0] + excess
            self.assertFalse(protocol.economic_gates(costs)["final_target_passed"])
        for bad in ([1, 2], [True, 2, 3], [-1, 2, 3], [float("nan"), 2, 3], [float("inf"), 2, 3]):
            with self.assertRaises(ValueError):
                protocol.economic_gates(bad)

    def test_ratio_of_means_not_mean_percentages(self):
        savings = [.045, .045, .0001]
        self.assertGreater(sum(savings) / 3, .03)
        self.assertFalse(protocol.economic_gates(
            [c * (1 - s) for c, s in zip(protocol.C3_COSTS, savings)])["final_target_passed"])

    def test_phase_idle_unfrozen_all_prerequisites_before_any_probe(self):
        state = self.phase()
        protocol.validate_phase(state)
        for key, value in (
            ("active_hypothesis", "H12"), ("active_runs", ["active"]), ("run_plan", ["old"]),
            ("formal_protocol", "old"), ("formal_protocol", 0), ("status", "registered"),
            ("authorized_slot_capacity", 3), ("authorized_slot_capacity", 4.),
            ("H13_development", {"enabled_host_engineering_accepted": False}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_phase({**state, key: value})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({**state, "active_runs": ["active"]}), encoding="utf-8")
            with patch.object(protocol, "probe") as probe, self.assertRaises(ValueError):
                protocol.freeze(path, ["h13s0a", "h13s1a", "h13s2a"])
            probe.assert_not_called()

    def test_prerequisite_checklist_and_auxiliary_exclusion(self):
        state = self.phase()
        for key, value in state["H13_checklist"].items():
            bad = copy.deepcopy(state)
            bad["H13_checklist"][key] = not value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_phase(bad)
        bad = copy.deepcopy(state)
        bad["execution_slots"]["arch715_0"]["state"] = "ready_idle"
        with self.assertRaises(ValueError):
            protocol.validate_phase(bad)

    def test_registration_exact_parent_root_inheritance_and_lineage(self):
        state = self.registration()
        protocol.validate_registration(state)
        for key, value in (
            ("parent_node_id", "H03"), ("exact_parent_code_commit", protocol.ANCESTOR),
            ("root_node_id", "H12"), ("child_delta_commits", [protocol.PARENT]),
            ("sole_new_delta", "target2"), ("inherited_mechanisms", []), ("715_required", True),
            ("physical_joint_budget_including_EA", 6000000.), ("parent_result_sha256", "wrong"),
            ("formal_protocol", "old"), ("never_rollback_retained_H12_or_support", False),
        ):
            bad = copy.deepcopy(state)
            bad["H13_registration"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_registration(bad)
        for key in state["lineage"]:
            bad = copy.deepcopy(state)
            bad["lineage"][key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_registration(bad)

    def test_current_state_gates_cannot_be_loosened_in_both_copies(self):
        state = self.registration()
        for key, value in (
            ("metric", "full_cost"), ("fixed_C3_costs", [400000.] * 3),
            ("mean_minimum_saving_fraction", .01), ("per_seed_minimum_saving_fraction", .03),
            ("arithmetic_mean_of_percentage_savings", True), ("seed_labels", [0, 1, 2.]),
        ):
            bad = copy.deepcopy(state)
            bad["current_economic_gate"][key] = value
            bad["H13_registration"]["final_gate"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_state_gates(bad)

    def test_rollback_never_parent_support_or_ledger(self):
        scope = self.scope()
        protocol.validate_rollback(protocol.CHILD, [protocol.CHILD], protocol.PARENT, scope, copy.deepcopy(scope))
        for commits, destination in (
            ([protocol.CHILD, "support"], protocol.PARENT), ([protocol.PARENT], protocol.PARENT),
            ([protocol.CHILD], protocol.ANCESTOR), ([protocol.CHILD], "C2_ROOT"),
        ):
            with self.assertRaises(ValueError):
                protocol.validate_rollback(protocol.CHILD, commits, destination, scope, scope)
        for key, value in (
            ("registered_child_code_paths", ["unknown"]),
            ("retained_H12_and_support_are_not_child_delta", False),
            ("preserve_current_ledger_document_paths", []),
        ):
            with self.assertRaises(ValueError):
                protocol.validate_rollback(protocol.CHILD, [protocol.CHILD], protocol.PARENT,
                                           {**scope, key: value}, scope)

    def test_exact151_raw_LF_only_two_support_additions_and_numeric_config(self):
        previous, current = self.host()
        protocol.compare_host(previous, current, 1, "local")
        for label in ("raw", "extra", "missing", "numeric", "config"):
            bad = copy.deepcopy(current)
            if label == "raw":
                bad["files"]["qualified/0.py"]["raw"] = "different-raw-same-LF"
            elif label == "extra":
                bad["files"]["unknown.py"] = {"raw": "extra", "lf": "extra"}
            elif label == "missing":
                bad["files"].pop(next(iter(protocol.ADDITIONS)))
            elif label == "numeric":
                bad["numeric_data"]["load"]["shape"] = [1, 24]
            else:
                bad["configs"]["child"]["actor_feature_layer_norm"] = False
            with self.subTest(label=label), self.assertRaises(ValueError):
                protocol.compare_host(previous, bad, 1, "local")

    def test_slot_stale_occupied_token_dirty_and_staged_scope(self):
        for host in protocol.HOSTS:
            previous, current = self.host(host)
            protocol.compare_host(previous, current, protocol.HOSTS[host][1], host)
            for key, value in (
                ("slots", [{"slot": 0, "state": "stale_lock"}]),
                ("slots", [{"slot": 0, "state": "occupied"}]), ("existing_tokens", ["h13s0a"]),
                ("git_status", [{"status": "??", "path": "unknown"}]),
                ("git_status", [{"status": "M ", "path": next(iter(protocol.MANUSCRIPTS))}]),
            ):
                with self.subTest(host=host, key=key), self.assertRaises(ValueError):
                    protocol.compare_host(previous, {**current, key: value}, protocol.HOSTS[host][1], host)
        previous, current = self.host()
        current["git_status"] = [{"status": " M", "path": p} for p in protocol.MANUSCRIPTS]
        protocol.compare_host(previous, current, 1, "local")

    def test_resource_dependencies_thread_and_aux_host_guards(self):
        previous, current = self.host()
        for key in ("root", "host", "platform", "python", "torch", "numpy", "pandas"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.compare_host(previous, {**current, key: "changed"}, 1, "local")
        for value in (True, 1., 0, 2):
            with self.assertRaises(ValueError):
                protocol.compare_host(previous, {**current, "torch_threads": value}, 1, "local")
        for value in (float("nan"), -1, 101, True):
            with self.assertRaises(ValueError):
                protocol.compare_host(previous, {**current, "cpu_busy_percent_snapshot": value}, 1, "local")
        with self.assertRaises(ValueError):
            protocol.compare_host(previous, {**current, "memory": {"available_bytes": 3 * 1024**3}}, 1, "local")
        previous["logical_cpus"] = current["logical_cpus"] = 4
        with self.assertRaises(ValueError):
            protocol.compare_host(previous, current, 1, "local")
        with patch.object(protocol.subprocess, "run") as ssh, self.assertRaises(ValueError):
            protocol.probe("715", "/unused", [], [], 1)
        ssh.assert_not_called()

    def test_focused_qualification_binds_both_files_count_exit_raw(self):
        _, current = self.host()
        checked = {"accepted": True, "focused_tests": protocol.FOCUSED_TESTS, "returncode": 0,
                   "raw_sha256": {p: p for p in protocol.ADDITIONS}}
        protocol.validate_tooling(current, checked)
        for key, value in (
            ("focused_tests", 0), ("focused_tests", float(protocol.FOCUSED_TESTS)),
            ("accepted", False), ("returncode", True), ("returncode", 1),
            ("raw_sha256", dict.fromkeys(protocol.ADDITIONS, "wrong")),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_tooling(current, {**checked, key: value})

    def test_transport_binary_stdin_original_bytes_before_json_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "results/c2opt21"
            base.mkdir(parents=True)
            result = SimpleNamespace(returncode=0, stdout=b'{"ok":true}\n', stderr=b"original\r\n")
            with patch.object(protocol, "ROOT", root), patch.object(
                    protocol.subprocess, "run", return_value=result) as run:
                value = protocol.probe("local", protocol.HOSTS["local"][0], ["h13s0z"], [], 1)
                self.assertIsInstance(run.call_args.kwargs["input"], bytes)
                self.assertEqual(value, {"ok": True})
                self.assertEqual((base / "h13s0z_freeze_local.stdout").read_bytes(), result.stdout)
                self.assertEqual((base / "h13s0z_freeze_local.stderr").read_bytes(), result.stderr)
                with self.assertRaises(ValueError):
                    protocol.probe("local", protocol.HOSTS["local"][0], ["h13s0z"], [], 1)
                self.assertEqual(run.call_count, 1)
            result = SimpleNamespace(returncode=0, stdout=b"not-json", stderr=b"original-error")
            with patch.object(protocol, "ROOT", root), patch.object(
                    protocol.subprocess, "run", return_value=result), self.assertRaises(json.JSONDecodeError):
                protocol.probe("local", protocol.HOSTS["local"][0], ["h13s1z"], [], 1)
            self.assertEqual((base / "h13s1z_freeze_local.stderr").read_bytes(), b"original-error")

    def test_engineering_hash_and_pinned_evidence_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "h13_engineering_acceptance_a.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "engineering acceptance raw"):
                protocol.validate_engineering(base)
            state = self.registration()
            with patch.object(protocol, "PINS", {"pin.json": "original"}):
                (base / "pin.json").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Pinned H13"):
                    protocol.validate_lineage(state, base)
            (base / "h01_protocol_b.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "original C3 freeze"):
                protocol.revalidate_c3_trace({"baseline_C3": {}}, base)

    def test_engineering_positive_and_wrong_norm_or_target_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "results"
            base.mkdir()
            completion = {
                "lineage_role": "H13_child_of_retained_H12", "actor_update_interval": 2,
                "target_update_interval": 1, "target_update_tau": .002, "scaled_huber_critic": True,
                "huber_delta": 1., "huber_multiplier": 2., "critic_optimizer_steps": 8,
                "actor_optimizer_steps": 4, "target_updates": 8, "finite_update_checks": 8,
                "train_updates": 8, "optimizer_post_step_finite_checks": {"critic": 8, "actor": 4},
                "exact_resume_supported": False, "joint_interactions": 12624,
                "agent_transitions": 37872, "ea_cycles": 11, "actor_feature_layer_norm": True,
                "actor_feature_norm_width": 48, "actor_feature_norm_eps": 1e-5, "actor_feature_norm_affine": True,
            }
            value = {
                "accepted": True, "implementation_commit": protocol.CHILD,
                "exact_parent_code_destination": protocol.PARENT, "formal_hosts": ["local", "707"],
                "formal_slots": 3, "qualified_file_count": 151, "input_sha256": {}, "smokes": {},
            }
            for token in ("h13ls0b", "h13rs0a", "h13rs1a"):
                path = base / (token + ".json")
                path.write_text(json.dumps({
                    "accepted": True, "completion": completion,
                    "launch": {"git_commit": protocol.CHILD, "config": "gavdac_c2_h13"},
                }), encoding="utf-8")
                value["smokes"][token] = {
                    "path": path.relative_to(root).as_posix(), "sha256": protocol.digest(path),
                    "completion": copy.deepcopy(completion),
                }
            path = base / "h13_engineering_acceptance_a.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with patch.object(protocol, "ROOT", root), patch.object(protocol, "ENGINEERING", protocol.digest(path)):
                self.assertTrue(protocol.validate_engineering(base)["accepted"])
            for key, changed in (("actor_feature_norm_eps", .001), ("target_update_interval", 2),
                                 ("optimizer_post_step_finite_checks", {"critic": 7, "actor": 4})):
                smoke_path = root / value["smokes"]["h13ls0b"]["path"]
                smoke = protocol.read(smoke_path)
                smoke["completion"] = {**completion, key: changed}
                smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
                value["smokes"]["h13ls0b"]["sha256"] = protocol.digest(smoke_path)
                path.write_text(json.dumps(value), encoding="utf-8")
                with patch.object(protocol, "ROOT", root), patch.object(
                        protocol, "ENGINEERING", protocol.digest(path)), self.assertRaises(ValueError):
                    protocol.validate_engineering(base)

    def test_protected_manifest_mp_exception_and_manuscripts_remain_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = {"src/config/default.yaml": b"zero-workers", "src/envs/maenv.py": b"original\r\n"}
            content.update({p: p.encode() for p in protocol.MANUSCRIPTS})
            for name, raw in content.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            manuscripts = {p: protocol.digest(root / p) for p in protocol.MANUSCRIPTS}
            default = protocol.digest(root / "src/config/default.yaml")
            state = {"protected_sha256": {"src/config/default.yaml": "pre-MP",
                                         "src/envs/maenv.py": protocol.digest(root / "src/envs/maenv.py")}}
            record = {"files": {"src/config/default.yaml": {"raw": default}},
                      "manuscript_actualraw": manuscripts}
            with patch.object(protocol, "ROOT", root), patch.object(protocol, "MP_DEFAULT", default), \
                 patch.object(protocol, "MANUSCRIPTS", manuscripts):
                self.assertTrue(protocol.validate_protection(state, record)["original_manifest_preserved"])
                (root / next(iter(manuscripts))).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    protocol.validate_protection(state, record)
            self.assertEqual(state["protected_sha256"]["src/config/default.yaml"], "pre-MP")

    def test_C3_raw_source_keys_metadata_pointer_config_and_weights(self):
        import torch
        for changed in (None, "source", "data", "pointer", "config", "metadata", "weights"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "models/best_policies/rl/100"
                checkpoint.mkdir(parents=True)
                source = root / "saved"
                source.mkdir()
                (source / "actor.py").write_bytes(b"original-python\r\n")
                (source / "config.yaml").write_bytes(b"original-yaml\r\n")
                data = root / "src/envs/data"
                data.mkdir(parents=True)
                (data / "load.csv").write_bytes(b"1,2\r\n")
                torch.save({"weight": torch.ones(2)}, checkpoint / "agent.th")
                (checkpoint / "config.yaml").write_bytes(b"original-config\r\n")
                pointer = checkpoint.parent.parent / "best_rl.json"
                pointer.write_text("{}", encoding="utf-8")
                metadata = {
                    "validation": {"rows": list(range(731, 739))}, "policy_source": "rl",
                    "checkpoint_step": 100, "policy_sha256": protocol.digest(checkpoint / "agent.th"),
                    "source_snapshot": str(source), "source_sha256": {"actor.py": protocol.digest(source / "actor.py")},
                    "data_sha256": {"load.csv": protocol.digest(data / "load.csv")},
                }
                (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                actor = {
                    "seed": 0, "checkpoint": str(checkpoint), "checkpoint_step": 100,
                    "metadata_sha256": protocol.digest(checkpoint / "metadata.json"),
                    "policy_sha256": metadata["policy_sha256"],
                    "config_sha256": protocol.digest(checkpoint / "config.yaml"),
                    "selection_pointer_sha256": protocol.digest(pointer),
                    "evaluation_source_sha256": {**metadata["source_sha256"],
                                                "config.yaml": protocol.digest(source / "config.yaml")},
                    "evaluation_data_sha256": metadata["data_sha256"],
                }
                if changed is None:
                    self.assertTrue(protocol.check_c3_actor(actor, root)[
                        "metadata_and_evaluation_source_keys_checked_individually"])
                else:
                    path = {"source": source / "config.yaml", "data": data / "load.csv",
                            "pointer": pointer, "config": checkpoint / "config.yaml",
                            "metadata": checkpoint / "metadata.json", "weights": checkpoint / "agent.th"}[changed]
                    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n") + b"\n")
                    with self.assertRaises(ValueError):
                        protocol.check_c3_actor(actor, root)


if __name__ == "__main__":
    unittest.main()
