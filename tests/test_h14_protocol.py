"""Independent H14 freeze validation with actual ancestors and synthetic child."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import freeze_h14_protocol as p


class H14ProtocolTest(unittest.TestCase):
    def entries(self):
        ancestors = [p.entry_config(n, {"rollout_workers": 2, "learner_workers": 2}) for n in p.CONFIG_NAMES[:3]]
        return [*ancestors, dict(ancestors[-1], name="gavdac_c2_h14", signed_mixer_hyperweights=True)]

    def phase(self):
        return dict(active_hypothesis="H14", status="h14_engineering_accepted_pending_independent_freeze",
                    active_runs=[], run_plan=[], formal_protocol=None, authorized_slot_capacity=4,
                    H14_development={"enabled_host_engineering_accepted": True},
                    execution_slots={"arch715_0": {"enabled": True, "state": "enabled_idle_pending_requalification"}},
                    H14_checklist={
                        **dict.fromkeys(("preregistered_and_normally_pushed", "parent_and_primary_review",
                                        "implementation", "local_engineering", "remote_engineering",
                                        "original_C3_readonly_audit"), True),
                        **dict.fromkeys(("independent_freeze", "formal_training_started", "formal_training_accepted",
                                        "primary_evaluation", "final_economic_gate", "parent_retention_gate"), False)})

    def registration(self):
        gate = dict(version="each_seed_better_mean_saving_3pct_v2", metric="economic_cost",
                    seed_labels=[0, 1, 2], per_seed_rule="candidate_cost < corresponding_C3_cost",
                    per_seed_minimum_saving_fraction=0.0, per_seed_comparison="strictly_greater",
                    mean_rule="1-mean(candidate_cost)/mean(C3_cost) >=0.03",
                    mean_minimum_saving_fraction=.03, arithmetic_mean_of_percentage_savings=False,
                    fixed_C3_costs=list(p.C3_COSTS))
        reg = dict(hypothesis="H14", preregistration_commit=p.PREREGISTRATION, normal_push_verified=True,
                   parent_node_id="H12", root_node_id="C2_ROOT", exact_parent_code_commit=p.PARENT,
                   parent_algorithm_commit=p.PARENT_ALGORITHM, parent_protocol_sha256=p.PARENT_PROTOCOL,
                   parent_result_sha256=p.PARENT_RESULT, inherited_mechanisms=deepcopy(p.INHERITED),
                   sole_new_delta=p.DELTA, implementation_commit=p.CHILD, child_delta_commits=[p.CHILD],
                   rollback_destination_commit=p.PARENT, never_rollback_retained_H12_or_support=True,
                   formal_seed_labels=[0, 1, 2], physical_joint_budget_including_EA=6000000,
                   formal_host_slots=[["707", 0, 0], ["707", 1, 1], ["local", 0, 2]],
                   workers={"rollout_workers": 2, "learner_workers": 2, "threads": 1},
                   formal_protocol=None, config_name="gavdac_c2_h14", parent_config_name="gavdac_c2_h12",
                   final_gate=deepcopy(gate), retention_gate=p.retention_protocol(), **{"715_required": False})
        return dict(current_economic_gate=gate, H14_registration=reg,
                    research_lineage_policy={"version": "one_increment_per_recorded_parent_v1"},
                    lineage=dict(working_checkout_node_id="H14", selected_parent_node_id="H12",
                                 active_child_node_id="H14", new_working_parent_commit=p.PARENT,
                                 registry="results/c2opt21/lineage_registry_x.json", registry_sha256=p.LINEAGE))

    def scope(self):
        return dict(child_reverse_commit_order=[p.CHILD], exact_parent_code_destination=p.PARENT,
                    registered_child_code_paths=sorted(p.CHILD_PATHS), algorithm_commit_excludes_ledger_docs=True,
                    retained_H12_and_support_are_not_child_delta=True,
                    preserve_current_ledger_document_paths=["docs/c2_hypotheses/H14.md", "docs/c2_hypotheses/README.md"])

    def host(self, host="local"):
        previous = dict(files={f"original/{i}": {"raw": str(i), "lf": str(i)} for i in range(152)},
                        root=p.HOSTS[host][0], host="actual-host", platform="actual-platform", logical_cpus=12,
                        python="qualified", torch="qualified", numpy="qualified", pandas="qualified", gymnasium="qualified",
                        numeric_data={"load": {"shape": [1095, 24], "dtype": "float64", "sha256": "data"}},
                        configs={"synthetic_child": {"signed_mixer_hyperweights": True}})
        current = dict(deepcopy(previous), slots=[{"slot": i, "state": "free"} for i in range(p.HOSTS[host][1])],
                       existing_tokens=[], git_status=[], torch_threads=1,
                       memory={"available_bytes": 12 * 1024**3}, cpu_busy_percent_snapshot=20.,
                       runtime_identity=p.runtime_contract())
        current["files"].update({n: {"raw": n, "lf": n} for n in p.ADDITIONS})
        return previous, current

    def test_01_actual_ancestors_and_synthetic_only_delta(self):
        p.validate_candidate(*self.entries())
        configs = self.entries()
        with self.assertRaises(ValueError):
            p.validate_candidate(configs[0], dict(configs[1], lr=.001), configs[2], configs[3])

    def test_02_boolean_flags_are_not_numbers_or_strings(self):
        configs = self.entries()
        for name in ("signed_mixer_hyperweights", "scaled_huber_critic"):
            for value in (False, 1, 1., "true", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    p.validate_candidate(*configs[:3], dict(configs[3], **{name: value}))

    def test_03_retired_flags_even_false(self):
        for flag in p.RETIRED:
            for value in (False, True, None):
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    p.validate_candidate(*[dict(c, **{flag: value}) for c in self.entries()])

    def test_04_shared_knob_drift_rejected(self):
        for name, value in (
            ("t_max", 6000000.), ("t_max", 12500), ("learner_workers", True), ("learner_workers", 2.),
            ("rollout_workers", 0), ("actor_update_interval", 1), ("target_update_interval", 2),
            ("target_update_tau", .004), ("rnn_hidden_dim", 64), ("mixer_weight_scale", 1.),
            ("use_cuda", True), ("EA", False), ("crossover_prob", .1), ("checkpoint_path", "old"),
            ("evaluate", True), ("load_step", 1), ("policy_checkpoint", "old"), ("wandb_mode", "offline"),
            ("save_model", False), ("save_best_policy", False), ("test_interval", 20000),
            ("fixed_eval_days", 182), ("evaluation_initial_soc", .8),
            ("validation_rows", [913]), ("validation_rows", list(map(float, range(731, 739)))),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                p.validate_candidate(*[dict(c, **{name: value}) for c in self.entries()])

    def test_05_child_extra_setting_and_mechanism_origin(self):
        configs = self.entries()
        for name, value in (("lr", .001), ("critic_lr", .001), ("learner", "GAVDACLearner")):
            with self.assertRaises(ValueError):
                p.validate_candidate(*configs[:3], dict(configs[3], **{name: value}))
        configs = [dict(c, signed_mixer_hyperweights=True) for c in self.entries()]
        with self.assertRaises(ValueError):
            p.validate_candidate(*configs)

    def test_06_runtime_explicit_class_and_inherited_functions(self):
        good = p.runtime_contract()
        p.validate_runtime(good)
        for name, value in (
            ("actor_class", "LayerNormRNNAgent"), ("learner_class", "HuberDelayedActorGAVDACLearner"),
            ("run_binding_exact", False), ("worker_explicit_signed_branch", False),
            ("parent_class_exact", False), ("inherited_train_exact", False), ("inherited_TD_loss_exact", False),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                p.validate_runtime(dict(good, **{name: value}))

    def test_07_runtime_map_scale_counts_tensor_rng_and_cpu(self):
        good = p.runtime_contract()
        for name, value in (
            ("mixer_operations", ["sigmoid", "signed_sigmoid"]), ("mixer_scales", [.1, 1.]),
            ("actor_parameters", 15554), ("critic_mixer_parameters", 65915.),
            ("initial_parameter_tensors_equal_parent", False), ("initial_rng_equal_parent", False),
            ("all_parameters_CPUfloat32_finite", False), ("actor_binding_exact", 1),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                p.validate_runtime(dict(good, **{name: value}))

    def test_08_probe_C3_original_zero_workers(self):
        configs = dict(zip(p.CONFIG_NAMES[:4], self.entries(), strict=True))
        configs["gavdac_mappo"] = p.entry_config("gavdac_mappo")
        def entry(name, options):
            self.assertEqual(options, {} if name == "gavdac_mappo" else {"rollout_workers": 2, "learner_workers": 2})
            return configs[name]
        with patch.object(p, "inventory", return_value={}), patch.object(p, "entry_config", side_effect=entry), \
             patch.object(p, "runtime_identity", return_value=p.runtime_contract()):
            value = p.inspect_host({})
        self.assertEqual(value["configs"]["gavdac_mappo"]["rollout_workers"], 0)
        self.assertEqual(value["configs"]["gavdac_mappo"]["learner_workers"], 0)

    def test_09_exact_three_plans(self):
        plans = p.make_plans(["h14s0a", "h14s1a", "h14s2a"])
        p.validate_plans(plans)
        self.assertEqual([(v["host"], v["slot"], v["seed"]) for v in plans], [("707", 0, 0), ("707", 1, 1), ("local", 0, 2)])

    def test_10_allocation_and_integer_types(self):
        for name, value in (("host", "715"), ("seed", True), ("slot", 1), ("capacity", 2.),
                            ("joint_budget", 6000000.), ("rollout_workers", 4),
                            ("learner_workers", True), ("threads", 2), ("config_name", "gavdac_c2_h12")):
            bad = p.make_plans(["h14s0a", "h14s1a", "h14s2a"])
            bad[0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                p.validate_plans(bad)

    def test_11_token_namespace_path_length_and_uniqueness(self):
        for value in (["h14s0a"] * 3, ["h14s0a", "../x", "h14s2a"], ["h13s0a", "h14s1a", "h14s2a"],
                      ["h14s0a", None, "h14s2a"], ["h14s0a", "h14s1a"], ["h14s" + "x" * 26, "h14s1a", "h14s2a"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.make_plans(value)

    def test_12_frozen_primary_inventory_metric_soc_and_strict_gate(self):
        p.validate_evaluation(p.evaluation_protocol())
        for name, value in (
            ("rows", [913]), ("mode", "daily_reset"), ("actor", "ea"), ("initial_soc", .8),
            ("terminal_soc", "cyclic"), ("hidden_and_previous_action_reset", "never"),
            ("primary_metric", "full_cost"), ("expected_agent_hours", 13104.),
            ("per_seed_minimum_saving_fraction", .03), ("mean_minimum_saving_fraction", .01),
            ("mean_of_percentage_savings", True),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                p.validate_evaluation(dict(p.evaluation_protocol(), **{name: value}))

    def test_13_retention_is_independent_and_strict(self):
        p.validate_retention(p.retention_protocol())
        for name, value in (("mean_cost_strictly_below_historical_parent", 343419.27471629175),
                            ("per_seed_parent_threshold", 0), ("all_parent_seed_regressions_disclosed", False),
                            ("all_seed_cost_strictly_below_C3", False)):
            with self.assertRaises(ValueError):
                p.validate_retention(dict(p.retention_protocol(), **{name: value}))
        self.assertFalse(p.economic_gates([p.PARENT_MEAN] * 3)["parent_retention_passed"])

    def test_14_each_strict_but_not_each_three_percent(self):
        costs = [c * r for c, r in zip(p.C3_COSTS, (.99, .95, .95))]
        self.assertTrue(p.economic_gates(costs)["final_target_passed"])
        for offset in (0, 1):
            costs[0] = p.C3_COSTS[0] + offset
            self.assertFalse(p.economic_gates(costs)["final_target_passed"])
        retained = p.economic_gates([344100., 342000., 343700.])
        self.assertTrue(retained["parent_retention_passed"])
        self.assertFalse(retained["final_target_passed"])

    def test_15_ratio_of_means_not_mean_percentages(self):
        savings = [.045, .045, .0001]
        self.assertGreater(sum(savings) / 3, .03)
        self.assertFalse(p.economic_gates([c * (1 - s) for c, s in zip(p.C3_COSTS, savings)])["final_target_passed"])

    def test_16_costs_must_be_three_finite_numeric_values(self):
        for costs in ([1, 2], [True, 2, 3], [-1, 2, 3], [float("nan"), 2, 3], [float("inf"), 2, 3]):
            with self.assertRaises(ValueError):
                p.economic_gates(costs)

    def test_17_phase_prevents_old_active_or_frozen_work(self):
        good = self.phase()
        p.validate_phase(good)
        for name, value in (("active_hypothesis", "H13"), ("status", "registered"),
                            ("active_runs", ["active"]), ("run_plan", ["old"]), ("formal_protocol", 0),
                            ("formal_protocol", "old"), ("authorized_slot_capacity", 4.)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                p.validate_phase(dict(good, **{name: value}))

    def test_18_each_prerequisite_and_auxiliary_exclusion(self):
        for key, value in self.phase()["H14_checklist"].items():
            bad = self.phase()
            bad["H14_checklist"][key] = not value
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.validate_phase(bad)
        bad = self.phase()
        bad["execution_slots"]["arch715_0"]["state"] = "ready_idle"
        with self.assertRaises(ValueError):
            p.validate_phase(bad)

    def test_19_invalid_phase_precedes_any_remote_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps(dict(self.phase(), active_runs=["active"])), encoding="utf-8")
            with patch.object(p, "probe") as probe, self.assertRaises(ValueError):
                p.freeze(path, ["h14s0a", "h14s1a", "h14s2a"])
            probe.assert_not_called()

    def test_20_registration_exact_parent_root_inheritance(self):
        good = self.registration()
        p.validate_registration(good)
        for name, value in (
            ("parent_node_id", "H03"), ("root_node_id", "H12"), ("exact_parent_code_commit", p.ANCESTOR),
            ("sole_new_delta", "target2"), ("inherited_mechanisms", []), ("child_delta_commits", [p.PARENT]),
            ("parent_result_sha256", "wrong"), ("physical_joint_budget_including_EA", 6000000.),
            ("715_required", True), ("formal_protocol", "old"), ("never_rollback_retained_H12_or_support", False),
        ):
            bad = deepcopy(good)
            bad["H14_registration"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                p.validate_registration(bad)
        for key in good["lineage"]:
            bad = deepcopy(good)
            bad["lineage"][key] = "wrong"
            with self.assertRaises(ValueError):
                p.validate_registration(bad)

    def test_21_loosened_gate_in_both_copies_still_rejects(self):
        for name, value in (("metric", "full_cost"), ("mean_minimum_saving_fraction", .01),
                            ("per_seed_minimum_saving_fraction", .03), ("fixed_C3_costs", [400000.] * 3),
                            ("seed_labels", [0, 1, 2.]), ("arithmetic_mean_of_percentage_savings", True)):
            bad = self.registration()
            bad["current_economic_gate"][name] = bad["H14_registration"]["final_gate"][name] = value
            with self.assertRaises(ValueError):
                p.validate_state_gates(bad)

    def test_22_child_only_rollback_exact14paths(self):
        good = self.scope()
        p.validate_rollback(p.CHILD, [p.CHILD], p.PARENT, good, deepcopy(good))
        for commits, parent in (([p.PARENT], p.PARENT), ([p.CHILD, "support"], p.PARENT),
                                ([p.CHILD], p.ANCESTOR), ([p.CHILD], "C2_ROOT")):
            with self.assertRaises(ValueError):
                p.validate_rollback(p.CHILD, commits, parent, good, good)
        bad = deepcopy(good)
        bad["registered_child_code_paths"][-1] = "unknown.py"
        with self.assertRaises(ValueError):
            p.validate_rollback(p.CHILD, [p.CHILD], p.PARENT, bad, bad)

    def test_23_original152raw_and_only_two_support_additions(self):
        previous, current = self.host()
        p.compare_host(previous, current, 1, "local")
        for kind in ("raw", "extra", "missing", "numeric", "config"):
            bad = deepcopy(current)
            if kind == "raw":
                bad["files"]["original/0"]["raw"] = "different-raw-same-LF"
            elif kind == "extra":
                bad["files"]["extra.py"] = {"raw": "x", "lf": "x"}
            elif kind == "missing":
                bad["files"].pop(next(iter(p.ADDITIONS)))
            elif kind == "numeric":
                bad["numeric_data"]["load"]["shape"] = [1, 24]
            else:
                bad["configs"]["synthetic_child"]["signed_mixer_hyperweights"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                p.compare_host(previous, bad, 1, "local")

    def test_24_lock_oldtoken_unknown_dirty_staged_support(self):
        for host in p.HOSTS:
            previous, current = self.host(host)
            for name, value in (
                ("slots", [{"slot": 0, "state": "stale_lock"}]), ("slots", [{"slot": 0, "state": "running"}]),
                ("existing_tokens", ["h14s0a"]), ("git_status", [{"status": "??", "path": "unknown"}]),
                ("git_status", [{"status": " M", "path": next(iter(p.ADDITIONS))}]),
                ("git_status", [{"status": "M ", "path": next(iter(p.MANUSCRIPTS))}]),
            ):
                with self.subTest(host=host, name=name), self.assertRaises(ValueError):
                    p.compare_host(previous, dict(current, **{name: value}), p.HOSTS[host][1], host)
        previous, current = self.host()
        current["git_status"] = [{"status": " M", "path": n} for n in p.MANUSCRIPTS]
        p.compare_host(previous, current, 1, "local")

    def test_25_resources_threads_dependencies_and_aux_host(self):
        previous, current = self.host()
        for key in ("root", "host", "platform", "python", "torch", "numpy", "pandas", "gymnasium"):
            with self.assertRaises(ValueError):
                p.compare_host(previous, dict(current, **{key: "changed"}), 1, "local")
        for value in (True, 1., 0, 2):
            with self.assertRaises(ValueError):
                p.compare_host(previous, dict(current, torch_threads=value), 1, "local")
        for value in (-1, 101, True, float("nan")):
            with self.assertRaises(ValueError):
                p.compare_host(previous, dict(current, cpu_busy_percent_snapshot=value), 1, "local")
        with self.assertRaises(ValueError):
            p.compare_host(previous, dict(current, memory={"available_bytes": 3 * 1024**3}), 1, "local")
        previous["logical_cpus"] = current["logical_cpus"] = 4
        with self.assertRaises(ValueError):
            p.compare_host(previous, current, 1, "local")
        with patch.object(p.subprocess, "run") as run, self.assertRaises(ValueError):
            p.probe("715", "/unused", [], [], 1)
        run.assert_not_called()

    def test_26_focused_count_exit_and_actualraw_binding(self):
        _, host = self.host()
        value = dict(accepted=True, focused_tests=p.FOCUSED_TESTS, returncode=0, raw_sha256={n: n for n in p.ADDITIONS})
        p.validate_tooling(host, value)
        for key, bad in (("accepted", False), ("focused_tests", 0), ("focused_tests", float(p.FOCUSED_TESTS)),
                         ("returncode", True), ("returncode", 1), ("raw_sha256", dict.fromkeys(p.ADDITIONS, "wrong"))):
            with self.assertRaises(ValueError):
                p.validate_tooling(host, dict(value, **{key: bad}))

    def test_27_transport_original_bytes_binary_stdin_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "results/c2opt21"
            base.mkdir(parents=True)
            response = SimpleNamespace(returncode=0, stdout=b'{"ok":true}\n', stderr=b"original\r\n")
            with patch.object(p, "ROOT", root), patch.object(p.subprocess, "run", return_value=response) as run:
                self.assertEqual(p.probe("local", p.HOSTS["local"][0], ["h14s0z"], [], 1), {"ok": True})
                self.assertIsInstance(run.call_args.kwargs["input"], bytes)
                self.assertEqual((base / "h14s0z_freeze_local.stderr").read_bytes(), b"original\r\n")
                with self.assertRaises(ValueError):
                    p.probe("local", p.HOSTS["local"][0], ["h14s0z"], [], 1)
                self.assertEqual(run.call_count, 1)
            bad = SimpleNamespace(returncode=0, stdout=b"bad-json", stderr=b"original-error")
            with patch.object(p, "ROOT", root), patch.object(p.subprocess, "run", return_value=bad), \
                 self.assertRaises(json.JSONDecodeError):
                p.probe("local", p.HOSTS["local"][0], ["h14s1z"], [], 1)
            self.assertEqual((base / "h14s1z_freeze_local.stderr").read_bytes(), b"original-error")

    def test_28_engineering_pins_smoke_map_counts_and_C3_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "results"
            base.mkdir()
            path = base / "h14_engineering_acceptance_a.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                p.validate_engineering(base)
            value = dict(accepted=True, implementation_commit=p.CHILD, exact_parent_code_destination=p.PARENT,
                         formal_hosts=["local", "707"], formal_slots=3, qualified_file_count=152, input_sha256={}, smokes={})
            for token in ("h14ls0a", "h14rs0a", "h14rs1a"):
                file = base / (token + ".json")
                file.write_text(json.dumps(dict(accepted=True, completion=p.completion_contract(),
                                                launch={"git_commit": p.CHILD, "config": "gavdac_c2_h14"})), encoding="utf-8")
                value["smokes"][token] = dict(path=file.relative_to(root).as_posix(), sha256=p.digest(file),
                                             completion=p.completion_contract())
            path.write_text(json.dumps(value), encoding="utf-8")
            with patch.object(p, "ROOT", root), patch.object(p, "ENGINEERING", p.digest(path)):
                self.assertTrue(p.validate_engineering(base)["accepted"])
            changed = root / value["smokes"]["h14ls0a"]["path"]
            smoke = p.read(changed)
            smoke["completion"]["target_updates"] = 4
            changed.write_text(json.dumps(smoke), encoding="utf-8")
            value["smokes"]["h14ls0a"]["sha256"] = p.digest(changed)
            path.write_text(json.dumps(value), encoding="utf-8")
            with patch.object(p, "ROOT", root), patch.object(p, "ENGINEERING", p.digest(path)), self.assertRaises(ValueError):
                p.validate_engineering(base)
            (base / "pin.json").write_text("changed", encoding="utf-8")
            with patch.object(p, "PINS", {"pin.json": "original"}), self.assertRaises(ValueError):
                p.validate_lineage(self.registration(), base)
            (base / "h01_protocol_b.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                p.revalidate_c3_trace({}, base)

    def test_29_protected_MP_exception_does_not_rewrite_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, raw in {"src/config/default.yaml": b"zero-workers",
                              "env.py": b"original\r\n", "method.tex": b"user1", "modeling.tex": b"user2"}.items():
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(raw)
            manuscripts = {n: p.digest(root / n) for n in ("method.tex", "modeling.tex")}
            default = p.digest(root / "src/config/default.yaml")
            state = {"protected_sha256": {"src/config/default.yaml": "pre-MP", "env.py": p.digest(root / "env.py")}}
            record = {"manuscript_actualraw": manuscripts, "files": {"src/config/default.yaml": {"raw": default}}}
            with patch.object(p, "ROOT", root), patch.object(p, "MANUSCRIPTS", manuscripts), patch.object(p, "MP_DEFAULT", default):
                self.assertTrue(p.validate_protection(state, record)["original_manifest_preserved"])
                (root / "method.tex").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    p.validate_protection(state, record)
            self.assertEqual(state["protected_sha256"]["src/config/default.yaml"], "pre-MP")

    def test_30_C3_actualraw_independent_source_keysets_and_CPUweights(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt = root / "models/best_policies/rl/100"
            ckpt.mkdir(parents=True)
            saved, data = root / "saved", root / "src/envs/data"
            saved.mkdir()
            data.mkdir(parents=True)
            for path, raw in ((saved / "actor.py", b"original\r\n"), (saved / "config.yaml", b"yaml\r\n"),
                              (data / "load.csv", b"1,2\r\n"), (ckpt / "config.yaml", b"original-config\r\n")):
                path.write_bytes(raw)
            torch.save({"weight": torch.ones(2)}, ckpt / "agent.th")
            pointer = ckpt.parent.parent / "best_rl.json"
            pointer.write_text("{}", encoding="utf-8")
            metadata = dict(validation={"rows": list(range(731, 739))}, policy_source="rl", checkpoint_step=100,
                            policy_sha256=p.digest(ckpt / "agent.th"), source_snapshot=str(saved),
                            source_sha256={"actor.py": p.digest(saved / "actor.py")},
                            data_sha256={"load.csv": p.digest(data / "load.csv")})
            (ckpt / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            actor = dict(seed=0, checkpoint=str(ckpt), checkpoint_step=100, policy_sha256=metadata["policy_sha256"],
                         metadata_sha256=p.digest(ckpt / "metadata.json"), config_sha256=p.digest(ckpt / "config.yaml"),
                         selection_pointer_sha256=p.digest(pointer), evaluation_data_sha256=metadata["data_sha256"],
                         evaluation_source_sha256={**metadata["source_sha256"], "config.yaml": p.digest(saved / "config.yaml")})
            self.assertTrue(p.check_c3_actor(actor, root)["metadata_and_evaluation_source_keys_checked_individually"])
            for file in (saved / "config.yaml", data / "load.csv", pointer, ckpt / "config.yaml",
                         ckpt / "metadata.json", ckpt / "agent.th"):
                raw = file.read_bytes()
                file.write_bytes(raw.replace(b"\r\n", b"\n") + b"\n")
                with self.subTest(file=file), self.assertRaises(ValueError):
                    p.check_c3_actor(actor, root)
                file.write_bytes(raw)


if __name__ == "__main__":
    unittest.main()
