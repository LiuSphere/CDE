"""Independent H12 guards using actual root/parent and a synthetic child."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import freeze_h12_protocol as protocol


class H12ProtocolTest(unittest.TestCase):
    def entries(self):
        workers = {"rollout_workers": 2, "learner_workers": 2}
        root = protocol.entry_config("gavdac_ea_main", workers)
        parent = protocol.entry_config("gavdac_c2_h11_parent", workers)
        return root, parent, {**parent, "name": "gavdac_c2_h12", "scaled_huber_critic": True}

    def test_exact_root_parent_and_only_huber_delta(self):
        root, parent, child = self.entries()
        protocol.validate_candidate(root, parent, child)
        for key, value in (("lr", .001), ("learner", "GAVDACLearner"), ("name", "gavdac_c2_h04")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_candidate(root, parent, {**child, key: value})
        with self.assertRaises(ValueError):
            protocol.validate_candidate(root, {**parent, "lr": .001}, child)

    def test_true_boolean_child_only_huber(self):
        root, parent, child = self.entries()
        for value in (False, 1, 1.0, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.validate_candidate(root, parent, {**child, "scaled_huber_critic": value})
        with self.assertRaises(ValueError):
            protocol.validate_candidate(root, {**parent, "scaled_huber_critic": False}, child)

    def test_all_other_retired_flags_rejected_even_false(self):
        root, parent, child = self.entries()
        for flag in protocol.RETIRED:
            for value in (True, False, None):
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    protocol.validate_candidate(
                        {**root, flag: value}, {**parent, flag: value}, {**child, flag: value})

    def test_fresh_budget_cpu_selection_and_integer_workers(self):
        root, parent, child = self.entries()
        for key, value in (
            ("t_max", 12500), ("t_max", 6000000.0), ("use_cuda", True),
            ("rollout_workers", 0), ("learner_workers", True), ("learner_workers", 2.0),
            ("checkpoint_path", "old"), ("policy_checkpoint", "old"), ("load_step", 100),
            ("evaluate", True), ("test_interval", 1000), ("fixed_eval_days", 182),
            ("evaluation_initial_soc", .8), ("save_model", False), ("save_best_policy", False),
            ("wandb_mode", "offline"), ("rnn_hidden_dim", 64), ("target_update_tau", .004),
            ("validation_rows", [913]), ("validation_rows", list(map(float, range(731, 739)))),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_candidate(
                    {**root, key: value}, {**parent, key: value}, {**child, key: value})

    def test_inherited_actor2_every_step_target1_no_h11_cadence(self):
        root, parent, child = self.entries()
        for key, valid in (("actor_update_interval", 2), ("target_update_interval", 1)):
            for value in (False, True, float(valid), str(valid), None, 3, 2 if valid == 1 else 1):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    protocol.validate_candidate(root, parent, {**child, key: value})

    def test_three_original_seed_host_slots_and_process_budget(self):
        plans = protocol.make_plans(["h12s0a", "h12s1a", "h12s2a"])
        protocol.validate_plans(plans)
        for key, value in (
            ("seed", True), ("seed", 0.0), ("host", "715"), ("root", "/wrong"), ("slot", 1),
            ("capacity", 1), ("joint_budget", 12500), ("rollout_workers", 0),
            ("learner_workers", 4), ("threads", 2), ("config_name", "gavdac_c2_h11"),
        ):
            bad = copy.deepcopy(plans)
            bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_plans(bad)

    def test_tokens_unique_short_safe_and_h12_only(self):
        for tokens in (
            ["h12s0a"] * 3, ["h12s0a", "../bad", "h12s2a"], ["h11s0a", "h12s1a", "h12s2a"],
            ["h12s0a", "h12s1a"], ["h12s0a", None, "h12s2a"],
            ["h12s0a", "h12s1a", "h12s" + "a" * 26],
        ):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                protocol.make_plans(tokens)

    def test_evaluation_and_retention_inventory_metric_actor_boundaries(self):
        protocol.validate_evaluation(protocol.evaluation_protocol())
        protocol.validate_retention(protocol.retention_protocol())
        for key, value in (
            ("primary_metric", "full_cost"), ("mode", "daily_reset"), ("actor", "ea"),
            ("rows", [913]), ("initial_soc", .8), ("terminal_soc", "cyclic"),
            ("expected_agent_hours", 13104.0), ("per_seed_minimum_saving_fraction", .03),
            ("mean_of_percentage_savings", True), ("each_seed_strictly_below_corresponding_C3", 1),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_evaluation({**protocol.evaluation_protocol(), key: value})
        for key, value in (("mean_cost_strictly_below_historical_parent", 344000.),
                           ("per_seed_parent_threshold", 0), ("all_parent_seed_regressions_disclosed", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_retention({**protocol.retention_protocol(), key: value})

    def test_final_gate_no_per_seed_three_percent_but_strict_each(self):
        costs = [v * ratio for v, ratio in zip(protocol.C3_COSTS, (.99, .95, .95))]
        self.assertTrue(protocol.economic_gates(costs)["final_target_passed"])
        for excess in (0, 1):
            costs[0] = protocol.C3_COSTS[0] + excess
            self.assertFalse(protocol.economic_gates(costs)["final_target_passed"])
        for costs in ([1, 2], [True, 2, 3], [float("nan"), 2, 3], [float("inf"), 2, 3], [-1, 2, 3]):
            with self.assertRaises(ValueError):
                protocol.economic_gates(costs)

    def test_ratio_of_means_and_separate_retention_with_parent_regression(self):
        savings = [.045, .045, .0001]
        costs = [v * (1 - s) for v, s in zip(protocol.C3_COSTS, savings)]
        self.assertGreater(sum(savings) / 3, .03)
        self.assertFalse(protocol.economic_gates(costs)["final_target_passed"])
        costs = [344413.5095480575, 342315.7419415101, 342828.5726593077]
        result = protocol.economic_gates(costs)
        self.assertTrue(result["parent_retention_passed"])
        self.assertFalse(result["final_target_passed"])

    def phase(self):
        return {
            "active_hypothesis": "H12", "active_runs": [], "run_plan": [], "formal_protocol": None,
            "status": "h12_engineering_accepted_pending_independent_freeze",
            "H12_development": {"enabled_host_engineering_accepted": True},
            "authorized_slot_capacity": 4,
            "execution_slots": {"arch715_0": {"enabled": True, "state": "enabled_idle_pending_requalification"}},
            "H12_checklist": {**dict.fromkeys((
                "preregistered_and_normally_pushed", "parent_and_primary_review", "implementation",
                "local_engineering", "remote_engineering", "original_C3_readonly_audit"), True),
                "independent_freeze": False, "formal_training_started": False},
        }

    def test_phase_active_freeze_and_auxiliary_guard_before_probe(self):
        state = self.phase()
        protocol.validate_phase(state)
        for key, value in (
            ("active_hypothesis", "H11"), ("active_runs", ["active"]), ("run_plan", ["old"]),
            ("formal_protocol", "old"), ("formal_protocol", 0), ("status", "registered"),
            ("authorized_slot_capacity", 3),
            ("H12_development", {"enabled_host_engineering_accepted": False}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_phase({**state, key: value})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({**state, "active_hypothesis": "H11"}))
            with patch.object(protocol, "probe") as probe, self.assertRaises(ValueError):
                protocol.freeze(path, ["h12s0a", "h12s1a", "h12s2a"])
            probe.assert_not_called()

    def test_exact_registered_rollback_excludes_parent_support_ledger(self):
        scope = {
            "child_reverse_commit_order": [protocol.CHILD], "exact_parent_code_destination": protocol.PARENT,
            "registered_child_code_paths": [f"code{i}" for i in range(11)],
            "algorithm_commit_excludes_ledger_docs": True, "parent_reconstruction_is_not_child_delta_or_support": True,
            "preserve_current_ledger_document_paths": ["docs/c2_hypotheses/H12.md", "docs/c2_hypotheses/README.md"],
        }
        protocol.validate_rollback(protocol.CHILD, [protocol.CHILD], protocol.PARENT, scope, copy.deepcopy(scope))
        for commits, destination in (([], protocol.PARENT), ([protocol.PARENT], protocol.PARENT),
                                     ([protocol.CHILD, "support"], protocol.PARENT), ([protocol.CHILD], "C2_ROOT")):
            with self.assertRaises(ValueError):
                protocol.validate_rollback(protocol.CHILD, commits, destination, scope, scope)
        with self.assertRaises(ValueError):
            protocol.validate_rollback(protocol.CHILD, [protocol.CHILD], protocol.PARENT,
                                       {**scope, "preserve_current_ledger_document_paths": []}, scope)

    def test_lineage_and_engineering_reject_wrong_identity(self):
        with self.assertRaisesRegex(ValueError, "registration identity"):
            protocol.validate_lineage({"H12_registration": {}}, Path("."))
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "h12_engineering_acceptance_a.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "engineering acceptance bytes"):
                protocol.validate_engineering(base)
        value = protocol.intervention()
        self.assertEqual(value["child_learner"], "HuberDelayedActorGAVDACLearner")
        self.assertEqual(value["worker_learner"], value["child_learner"])
        self.assertEqual(value["target_update_interval"], 1)

    def host(self):
        previous = {
            "files": {"src/run.py": {"raw": "actual", "lf": "shared"}},
            "root": "qualified", "host": "qualified", "platform": "qualified", "logical_cpus": 12,
            "python": "qualified", "torch": "qualified", "numpy": "qualified", "pandas": "qualified",
            "numeric_data": {"load": {"shape": [1095, 24], "dtype": "float64", "sha256": "numeric"}},
        }
        current = {**copy.deepcopy(previous), "slots": [{"slot": 0, "state": "free"}],
                   "existing_tokens": [], "git_status": [{"status": " M", "path": p} for p in protocol.MANUSCRIPTS],
                   "torch_threads": 1, "memory": {"available_bytes": 4 * 1024**3}, "cpu_busy_percent_snapshot": 10.}
        current["files"].update({p: {"raw": p, "lf": p} for p in protocol.ADDITIONS})
        return previous, current

    def test_host_original_raw_numeric_data_and_only_two_additions(self):
        previous, current = self.host()
        protocol.compare_host(previous, current, 1, "local")
        for label in ("raw", "numeric", "extra", "missing"):
            bad = copy.deepcopy(current)
            if label == "raw":
                bad["files"]["src/run.py"]["raw"] = "same-lf-different-raw"
            elif label == "numeric":
                bad["numeric_data"]["load"]["shape"] = [1, 24]
            elif label == "extra":
                bad["files"]["tools/unknown.py"] = {"raw": "extra", "lf": "extra"}
            else:
                bad["files"].pop(next(iter(protocol.ADDITIONS)))
            with self.subTest(label=label), self.assertRaises(ValueError):
                protocol.compare_host(previous, bad, 1, "local")

    def test_host_dirty_staged_lock_token_and_insufficient_ram(self):
        previous, current = self.host()
        for key, value in (
            ("git_status", [{"status": "??", "path": "unknown"}]),
            ("git_status", [{"status": "M ", "path": next(iter(protocol.MANUSCRIPTS))}]),
            ("slots", [{"slot": 0, "state": "stale_lock"}]), ("existing_tokens", ["h12s0a"]),
            ("memory", {"available_bytes": 3 * 1024**3}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.compare_host(previous, {**current, key: value}, 1, "local")

    def test_host_dependency_threads_cpu_and_auxiliary_not_formal(self):
        previous, current = self.host()
        for key in ("root", "host", "platform", "logical_cpus", "python", "torch", "numpy", "pandas"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.compare_host(previous, {**current, key: "changed"}, 1, "local")
        for value in (True, 1., 0, 2):
            with self.assertRaises(ValueError):
                protocol.compare_host(previous, {**current, "torch_threads": value}, 1, "local")
        for value in (float("nan"), -1, 101, True):
            with self.assertRaises(ValueError):
                protocol.compare_host(previous, {**current, "cpu_busy_percent_snapshot": value}, 1, "local")
        with patch.object(protocol.subprocess, "run") as ssh, self.assertRaises(ValueError):
            protocol.probe("715", "/unused", [], [], 1)
        ssh.assert_not_called()

    def test_focused_qualification_binds_count_exit_and_actual_host_support_raw(self):
        _, current = self.host()
        checked = {"accepted": True, "focused_tests": protocol.FOCUSED_TESTS, "returncode": 0,
                   "raw_sha256": {p: p for p in protocol.ADDITIONS}}
        protocol.validate_tooling(current, checked)
        for key, value in (("focused_tests", 0), ("focused_tests", float(protocol.FOCUSED_TESTS)),
                           ("accepted", False), ("returncode", 1), ("raw_sha256", dict.fromkeys(protocol.ADDITIONS, "bad"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_tooling(current, {**checked, key: value})

    def test_original_c3_raw_and_mp_protected_exception_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "h01_protocol_b.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "original C3 freeze"):
                protocol.revalidate_c3_trace({"baseline_C3": {}}, base)
        state = {"protected_sha256": {}}
        with patch.object(protocol, "read", return_value={"protected": {
            "MP_default_exception": {"new_raw": "wrong"}}}):
            with self.assertRaisesRegex(ValueError, "MP exception"):
                protocol.validate_protection(state, {})

    def snapshot(self, root):
        import torch
        checkpoint = root / "models/best_policies/rl/100"
        checkpoint.mkdir(parents=True)
        source = root / "saved"
        source.mkdir()
        (source / "actor.py").write_bytes(b"original-python\r\n")
        (source / "config.yaml").write_bytes(b"original-yaml\r\n")
        data = root / "src/envs/data"
        data.mkdir(parents=True)
        (data / "load.csv").write_bytes(b"1,2\r\n")
        torch.save({"weight": torch.ones(2, dtype=torch.float32)}, checkpoint / "agent.th")
        (checkpoint / "config.yaml").write_bytes(b"original-config\r\n")
        pointer = checkpoint.parent.parent / "best_rl.json"
        pointer.write_text("{}")
        metadata = {
            "validation": {"rows": list(range(731, 739))}, "policy_source": "rl",
            "checkpoint_step": 100, "policy_sha256": protocol.digest(checkpoint / "agent.th"),
            "source_snapshot": str(source), "source_sha256": {"actor.py": protocol.digest(source / "actor.py")},
            "data_sha256": {"load.csv": protocol.digest(data / "load.csv")},
        }
        (checkpoint / "metadata.json").write_text(json.dumps(metadata))
        return {
            "seed": 0, "checkpoint": str(checkpoint), "checkpoint_step": 100,
            "metadata_sha256": protocol.digest(checkpoint / "metadata.json"),
            "policy_sha256": metadata["policy_sha256"],
            "config_sha256": protocol.digest(checkpoint / "config.yaml"),
            "selection_pointer_sha256": protocol.digest(pointer),
            "evaluation_source_sha256": {**metadata["source_sha256"],
                                        "config.yaml": protocol.digest(source / "config.yaml")},
            "evaluation_data_sha256": metadata["data_sha256"],
        }, checkpoint, source

    def test_c3_original_raw_not_lf_and_independent_source_key_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, _, source = self.snapshot(root)
            self.assertTrue(protocol.check_c3_actor(actor, root)[
                "metadata_and_evaluation_source_keys_checked_individually"])
            (source / "config.yaml").write_bytes(b"original-yaml\n")
            with self.assertRaisesRegex(ValueError, "saved source raw"):
                protocol.check_c3_actor(actor, root)

    def test_c3_actor_metadata_pointer_config_and_data_corruption(self):
        for label in ("agent.th", "metadata.json", "config.yaml", "pointer", "data"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                actor, checkpoint, _ = self.snapshot(root)
                path = (checkpoint.parent.parent / "best_rl.json" if label == "pointer" else
                        root / "src/envs/data/load.csv" if label == "data" else checkpoint / label)
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaises(ValueError):
                    protocol.check_c3_actor(actor, root)

    def test_engineering_positive_and_h11_target_cadence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "results"
            base.mkdir()
            completion = {
                "lineage_role": "H12_child_of_reconstructed_H03", "actor_update_interval": 2,
                "target_update_interval": 1, "target_update_tau": .002, "scaled_huber_critic": True,
                "huber_delta": 1., "huber_multiplier": 2., "critic_optimizer_steps": 8,
                "actor_optimizer_steps": 4, "target_updates": 8, "finite_update_checks": 8,
                "train_updates": 8, "optimizer_post_step_finite_checks": {"critic": 8, "actor": 4},
                "exact_resume_supported": False, "joint_interactions": 12624,
                "agent_transitions": 37872, "ea_cycles": 11,
            }
            document = {
                "accepted": True, "implementation_commit": protocol.CHILD,
                "exact_parent_code_destination": protocol.PARENT, "formal_hosts": ["local", "707"],
                "formal_slots": 3, "qualified_file_count": 144, "bound_original_artifacts": {}, "smokes": {},
            }
            for token in ("h12ls0a", "h12rs0a", "h12rs1a"):
                smoke = {"accepted": True, "completion": completion,
                         "launch": {"git_commit": protocol.CHILD, "config": "gavdac_c2_h12"}}
                path = base / (token + ".json")
                path.write_text(json.dumps(smoke))
                document["smokes"][token] = {
                    "path": path.relative_to(root).as_posix(), "sha256": protocol.digest(path),
                    "completion": completion,
                }
            path = base / "h12_engineering_acceptance_a.json"
            path.write_text(json.dumps(document))
            with patch.object(protocol, "ROOT", root), patch.object(protocol, "PINS", {}), \
                 patch.object(protocol, "ENGINEERING", protocol.digest(path)):
                self.assertTrue(protocol.validate_engineering(base)["accepted"])
            smoke_path = root / document["smokes"]["h12ls0a"]["path"]
            bad = protocol.read(smoke_path)
            bad["completion"]["target_update_interval"] = 2
            smoke_path.write_text(json.dumps(bad))
            document["smokes"]["h12ls0a"]["sha256"] = protocol.digest(smoke_path)
            path.write_text(json.dumps(document))
            with patch.object(protocol, "ROOT", root), patch.object(protocol, "PINS", {}), \
                 patch.object(protocol, "ENGINEERING", protocol.digest(path)), self.assertRaises(ValueError):
                protocol.validate_engineering(base)

    def test_protection_positive_preserves_original_manifest_and_manuscripts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"src/config/default.yaml": b"two-zero-defaults", "src/envs/maenv.py": b"original"}
            paths.update(dict.fromkeys(protocol.MANUSCRIPTS, b"user-dirty"))
            for name, raw in paths.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            new_raw = protocol.digest(root / "src/config/default.yaml")
            manuscripts = {name: protocol.digest(root / name) for name in protocol.MANUSCRIPTS}
            previous = {"protected": {"MP_default_exception": {"new_raw": new_raw},
                                      "manuscript_actual_raw": manuscripts}}
            state = {"protected_sha256": {"src/config/default.yaml": "original-pre-MP",
                                         "src/envs/maenv.py": protocol.digest(root / "src/envs/maenv.py")}}
            record = {"qualified_local_raw_sha256": {"src/config/default.yaml": new_raw}}
            with patch.object(protocol, "ROOT", root), patch.object(protocol, "read", return_value=previous), \
                 patch.object(protocol, "MP_DEFAULT", new_raw):
                self.assertTrue(protocol.validate_protection(state, record)["original_manifest_preserved"])
                (root / next(iter(protocol.MANUSCRIPTS))).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    protocol.validate_protection(state, record)
            self.assertEqual(state["protected_sha256"]["src/config/default.yaml"], "original-pre-MP")


if __name__ == "__main__":
    unittest.main()
