"""Retired H11 guards: actual root/parent and synthetic child, no rejected runtime."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import freeze_h11_protocol as protocol


class H11ProtocolTest(unittest.TestCase):
    def entries(self):
        workers = {"rollout_workers": 2, "learner_workers": 2}
        root = protocol.entry_config("gavdac_ea_main", workers)
        parent = protocol.entry_config("gavdac_c2_h11_parent", workers)
        return root, parent, {**parent, "name": "gavdac_c2_h11", "target_update_interval": 2}

    def test_actual_parent_child_entries_only_registered_delta(self):
        root, parent, child = self.entries()
        protocol.validate_candidate(root, parent, child)
        for key, value in (("lr", .001), ("learner", "GAVDACLearner"),
                           ("name", "gavdac_c2_h10"), ("target_update_tau", .004)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_candidate(root, parent, {**child, key: value})
        with self.assertRaises(ValueError):
            protocol.validate_candidate(root, {**parent, "lr": .001}, child)

    def test_all_retired_flags_rejected_even_false_in_parent(self):
        root, parent, child = self.entries()
        self.assertEqual(len(protocol.RETIRED), 9)
        for flag in protocol.RETIRED:
            for value in (False, True, None):
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    protocol.validate_candidate(
                        {**root, flag: value}, {**parent, flag: value}, {**child, flag: value})

    def test_fresh_budget_inventory_selection_workers_and_width(self):
        root, parent, child = self.entries()
        for key, value in (
            ("t_max", 12500), ("t_max", 6000000.0), ("use_cuda", True),
            ("rollout_workers", 0), ("learner_workers", 2.0), ("learner_workers", True),
            ("checkpoint_path", "old"), ("policy_checkpoint", "old"), ("evaluate", True),
            ("test_interval", 1000), ("fixed_eval_days", 182), ("evaluation_initial_soc", .8),
            ("save_model", False), ("save_best_policy", False), ("wandb_mode", "offline"),
            ("rnn_hidden_dim", 64), ("target_update_tau", .004),
            ("validation_rows", [913]), ("validation_rows", list(map(float, range(731, 739)))),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_candidate(
                    {**root, key: value}, {**parent, key: value}, {**child, key: value})

    def test_intervals_reject_bool_float_missing_and_unsupported_combinations(self):
        root, parent, child = self.entries()
        for value in (False, True, 2.0, "2", None, 1, 3):
            for key in ("actor_update_interval", "target_update_interval"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    protocol.validate_candidate(root, parent, {**child, key: value})
        missing = dict(child)
        missing.pop("target_update_interval")
        with self.assertRaises(ValueError):
            protocol.validate_candidate(root, parent, missing)

    def test_original_seed_host_slot_budget_and_process_assignments(self):
        plans = protocol.make_plans(["h11s0a", "h11s1a", "h11s2a"])
        protocol.validate_plans(plans)
        for key, value in (
            ("seed", True), ("seed", 0.0), ("host", "715"), ("root", "/other"),
            ("slot", 1), ("capacity", 1), ("joint_budget", 12500),
            ("rollout_workers", 0), ("learner_workers", 4), ("threads", 2),
            ("config_name", "gavdac_c2_h11_parent"),
        ):
            bad = copy.deepcopy(plans)
            bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_plans(bad)

    def test_tokens_new_round_short_unique_and_safe(self):
        for tokens in (
            ["h11s0a"] * 3, ["h11s0a", "../unsafe", "h11s2a"],
            ["h10s0a", "h11s1a", "h11s2a"], ["h03s0a", "h11s1a", "h11s2a"],
            ["h11s0a", "h11s1a"], ["h11s0a", None, "h11s2a"],
            ["h11s0a", "h11s1a", "h11s" + "a" * 26],
        ):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                protocol.make_plans(tokens)

    def test_evaluation_actor_inventory_metric_and_mean_only_three_percent(self):
        protocol.validate_evaluation(protocol.evaluation_protocol())
        for key, value in (
            ("primary_metric", "full_cost"), ("mode", "daily_reset"), ("actor", "ea"),
            ("rows", [913]), ("initial_soc", .8), ("terminal_soc", "cyclic"),
            ("expected_agent_hours", 13104.0), ("per_seed_minimum_saving_fraction", .03),
            ("mean_minimum_saving_fraction", .01), ("mean_of_percentage_savings", True),
            ("each_seed_strictly_below_corresponding_C3", 1),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_evaluation({**protocol.evaluation_protocol(), key: value})

    def test_parent_retention_contract_not_extra_per_seed_parent_gate(self):
        protocol.validate_retention(protocol.retention_protocol())
        for key, value in (
            ("mean_cost_strictly_below_historical_parent", 344000.0),
            ("all_seed_cost_strictly_below_C3", 1), ("per_seed_parent_threshold", .0),
            ("all_parent_seed_regressions_disclosed", False),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_retention({**protocol.retention_protocol(), key: value})

    def test_final_gate_accepts_one_seed_below_three_percent(self):
        costs = [value * ratio for value, ratio in zip(protocol.C3_COSTS, (.99, .95, .95))]
        result = protocol.economic_gates(costs)
        self.assertTrue(result["final_target_passed"])
        self.assertLess(1 - costs[0] / protocol.C3_COSTS[0], .03)
        self.assertGreaterEqual(result["ratio_of_means_saving"], .03)

    def test_final_gate_requires_strict_each_seed_and_finite_complete_inputs(self):
        for excess in (0, 1):
            costs = [protocol.C3_COSTS[0] + excess,
                     protocol.C3_COSTS[1] * .90, protocol.C3_COSTS[2] * .90]
            self.assertFalse(protocol.economic_gates(costs)["final_target_passed"])
        for costs in ([1, 2], [True, 2, 3], [float("nan"), 2, 3], [float("inf"), 2, 3], [-1, 2, 3]):
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                protocol.economic_gates(costs)

    def test_gate_uses_ratio_of_means_not_mean_percentages(self):
        savings = [.045, .045, .0001]
        costs = [v * (1 - s) for v, s in zip(protocol.C3_COSTS, savings)]
        self.assertGreater(sum(savings) / 3, .03)
        result = protocol.economic_gates(costs)
        self.assertTrue(result["each_seed_better"])
        self.assertLess(result["ratio_of_means_saving"], .03)
        self.assertFalse(result["final_target_passed"])

    def test_retention_can_pass_with_parent_seed_regression_but_final_fail(self):
        historical = [344313.5095480575, 342715.7419415101, 343228.5726593077]
        costs = [historical[0] + 100, historical[1] - 400, historical[2] - 400]
        result = protocol.economic_gates(costs)
        self.assertTrue(result["parent_retention_passed"])
        self.assertFalse(result["final_target_passed"])
        self.assertGreater(costs[0], historical[0])
        self.assertFalse(protocol.economic_gates(protocol.C3_COSTS)["parent_retention_passed"])

    def phase_fixture(self):
        return {
            "active_hypothesis": "H11", "active_runs": [], "run_plan": [], "formal_protocol": None,
            "status": "h11_enabled_engineering_accepted_pending_independent_freeze",
            "H11_development": {"enabled_host_engineering_accepted": True},
            "authorized_slot_capacity": 3, "execution_slots": {"arch715_0": {"enabled": False}},
            "execution_host_amendment": {"enabled_slot_keys": ["local0", "707_0", "707_1"]},
        }

    def test_wrong_phase_or_reenabled715_rejects_before_probe(self):
        state = self.phase_fixture()
        protocol.validate_phase(state)
        for key, value in (
            ("active_hypothesis", "H10"), ("active_runs", ["running"]), ("run_plan", ["old"]),
            ("formal_protocol", "old"), ("formal_protocol", 0), ("status", "registered"),
            ("authorized_slot_capacity", 4), ("execution_slots", {"arch715_0": {"enabled": True}}),
            ("H11_development", {"enabled_host_engineering_accepted": False}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_phase({**state, key: value})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({**state, "active_hypothesis": "H10"}))
            with patch.object(protocol, "probe") as probe, self.assertRaises(ValueError):
                protocol.freeze(path, ["h11s0a", "h11s1a", "h11s2a"])
            probe.assert_not_called()

    def test_exact_parent_child_rollback_and_ledger_preservation(self):
        scope = {
            "child_reverse_commit_order": [protocol.CHILD],
            "exact_parent_code_destination": protocol.PARENT,
            "registered_child_code_paths": [f"code{i}" for i in range(11)],
            "preserve_current_ledger_document_paths": [
                "docs/c2_hypotheses/H11.md", "docs/c2_hypotheses/README.md"],
            "parent_reconstruction_is_not_child_delta_or_support": True,
        }
        protocol.validate_rollback(protocol.CHILD, [protocol.CHILD], protocol.PARENT, scope, copy.deepcopy(scope))
        for commits, destination in (
            ([], protocol.PARENT), ([protocol.PARENT], protocol.PARENT),
            ([protocol.CHILD, "support"], protocol.PARENT), ([protocol.CHILD], "C2_ROOT"),
        ):
            with self.subTest(commits=commits), self.assertRaises(ValueError):
                protocol.validate_rollback(protocol.CHILD, commits, destination, scope, scope)
        bad = {**scope, "preserve_current_ledger_document_paths": []}
        with self.assertRaises(ValueError):
            protocol.validate_rollback(protocol.CHILD, [protocol.CHILD], protocol.PARENT, bad, scope)

    def test_lineage_wrong_parent_root_child_or_registration_rejected(self):
        registration = {
            "preregistration_commit": protocol.PREREGISTRATION, "parent_node_id": "H03",
            "root_node_id": "C2_ROOT", "parent_reconstruction_commits": [protocol.PARENT],
            "child_delta_commits": [protocol.CHILD], "rollback_destination_commit": protocol.PARENT,
        }
        for key, value in (
            ("preregistration_commit", "later"), ("parent_node_id", "H10"),
            ("root_node_id", "other"), ("parent_reconstruction_commits", []),
            ("child_delta_commits", [protocol.PARENT]), ("rollback_destination_commit", "root"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_lineage({"H11_registration": {**registration, key: value}}, Path("."))

    def host_fixture(self):
        previous = {
            "files": {"src/run.py": {"raw": "host-raw", "lf": "shared-lf"},
                      "src/envs/data/load.csv": {"raw": "actual-data", "lf": "shared-data"}},
            "platform": "qualified", "root": "qualified-root", "logical_cpus": 12,
            "python": "3.12.13", "torch": "qualified-torch", "host": "qualified-host",
            "numpy": "qualified-numpy", "pandas": "qualified-pandas",
        }
        current = {
            **copy.deepcopy(previous), "slots": [{"slot": 0, "state": "free"}],
            "git_status": [{"status": " M", "path": name} for name in protocol.MANUSCRIPTS],
            "existing_tokens": [], "memory": {"available_bytes": 4 * 1024**3},
            "torch_threads": 1, "cpu_busy_percent_snapshot": 15.5,
        }
        current["files"].update({name: {"raw": name, "lf": name} for name in protocol.ADDITIONS})
        return previous, current

    def test_original_host_raw_data_extra_source_and_exact_two_additions(self):
        previous, current = self.host_fixture()
        protocol.compare_host(previous, current, 1, "local")
        for label in ("raw", "data", "extra", "missing"):
            bad = copy.deepcopy(current)
            if label in ("raw", "data"):
                name = "src/run.py" if label == "raw" else "src/envs/data/load.csv"
                bad["files"][name]["raw"] = "different-with-same-lf"
            elif label == "extra":
                bad["files"]["tools/unknown.py"] = {"raw": "x", "lf": "x"}
            else:
                bad["files"].pop(next(iter(protocol.ADDITIONS)))
            with self.subTest(label=label), self.assertRaises(ValueError):
                protocol.compare_host(previous, bad, 1, "local")

    def test_host_dirty_scope_locks_existing_tokens_and_ram(self):
        previous, current = self.host_fixture()
        for label in ("dirty", "staged", "lock", "token", "ram"):
            bad = copy.deepcopy(current)
            if label == "dirty":
                bad["git_status"].append({"status": "??", "path": "unknown"})
            elif label == "staged":
                bad["git_status"][0]["status"] = "M "
            elif label == "lock":
                bad["slots"][0]["state"] = "stale_lock"
            elif label == "token":
                bad["existing_tokens"] = ["h11s0a"]
            else:
                bad["memory"]["available_bytes"] = 3 * 1024**3
            with self.subTest(label=label), self.assertRaises(ValueError):
                protocol.compare_host(previous, bad, 1, "local")

    def test_host_dependencies_cpu_threads_and_disabled_host(self):
        previous, current = self.host_fixture()
        for name in ("platform", "root", "logical_cpus", "python", "torch", "numpy", "pandas", "host"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                protocol.compare_host(previous, {**current, name: "changed"}, 1, "local")
        for value in (True, 1.0, 0, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.compare_host(previous, {**current, "torch_threads": value}, 1, "local")
        for value in (float("nan"), -1, 101, True):
            with self.subTest(busy=value), self.assertRaises(ValueError):
                protocol.compare_host(previous, {**current, "cpu_busy_percent_snapshot": value}, 1, "local")
        for host, capacity in (("715", 1), ("707", 1), ("local", True), ("other", 1)):
            with self.subTest(host=host), self.assertRaises(ValueError):
                protocol.compare_host(previous, current, capacity, host)
        with patch.object(protocol.subprocess, "run") as ssh, self.assertRaises(ValueError):
            protocol.probe("715", "/unused", [], [], 1)
        ssh.assert_not_called()

    def test_incremental_tooling_count_exit_and_actual_host_raw_bound(self):
        current = {"files": {name: {"raw": name, "lf": name} for name in protocol.ADDITIONS}}
        checked = {"accepted": True, "focused_tests": protocol.FOCUSED_TESTS, "returncode": 0,
                   "raw_sha256": {name: name for name in protocol.ADDITIONS}}
        protocol.validate_tooling(current, checked)
        for key, value in (("focused_tests", 0), ("focused_tests", float(protocol.FOCUSED_TESTS)),
                           ("accepted", False), ("returncode", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_tooling(current, {**checked, key: value})
        bad = copy.deepcopy(checked)
        bad["raw_sha256"][next(iter(protocol.ADDITIONS))] = "changed"
        with self.assertRaises(ValueError):
            protocol.validate_tooling(current, bad)

    def test_original_C3_freeze_baseline_and_engineering_pins(self):
        original = {str(seed): {"economic_cost": cost, "candidate_maximum_cost": .97 * cost,
                                "policy_sha256": "original"}
                    for seed, cost in enumerate(protocol.C3_COSTS)}
        protocol.validate_baseline(copy.deepcopy(original), original)
        for field in ("economic_cost", "candidate_maximum_cost", "policy_sha256"):
            changed = copy.deepcopy(original)
            changed["1"][field] = "tampered"
            with self.subTest(field=field), self.assertRaises(ValueError):
                protocol.validate_baseline(changed, original)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "h01_protocol_b.json").write_text("{}")
            (base / "h11_engineering_acceptance_a.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "original C3 freeze"):
                protocol.revalidate_c3_trace({"baseline_C3": {}}, base)
            with self.assertRaisesRegex(ValueError, "engineering acceptance bytes"):
                protocol.validate_engineering(base)

    def engineering_fixture(self, root):
        base = root / "results"
        base.mkdir()
        smokes = {}
        for token in ("h11ls0a", "h11rs0a", "h11rs1a"):
            completion = {
                "actor_update_interval": 2, "target_update_interval": 2, "target_update_tau": .002,
                "critic_optimizer_steps": 8, "actor_optimizer_steps": 4, "target_updates": 4,
                "finite_update_checks": 8, "train_updates": 8, "exact_resume_supported": False,
                "optimizer_post_step_finite_checks": {"critic": 8, "actor": 4},
                "joint_interactions": 12624, "agent_transitions": 37872, "ea_cycles": 11,
                "lineage_role": "H11_child_of_reconstructed_H03",
            }
            path = base / (token + ".json")
            path.write_text(json.dumps({
                "accepted": True, "completion": completion,
                "launch": {"git_commit": protocol.CHILD, "config": "gavdac_c2_h11"}}))
            smokes[token] = {"path": path.relative_to(root).as_posix(), "sha256": protocol.digest(path)}
        document = {
            "accepted": True, "implementation_commit": protocol.CHILD,
            "parent_reconstruction_commit": protocol.PARENT, "enabled_hosts": ["local", "707"],
            "enabled_slot_count": 3, "smokes": smokes, "bound_original_artifacts": {},
        }
        path = base / "h11_engineering_acceptance_a.json"
        path.write_text(json.dumps(document))
        return base, document, path

    def test_engineering_actual_H11_cadence_counts_and_runtime_required(self):
        for field, value in (
            ("actor_update_interval", 1), ("target_update_interval", 1), ("target_update_tau", .004),
            ("critic_optimizer_steps", 8.0), ("actor_optimizer_steps", 8), ("target_updates", 8),
            ("finite_update_checks", 4), ("joint_interactions", 12500),
            ("exact_resume_supported", True),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, document, path = self.engineering_fixture(root)
                with patch.object(protocol, "ROOT", root), patch.object(protocol, "PINS", {}), \
                     patch.object(protocol, "ENGINEERING", protocol.digest(path)):
                    self.assertTrue(protocol.validate_engineering(base)["accepted"])
                smoke_path = root / document["smokes"]["h11ls0a"]["path"]
                smoke = json.loads(smoke_path.read_text())
                smoke["completion"][field] = value
                smoke_path.write_text(json.dumps(smoke))
                document["smokes"]["h11ls0a"]["sha256"] = protocol.digest(smoke_path)
                path.write_text(json.dumps(document))
                with patch.object(protocol, "ROOT", root), patch.object(protocol, "PINS", {}), \
                     patch.object(protocol, "ENGINEERING", protocol.digest(path)), self.assertRaises(ValueError):
                    protocol.validate_engineering(base)

    def snapshot(self, root):
        import torch
        checkpoint = root / "models/best_policies/rl/100"
        checkpoint.mkdir(parents=True)
        source = root / "saved-source"
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
            "validation": {"rows": list(range(731, 739))}, "policy_source": "rl", "checkpoint_step": 100,
            "policy_sha256": protocol.digest(checkpoint / "agent.th"), "source_snapshot": str(source),
            "source_sha256": {"actor.py": protocol.digest(source / "actor.py")},
            "data_sha256": {"load.csv": protocol.digest(data / "load.csv")},
        }
        (checkpoint / "metadata.json").write_text(json.dumps(metadata))
        actor = {
            "seed": 0, "checkpoint": str(checkpoint), "checkpoint_step": 100,
            "metadata_sha256": protocol.digest(checkpoint / "metadata.json"),
            "policy_sha256": metadata["policy_sha256"], "config_sha256": protocol.digest(checkpoint / "config.yaml"),
            "selection_pointer_sha256": protocol.digest(pointer),
            "evaluation_source_sha256": {**metadata["source_sha256"],
                                        "config.yaml": protocol.digest(source / "config.yaml")},
            "evaluation_data_sha256": metadata["data_sha256"],
        }
        return actor, checkpoint, source

    def test_C3_independent_Python_YAML_keys_and_original_raw_not_LF(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, _, source = self.snapshot(root)
            self.assertTrue(protocol.check_c3_actor(actor, root)[
                "metadata_and_evaluation_source_keys_checked_individually"])
            (source / "config.yaml").write_bytes(b"original-yaml\n")
            with self.assertRaisesRegex(ValueError, "saved source raw"):
                protocol.check_c3_actor(actor, root)

    def test_C3_original_Actor_config_metadata_pointer_data_raw_corruption(self):
        for label in ("agent.th", "config.yaml", "metadata.json", "pointer", "data"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                actor, checkpoint, _ = self.snapshot(root)
                path = (checkpoint.parent.parent / "best_rl.json" if label == "pointer" else
                        root / "src/envs/data/load.csv" if label == "data" else checkpoint / label)
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaises(ValueError):
                    protocol.check_c3_actor(actor, root)

    def test_C3_source_escape_nonfinite_dtype_and_summary_data_rejected(self):
        import torch
        for label in ("escape", "nonfinite", "dtype", "summary_data"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                actor, checkpoint, _ = self.snapshot(root)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                if label == "escape":
                    (root / "outside.py").write_bytes(b"outside")
                    actor["evaluation_source_sha256"]["../outside.py"] = protocol.digest(root / "outside.py")
                elif label == "summary_data":
                    actor["evaluation_data_sha256"] = {"load.csv": "changed"}
                else:
                    torch.save({"weight": torch.tensor([float("nan")]) if label == "nonfinite"
                                else torch.ones(2, dtype=torch.float64)}, checkpoint / "agent.th")
                    actor["policy_sha256"] = metadata["policy_sha256"] = protocol.digest(checkpoint / "agent.th")
                    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
                    actor["metadata_sha256"] = protocol.digest(checkpoint / "metadata.json")
                with self.assertRaises(ValueError):
                    protocol.check_c3_actor(actor, root)


if __name__ == "__main__":
    unittest.main()
