"""Independent H10 protocol guards; no training, optimizer or environment."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.freeze_h10_protocol import (
    ADDITIONS, C3_COSTS, EVIDENCE, FOCUSED_TESTS, HOST_PROBE, IMPLEMENTATION, MANUSCRIPTS,
    RETIRED, check_c3_actor, compare_host, digest, entry_config, evaluation_protocol, freeze,
    intervention, make_plans, revalidate_C3, validate_baseline, validate_candidate,
    validate_engineering, validate_evaluation, validate_phase, validate_plans, validate_rollback,
    validate_tooling,
)


class H10ProtocolTest(unittest.TestCase):
    def entries(self):
        workers = {"rollout_workers": 2, "learner_workers": 2}
        return tuple(entry_config(name, workers) for name in ("gavdac_ea_main", "gavdac_c2_h10"))

    def test_real_entry_only_critic_generator_centering_and_retired_flags(self):
        parent, candidate = self.entries()
        validate_candidate(parent, candidate)
        for key, value in (("critic_center_generator_action", False), ("critic_center_generator_action", 1),
                           ("critic_center_generator_action", "true"), ("critic_center_generator_action", None),
                           ("lr", .001)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_candidate(parent, {**candidate, key: value})
        self.assertEqual(len(RETIRED), 9)
        self.assertIn("critic_small_output_init", RETIRED)
        self.assertIn("actor_no_action_l2", RETIRED)
        for key in RETIRED:
            for value in (False, True):
                with self.subTest(retired=key, value=value), self.assertRaises(ValueError):
                    validate_candidate({**parent, key: value}, {**candidate, key: value})

    def test_fresh_cpu_budget_selection_workers_online_width_fixed(self):
        parent, candidate = self.entries()
        for key, value in (
            ("t_max", 12500), ("t_max", 6000000.0), ("use_cuda", True),
            ("learner_workers", 0), ("learner_workers", 2.0), ("rollout_workers", 4),
            ("checkpoint_path", "old"), ("policy_checkpoint", "old"), ("evaluate", True),
            ("validation_rows", [913]), ("validation_rows", list(map(float, range(731, 739)))),
            ("test_interval", 1000), ("evaluation_initial_soc", .8), ("save_model", False),
            ("save_best_policy", False), ("wandb_mode", "offline"), ("rnn_hidden_dim", 64),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_candidate({**parent, key: value}, {**candidate, key: value})

    def test_original_seed_host_slot_budget_process_contract(self):
        plans = make_plans(["h10s0a", "h10s1a", "h10s2a"])
        validate_plans(plans)
        for key, value in (
            ("seed", True), ("seed", 2.0), ("host", "715"), ("root", "/other"),
            ("slot", 1), ("capacity", 4), ("joint_budget", 6000001),
            ("learner_workers", 0), ("rollout_workers", 4), ("threads", 2),
            ("config_name", "gavdac_ea_main"),
        ):
            changed = copy.deepcopy(plans)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plans(changed)

    def test_tokens_short_unique_h10_not_old_or_unsafe(self):
        for tokens in (
            ["h10s0a"] * 3, ["h10s0a", "../unsafe", "h10s2a"],
            ["h09s0a", "h10s1a", "h10s2a"], ["h10s0a", "h10s1a"],
            ["h10s0a", "h10s1a", "h10s" + "a" * 26], ["h10s0a", None, "h10s2a"],
        ):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                make_plans(tokens)

    def test_evaluation_actor_inventory_metric_gate_fixed(self):
        validate_evaluation(evaluation_protocol())
        for key, value in (
            ("primary_metric", "full_cost"), ("mode", "daily_reset"), ("actor", "ea"),
            ("minimum_saving_fraction", .01), ("rows", [913]), ("initial_soc", .8),
            ("terminal_soc", "cyclic"), ("expected_agent_hours", 13103),
            ("require_each_seed_and_ratio_of_means", False),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_evaluation({**evaluation_protocol(), key: value})

    def test_evaluation_numeric_types_not_bool_equivalence(self):
        for key, value in (
            ("require_each_seed_and_ratio_of_means", 1),
            ("retrospective_retained_development", 1), ("expected_agent_hours", 13104.0),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_evaluation({**evaluation_protocol(), key: value})

    def test_only_registered_h10_algorithm_revertible(self):
        validate_rollback(IMPLEMENTATION, [IMPLEMENTATION])
        for commits in ([], ["9f03926"], [IMPLEMENTATION, "8db8428"], ["ab86793"]):
            with self.assertRaises(ValueError):
                validate_rollback(IMPLEMENTATION, commits)
        with self.assertRaises(ValueError):
            validate_rollback("support-tool", [IMPLEMENTATION])

    def host_fixture(self):
        previous = {
            "files": {"src/run.py": {"raw": "host-raw", "lf": "shared-lf"},
                      "src/envs/data/load.csv": {"raw": "actual-data", "lf": "shared-data"}},
            "platform": "qualified", "root": "root", "logical_cpus": 12,
            "python": "3.12.13", "torch": "qualified-torch", "host": "qualified-host"}
        current = {
            **copy.deepcopy(previous), "slots": [{"slot": 0, "state": "free"}],
            "git_status": [{"status": " M", "path": name} for name in MANUSCRIPTS],
            "existing_tokens": [], "memory": {"available_bytes": 4 * 1024**3}, "torch_threads": 1}
        current["files"].update({name: {"raw": name, "lf": name} for name in ADDITIONS})
        return previous, current

    def test_original_host_raw_dirty_locks_tokens_ram_additions(self):
        previous, current = self.host_fixture()
        compare_host(previous, current, 1, "local")
        for label in ("raw", "data", "dirty", "staged", "token", "slot", "memory",
                      "addition", "missing_tool", "python", "torch", "root", "logical_cpus", "host"):
            changed = copy.deepcopy(current)
            if label in ("raw", "data"):
                name = "src/run.py" if label == "raw" else "src/envs/data/load.csv"
                changed["files"][name]["raw"] = "changed-with-same-lf"
            elif label == "dirty":
                changed["git_status"].append({"status": "??", "path": "unknown.txt"})
            elif label == "staged":
                changed["git_status"][0]["status"] = "M "
            elif label == "token":
                changed["existing_tokens"] = ["h10s0a"]
            elif label == "slot":
                changed["slots"][0]["state"] = "stale_lock"
            elif label == "memory":
                changed["memory"]["available_bytes"] = 3 * 1024**3
            elif label == "addition":
                changed["files"]["tools/unknown.py"] = {"raw": "x", "lf": "x"}
            elif label == "missing_tool":
                changed["files"].pop(next(iter(ADDITIONS)))
            else:
                changed[label] = "different"
            with self.subTest(label=label), self.assertRaises(ValueError):
                compare_host(previous, changed, 1, "local")
        for host, capacity in (("715", 1), ("707", 1), ("other", 1), ("local", True)):
            with self.subTest(host=host, capacity=capacity), self.assertRaises(ValueError):
                compare_host(previous, current, capacity, host)

    def test_original_c3_identity_cost_threshold_immutable(self):
        original = {str(seed): {"economic_cost": cost, "candidate_maximum_cost": .97 * cost}
                    for seed, cost in enumerate(C3_COSTS)}
        validate_baseline(copy.deepcopy(original), original)
        for field in ("economic_cost", "candidate_maximum_cost", "policy_sha256"):
            changed = copy.deepcopy(original)
            changed["1"][field] = "tampered"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_baseline(changed, original)
        changed = copy.deepcopy(original)
        changed["1"]["economic_cost"] -= 1
        changed["1"]["candidate_maximum_cost"] = .97 * changed["1"]["economic_cost"]
        with self.assertRaises(ValueError):
            validate_baseline(changed, changed)

    def test_original_freeze_and_engineering_pins_reject_changed_bytes(self):
        compile(HOST_PROBE, "<H10 read-only host probe>", "exec")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "h01_protocol_b.json").write_text("{}", encoding="utf-8")
            (base / "h10_engineering_acceptance_a.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "original C3 freeze"):
                revalidate_C3({"baseline_C3": {}}, base)
            with self.assertRaisesRegex(ValueError, "engineering bytes"):
                validate_engineering(base)
        self.assertIn("h10_host_707_a.json", EVIDENCE)
        self.assertIn("h10_707_smoke_live_overlap_a.json", EVIDENCE)

    def test_focused_tool_count_and_original_host_raw_bound(self):
        current = {"files": {name: {"raw": name, "lf": name} for name in ADDITIONS}}
        checked = {"accepted": True, "focused_tests": FOCUSED_TESTS,
                   "raw_sha256": {name: name for name in ADDITIONS}}
        validate_tooling(current, checked)
        for change in ("count", "float_count", "bytes", "accepted"):
            bad = copy.deepcopy(checked)
            if change == "count":
                bad["focused_tests"] -= 1
            elif change == "float_count":
                bad["focused_tests"] = float(FOCUSED_TESTS)
            elif change == "bytes":
                bad["raw_sha256"][next(iter(ADDITIONS))] = "wrong"
            else:
                bad["accepted"] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_tooling(current, bad)

    def test_wrong_phase_rejects_before_remote_probe(self):
        state = {"active_hypothesis": "H10", "active_runs": [], "run_plan": [], "formal_protocol": None,
                 "status": "h10_all_host_engineering_accepted_pending_independent_protocol_freeze",
                 "H10_development": {"all_host_engineering_accepted": True}}
        validate_phase(state)
        for key, value in (
            ("active_hypothesis", "H09"), ("active_runs", ["running"]), ("run_plan", ["old"]),
            ("formal_protocol", "old"), ("formal_protocol", 0), ("status", "registered"),
            ("H10_development", {"all_host_engineering_accepted": False}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_phase({**state, key: value})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps({**state, "active_hypothesis": "H09"}), encoding="utf-8")
            with patch("tools.freeze_h10_protocol.probe") as remote:
                with self.assertRaises(ValueError):
                    freeze(path, ["h10s0a", "h10s1a", "h10s2a"])
                remote.assert_not_called()

    def snapshot(self, root):
        import torch
        checkpoint = root / "model/best_policies/rl/100"
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
        pointer.write_text("{}", encoding="utf-8")
        metadata = {
            "validation": {"rows": list(range(731, 739))}, "policy_source": "rl", "checkpoint_step": 100,
            "policy_sha256": digest(checkpoint / "agent.th"), "source_snapshot": str(source),
            "source_sha256": {"actor.py": digest(source / "actor.py")},
            "data_sha256": {"load.csv": digest(data / "load.csv")}}
        (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        actor = {"seed": 0, "checkpoint": str(checkpoint), "checkpoint_step": 100,
                 "metadata_sha256": digest(checkpoint / "metadata.json"),
                 "policy_sha256": metadata["policy_sha256"],
                 "config_sha256": digest(checkpoint / "config.yaml"),
                 "selection_pointer_sha256": digest(pointer),
                 "evaluation_source_sha256": {**metadata["source_sha256"],
                                              "config.yaml": digest(source / "config.yaml")},
                 "evaluation_data_sha256": metadata["data_sha256"]}
        return actor, checkpoint, source

    def test_c3_python_metadata_and_yaml_summary_individual_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, _, source = self.snapshot(root)
            result = check_c3_actor(actor, root)
            self.assertTrue(result["metadata_and_evaluation_source_keys_checked_individually"])
            (source / "config.yaml").write_bytes(b"original-yaml\n")
            with self.assertRaisesRegex(ValueError, "saved source"):
                check_c3_actor(actor, root)

    def test_c3_raw_actor_config_pointer_data_corruption_rejected(self):
        for label in ("metadata.json", "agent.th", "config.yaml", "pointer", "data"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                actor, checkpoint, _ = self.snapshot(root)
                path = (checkpoint.parent.parent / "best_rl.json" if label == "pointer" else
                        root / "src/envs/data/load.csv" if label == "data" else checkpoint / label)
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaises(ValueError):
                    check_c3_actor(actor, root)

    def test_c3_source_escape_nonfinite_dtype_and_data_rejected(self):
        import torch
        for label in ("escape", "nonfinite", "dtype", "evaluation_data"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                actor, checkpoint, _ = self.snapshot(root)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                if label == "escape":
                    (root / "outside.py").write_bytes(b"outside")
                    actor["evaluation_source_sha256"]["../outside.py"] = digest(root / "outside.py")
                elif label == "evaluation_data":
                    actor["evaluation_data_sha256"] = {"load.csv": "different"}
                else:
                    torch.save({"weight": torch.tensor([float("nan")]) if label == "nonfinite"
                                else torch.ones(2, dtype=torch.float64)}, checkpoint / "agent.th")
                    actor["policy_sha256"] = metadata["policy_sha256"] = digest(checkpoint / "agent.th")
                    (checkpoint / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                    actor["metadata_sha256"] = digest(checkpoint / "metadata.json")
                with self.assertRaises(ValueError):
                    check_c3_actor(actor, root)

    def test_only_generator_fraction_coordinate_not_retired_objectives(self):
        value = intervention()
        self.assertIs(value["critic_center_generator_action"], True)
        self.assertEqual(value["critic_action_map"], "[2*a0-1,a1]")
        self.assertEqual(value["actor_action_l2_coefficient"], .001)
        self.assertEqual(value["action_Box_low"], [0., -1.])
        self.assertEqual(value["action_Box_high"], [1., 1.])
        self.assertEqual(value["learner"], "CenteredGeneratorGAVDACLearner")
        self.assertEqual(value["critic"], "CenteredGeneratorGAVDACCritic")
        self.assertTrue(value["live_mixer_context_action_recurrent_gradients_preserved"])
        self.assertTrue(value["critic_before_actor_every_step_actor_target_tau_preserved"])
        self.assertTrue(value["replay_actor_history_environment_coordinates_unchanged"])
        self.assertFalse(value["physical_kW_normalization"])
        self.assertFalse(value["new_parameters_or_buffers"])
        self.assertFalse(value["weight_compensation"])
        self.assertEqual(value["additional_RNG_draws"], 0)
        self.assertTrue(value["single_critic"])
        self.assertFalse(value["exact_resume_supported"])
        self.assertNotIn("actor_no_action_l2", value)
        self.assertNotIn("critic_small_output_init", value)

    def test_synthetic_engineering_schema_binds_mapping_and_actual_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "results"
            base.mkdir()
            records = {}
            for token in ("h10ls0a", "h10rs0a", "h10rs1a", "h10715a"):
                path = base / (token + ".json")
                completion = {
                    "critic_center_generator_action": True, "critic_action_map": "[2*a0-1,a1]",
                    "exact_resume_supported": False,
                    **dict.fromkeys(("critic_steps", "actor_steps", "target_steps", "finite_updates", "train_updates"), 8)}
                path.write_text(json.dumps({
                    "accepted": True, "completion": completion,
                    "launch": {"git_commit": IMPLEMENTATION if token == "h10ls0a" else
                               "6b060c6350811677edc707fdeadd8e95748ac43e"}}), encoding="utf-8")
                records[token] = {"report": path.relative_to(root).as_posix(), "sha256": digest(path)}
            path = base / "h10_engineering_acceptance_a.json"
            document = {"accepted": True, "sole_algorithm_rollback_commit": IMPLEMENTATION, "exclusive_smokes": records}
            path.write_text(json.dumps(document), encoding="utf-8")
            with patch("tools.freeze_h10_protocol.ROOT", root), \
                 patch("tools.freeze_h10_protocol.EVIDENCE", {}), \
                 patch("tools.freeze_h10_protocol.ENGINEERING", digest(path)):
                self.assertTrue(validate_engineering(base)["accepted"])
            for key, value in (("critic_center_generator_action", False),
                               ("critic_action_map", "[a0,a1]"), ("finite_updates", 8.0)):
                smoke_path = base / "h10ls0a.json"
                smoke = json.loads(smoke_path.read_text())
                original = copy.deepcopy(smoke)
                smoke["completion"][key] = value
                smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
                document["exclusive_smokes"]["h10ls0a"]["sha256"] = digest(smoke_path)
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(key=key), patch("tools.freeze_h10_protocol.ROOT", root), \
                     patch("tools.freeze_h10_protocol.EVIDENCE", {}), \
                     patch("tools.freeze_h10_protocol.ENGINEERING", digest(path)), \
                     self.assertRaises(ValueError):
                    validate_engineering(base)
                smoke_path.write_text(json.dumps(original), encoding="utf-8")

    def test_original_host_thread_contract_and_probe_module(self):
        previous, current = self.host_fixture()
        for value in (0, 2, True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_host(previous, {**current, "torch_threads": value}, 1, "local")
        self.assertIn("tools.freeze_h10_protocol", HOST_PROBE)
        self.assertNotIn("freeze_h09", HOST_PROBE)
        self.assertEqual(ADDITIONS, {"tools/freeze_h10_protocol.py", "tests/test_h10_protocol.py"})


if __name__ == "__main__":
    unittest.main()
