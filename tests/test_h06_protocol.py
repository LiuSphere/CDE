"""Independent H06 freeze guards; these tests never train, freeze or launch."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.freeze_h06_protocol import (
    ADDITIONS, C3_COSTS, FOCUSED_TESTS, HOST_PROBE, IMPLEMENTATION, MANUSCRIPTS,
    compare_host, entry_config, evaluation_protocol, freeze, make_plans, revalidate_C3,
    validate_baseline, validate_candidate, validate_evaluation, validate_phase, validate_plans,
    validate_rollback, validate_tooling,
)


class H06ProtocolTest(unittest.TestCase):
    def entries(self):
        workers = {"rollout_workers": 2, "learner_workers": 2}
        return (entry_config(name, workers) for name in ("gavdac_ea_main", "gavdac_c2_h06"))

    def test_actual_entry_name_and_LayerNorm_only(self):
        parent, candidate = self.entries()
        validate_candidate(parent, candidate)
        for key, value in (
            ("critic_layer_norm", False), ("critic_layer_norm", 1), ("lr", .001),
            ("actor_update_interval", 2), ("scaled_huber_critic", True),
            ("clipped_double_critic", True), ("actor_mixer_context_grad", False),
            ("five_step_critic", True), ("rnn_hidden_dim", 64),
            ("rollout_workers", 4), ("learner_workers", 0), ("t_max", 12500),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_candidate(parent, {**candidate, key: value})

    def test_budget_CPU_fresh_weights_selection_and_online_fixed(self):
        parent, candidate = self.entries()
        for key, value in (
            ("t_max", 500000), ("t_max", 6000000.0), ("use_cuda", True),
            ("learner_workers", 2.0), ("checkpoint_path", "old"), ("policy_checkpoint", "old"),
            ("validation_rows", [913]), ("test_interval", 1000), ("evaluation_initial_soc", .8),
            ("save_model", False), ("wandb_mode", "offline"), ("evaluate", True),
            ("rnn_hidden_dim", 64),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_candidate({**parent, key: value}, {**candidate, key: value})
        for flag in ("five_step_critic", "scaled_huber_critic", "clipped_double_critic"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                validate_candidate({**parent, flag: False}, {**candidate, flag: False})

    def test_original_seed_host_slot_workers_and_joint_budget(self):
        plans = make_plans(["h06s0a", "h06s1a", "h06s2a"])
        validate_plans(plans)
        for key, value in (
            ("seed", True), ("host", "715"), ("root", "/other"), ("slot", 1),
            ("capacity", 4), ("joint_budget", 6000001), ("learner_workers", 0),
            ("rollout_workers", 4), ("threads", 2), ("seed", 2.0),
        ):
            changed = copy.deepcopy(plans)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plans(changed)

    def test_tokens_unique_short_H06_only(self):
        for tokens in (
            ["h06s0a"] * 3, ["h06s0a", "../unsafe", "h06s2a"],
            ["h05s0a", "h06s1a", "h06s2a"], ["h06s0a", "h06s1a"],
            ["h06s0a", "h06s1a", "h06s" + "a" * 26], ["h06s0a", None, "h06s2a"],
        ):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                make_plans(tokens)

    def test_evaluation_inventory_and_gate_cannot_change(self):
        validate_evaluation(evaluation_protocol())
        for key, value in (
            ("primary_metric", "full_cost"), ("mode", "daily_reset"), ("actor", "ea"),
            ("minimum_saving_fraction", .01), ("rows", [913]), ("initial_soc", .8),
            ("terminal_soc", "cyclic"), ("expected_agent_hours", 13103),
            ("require_each_seed_and_ratio_of_means", False),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_evaluation({**evaluation_protocol(), key: value})

    def test_evaluation_bool_and_integer_types_not_interchangeable(self):
        for key, value in (
            ("require_each_seed_and_ratio_of_means", 1),
            ("retrospective_retained_development", 1), ("expected_agent_hours", 13104.0),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_evaluation({**evaluation_protocol(), key: value})

    def test_only_registered_H06_algorithm_revertible(self):
        validate_rollback(IMPLEMENTATION, [IMPLEMENTATION])
        for commits in ([], ["9f03926"], [IMPLEMENTATION, "8db8428"], ["28170d8"]):
            with self.assertRaises(ValueError):
                validate_rollback(IMPLEMENTATION, commits)
        with self.assertRaises(ValueError):
            validate_rollback("support-tool", [IMPLEMENTATION])

    def test_raw_resources_dirty_locks_tokens_runtime_and_additions(self):
        previous = {
            "files": {
                "src/run.py": {"raw": "host-raw", "lf": "shared-lf"},
                "src/envs/data/load.csv": {"raw": "actual-data", "lf": "shared-data"},
            },
            "platform": "qualified", "root": "root", "logical_cpus": 12,
            "python": "3.12.13", "torch": "qualified-torch",
        }
        current = {
            **copy.deepcopy(previous), "slots": [{"slot": 0, "state": "free"}],
            "git_status": [{"status": " M", "path": name} for name in MANUSCRIPTS],
            "existing_tokens": [], "memory": {"available_bytes": 4 * 1024**3},
        }
        current["files"].update({name: {"raw": name, "lf": name} for name in ADDITIONS})
        compare_host(previous, current, 1, "local")
        for label in ("raw", "data", "dirty", "staged", "token", "slot", "memory",
                      "addition", "missing_tool", "python", "torch", "root", "logical_cpus"):
            changed = copy.deepcopy(current)
            if label in ("raw", "data"):
                name = "src/run.py" if label == "raw" else "src/envs/data/load.csv"
                changed["files"][name]["raw"] = "changed-with-same-lf"
            elif label == "dirty":
                changed["git_status"].append({"status": "??", "path": "unknown.txt"})
            elif label == "staged":
                changed["git_status"][0]["status"] = "M "
            elif label == "token":
                changed["existing_tokens"] = ["h06s0a"]
            elif label == "slot":
                changed["slots"][0]["state"] = "stale_lock"
            elif label == "memory":
                changed["memory"]["available_bytes"] = 1024**3
            elif label == "addition":
                changed["files"]["tools/unknown.py"] = {"raw": "x", "lf": "x"}
            elif label == "missing_tool":
                changed["files"].pop(next(iter(ADDITIONS)))
            else:
                changed[label] = "different"
            with self.subTest(label=label), self.assertRaises(ValueError):
                compare_host(previous, changed, 1, "local")
        for host, capacity in (("715", 1), ("707", 1), ("other", 1)):
            with self.subTest(host=host), self.assertRaises(ValueError):
                compare_host(previous, current, capacity, host)

    def test_C3_identity_cost_and_threshold_immutable(self):
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

    def test_original_freeze_checked_before_C3_audit_and_probe_compiles(self):
        compile(HOST_PROBE, "<H06 original-host read-only probe>", "exec")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "h01_protocol_b.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "original C3 freeze"):
                revalidate_C3({"baseline_C3": {}}, base)

    def test_original_host_tool_acceptance_bound_to_rawbytes(self):
        current = {"files": {name: {"raw": name, "lf": name} for name in ADDITIONS}}
        checked = {"accepted": True, "focused_tests": FOCUSED_TESTS,
                   "raw_sha256": {name: name for name in ADDITIONS}}
        validate_tooling(current, checked)
        for change in ("count", "bytes", "accepted"):
            bad = copy.deepcopy(checked)
            if change == "count":
                bad["focused_tests"] -= 1
            elif change == "bytes":
                bad["raw_sha256"][next(iter(ADDITIONS))] = "wrong"
            else:
                bad["accepted"] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_tooling(current, bad)

    def test_wrong_phase_rejects_before_entry_probe_or_output(self):
        state = {
            "active_hypothesis": "H06", "active_runs": [], "run_plan": [], "formal_protocol": None,
            "status": "h06_all_host_engineering_accepted_pending_formal_protocol_freeze",
        }
        validate_phase(state)
        for key, value in (
            ("active_hypothesis", "H05"), ("active_runs", ["running"]),
            ("run_plan", ["old"]), ("formal_protocol", "old"), ("status", "registered"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_phase({**state, key: value})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "state.json"
            source.write_text('{"active_hypothesis":"H05","active_runs":[],"run_plan":[],'
                              '"formal_protocol":null,"status":"registered"}', encoding="utf-8")
            with patch("tools.freeze_h06_protocol.probe") as remote:
                with self.assertRaises(ValueError):
                    freeze(source, ["h06s0a", "h06s1a", "h06s2a"])
                remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
