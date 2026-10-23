"""Retired H05 guards using real baseline and synthetic candidate, not failed runtime."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.freeze_h05_protocol import (
    ADDITIONS, C3_COSTS, FOCUSED_TESTS, HOST_PROBE, IMPLEMENTATION, MANUSCRIPTS,
    compare_host, entry_config, evaluation_protocol, make_plans, revalidate_C3,
    validate_baseline, validate_candidate, validate_evaluation, validate_plans,
    validate_rollback, validate_tooling,
)


class H05ProtocolTest(unittest.TestCase):
    def test_baseline_entry_and_synthetic_retired_H05_guards(self):
        workers = {"rollout_workers": 2, "learner_workers": 2}
        parent = entry_config("gavdac_ea_main", workers)
        candidate = {**parent, "name": "gavdac_c2_h05", "five_step_critic": True}
        validate_candidate(parent, candidate)
        for key, value in (
            ("five_step_critic", False), ("five_step_critic", 1), ("lr", .001),
            ("actor_update_interval", 2), ("scaled_huber_critic", True),
            ("clipped_double_critic", True), ("actor_mixer_context_grad", False),
            ("rollout_workers", 4), ("learner_workers", 0), ("t_max", 12500),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_candidate(parent, {**candidate, key: value})

    def test_joint_budget_CPU_fresh_weights_selection_and_online_are_fixed(self):
        workers = {"rollout_workers": 2, "learner_workers": 2}
        parent = entry_config("gavdac_ea_main", workers)
        candidate = {**parent, "name": "gavdac_c2_h05", "five_step_critic": True}
        for key, value in (
            ("t_max", 500000), ("t_max", 6000000.0), ("use_cuda", True),
            ("learner_workers", 2.0), ("checkpoint_path", "old"), ("policy_checkpoint", "old"),
            ("validation_rows", [913]), ("test_interval", 1000), ("evaluation_initial_soc", .8),
            ("save_model", False), ("wandb_mode", "offline"), ("evaluate", True),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_candidate({**parent, key: value}, {**candidate, key: value})
        with self.assertRaises(ValueError):
            validate_candidate({**parent, "scaled_huber_critic": True},
                               {**candidate, "scaled_huber_critic": True})

    def test_seed_host_slot_worker_and_budget_assignments(self):
        plans = make_plans(["h05s0a", "h05s1a", "h05s2a"])
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

    def test_tokens_are_unique_short_H05_only(self):
        for tokens in (
            ["h05s0a"] * 3, ["h05s0a", "../unsafe", "h05s2a"],
            ["h04s0a", "h05s1a", "h05s2a"], ["h05s0a", "h05s1a"],
            ["h05s0a", "h05s1a", "h05s" + "a" * 26], ["h05s0a", None, "h05s2a"],
        ):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                make_plans(tokens)

    def test_evaluation_and_economic_gate_cannot_change(self):
        validate_evaluation(evaluation_protocol())
        for key, value in (
            ("primary_metric", "full_cost"), ("mode", "daily_reset"), ("actor", "ea"),
            ("minimum_saving_fraction", .01), ("rows", [913]), ("initial_soc", .8),
            ("terminal_soc", "cyclic"), ("expected_agent_hours", 13103),
            ("require_each_seed_and_ratio_of_means", False),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_evaluation({**evaluation_protocol(), key: value})

    def test_only_registered_H05_algorithm_is_revertible(self):
        validate_rollback(IMPLEMENTATION, [IMPLEMENTATION])
        for commits in ([], ["9f03926"], [IMPLEMENTATION, "8db8428"]):
            with self.assertRaises(ValueError):
                validate_rollback(IMPLEMENTATION, commits)
        with self.assertRaises(ValueError):
            validate_rollback("support-tool", [IMPLEMENTATION])

    def test_actualraw_resources_dirty_scope_locks_and_tokens(self):
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
        for label in ("raw", "data", "dirty", "staged_manuscript", "token", "slot", "memory",
                      "addition", "missing_tool", "python", "torch", "root", "logical_cpus"):
            changed = copy.deepcopy(current)
            if label == "raw":
                changed["files"]["src/run.py"]["raw"] = "different-raw-same-lf"
            elif label == "data":
                changed["files"]["src/envs/data/load.csv"]["raw"] = "changed-same-lf"
            elif label == "dirty":
                changed["git_status"].append({"status": "??", "path": "unknown.txt"})
            elif label == "staged_manuscript":
                changed["git_status"][0]["status"] = "M "
            elif label == "token":
                changed["existing_tokens"] = ["h05s0a"]
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
        with self.assertRaises(ValueError):
            compare_host(previous, current, 1, "715")

    def test_C3_identity_cost_and_threshold_immutability(self):
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

    def test_original_freeze_raw_checked_before_C3_audit_and_probe_compiles(self):
        compile(HOST_PROBE, "<H05 original-host read-only probe>", "exec")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "h01_protocol_b.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "original C3 freeze"):
                revalidate_C3({"baseline_C3": {}}, base)

    def test_original_host_focused_tool_acceptance_is_bound_to_rawbytes(self):
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


if __name__ == "__main__":
    unittest.main()
