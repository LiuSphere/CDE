"""Retained freeze guards for the retired H03 candidate, without training."""

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.freeze_h03_protocol import (
    ADDITIONS, HOST_PROBE, IMPLEMENTATION, compare_host, entry_config, evaluation_protocol,
    make_plans, validate_baseline, validate_candidate, validate_evaluation,
    validate_plans, validate_rollback,
)


class H03ProtocolTest(unittest.TestCase):
    def test_entry_config_allows_only_registered_algorithm_difference(self):
        compile(HOST_PROBE, "<H03 original-host probe>", "exec")
        workers = {"rollout_workers": 2, "learner_workers": 2}
        parent = entry_config("gavdac_ea_main", workers)
        # Preserve protocol checks without restoring the failed opt-in runtime.
        candidate = {**parent, "name": "gavdac_c2_h03", "actor_update_interval": 2}
        validate_candidate(parent, candidate)
        for key, value in (
            ("actor_update_interval", 1), ("lr", .001), ("learner_workers", 0),
            ("rollout_workers", 4), ("t_max", 12500), ("checkpoint_path", "old"),
            ("learner_workers", 2.0),
        ):
            changed = {**candidate, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_candidate(parent, changed)
        # Even identical changes to both configs cannot weaken absolute campaign guards.
        for key, value in (("t_max", 500000), ("use_cuda", True), ("learner_workers", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_candidate({**parent, key: value}, {**candidate, key: value})

    def test_exact_seed_host_slot_budget_and_token_guards(self):
        plans = make_plans(["h03s0a", "h03s1a", "h03s2a"])
        validate_plans(plans)
        for key, value in (("seed", 2), ("host", "715"), ("slot", 1),
                           ("threads", 2), ("joint_budget", 7000000), ("learner_workers", 0)):
            changed = copy.deepcopy(plans)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plans(changed)
        changed = copy.deepcopy(plans)
        changed[1]["seed"] = True
        with self.assertRaises(ValueError):
            validate_plans(changed)
        with self.assertRaises(ValueError):
            make_plans(["h03s0a"] * 3)
        with self.assertRaises(ValueError):
            make_plans(["h03s0a", "../unsafe", "h03s2a"])

    def test_evaluation_and_rollback_cannot_be_relaxed(self):
        validate_evaluation(evaluation_protocol())
        for key, value in (("primary_metric", "full_cost"), ("mode", "daily_reset"),
                           ("minimum_saving_fraction", .01), ("initial_soc", .8),
                           ("terminal_soc", "cyclic"), ("require_each_seed_and_ratio_of_means", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_evaluation({**evaluation_protocol(), key: value})
        validate_rollback(IMPLEMENTATION, [IMPLEMENTATION])
        for commits in ([], ["9f03926"], [IMPLEMENTATION, "8db8428"]):
            with self.assertRaises(ValueError):
                validate_rollback(IMPLEMENTATION, commits)

    def test_compare_requires_original_raw_bytes_and_only_reviewed_additions(self):
        previous = {
            "files": {"src/run.py": {"raw": "host-raw", "lf": "shared-lf"}},
            "platform": "qualified", "root": "root", "logical_cpus": 12,
        }
        current = {
            **copy.deepcopy(previous),
            "slots": [{"slot": 0, "state": "free"}], "dirty_scope": "", "existing_tokens": [],
            "memory": {"available_bytes": 4 * 1024**3},
        }
        current["files"].update({name: {"raw": name, "lf": name} for name in ADDITIONS})
        compare_host(previous, current, 1, "local")
        for label in ("raw", "dirty", "token", "slot", "memory", "unknown_file", "missing_tool"):
            changed = copy.deepcopy(current)
            if label == "raw":
                changed["files"]["src/run.py"]["raw"] = "different-raw-same-lf"
            elif label == "dirty":
                changed["dirty_scope"] = " M src/run.py"
            elif label == "token":
                changed["existing_tokens"] = ["h03s0a"]
            elif label == "slot":
                changed["slots"][0]["state"] = "stale_lock"
            elif label == "memory":
                changed["memory"]["available_bytes"] = 1024**3
            elif label == "unknown_file":
                changed["files"]["tools/unknown.py"] = {"raw": "x", "lf": "x"}
            else:
                changed["files"].pop(next(iter(ADDITIONS)))
            with self.subTest(label=label), self.assertRaises(ValueError):
                compare_host(previous, changed, 1, "local")

    def test_C3_identities_cost_and_threshold_are_immutable(self):
        from tools.freeze_h03_protocol import C3_COSTS
        original = {str(seed): {"economic_cost": cost, "candidate_maximum_cost": .97 * cost}
                    for seed, cost in enumerate(C3_COSTS)}
        validate_baseline(copy.deepcopy(original), original)
        changed = copy.deepcopy(original)
        changed["1"]["economic_cost"] -= 1
        with self.assertRaises(ValueError):
            validate_baseline(changed, original)
        # Coordinated state/original edits still cannot alter the registered costs.
        changed["1"]["candidate_maximum_cost"] = .97 * changed["1"]["economic_cost"]
        with self.assertRaises(ValueError):
            validate_baseline(changed, changed)


if __name__ == "__main__":
    unittest.main()
