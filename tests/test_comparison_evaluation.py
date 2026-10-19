"""Fresh-campaign inventory boundaries, rule causality and independent accounting."""

import sys
import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import direct_main
from utils.comparison_evaluation import (
    evaluate_inventory, make_evaluation_runner, summarize_trajectories,
)
from utils.rule_policy import TariffSOCPolicy
from utils.policy_evaluation import BestPolicyStore


def entry_config(name="gavdac_ea_main"):
    loaded = []
    with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
         patch.object(direct_main, "run", lambda config, logger: loaded.append(config)):
        direct_main.main()
    return loaded[0]


class ConstantPolicy:
    def __init__(self):
        self.resets = 0
        self.previous_actions = []

    def init_hidden(self, batch_size):
        self.resets += 1

    def select_actions(self, batch, t_ep, t_env, test_mode=False):
        if not test_mode:
            raise AssertionError("Exploration enabled in evaluation")
        if t_ep == 0:
            self.previous_actions.append(batch["actions"][0, 0].clone())
        actions = torch.zeros(1, 3, 2)
        actions[:, :, 1] = 0.10
        return actions


class ComparisonEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(self.directory.cleanup)
        self.config = entry_config()
        self.config["env_args"].update(
            result_dir=self.directory.name, run_id="comparison_tests", seed=0,
        )
        self.rows = [731, 732]

    def run_policy(self, mode, policy=None):
        runner, mac = make_evaluation_runner(self.config, policy)
        summary, records = evaluate_inventory(
            runner, mac, self.rows, 0.35, "validation", mode,
        )
        return runner, mac, summary, records

    def test_matched_config_has_only_ea_and_name_differences(self):
        parent = entry_config()
        child = entry_config("gavdac_ea_main_no_ea")
        self.assertEqual(child, {**parent, "name": "gavdac_ea_main_no_ea", "EA": False})
        self.assertEqual(child["crossover_prob"], 0)
        self.assertEqual(child["t_max"], 6_000_000)

    def test_real_no_ea_training_crosses_warmup_without_population(self):
        import run as training
        from controllers.cqmix_controller import CQMixMAC
        from ea.mod_neuro_evo import SSNE
        from learners.gavdac_learner import GAVDACLearner

        train = GAVDACLearner.train
        counts = {"learner": 0}

        def learn(learner, batch, t_env, episode):
            self.assertFalse(learner.args.EA)
            self.assertEqual(learner.args.crossover_prob, 0)
            self.assertGreaterEqual(t_env, 512 * 24)
            self.assertEqual(batch.batch_size, 128)
            self.assertTrue(torch.equal(learner.mixer.communication_adjacency,
                                        torch.ones(3, 3)))
            result = train(learner, batch, t_env, episode)
            for parameter in learner.mac.parameters():
                self.assertTrue(bool(torch.isfinite(parameter).all()))
            counts["learner"] += 1
            return result

        argv = ["direct_main.py", "--config", "gavdac_ea_main_no_ea"]
        overrides = {
            "t_max": 12500, "wandb_mode": "disabled", "save_model": False,
            "showcase_enabled": False, "unique_token": "no_ea_check",
            "local_results_path": self.directory.name,
            "env_args.result_dir": self.directory.name,
            "wandb_dir": f"{self.directory.name}/wandb",
        }
        for key, value in overrides.items():
            argv += ["--override", f"{key}={value}"]
        with patch.object(sys, "argv", argv), \
             patch.object(training, "CQMixMAC", wraps=CQMixMAC) as construct, \
             patch.object(SSNE, "epoch", side_effect=AssertionError("EA executed")), \
             patch.object(GAVDACLearner, "train", learn):
            direct_main.main()
        self.assertEqual(construct.call_count, 1)
        self.assertGreater(counts["learner"], 0)

    def test_freeze_exports_fresh_matrix_and_predeclared_decisions(self):
        from tools.freeze_paper_protocol import freeze
        protocol = freeze()
        self.assertFalse(protocol["historical_reuse"])
        self.assertEqual(protocol["formal_training_runs"], 9)
        self.assertEqual(set(protocol["configs"]),
                         {"gavdac_ea_main", "gavdac_ea_main_no_ea"})
        self.assertEqual(protocol["splits"]["retrospective_holdout"], [913, 1094])
        self.assertEqual(protocol["decision_protocol"]["practical_improvement_fraction"], 0.01)

    def test_continuous_soc_and_daily_policy_resets(self):
        policy = ConstantPolicy()
        runner, _, summary, records = self.run_policy("continuous", policy)
        self.assertEqual(policy.resets, 2)
        self.assertTrue(all(torch.count_nonzero(value) == 0
                            for value in policy.previous_actions))
        self.assertEqual(summary["days"][0]["terminal_soc"], summary["days"][1]["initial_soc"])
        self.assertNotEqual(summary["days"][1]["initial_soc"], [0.35] * 3)
        self.assertEqual(runner.t_env, 0)
        self.assertEqual(len(records), 144)
        self.assertTrue(summary["physical_checks_passed"])
        self.assertLess(summary["max_errors"]["balance"], 1e-10)
        self.assertEqual(summary["max_errors"]["cost"], 0)

    def test_daily_reset_is_a_different_inventory_task(self):
        _, _, continuous, _ = self.run_policy("continuous", ConstantPolicy())
        _, _, daily, _ = self.run_policy("daily_reset", ConstantPolicy())
        self.assertEqual(daily["days"][1]["initial_soc"], [0.35] * 3)
        self.assertNotEqual(continuous["totals"]["economic_cost"], daily["totals"]["economic_cost"])
        self.assertAlmostEqual(sum(day["full_cost"] for day in continuous["days"]),
                               continuous["totals"]["full_cost"], places=9)

    def test_independent_accounting_detects_cost_and_inventory_corruption(self):
        runner, _, _, records = self.run_policy("continuous", ConstantPolicy())
        for field in ("full_cost", "soc_init", "pd", "peer_import", "load"):
            corrupt = deepcopy(records)
            corrupt[0][field] += 100
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_trajectories(corrupt, runner.env, self.rows, 0.35, "continuous")
        corrupt = deepcopy(records)
        corrupt[0]["full_cost"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            summarize_trajectories(corrupt, runner.env, self.rows, 0.35, "continuous")

    def test_duplicate_missing_and_nonadjacent_rows_are_rejected(self):
        runner, mac = make_evaluation_runner(self.config, ConstantPolicy())
        for rows in ([], [731, 731], [732, 731], [731, 733], [0, 1], [731.0, 732.0]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate_inventory(runner, mac, rows, 0.35, "validation", "continuous")
        _, _, _, records = self.run_policy("continuous", ConstantPolicy())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_trajectories(records[:-1] + [records[0]], runner.env,
                                   self.rows, 0.35, "continuous")

    def test_real_actor_is_repeatable_and_does_not_touch_training_env(self):
        training, actor = make_evaluation_runner(self.config)
        training.run(actor, test_mode=False)
        training_state = training.env.training_scenario()
        rng_state = deepcopy(training.env.rng.bit_generator.state)
        training_steps = training.t_env
        runner, mac = make_evaluation_runner(self.config)
        mac.load_state(actor)
        first, first_records = evaluate_inventory(
            runner, mac, self.rows, 0.35, "validation", "continuous",
        )
        second, second_records = evaluate_inventory(
            runner, mac, self.rows, 0.35, "validation", "continuous",
        )
        self.assertEqual(first, second)
        # Episode counters differ, but the policy and physical trajectories do not.
        for left, right in zip(first_records, second_records):
            self.assertEqual(left["action_pd"], right["action_pd"])
            self.assertEqual(left["action_pb"], right["action_pb"])
            self.assertEqual(left["soc"], right["soc"])
        self.assertEqual(training.env.training_scenario(), training_state)
        self.assertEqual(training.env.rng.bit_generator.state, rng_state)
        self.assertEqual(training.t_env, training_steps)
        self.assertEqual(runner.batch["obs"].dtype, torch.float32)
        self.assertEqual(runner.batch["obs"].device.type, "cpu")

    def test_rule_is_causal_train_thresholded_and_uses_runner_interface(self):
        runner, _ = make_evaluation_runner(self.config)
        settings = direct_main.load_yaml(
            ROOT / "src/config/baselines/microgrid_rule.yaml",
        )
        policy = TariffSOCPolicy(runner.env, settings)
        runner.env.reset_test(source="validation", row_idx=731, initial_soc=0.35)
        observation = np.asarray(runner.env.get_obs())
        original = policy.actions(observation)
        runner.env.agent_data["load"][732:] *= 10
        runner.env.agent_data["pv"][732:] *= 10
        runner.env.data_dict["price"][732:] *= 10
        np.testing.assert_array_equal(policy.actions(observation), original)
        rebuilt = TariffSOCPolicy(runner.env, settings)
        self.assertEqual(policy.specification(), rebuilt.specification())
        self.assertTrue(np.all(original >= runner.env.action_lows))
        self.assertTrue(np.all(original <= runner.env.action_highs))
        fresh, _ = make_evaluation_runner(self.config, policy)
        summary, _ = evaluate_inventory(
            fresh, policy, self.rows, 0.35, "validation", "continuous", policy_source="rule",
        )
        self.assertTrue(summary["physical_checks_passed"])
        self.assertGreater(summary["totals"]["charge_kwh"], 0)
        self.assertGreater(summary["totals"]["discharge_kwh"], 0)

    def test_rule_high_tariff_threshold_survives_float32_observations(self):
        runner, _ = make_evaluation_runner(self.config)
        settings = direct_main.load_yaml(ROOT / "src/config/baselines/microgrid_rule.yaml")
        policy = TariffSOCPolicy(runner.env, settings)
        obs = np.asarray(runner.env.get_obs())
        obs[:, 0] = policy.high_price / runner.env.observation_price_scale
        obs[:, 1] = 100 / runner.env.observation_power_scale
        obs[:, 2] = 0
        obs[:, 3] = 0.5
        actions = policy.actions(obs.astype(float))
        self.assertTrue(np.all(actions[:, 1] > 0))
        np.testing.assert_array_equal(actions, policy.actions(obs))

    def test_command_line_rule_and_checkpoint_with_strict_snapshot_checks(self):
        root = Path(self.directory.name)
        command = [
            "uv", "run", "python", str(ROOT / "tools/evaluate_microgrid_comparison.py"),
            "--mode", "continuous", "--rows", "731", "732",
        ]
        rule_run = subprocess.run(
            [*command, "--rule", "--output", str(root / "rule")],
            cwd=ROOT, text=True, capture_output=True, check=True, timeout=60,
        )
        self.assertIn('"physical_checks_passed": true', rule_run.stdout)
        runner, mac = make_evaluation_runner(self.config)
        evaluation = {"rows": list(range(731, 739)), "mean_return": 0.0}
        store = BestPolicyStore(root / "best", self.config)
        checkpoint = store.consider(mac, "rl", 0, evaluation, runner.env)
        subprocess.run(
            [*command, "--checkpoint", str(checkpoint), "--output", str(root / "actor")],
            cwd=ROOT, text=True, capture_output=True, check=True, timeout=60,
        )
        summary = json.loads((root / "actor/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["policy_source"], "rl")
        self.assertEqual(summary["policy_sha256"], store.best["rl"]["policy_sha256"])
        self.assertTrue((root / "actor/runtime/envs/data/load.csv").is_file())
        # Corruption stays in an owned test fixture, never in current source.
        snapshot = root / "best/source/utils/rule_policy.py"
        snapshot.write_text("# tampered fixture\n", encoding="utf-8")
        corrupt = subprocess.run(
            [*command, "--checkpoint", str(checkpoint), "--output", str(root / "corrupt")],
            cwd=ROOT, text=True, capture_output=True, timeout=60,
        )
        self.assertNotEqual(corrupt.returncode, 0)
        self.assertIn("Checkpoint source changed", corrupt.stderr)

    def test_command_line_requires_explicit_holdout_gate(self):
        result = subprocess.run(
            ["uv", "run", "python", str(ROOT / "tools/evaluate_microgrid_comparison.py"),
             "--rule", "--mode", "continuous", "--split", "test", "--all-days",
             "--output", str(Path(self.directory.name) / "blocked")],
            cwd=ROOT, text=True, capture_output=True, timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--retrospective-holdout", result.stderr)
        self.assertFalse((Path(self.directory.name) / "blocked").exists())


if __name__ == "__main__":
    unittest.main()
