"""Best-case displays preserve training state and never hide unqualified days."""

import json
import random
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_policy_evaluation import setup_runner
from utils.dispatch_showcase import DispatchShowcase, day_metrics, plot_day, select_days
from utils.logging import Logger
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed


class ShowcaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_select_qualified_days_before_lucrative_inventory_depletion(self):
        metrics = [
            {"row_idx": 740, "qualified": False, "inventory_adjusted_saving_ratio": 0.8,
             "max_terminal_soc_error": 0.15},
            {"row_idx": 766, "qualified": True, "inventory_adjusted_saving_ratio": 0.2,
             "max_terminal_soc_error": 0.01},
            {"row_idx": 808, "qualified": True, "inventory_adjusted_saving_ratio": 0.3,
             "max_terminal_soc_error": 0.02},
        ]
        self.assertEqual([d["row_idx"] for d in select_days(metrics, 3)], [808, 766])
        diagnostics = select_days(metrics[:1], 2)
        self.assertFalse(diagnostics[0]["qualified"])
        with self.assertRaises(ValueError):
            select_days(metrics, 0)

    def test_daily_metrics_expose_depletion_and_balance_failures(self):
        _, runner, mac = setup_runner(self.path)
        evaluate_fixed(runner, mac, [740], 0.35, "validation", "rl", 1)
        records = deepcopy(runner.env.ep_buffer)
        metric = day_metrics(records, runner.env, 0.35, 0.03)
        self.assertLess(metric["max_balance_error"], 1e-8)
        records[-1]["soc"] = 0.2
        self.assertFalse(day_metrics(records, runner.env, 0.35, 0.03)["qualified"])
        records[0]["pg"] += 1
        self.assertGreater(day_metrics(records, runner.env, 0.35, 0.03)["max_balance_error"], 0.9)
        with self.assertRaises(ValueError):
            day_metrics(records[:-1], runner.env, 0.35, 0.03)

    def test_publish_preserves_rng_clock_weights_and_uses_immutable_checkpoint(self):
        config, runner, mac = setup_runner(self.path)
        runner.args.showcase_candidate_rows = [740, 766]
        runner.args.showcase_days = 1
        evaluation = evaluate_fixed(runner, mac, [731, 732], 0.35, "validation", "rl", 12)
        store = BestPolicyStore(self.path / "best", config)
        checkpoint = store.consider(mac, "rl", 12, evaluation, runner.env)
        before = {k: v.clone() for k, v in mac.agent.state_dict().items()}
        with torch.no_grad():
            for parameter in mac.parameters():
                parameter.add_(0.1)
        live = {k: v.clone() for k, v in mac.agent.state_dict().items()}
        # Reproduce the hidden state left by differentiable learner inference.
        mac.hidden_states = torch.ones(1, 3, 64, requires_grad=True) * 2
        training_state = runner.env.training_scenario()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        torch_state = torch.get_rng_state().clone()
        hidden = mac.hidden_states
        logger = MagicMock()
        reporter = DispatchShowcase(self.path / "showcase", logger, runner.args)
        reporter.publish_final(runner, mac, store, 48)
        self.assertEqual(runner.t_env, 0)
        self.assertEqual(training_state, runner.env.training_scenario())
        self.assertIs(hidden, mac.hidden_states)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        for name, value in mac.agent.state_dict().items():
            torch.testing.assert_close(value, live[name], rtol=0, atol=0)
        directory, report, images, step = logger.log_dispatch_report.call_args.args
        self.assertEqual(step, 48)
        self.assertEqual(report["checkpoint_step"], 12)
        self.assertEqual(report["candidate_rows"], [740, 766])
        self.assertEqual(len(images[1]), 5)
        self.assertIn("not typical", report["scope"])
        saved = torch.load(directory / "checkpoint/agent.th", weights_only=True)
        for name, value in saved.items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertEqual(len(list(directory.rglob("*.png"))), 5)
        self.assertTrue((directory / "candidate_trajectories.csv").is_file())
        reporter.publish_final(runner, mac, store, 1000000)
        self.assertEqual(logger.log_dispatch_report.call_count, 1)
        runner.close_env()

    def test_display_rejects_test_day_selection(self):
        config, runner, mac = setup_runner(self.path)
        runner.args.showcase_candidate_rows = [913]
        evaluation = evaluate_fixed(runner, mac, [731], 0.35, "validation", "rl", 12)
        store = BestPolicyStore(self.path / "best", config)
        store.consider(mac, "rl", 12, evaluation, runner.env)
        reporter = DispatchShowcase(self.path / "showcase", MagicMock(), runner.args)
        with self.assertRaisesRegex(ValueError, "validation"):
            reporter.publish_final(runner, mac, store, 48)

    def test_wandb_media_is_not_averaged_as_a_scalar_or_logged_at_old_checkpoint_step(self):
        logger = Logger(MagicMock())
        logger.wandb_run = MagicMock()
        logger.wandb_run.id = "test123"
        report = {
            "policy_source": "rl", "checkpoint_step": 24, "logged_step": 100,
            "policy_sha256": "abc", "candidate_rows": [740],
            "scope": "Selected best case, not typical", "qualified_count": 0,
            "initial_soc": 0.35,
            "candidates": [{"row_idx": 740, "qualified": False}],
            "selected": [{"row_idx": 740, "qualified": False, "max_terminal_soc_error": 0.15}],
        }
        logger.log_stat("train/loss", 1.2, 100)
        with patch("utils.logging.wandb.Image") as image, \
             patch("utils.logging.wandb.Table") as table, \
             patch("utils.logging.wandb.Artifact") as artifact:
            logger.log_dispatch_report(self.path, report, {1: {"battery": self.path / "a.png"}}, 100)
        logger.wandb_run.log.assert_not_called()
        logger.flush()
        self.assertNotIn("step", logger.wandb_run.log.call_args.kwargs)
        payload = logger.wandb_run.log.call_args.args[0]
        self.assertEqual(payload["train/loss"], 1.2)
        self.assertEqual(payload["environment_step"], 100)
        self.assertNotIn("dispatch/rl/best_case_1/battery", logger.stats)
        self.assertIn("qualified=False", image.call_args.kwargs["caption"])
        artifact.assert_not_called()
        logger.wandb_run.log_artifact.assert_not_called()
        table.assert_called_once()

    def test_both_policy_reports_and_later_scalars_share_one_step(self):
        logger = Logger(MagicMock())
        logger.wandb_run = MagicMock()
        report = {
            "policy_source": "rl", "checkpoint_step": 24, "logged_step": 100,
            "policy_sha256": "abc", "candidate_rows": [740],
            "scope": "Selected validation best cases", "qualified_count": 0,
            "candidates": [{"row_idx": 740, "qualified": False}], "selected": [],
        }
        with patch("utils.logging.wandb.Table"), patch("utils.logging.wandb.Artifact"):
            logger.log_dispatch_report(self.path, report, {}, 100)
            logger.log_dispatch_report(self.path, {**report, "policy_source": "ea"}, {}, 100)
        logger.log_stat("train/loss", 1.2, 100)
        logger.wandb_run.log.assert_not_called()
        logger.flush()
        logger.wandb_run.log.assert_called_once()
        payload = logger.wandb_run.log.call_args.args[0]
        self.assertIn("dispatch/rl/candidate_days", payload)
        self.assertIn("dispatch/ea/candidate_days", payload)
        self.assertEqual(payload["train/loss"], 1.2)
        self.assertEqual(payload["environment_step"], 100)

    def test_upload_filter_keeps_analysis_scalars_and_full_local_diagnostics(self):
        import yaml

        config = yaml.safe_load((ROOT / "src/config/default.yaml").read_text(encoding="utf-8"))
        logger = Logger(MagicMock())
        with patch("utils.logging.wandb.init", return_value=MagicMock()) as init:
            logger.setup_wandb(config, "test", "test", None, "disabled", str(self.path))
        for key in (
            "eval/rl_reward", "eval/ea_reward", "loss/critic_loss",
            "train/return_mean", "test/full_cost", "train/battery_charge_energy",
            "test/soc_violation", "test/spill_energy",
        ):
            logger.log_stat(key, 1.0, 100)
            self.assertIn(key, logger.pending_metrics)
        for key in ("train/t_env", "test/episode_limit", "train/buy_cost_reward"):
            logger.log_stat(key, 2.0, 100)
            self.assertNotIn(key, logger.pending_metrics)
            self.assertIn(key, logger.stats)
        logger.flush()
        payload = init.return_value.log.call_args.args[0]
        self.assertEqual(payload["environment_step"], 100)
        self.assertNotIn("train/buy_cost_reward", payload)

    def test_real_entry_point_reports_only_after_training_finishes(self):
        import direct_main

        limit = 12500
        calls = []
        original = DispatchShowcase.publish_final

        def at_completion(reporter, runner, mac, store, step):
            self.assertGreater(step, limit)
            self.assertEqual(runner.t_env, step)
            self.assertIn("rl", store.best)
            self.assertIn("ea", store.best)
            calls.append(step)
            return original(reporter, runner, mac, store, step)

        argv = [
            "direct_main.py", "--config", "gavdac_ea_tune_v03_long",
            "--env-config", "microgrid",
            "--override", "wandb_mode=disabled",
            "--override", f"t_max={limit}",
            "--override", "test_interval=2000",
            "--override", "fixed_eval_days=2",
            "--override", "showcase_candidate_rows=[740,766]",
            "--override", "showcase_days=1",
            "--override", f"local_results_path={self.path}",
            "--override", f"env_args.result_dir={self.path}",
            "--override", f"wandb_dir={self.path}/wandb",
        ]
        with patch.object(sys, "argv", argv), \
             patch.object(DispatchShowcase, "publish_final", at_completion):
            direct_main.main()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(list((self.path / "dispatch_showcases").rglob("*.png"))), 10)


if __name__ == "__main__":
    unittest.main()
