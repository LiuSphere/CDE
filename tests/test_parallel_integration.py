"""Main-branch compatibility gates for the optional execution pools."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from learners.gavdac_learner import GAVDACLearner
from test_gavdac_sequence import LossLog, fixture
from test_policy_evaluation import setup_runner
from utils.policy_evaluation import BestPolicyStore, evaluate_fixed
from tools.experiment_slot import validate_job, validate_worker_options


class ParallelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_retired_hypotheses_cannot_be_dispatched_after_execution_merge(self):
        for name in ("gavdac_c2_h01", "gavdac_c2_h02"):
            self.assertFalse((ROOT / "src/config/algs" / (name + ".yaml")).exists())
            with self.assertRaisesRegex(ValueError, "formal matrix"):
                validate_job(name, 0, "retiredcheck", 0, 1)
        self.assertFalse((ROOT / "src/learners/clipped_double_gavdac_learner.py").exists())
        validate_worker_options("gavdac_ea_main", 1, 2, 2)

    def test_baseline_keeps_main_actor_context_gradient(self):
        batch, mac, args, scheme = fixture(use_cuda=False)
        learner = GAVDACLearner(mac, scheme, LossLog(), args)
        contexts = []
        hook = learner.mixer.register_forward_pre_hook(
            lambda module, items, kwargs: contexts.append(
                kwargs["hidden_states"].requires_grad,
            ), with_kwargs=True,
        )
        try:
            learner.train(batch, 0, 0)
            self.assertEqual(contexts, [False, True])
            self.assertIsNone(learner.gradient_pool)
        finally:
            hook.remove()
            learner.close()

    def test_parallel_validation_checkpoint_uses_its_strict_saved_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            config, runner, mac = setup_runner(directory)
            config.update(rollout_workers=2, learner_workers=2)
            runner.args.rollout_workers = 2
            runner.args.learner_workers = 2
            try:
                evaluation = evaluate_fixed(
                    runner, mac, [731, 732], .35, "validation", "rl", 24)
                store = BestPolicyStore(Path(directory) / "best", config)
                checkpoint = store.consider(mac, "rl", 24, evaluation, runner.env)
                metadata = json.loads((checkpoint / "metadata.json").read_text())
                for name in ("learners/gradient_pool.py", "runners/rollout_pool.py"):
                    self.assertIn(name, metadata["source_sha256"])
            finally:
                runner.close_env()
            output = Path(directory) / "isolated_validation"
            result = subprocess.run(
                ["uv", "run", "--frozen", "python", "tools/evaluate_microgrid_comparison.py",
                 "--checkpoint", str(checkpoint), "--split", "validation", "--mode", "continuous",
                 "--rows", "731", "732", "--output", str(output)],
                cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                   OPENBLAS_NUM_THREADS="1"),
                capture_output=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["physical_checks_passed"])
            self.assertEqual(summary["policy_sha256"], metadata["policy_sha256"])
            for name, digest in metadata["source_sha256"].items():
                self.assertEqual(hashlib.sha256((output / "runtime" / name).read_bytes()).hexdigest(),
                                 digest)


if __name__ == "__main__":
    unittest.main()
