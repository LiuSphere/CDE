import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_comparison_queue as queue
from tools import experiment_slot


class ComparisonQueueTest(unittest.TestCase):
    def test_commands_use_full_holdout_original_checkpoint_and_inventory_mode(self):
        rows = list(range(913, 1095))
        job = dict(kind="policy", mode="continuous", output="results/new",
                   checkpoint="/original/host/snapshot/selected")
        command = queue.job_command(job, rows)
        self.assertIn("--retrospective-holdout", command)
        self.assertIn("--all-days", command)
        self.assertEqual(command[-1], job["checkpoint"])
        job.update(kind="milp", mode="daily_reset", segments=32, time_limit=60, mip_gap=1e-7)
        command = queue.job_command(job, rows)
        self.assertIn("daily_reset", command)
        self.assertIn("1094", command)
        self.assertIn("913", command)
        self.assertNotIn("--checkpoint", command)

    def test_freeze_checks_metadata_actor_and_output_uniqueness(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            file = root / "implementation.py"
            file.write_text("pass\n")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "agent.th").write_bytes(b"actor")
            (checkpoint / "metadata.json").write_text("{}")
            (checkpoint / "config.yaml").write_bytes(b"config")
            job = dict(
                id="c1", kind="policy", mode="continuous", seed=0,
                output=str(root / "results/out"), checkpoint=str(checkpoint),
                metadata_sha256=hashlib.sha256(b"{}").hexdigest(),
                config_sha256=hashlib.sha256(b"config").hexdigest(),
                policy_sha256=hashlib.sha256(b"actor").hexdigest())
            manifest = dict(
                rows=list(range(913, 1095)),
                implementation_sha256={"implementation.py": hashlib.sha256(file.read_bytes()).hexdigest()},
                jobs=[job])
            with patch.object(queue, "ROOT", root):
                queue.verify_manifest(manifest)
                duplicate = copy.deepcopy(manifest)
                duplicate["jobs"].append({**job, "id": "different"})
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    queue.verify_manifest(duplicate)
                (checkpoint / "agent.th").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "policy changed"):
                    queue.verify_manifest(manifest)

    def test_supervisor_rejects_evaluation_training_flag_mix(self):
        with self.assertRaisesRegex(ValueError, "Evaluation cannot"):
            experiment_slot.supervise("gavdac_mappo", 0, "not_used", 0, 1,
                                      engineering=True, evaluation_manifest="unused")

    def test_queue_runs_isolated_command_and_persists_completion(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            root = Path(directory)
            outer = root / "results/worker"
            outer.mkdir(parents=True)
            output = outer / "rule"
            manifest = dict(
                rows=list(range(913, 1095)), implementation_sha256={},
                jobs=[dict(id="rule", kind="rule", mode="continuous",
                           output=str(output), seed=None)])
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))

            def run(command, **kwargs):
                output.mkdir()
                (output / "summary.json").write_text(json.dumps({
                    "rows": manifest["rows"], "inventory_mode": "continuous",
                    "physical_checks_passed": True}))
                return type("Result", (), {"returncode": 0})()

            with patch.object(queue, "ROOT", root), patch.object(queue.subprocess, "run", run):
                queue.execute(path, 2, "worker")
            result = json.loads((outer / "completion_checks.json").read_text())
            self.assertTrue(result["all_jobs_accepted"])
            self.assertEqual(result["jobs"][0]["command"][0:3], ["uv", "run", "--frozen"])
            with self.assertRaises(FileExistsError):
                with patch.object(queue, "ROOT", root):
                    queue.execute(path, 2, "worker")


if __name__ == "__main__":
    unittest.main()
