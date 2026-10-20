"""Slot exclusion, process identities and credential handling without real secrets."""

import json
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import experiment_slot, sync_remote_wandb, validate_experiment_host


class ExperimentSlotTest(unittest.TestCase):
    def test_engineering_budget_cannot_modify_main_algorithm(self):
        with self.assertRaisesRegex(ValueError, "Engineering budget"):
            experiment_slot.supervise("gavdac_ea_main", 0, "bad_engineer", 0, 1,
                                      engineering=True)
        with self.assertRaisesRegex(ValueError, "Engineering budget"):
            experiment_slot.supervise("gavdac_mappo", 0, "bad_both", 0, 1,
                                      smoke=True, engineering=True)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "results")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_live_identity_is_stable_and_missing_pid_is_absent(self):
        first = experiment_slot.process_identity(os.getpid())
        self.assertIsNotNone(first)
        self.assertEqual(first, experiment_slot.process_identity(os.getpid()))
        self.assertIsNone(experiment_slot.process_identity(2147483647))

    def test_task_credential_overrides_conflicting_environment_without_logging_key(self):
        path = self.root / "credential.json"
        path.write_text(json.dumps({"base_url": "https://api.wandb.ai",
                                    "api_key": "fake_task_credential"}))
        path.chmod(0o600)
        with patch.dict(os.environ, {"WANDB_API_KEY": "fake_other_project"}):
            env = experiment_slot.job_environment(1, path)
        self.assertEqual(env["WANDB_API_KEY"], "fake_task_credential")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["MKL_NUM_THREADS"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        with self.assertRaises(ValueError):
            experiment_slot.job_environment(0, path)

    def test_status_distinguishes_free_live_and_stale_slots(self):
        locks = self.root / "results/cmp20/slot_locks"
        locks.mkdir(parents=True)
        lock = {"token": "test", "supervisor_pid": os.getpid(),
                "supervisor_identity": experiment_slot.process_identity(os.getpid())}
        (locks / "slot0.json").write_text(json.dumps(lock))
        with patch.object(experiment_slot, "ROOT", self.root):
            self.assertEqual([s["state"] for s in experiment_slot.slot_status(2)],
                             ["running", "free"])
            lock["supervisor_identity"] = "not_the_same_process"
            (locks / "slot0.json").write_text(json.dumps(lock))
            self.assertEqual(experiment_slot.slot_status(2)[0]["state"], "stale_lock")

    def test_existing_slot_lock_blocks_a_second_launch(self):
        locks = self.root / "results/cmp20/slot_locks"
        locks.mkdir(parents=True)
        (locks / "slot0.json").write_text('{"owner": "existing_job"}')
        config = self.root / "src/config/algs/gavdac_ea_main.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("name: gavdac_ea_main\n")
        with patch.object(experiment_slot, "ROOT", self.root), \
             patch.object(experiment_slot.shutil, "which", return_value="uv"), \
             patch.object(experiment_slot.subprocess, "Popen") as spawn, \
             self.assertRaises(FileExistsError):
            experiment_slot.supervise("gavdac_ea_main", 0, "test_new", 0, 2)
        spawn.assert_not_called()
        self.assertEqual(json.loads((locks / "slot0.json").read_text()), {"owner": "existing_job"})

    def test_supervisor_records_exit_but_never_claims_training_success(self):
        config = self.root / "src/config/algs/gavdac_ea_main.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("name: gavdac_ea_main\n")
        process = SimpleNamespace(pid=os.getpid(), wait=lambda: 7, poll=lambda: 7)
        with patch.object(experiment_slot, "ROOT", self.root), \
             patch.object(experiment_slot.shutil, "which", return_value="uv"), \
             patch.object(experiment_slot.subprocess, "Popen", return_value=process), \
             patch.object(experiment_slot.subprocess, "check_output", return_value="fake_commit"):
            result = experiment_slot.supervise("gavdac_ea_main", 0, "test_exit", 0, 2)
        self.assertEqual(result, 7)
        launch = json.loads((self.root / "results/test_exit/launch.json").read_text())
        self.assertNotIn("WANDB_API_KEY", launch)
        self.assertFalse(launch["smoke"])
        self.assertIn("wandb_mode=\"online\"", launch["command"])
        exit_record = json.loads((self.root / "results/test_exit/exit.json").read_text())
        self.assertEqual(exit_record["returncode"], 7)
        self.assertEqual(exit_record["status"], "exited_pending_validation")
        self.assertFalse((self.root / "results/cmp20/slot_locks/slot0.json").exists())

    def test_interrupted_supervisor_keeps_lock_while_child_is_alive(self):
        config = self.root / "src/config/algs/gavdac_ea_main.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("name: gavdac_ea_main\n")

        def interrupt():
            raise KeyboardInterrupt()

        process = SimpleNamespace(pid=os.getpid(), wait=interrupt, poll=lambda: None)
        with patch.object(experiment_slot, "ROOT", self.root), \
             patch.object(experiment_slot.shutil, "which", return_value="uv"), \
             patch.object(experiment_slot.subprocess, "Popen", return_value=process), \
             patch.object(experiment_slot.subprocess, "check_output", return_value="fake_commit"), \
             self.assertRaises(KeyboardInterrupt):
            experiment_slot.supervise("gavdac_ea_main", 0, "test_interrupt", 0, 2)
        self.assertTrue((self.root / "results/cmp20/slot_locks/slot0.json").exists())
        self.assertFalse((self.root / "results/test_interrupt/exit.json").exists())

    def test_sync_sends_key_on_stdin_only(self):
        key = "fake_key_for_test"
        settings = SimpleNamespace(base_url="https://api.wandb.ai", api_key=key)
        output = {"verified": True, "matches_local": True}
        completed = SimpleNamespace(returncode=0, stdout=json.dumps(output), stderr="")
        with patch.dict(os.environ, {"WANDB_API_KEY": ""}), \
             patch.object(sync_remote_wandb.wandb, "setup", return_value=SimpleNamespace(settings=settings)), \
             patch.object(sync_remote_wandb.subprocess, "run", return_value=completed) as execute:
            self.assertEqual(sync_remote_wandb.sync("707", "/home/arch707/data/CDE"), output)
        positional, keywords = execute.call_args
        self.assertTrue(all(key not in str(arg) for arg in positional[0]))
        self.assertEqual(json.loads(keywords["input"])["api_key"], key)

    def test_sync_error_redacts_credential(self):
        key = "fake_secret_do_not_log"
        settings = SimpleNamespace(base_url="https://api.wandb.ai", api_key=key)
        completed = SimpleNamespace(returncode=1, stdout="", stderr=f"rejected {key}")
        with patch.dict(os.environ, {"WANDB_API_KEY": ""}), \
             patch.object(sync_remote_wandb.wandb, "setup", return_value=SimpleNamespace(settings=settings)), \
             patch.object(sync_remote_wandb.subprocess, "run", return_value=completed), \
             self.assertRaises(RuntimeError) as raised:
            sync_remote_wandb.sync("707", "/home/arch707/data/CDE")
        self.assertNotIn(key, str(raised.exception))
        self.assertIn("[REDACTED]", str(raised.exception))

    def test_host_validation_lists_only_the_explicit_external_fixture_exclusion(self):
        class Case(unittest.TestCase):
            def __init__(self, identifier):
                super().__init__()
                self.identifier = identifier

            def id(self):
                return self.identifier

            def runTest(self):
                pass

        old = "test_sb3_microgrid_env.SB3ReferenceEnvTest.test_external"
        current = "test_graph_mixer.GraphMixerTest.test_current"
        output = self.root / "qualification.json"
        argv = ["validate_experiment_host.py", "--without-desktop-sb3-reference",
                "--output", str(output)]
        suite = unittest.TestSuite([Case(old), Case(current)])
        with patch.object(sys, "argv", argv), \
             patch.object(unittest.defaultTestLoader, "discover", return_value=suite), \
             patch.object(sys, "stdout", io.StringIO()), \
             self.assertRaises(SystemExit) as exited:
            validate_experiment_host.main()
        self.assertEqual(exited.exception.code, 0)
        report = json.loads(output.read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["tests_run"], 1)
        self.assertEqual(report["excluded_not_passed"], [old])


if __name__ == "__main__":
    unittest.main()
