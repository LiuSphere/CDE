"""Only H11's registered child is newly dispatchable; no old identities reopen."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experiment_slot


class H11LaunchTest(unittest.TestCase):
    def test_only_registered_child_added_parent_and_old_rounds_remain_rejected(self):
        experiment_slot.validate_job("gavdac_c2_h11", 2, "h11_guard", 0, 1)
        for name in ["gavdac_c2_h11_parent", *[f"gavdac_c2_h{i:02d}" for i in range(1, 11)]]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "formal matrix"):
                experiment_slot.validate_job(name, 0, "retired_guard", 0, 1)
        with self.assertRaisesRegex(ValueError, "Engineering budget"):
            experiment_slot.supervise("gavdac_c2_h11", 0, "no_pilot", 0, 1,
                                      engineering=True)
        with self.assertRaises(ValueError):
            experiment_slot.validate_worker_options("gavdac_c2_h11", 1, 2, 2, "frozen.json")

    def test_H11_supervisor_keeps_original_exclusive_smoke_and_worker_contract(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results", prefix="h11_launch_guard_") as directory:
            root = Path(directory)
            config = root / "src/config/algs/gavdac_c2_h11.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("name: gavdac_c2_h11\n")
            process = SimpleNamespace(pid=123, wait=lambda: 0, poll=lambda: 0)
            with patch.object(experiment_slot, "ROOT", root), \
                 patch.object(experiment_slot, "process_identity", return_value="fixture_creation"), \
                 patch.object(experiment_slot.shutil, "which", return_value="uv"), \
                 patch.object(experiment_slot.subprocess, "Popen", return_value=process), \
                 patch.object(experiment_slot.subprocess, "check_output", return_value="fixture_commit"):
                result = experiment_slot.supervise(
                    "gavdac_c2_h11", 2, "h11_guard", 0, 1, smoke=True,
                    rollout_workers=2, learner_workers=2, threads=1,
                )
            self.assertEqual(result, 0)
            launch = json.loads((root / "results/h11_guard/launch.json").read_text())
            overrides = {}
            for index, arg in enumerate(launch["command"]):
                if arg == "--override":
                    key, value = launch["command"][index + 1].split("=", 1)
                    overrides[key] = json.loads(value)
            self.assertEqual((launch["rollout_workers"], launch["learner_workers"],
                              launch["cpu_threads"]), (2, 2, 1))
            self.assertEqual(overrides["t_max"], 12500)
            self.assertEqual(overrides["wandb_mode"], "disabled")
            self.assertFalse(overrides["save_model"])
            self.assertFalse(overrides["showcase_enabled"])
            self.assertFalse(overrides["use_cuda"])
            self.assertFalse((root / "results/cmp20/slot_locks/slot0.json").exists())
            self.assertEqual(json.loads((root / "results/h11_guard/exit.json").read_text())["status"],
                             "exited_pending_validation")


if __name__ == "__main__":
    unittest.main()
