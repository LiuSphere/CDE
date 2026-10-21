"""Round registration is identity-checked and never starts a process."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import freeze_c2_round


class RoundRegistrationTest(unittest.TestCase):
    def test_registers_all_hosts_and_chooses_the_earliest_eta(self):
        self.register_fixture()

    def test_rejects_a_different_process_creation_identity(self):
        self.register_fixture(wrong_identity=True)

    def register_fixture(self, wrong_identity=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.json"
            plans, launches = [], []
            for seed in range(3):
                token = f"h01s{seed}a"
                plans.append(dict(seed=seed, token=token,
                                  host="local" if seed == 2 else "707",
                                  root="/project", slot=0 if seed == 2 else seed))
                launch = dict(
                    token=token, seed=seed, config="gavdac_c2_h01", cpu_threads=1,
                    git_commit="frozen", supervisor_pid=12, supervisor_identity="id12",
                    child_pid=13, child_identity="id13",
                    created_at="2026-10-21T00:00:00+00:00")
                launches.append(launch)
                (root / f"{token}_detach.json").write_text(json.dumps(
                    dict(manager_pid=11, manager_identity="id11")))
            local = root / "results/h01s2a"
            local.mkdir(parents=True)
            (local / "launch.json").write_text(json.dumps(launches[2]))
            state = dict(status="H01_verified_ready_to_launch", active_runs=[],
                         implementation_commit="frozen", run_plan=plans, scheduling={})
            state_path.write_text(json.dumps(state))
            identities = json.dumps({"11": "id11", "12": "id12", "13": "id13"})
            with patch.object(freeze_c2_round, "ROOT", root), \
                 patch.object(freeze_c2_round.subprocess, "check_output", side_effect=[
                     json.dumps(launches[0]), identities, json.dumps(launches[1]), identities]), \
                 patch("tools.experiment_slot.process_identity",
                       side_effect=lambda pid: "wrong" if wrong_identity else f"id{pid}"):
                if wrong_identity:
                    with self.assertRaisesRegex(ValueError, "creation identity"):
                        freeze_c2_round.register_runs(state_path)
                    self.assertEqual(json.loads(state_path.read_text()), state)
                    return
                freeze_c2_round.register_runs(state_path)
            saved = json.loads(state_path.read_text())
            self.assertEqual(len(saved["active_runs"]), 3)
            self.assertEqual(saved["status"], "H01_three_seeds_running")
            self.assertEqual(saved["scheduling"]["earliest_eta_utc"],
                             saved["active_runs"][2]["eta_utc"])

    def test_already_registered_state_cannot_be_registered_again(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps(dict(status="H01_three_seeds_running",
                                            active_runs=[{"token": "existing"}])))
            with self.assertRaisesRegex(ValueError, "exactly once"):
                freeze_c2_round.register_runs(path)


if __name__ == "__main__":
    unittest.main()
