"""Paired timing order, frozen inputs and actual-budget guards, without training."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_parallel_rollouts as benchmark


class ExecutionBenchmarkTest(unittest.TestCase):
    def test_paired_order_hashes_and_gains(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            directory = Path(directory)
            roots = [directory / name for name in ("candidate", "reference")]
            files = [
                "src/envs/maenv.py", "src/config/default.yaml",
                "src/config/envs/microgrid.yaml", "src/config/algs/gavdac_ea_main.yaml",
                "uv.lock", "pyproject.toml", "tests/test_training_timing.py",
                "tools/benchmark_parallel_rollouts.py",
                *(f"src/envs/data/{key}.csv" for key in ("load", "pv", "price")),
            ]
            for index, root in enumerate(roots):
                for name in files:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"same\r\n" if index == 0 else b"same\n")
            order = []

            def execute(command, *, cwd, **kwargs):
                order.append(cwd.name)
                seconds = 10 if cwd == roots[1] else 8
                path = Path(command[command.index("--output") + 1])
                rows = [
                    {"kind": kind, "seconds": seconds, "t_env": 28800}
                    for kind in ("wall_clock", "training_wall_clock", "post_warmup_wall_clock")
                ]
                rows += [{"kind": "learner_update", "seconds": 1, "t_env": 28800}] * 2
                with path.open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=("kind", "seconds", "t_env"))
                    writer.writeheader()
                    writer.writerows(rows)

            argv = ["benchmark", "--output", str(directory / "output"),
                    "--reference-root", str(roots[1]), "--setting", "4", "4",
                    "--timesteps", "28800", "--repeats", "2"]
            with patch.object(benchmark, "ROOT", roots[0]), \
                 patch.object(sys, "argv", argv), \
                 patch.object(benchmark.subprocess, "run", execute):
                benchmark.main()
            self.assertEqual(order, ["reference", "candidate", "candidate", "reference"])
            paired = json.loads((directory / "output/paired.json").read_text())[0]
            self.assertEqual(paired["each_pair_total_throughput_gain"], [.25, .25])
            self.assertEqual(paired["wall_clock_throughput_gain"], .25)
            manifest = json.loads((directory / "output/manifest.json").read_text())
            self.assertNotEqual(manifest["input_sha256"]["candidate"],
                                manifest["input_sha256"]["reference"])
            (roots[1] / "src/envs/data/load.csv").write_text("different")
            with self.assertRaisesRegex(ValueError, "shared input differs"):
                benchmark.check_shared_inputs(roots[0], roots[1], "gavdac_ea_main")


if __name__ == "__main__":
    unittest.main()
