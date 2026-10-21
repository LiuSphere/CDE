import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.report_comparison_results import aggregate, load_evaluation


class ComparisonReportTest(unittest.TestCase):
    def test_recomputes_cost_and_keeps_inventory_modes_and_actors_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = []
            for group, seed, mode, cost in [
                ("C0", None, "continuous", 10), ("C0", None, "daily_reset", 12),
                ("C2", 0, "continuous", 8), ("C2", 1, "continuous", 6)]:
                folder = root / f"{group}_{seed}_{mode}"
                folder.mkdir()
                records = [
                    dict(row_idx=row, col_idx=hour, agent=f"agent{agent}",
                         **{key: cost for key in
                            ("buy_cost", "gen_cost", "bess_cost", "exchange_cost",
                             "economic_cost", "full_cost")})
                    for row in range(913, 1095) for hour in range(24) for agent in range(1, 4)]
                with (folder / "trajectories.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
                summary = dict(
                    rows=list(range(913, 1095)), physical_checks_passed=True,
                    inventory_mode=mode, initial_soc=[.35] * 3, terminal_soc=[.2] * 3,
                    totals={key: cost * len(records) for key in
                            ("buy_cost", "gen_cost", "bess_cost", "exchange_cost",
                             "economic_cost", "full_cost")},
                    days=[dict(row_idx=row, economic_cost=cost * 72)
                          for row in range(913, 1095)])
                (folder / "summary.json").write_text(json.dumps(summary))
                items.append(dict(
                    group=group, seed=seed, policy_source="rule" if group == "C0" else "rl",
                    mode=mode, directory=str(folder), battery_capacities=[220, 180, 600],
                    summary_sha256=hashlib.sha256((folder / "summary.json").read_bytes()).hexdigest(),
                    trajectory_sha256=hashlib.sha256((folder / "trajectories.csv").read_bytes()).hexdigest()))
            rows, stats, _ = aggregate(items)
            c2 = next(s for s in stats if s["group"] == "C2")
            self.assertAlmostEqual(c2["mean_saving_vs_rule"], .3)
            self.assertEqual(c2["seeds"], [0, 1])
            self.assertGreater(c2["sample_std"], 0)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                aggregate(items + [items[0]])
            (Path(items[0]["directory"]) / "trajectories.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_evaluation(items[0])


if __name__ == "__main__":
    unittest.main()
