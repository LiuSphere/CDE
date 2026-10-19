"""Freeze the actual entry-loaded paper configs and source/input identities."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import direct_main


CONFIGS = ["gavdac_ea_main_no_ea", "gavdac_ea_main"]


def hashes(paths):
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def freeze():
    configs = {}
    for name in CONFIGS:
        with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
             patch.object(direct_main, "run", lambda config, logger: configs.update({name: config})):
            direct_main.main()
    with (ROOT / "src/envs/data/load.csv").open(encoding="utf-8") as file:
        days = sum(1 for _ in csv.reader(file))
    validation_end = 731 + (days - 731 + 1) // 2
    if days != 1095:
        raise ValueError("Review the frozen split protocol before using different input dimensions")
    parent = configs["gavdac_ea_main"]
    child = configs["gavdac_ea_main_no_ea"]
    if child != {**parent, "name": "gavdac_ea_main_no_ea", "EA": False}:
        raise ValueError("C1 differs from C2 beyond EA and its name")
    return {
        "campaign": "Fresh full-connectivity GAVDAC comparison",
        "historical_reuse": False,
        "formal_matrix": {"C0": "rule", "C1": CONFIGS[0], "C2": CONFIGS[1],
                          "C3": "MAPPO, implementation/configuration pending P3",
                          "C4": "physical MILP, continuous boundary validation pending P1"},
        "formal_seeds": [0, 1, 2],
        "formal_training_runs": 9,
        "stage_scope": "P0 C0/C1/C2 freeze; P1/P3 need separate verified implementation records",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        ),
        "configs": configs,
        "source_sha256": hashes(
            [p for p in (ROOT / "src").rglob("*") if p.suffix in {".py", ".yaml"}]
        ),
        "data_sha256": hashes((ROOT / "src/envs/data").glob("*.csv")),
        "tool_sha256": hashes([
            ROOT / "tools/freeze_paper_protocol.py",
            ROOT / "tools/evaluate_microgrid_comparison.py",
            ROOT / "tools/run_physical_microgrid_milp.py",
        ]),
        "document_sha256": hashes([
            ROOT / "docs/paper_experiment_plan.md", ROOT / "docs/paper_execution.md",
        ]),
        "rule_settings": direct_main.load_yaml(
            ROOT / "src/config/baselines/microgrid_rule.yaml",
        ),
        "splits": {
            "train": [0, 730], "validation": [731, validation_end - 1],
            "retrospective_holdout": [validation_end, days - 1],
            "checkpoint_validation": list(range(731, 739)),
            "candidate_validation": [740, 766, 808, 862, 880, 904],
        },
        "test_limitation": "Historically reused for physical/environment diagnostics; not untouched external data",
        "rng_protocol": "Existing NumPy/PyTorch/MGEnv seeding; Python/fastrand/Gym spaces not fully seeded",
        "training_budget": "6M total non-test environment steps, including EA population interactions",
        "primary_actor": "Validation-selected RL Actor; EA separately supplementary",
        "inventory_protocol": {
            "primary": "continuous SOC", "secondary": "daily reset SOC",
            "initial_soc": 0.35, "terminal_soc": "free within device bounds",
            "hidden_state_and_previous_action": "reset daily for all learning actors",
            "checkpoint_selection": "8 independent days, mean validation return",
        },
        "decision_protocol": {
            "primary_metric": "whole-period economic_cost",
            "secondary_metric": "whole-period full_cost",
            "practical_improvement_fraction": 0.01,
            "direction_requirement": "all three seeds lower than C0; all three C2/C1 label differences lower",
            "interpretation": "1% practical threshold and three-seed consistency are descriptive, not significance",
            "statistics": "per-seed whole-period totals, mean, sample standard deviation (ddof=1)",
            "daily_comparisons": "descriptive only; consecutive days are not independent training repeats",
        },
        "tolerances": {"balance_power": 1e-8, "soc_transition": 1e-10,
                       "cost_per_agent_step": 1e-8, "action_box": 1e-7},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Generated protocol must stay under results/")
    payload = freeze()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({"output": str(output), "splits": payload["splits"]}, indent=2))


if __name__ == "__main__":
    main()
