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


CONFIGS = [
    "gavdac_ea_tune_v03_long", "gavdac_v03_no_ea",
    "gavdac_v03_no_mutation", "gavdac_v03_no_crossover",
]


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
    return {
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
        "crossover_fix": "p=0 now disables clone recombination; p=1 operations and RNG sequence unchanged",
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
