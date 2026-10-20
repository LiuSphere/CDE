"""Freeze the actual validation-selected Actors and the full evaluation queue."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_comparison_queue import verify_manifest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(seed, tokens, queue_token, baselines=False):
    if len(tokens) != 3 or len(set(tokens)) != 3:
        raise ValueError("Provide exactly the fresh C1/C2/C3 token for this seed")
    from tools.experiment_slot import validate_job
    for token in [*tokens, queue_token]:
        validate_job("gavdac_mappo", seed, token, 0, 1)
    output_root = ROOT / "results" / queue_token
    if output_root.exists():
        raise FileExistsError("Use a new evaluation queue token")
    jobs = []
    selected = []
    for group, token in zip(("C1", "C2", "C3"), tokens):
        acceptance = ROOT / "results/cmp20" / f"{token}_acceptance.json"
        accepted = json.loads(acceptance.read_text())
        if (not accepted["accepted"] or accepted["seed"] != seed
                or accepted["group"] != group):
            raise ValueError("Formal run acceptance/seed mismatch")
        sources = ("rl", "ea") if group == "C2" else ("rl",)
        for source in sources:
            pointer = ROOT / "results" / token / "models" / token / "best_policies" / f"best_{source}.json"
            metadata = json.loads(pointer.read_text())
            checkpoint = Path(metadata["checkpoint"])
            actual = json.loads((checkpoint / "metadata.json").read_text())
            config = yaml.safe_load((checkpoint / "config.yaml").read_text(encoding="utf-8"))
            expected_name = {
                "C1": "gavdac_ea_main_no_ea", "C2": "gavdac_ea_main", "C3": "gavdac_mappo",
            }[group]
            if config["seed"] != seed or config["name"] != expected_name:
                raise ValueError("Selected checkpoint does not match the formal method/seed")
            if metadata != actual or metadata["validation"]["rows"] != list(range(731, 739)):
                raise ValueError("Selected checkpoint differs from validation-only pointer")
            if sha256(checkpoint / "agent.th") != metadata["policy_sha256"]:
                raise ValueError("Frozen actor integrity failure")
            for name, expected in metadata["source_sha256"].items():
                if sha256(Path(metadata["source_snapshot"]) / name) != expected:
                    raise ValueError(f"Original selected-policy source changed: {name}")
            for name, expected in metadata["data_sha256"].items():
                if sha256(ROOT / "src/envs/data" / name) != expected:
                    raise ValueError(f"Original input data changed: {name}")
            identity = {
                "group": group, "seed": seed, "policy_source": source, "token": token,
                "checkpoint": str(checkpoint), "checkpoint_step": metadata["checkpoint_step"],
                "metadata_sha256": sha256(checkpoint / "metadata.json"),
                "config_sha256": sha256(checkpoint / "config.yaml"),
                "policy_sha256": metadata["policy_sha256"],
                "selection_pointer_sha256": sha256(pointer),
                "acceptance_sha256": sha256(acceptance),
            }
            selected.append(identity)
            for mode in ("continuous", "daily_reset"):
                job_id = f"{group.lower()}_{source}_{mode}"
                jobs.append({
                    **identity, "kind": "policy", "mode": mode, "id": job_id,
                    "output": str(output_root / job_id),
                })
    if baselines:
        baseline_jobs = []
        for kind, group in (("rule", "C0"), ("milp", "C4")):
            for mode in ("continuous", "daily_reset"):
                job_id = f"{group.lower()}_{mode}"
                baseline_jobs.append({
                    "kind": kind, "group": group, "seed": None, "id": job_id, "mode": mode,
                    "output": str(output_root / job_id),
                    **({"segments": 32, "time_limit": 60, "mip_gap": 1e-7}
                       if kind == "milp" else {}),
                })
        jobs = baseline_jobs + jobs
    files = [
        ROOT / "tools/run_comparison_queue.py",
        ROOT / "tools/evaluate_microgrid_comparison.py",
        ROOT / "tools/run_continuous_microgrid_milp.py",
        ROOT / "tools/run_physical_microgrid_milp.py",
        ROOT / "tools/run_current_model_milp.py",
        ROOT / "tools/freeze_comparison_evaluation.py",
        ROOT / "tools/experiment_slot.py",
        ROOT / "uv.lock", ROOT / "pyproject.toml",
        *[p for p in (ROOT / "src").rglob("*") if p.suffix in {".py", ".yaml", ".csv"}],
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT), "platform": platform.platform(), "seed": seed,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "rows": list(range(913, 1095)), "selected": selected, "jobs": jobs,
        "initial_soc": .35, "final_soc": "free within equipment bounds",
        "primary_metric": "economic_cost", "secondary_metric": "full_cost",
        "primary_actor": "rl", "supplementary_actor": "C2_ea, never selected per day",
        "information": "C0 causal4h features; C4 perfect realized complete horizon",
        "selection": "Validation-only rows731-738; freeze before any retained comparison",
        "limitation": "Retrospective holdout historically used for diagnostics, not untouched external test",
        "statistics": "Per-seed whole-period cost then mean/sample std(ddof1); no independence claim for days",
        "practical_threshold": .01, "all_seed_direction_report": True,
        "representative_rule": "Choose common maximum community load,PV,and tariff daily rows; ties earliest row; exogenous only",
        "solver_protocol": "Economic PWL32,relative gap1e-7,time limit60s per MILP; joint182day main,free-terminal daily reset secondary; never silently replace joint horizon",
        "implementation_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
    }
    verify_manifest(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--tokens", nargs=3, required=True)
    parser.add_argument("--queue-token", required=True)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Freeze manifest must be a new results artifact")
    manifest = freeze(args.seed, args.tokens, args.queue_token, args.baselines)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
    print(json.dumps({"manifest": str(out), "jobs": len(manifest["jobs"]),
                      "selected_actor_count": len(manifest["selected"])}))


if __name__ == "__main__":
    main()
