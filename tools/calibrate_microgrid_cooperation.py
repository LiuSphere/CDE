"""Select community installation assumptions on validation medoids only."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_physical_microgrid_milp import (
    comparison, cooperation_acceptance, make_env, representative_days,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--round", choices=["capacity", "link-cost"], default="capacity")
    args = parser.parse_args()
    out = ROOT / "results/oracle_milp" / args.output_id
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(
        (ROOT / "src/config/envs/microgrid.yaml").read_text(encoding="utf-8"),
    )["env_args"]
    env = make_env(config, out)
    representatives = representative_days(env, "validation")
    # A predeclared capacity sweep; no test-day outcomes select installations.
    trials = (
        [(145, 4, [0.45, 0.2, 0.35], 120, 0.2),
         (240, 4, [0.55, 0.2, 0.25], 120, 0.2),
         (280, 4, [0.55, 0.2, 0.25], 120, 0.2),
         (280, 5, [0.55, 0.2, 0.25], 120, 0.2),
         (280, 6, [0.55, 0.2, 0.25], 120, 0.2)]
        if args.round == "capacity" else
        [(240, 4, [0.55, 0.2, 0.25], 160, 0.18),
         (240, 4, [0.55, 0.2, 0.25], 200, 0.165),
         (240, 4, [0.55, 0.2, 0.25], 240, 0.15),
         (280, 4, [0.55, 0.2, 0.25], 200, 0.165)]
    )
    history, selected = [], None
    for rating, pv, shares, solar_link, solar_cost in trials:
        candidate = deepcopy(config)
        candidate["pv_scale"] = pv
        for p, share in zip(candidate["data"], shares):
            p["load_share"] = share
        candidate["data"][2]["dg_max"] = rating
        candidate["data"][0]["exchange_max"] = 100
        candidate["data"][1]["exchange_max"] = 120
        candidate["data"][1]["costb"] = 0.2
        candidate["data"][2]["exchange_max"] = 120
        if args.round == "link-cost":
            candidate["data"][0]["exchange_max"] = 160
            candidate["data"][1]["exchange_max"] = solar_link
            candidate["data"][1]["costb"] = solar_cost
            candidate["data"][2]["exchange_max"] = 160
        env = make_env(candidate, out)
        results = [
            comparison(env, candidate, r["row_idx"], "validation", 0.35)
            for r in representatives
        ]
        record = {
            "dg3": rating, "pv_scale": pv, "load_shares": shares,
            "solar_link": solar_link, "solar_linear_cost": solar_cost,
            "acceptance": cooperation_acceptance(results),
            "mean_cooperation_saving": float(np.mean([r["cooperation_saving_fraction"] for r in results])),
            "min_cooperation_saving": min(r["cooperation_saving_fraction"] for r in results),
            "min_battery_saving": min(r["battery_cost_saving_fraction"] for r in results),
            "min_transfer_fraction": min(r["transfer_fraction"] for r in results),
            "days": [{k: v for k, v in r.items() if k not in ("records", "solutions", "ablations")}
                     for r in results],
        }
        history.append(record)
        if selected is None and record["acceptance"]["passed"]:
            selected = candidate
        print(json.dumps({k: v for k, v in record.items() if k != "days"}), flush=True)
        with (out / f"trial_{len(history)}.json").open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2, allow_nan=False)
    with (out / "calibration.json").open("x", encoding="utf-8") as file:
        json.dump({"round": args.round, "base_config": config,
                   "representatives": representatives, "history": history, "selected": selected},
                  file, indent=2, allow_nan=False)
    if selected is None:
        raise SystemExit("No candidate passed; continue calibration on validation data")
    with (out / "selected_env.yaml").open("x", encoding="utf-8") as file:
        yaml.safe_dump(selected, file, sort_keys=False)
    print(f"CALIBRATION={out}", flush=True)


if __name__ == "__main__":
    main()
