"""Rebuild descriptive comparison tables from integrity-checked trajectories."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ["buy_cost", "gen_cost", "bess_cost", "exchange_cost", "economic_cost", "full_cost"]


def load_evaluation(item):
    directory = Path(item["directory"])
    summary_file = directory / "summary.json"
    trace_file = directory / "trajectories.csv"
    for file, key in ((summary_file, "summary_sha256"), (trace_file, "trajectory_sha256")):
        if hashlib.sha256(file.read_bytes()).hexdigest() != item[key]:
            raise ValueError(f"Evaluation artifact changed: {file}")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if (summary["rows"] != list(range(913, 1095)) or not summary["physical_checks_passed"]
            or summary["inventory_mode"] != item["mode"]):
        raise ValueError("Invalid evaluation scope or physical acceptance")
    with trace_file.open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    if len(records) != 182 * 24 * 3:
        raise ValueError("Missing full-period agent trajectories")
    keys = {(int(r["row_idx"]), int(r["col_idx"]), r["agent"]) for r in records}
    if len(keys) != len(records):
        raise ValueError("Duplicate agent-hour records")
    agents = {r["agent"] for r in records}
    expected = {(row, hour, agent) for row in range(913, 1095)
                for hour in range(24) for agent in agents}
    if len(agents) != 3 or keys != expected:
        raise ValueError("Invalid agent-hour coverage")
    if [d["row_idx"] for d in summary["days"]] != list(range(913, 1095)):
        raise ValueError("Invalid daily summary coverage")
    totals = {key: sum(float(r[key]) for r in records) for key in COMPONENTS}
    for key, value in totals.items():
        if not np.isfinite(value) or not np.isclose(
                value, summary["totals"][key], rtol=0, atol=1e-6):
            raise ValueError(f"Independent trajectory cost mismatch: {key}")
    daily_costs = {row: 0.0 for row in range(913, 1095)}
    for record in records:
        daily_costs[int(record["row_idx"])] += float(record["economic_cost"])
    if any(not np.isclose(d["economic_cost"], daily_costs[d["row_idx"]],
                          rtol=0, atol=1e-6) for d in summary["days"]):
        raise ValueError("Daily economic cost mismatch")
    return summary, records


def aggregate(items):
    seen, loaded = set(), []
    for item in items:
        identity = (item["group"], item["seed"], item["policy_source"], item["mode"])
        if identity in seen:
            raise ValueError("Duplicate comparison group/seed/actor/mode")
        seen.add(identity)
        summary, records = load_evaluation(item)
        loaded.append((item, summary, records))
    rows = []
    for item, summary, _ in loaded:
        rule = next(s for i, s, _ in loaded
                    if i["group"] == "C0" and i["mode"] == item["mode"])
        rows.append({
            "group": item["group"], "seed": item["seed"],
            "policy_source": item["policy_source"], "mode": item["mode"],
            **{key: summary["totals"][key] for key in COMPONENTS},
            "saving_vs_rule": 1 - summary["totals"]["economic_cost"] / rule["totals"]["economic_cost"],
            "initial_soc": summary["initial_soc"], "terminal_soc": summary["terminal_soc"],
            "worst_day_economic_cost": max(d["economic_cost"] for d in summary["days"]),
            "initial_kwh": float(np.dot(summary["initial_soc"], item["battery_capacities"])),
            "terminal_kwh": float(np.dot(summary["terminal_soc"], item["battery_capacities"])),
        })
    statistics = []
    for mode in ("continuous", "daily_reset"):
        for group, source in (("C0", "rule"), ("C1", "rl"), ("C2", "rl"),
                              ("C2", "ea"), ("C3", "rl"), ("C4", "milp")):
            subset = [r for r in rows if r["group"] == group
                      and r["policy_source"] == source and r["mode"] == mode]
            if not subset:
                continue
            costs = [r["economic_cost"] for r in subset]
            statistics.append({
                "group": group, "policy_source": source, "mode": mode,
                "n": len(subset), "seeds": sorted(
                    [r["seed"] for r in subset], key=lambda s: -1 if s is None else s),
                "mean_economic_cost": float(np.mean(costs)),
                "sample_std": float(np.std(costs, ddof=1)) if len(costs) > 1 else None,
                "mean_saving_vs_rule": float(np.mean([r["saving_vs_rule"] for r in subset])),
            })
    return rows, statistics, loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Reports must use a new results directory")
    index = json.loads(args.index.read_text(encoding="utf-8"))
    rows, statistics, loaded = aggregate(index["evaluations"])
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (("per_seed.json", rows), ("statistics.json", statistics)):
        with (output / name).open("x", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
    with (output / "per_seed.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired = []
    for mode in ("continuous", "daily_reset"):
        for seed in (0, 1, 2):
            c1 = next((s for i, s, _ in loaded if i["group"] == "C1"
                       and i["seed"] == seed and i["mode"] == mode), None)
            c2 = next((s for i, s, _ in loaded if i["group"] == "C2"
                       and i["policy_source"] == "rl" and i["seed"] == seed
                       and i["mode"] == mode), None)
            if c1 is None or c2 is None:
                continue
            difference = np.array([b["economic_cost"] - a["economic_cost"]
                                   for a, b in zip(c1["days"], c2["days"])])
            paired.append({
                "seed": seed, "mode": mode, "C2_minus_C1": float(difference.sum()),
                "saving_vs_C1": 1 - c2["totals"]["economic_cost"] / c1["totals"]["economic_cost"],
                "daily_C2_wins": int((difference < 0).sum()),
                "first7_difference": float(difference[:7].sum()),
                "middle168_difference": float(difference[7:-7].sum()),
                "last7_difference": float(difference[-7:].sum()),
                "component_C2_minus_C1": {key: c2["totals"][key] - c1["totals"][key]
                                         for key in COMPONENTS},
                "daily_differences": difference.tolist(),
                "interpretation": "Descriptive seed-label comparison, not matched randomness or independent daily samples",
            })
    with (output / "ea_pairwise.json").open("x") as file:
        json.dump(paired, file, indent=2, allow_nan=False)
    lines = [
        "# Frozen Comparison Results", "",
        "Retrospective retained evaluation, rows913-1094,182days.",
        "Economic cost is monetary; full_cost includes emissions/penalties.",
        "RL Actors are primary; EA supplemental. No per-day actor selection.",
        "All costs independently recomputed from immutable trajectories.",
        "Statistics are descriptive; three seeds do not establish significance.", "",
    ]
    for mode in ("continuous", "daily_reset"):
        lines += [f"## {mode}", "",
                  "| Method/Actor | Seeds | Mean Economic Cost | Sample SD | Saving vs Rule |",
                  "|---|---|---:|---:|---:|"]
        subset = [s for s in statistics if s["mode"] == mode]
        for s in subset:
            sd = f"{s['sample_std']:.2f}" if s["sample_std"] is not None else "N/A"
            lines.append(f"| {s['group']}/{s['policy_source']} | {s['seeds']} | "
                         f"{s['mean_economic_cost']:.2f} | {sd} | "
                         f"{s['mean_saving_vs_rule']:.2%} |")
        missing = [group for group in ("C1", "C2", "C3") if not any(
            s["group"] == group and s["policy_source"] == "rl"
            and s["seeds"] == [0, 1, 2] for s in subset)]
        if missing:
            lines += ["", f"INCOMPLETE three-seed main matrix: {missing}."]
        if not any(s["group"] == "C4" for s in subset):
            lines += ["", "MILP not yet accepted; no optimality-distance claim."]
        fig, ax = plt.subplots(figsize=(9, 4))
        main = [s for s in subset if s["policy_source"] != "ea"]
        ax.bar([s["group"] for s in main],
               [s["mean_economic_cost"] for s in main],
               color=["#526d82", "#258876", "#ba5268", "#7164a7", "#d3aa32"][:len(main)])
        ax.set_ylabel("Whole-period economic cost (CU)")
        ax.set_title(f"182 days / {mode}")
        fig.tight_layout()
        fig.savefig(output / f"economic_{mode}.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        for p in paired:
            if p["mode"] == mode:
                ax.plot(range(913, 1095), np.cumsum(p["daily_differences"]),
                        label=f"seed{p['seed']}")
        ax.axhline(0, color="#444444", linewidth=.7)
        ax.set(xlabel="Dataset row", ylabel="Cumulative C2 minus C1 economic cost",
               title=f"EA whole-method comparison / {mode}")
        if any(p["mode"] == mode for p in paired):
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"ea_cumulative_{mode}.png", dpi=160)
        plt.close(fig)
        lines += ["", "### EA Seed-Label Differences", "",
                  "| Seed | Saving vs C1 | C2 Wins/182 Days | First7 Cost Difference | Middle168 | Last7 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for p in paired:
            if p["mode"] == mode:
                lines.append(f"| {p['seed']} | {p['saving_vs_C1']:.2%} | {p['daily_C2_wins']} | "
                             f"{p['first7_difference']:.2f} | {p['middle168_difference']:.2f} | "
                             f"{p['last7_difference']:.2f} |")
    if index.get("training"):
        lines += ["", "## Training Cost", "",
                  "| Method | Seed | Host | Joint Steps | Wall Hours | Updates |",
                  "|---|---|---|---:|---:|---:|"]
        for t in index["training"]:
            lines.append(f"| {t['group']} | {t['seed']} | {t['host']} | "
                         f"{t['actual_interactions']} | {t['elapsed_seconds']/3600:.3f} | "
                         f"{t.get('optimizer_updates', 'not directly recorded')} |")
        lines += ["", "Compare timings within the same host only; same steps are not same update counts.",
                  "This table includes validation, saving and uploads in supervised wall time.",
                  "Missing GAVDAC update counts must be reconstructed from frozen schedule, not guessed."]
    lines += [
        "", "## Interpretation Boundaries", "",
        "Compare C2-C1 by seed label, without claiming matched random trajectories.",
        "Report initial/final energy and daily cost differences without arbitrary terminal pricing.",
        "Daily reset grants/withdraws energy at each boundary; do not pool with continuous SOC.",
        "Perfect-foresight MILP is an ideal-information reference, not an information-matched actor.",
        "Do not attribute a total-method difference to attention, mutation or credit assignment.",
        "Wall time comparisons must use the same host and report algorithm/update differences.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "evaluations": len(loaded)}))


if __name__ == "__main__":
    main()
