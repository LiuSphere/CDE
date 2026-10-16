"""Paired seed-level cost comparisons with explicit no-EA and actor-only results."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSIONS = {"heterogeneous_exchange_v2": "v2", "facmac_dispatch_v3": "v3"}


def final_policies(run):
    expected = {"rl", "ea"} if run["algorithm"] == "facmac_ea" else {"rl"}
    evaluations = [x for x in run["history"] if x.get("phase") == "final"]
    if len(evaluations) != len(expected) or {x["policy"] for x in evaluations} != expected:
        raise ValueError("Final evaluations are missing or duplicated")
    costs = {}
    for evaluation in evaluations:
        if evaluation["step"] != run["actual_steps"]:
            raise ValueError("A scheduled evaluation cannot stand in for the final evaluation")
        if [x["row_idx"] for x in evaluation["rows"]] != run["validation_rows"]:
            raise ValueError("Validation rows do not match")
        cost = float(np.mean([
            x["full_cost"] + x["battery_punishment_reward"] for x in evaluation["rows"]
        ]))
        if not np.isfinite(cost) or cost <= 0:
            raise ValueError("Relative comparisons require finite positive operating costs")
        if not np.isclose(cost, evaluation["mean_operating_cost"], rtol=0, atol=1e-8):
            raise ValueError("Operating-cost decomposition mismatch")
        costs[evaluation["policy"]] = cost
    return costs


def resource_metrics(run, policy):
    evaluation = next(x for x in run["history"]
                      if x.get("phase") == "final" and x["policy"] == policy)
    rows = evaluation["rows"]
    return {
        "mean_terminal_soc": float(np.mean([x["terminal_soc"] for x in rows])),
        "mean_charge_kwh": float(np.mean([x["battery_charge_energy"] for x in rows])),
        "mean_discharge_kwh": float(np.mean([x["battery_discharge_energy"] for x in rows])),
    }


def compare_runs(runs):
    index = {}
    for run in runs:
        key = (run["seed"], VERSIONS[run["model_version"]], run["algorithm"])
        if key in index:
            raise ValueError(f"Duplicate run: {key}")
        if not run.get("pre_rollout_reseed"):
            raise ValueError("All random sources must be reset immediately before model construction")
        index[key] = run
    seeds = sorted({key[0] for key in index})
    if len(index) != 4 * len(seeds):
        raise ValueError("Each seed requires v2/v3 and EA/no-EA")
    paired = []
    for seed in seeds:
        pair = {"seed": seed}
        for algorithm in ("facmac_no_ea", "facmac_ea"):
            old, new = [index[seed, version, algorithm] for version in ("v2", "v3")]
            for field in ("requested_steps", "actual_steps", "validation_rows", "cpu_threads",
                          "rng_protocol", "deterministic_algorithms"):
                if old[field] != new[field]:
                    raise ValueError(f"Unmatched {field} for seed {seed} / {algorithm}")
            old_cost, new_cost = final_policies(old), final_policies(new)
            categories = (("facmac_only", "rl"),) if algorithm == "facmac_no_ea" else (
                ("actor_with_ea", "rl"), ("ea_elite", "ea"),
            )
            for category, policy in categories:
                pair[category] = {
                    "v2_cost": old_cost[policy], "v3_cost": new_cost[policy],
                    "improvement_pct": 100 * (1 - new_cost[policy] / old_cost[policy]),
                    "actual_steps": old["actual_steps"],
                    "v2_resources": resource_metrics(old, policy),
                    "v3_resources": resource_metrics(new, policy),
                }
            if algorithm == "facmac_ea":
                a, b = min(old_cost.values()), min(new_cost.values())
                old_policy, new_policy = min(old_cost, key=old_cost.get), min(new_cost, key=new_cost.get)
                pair["best_with_ea"] = {
                    "v2_cost": a, "v3_cost": b, "improvement_pct": 100 * (1 - b / a),
                    "actual_steps": old["actual_steps"],
                    "v2_selected_policy": old_policy, "v3_selected_policy": new_policy,
                    "v2_resources": resource_metrics(old, old_policy),
                    "v3_resources": resource_metrics(new, new_policy),
                }
        paired.append(pair)
    aggregate = {}
    for category in ("facmac_only", "actor_with_ea", "ea_elite", "best_with_ea"):
        values = np.array([row[category]["improvement_pct"] for row in paired])
        aggregate[category] = {
            "v2_mean_cost": float(np.mean([row[category]["v2_cost"] for row in paired])),
            "v3_mean_cost": float(np.mean([row[category]["v3_cost"] for row in paired])),
            "mean_paired_improvement_pct": float(values.mean()),
            "std_paired_improvement_pct": float(values.std(ddof=1)) if len(values) > 1 else 0,
            "improved_seeds": int((values > 1e-6).sum()), "seed_count": len(seeds),
            "unchanged_seeds": int((np.abs(values) <= 1e-6).sum()),
            "worse_seeds": int((values < -1e-6).sum()),
            "worst_improvement_pct": float(values.min()),
            "best_improvement_pct": float(values.max()),
            "v2_mean_terminal_soc": float(np.mean([
                row[category]["v2_resources"]["mean_terminal_soc"] for row in paired
            ])),
            "v3_mean_terminal_soc": float(np.mean([
                row[category]["v3_resources"]["mean_terminal_soc"] for row in paired
            ])),
        }
    return {"seeds": seeds, "paired": paired, "aggregate": aggregate}


def plot_comparison(summary, path):
    categories = ("facmac_only", "actor_with_ea", "ea_elite")
    labels = ("FACMAC only\n(no EA)", "FACMAC actor\n(with EA)", "EA elite\n(with EA)")
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    seeds = summary["seeds"]
    offsets = np.linspace(-0.20, 0.20, len(seeds))
    for x, category in enumerate(categories):
        values = [row[category]["improvement_pct"] for row in summary["paired"]]
        colors = ["#277d58" if value > 1e-6 else "#b33e65" if value < -1e-6 else "#888888"
                  for value in values]
        ax.scatter(x + offsets, values, color=colors, s=45, zorder=3)
        for dx, value, seed in zip(offsets, values, seeds):
            ax.annotate(str(seed), (x + dx, value), xytext=(4, 5),
                        textcoords="offset points", fontsize=9)
        stats = summary["aggregate"][category]
        ax.errorbar(x, stats["mean_paired_improvement_pct"],
                    yerr=stats["std_paired_improvement_pct"], color="#333333",
                    fmt="D", capsize=6, markersize=5, zorder=4)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set(xticks=range(3), xticklabels=labels, xlim=(-0.5, 2.5),
           ylabel="Operating-cost reduction: v3 vs v2 (%)")
    ax.margins(y=0.18)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{len(seeds)} paired seeds | positive = v3 lower cost", fontsize=15, pad=16)
    fig.supxlabel("Green: v3 cheaper. Red: v3 costlier. Number: seed. Diamond/bar: mean +/- SD.",
                  fontsize=10)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    directory = args.experiment.resolve(strict=True)
    completed = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
    runs = [json.loads(Path(x["result"]).read_text(encoding="utf-8")) for x in completed["runs"]]
    summary = compare_runs(runs)
    summary.update(
        scope="short paired learning check; no convergence or algorithm-superiority claim",
        units="scenario CU/day; mean of six fixed validation days; request penalties excluded",
        environment_tuning=False, test_set_used=False,
    )
    out = directory / "analysis"
    out.mkdir(exist_ok=False)
    plot_comparison(summary, out / "paired_improvements.png")
    with (out / "summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"SUMMARY={out}")


if __name__ == "__main__":
    main()
