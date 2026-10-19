"""Compare fixed-budget environment checks in physical cost, not reward units."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def operating_cost(evaluation):
    return float(np.mean([
        row["full_cost"] + row["battery_punishment_reward"] for row in evaluation["rows"]
    ]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, nargs="+", required=True)
    parser.add_argument("--after", type=Path, nargs="+", required=True)
    parser.add_argument("--output-id", required=True)
    args = parser.parse_args()
    out = ROOT / "results/gavdac_environment" / args.output_id
    out.mkdir(parents=True, exist_ok=False)
    groups = [
        {run["seed"]: run for run in [json.loads(p.read_text(encoding="utf-8")) for p in files]}
        for files in (args.before, args.after)
    ]
    if set(groups[0]) != set(groups[1]):
        raise ValueError("Before/after seeds must match")
    if any(len(group) != len(files) for group, files in zip(groups, (args.before, args.after))):
        raise ValueError("Duplicate seeds cannot be silently combined")
    seeds = sorted(groups[0])
    fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 4.8),
                             sharey=True, squeeze=False)
    paired = []
    for ax, seed in zip(axes.flat, seeds):
        before, after = [g[seed] for g in groups]
        for key in ("requested_steps", "algorithm", "validation_rows"):
            if before[key] != after[key]:
                raise ValueError(f"Mismatched {key} for seed {seed}")
        series = {}
        for label, run, style in (("v2", before, "--"), ("v3", after, "-")):
            for policy, color, marker in (("rl", "#176f85", "o"), ("ea", "#b33e65", "^")):
                values = sorted(
                    (x["step"], operating_cost(x)) for x in run["history"] if x["policy"] == policy
                )
                series[label, policy] = dict(values)
                ax.plot([v[0] for v in values], [v[1] for v in values],
                        label=f"{label} {policy.upper()}", color=color, linestyle=style,
                        marker=marker, markersize=4, linewidth=1.5)
        common = set.intersection(*(set(v) for v in series.values()))
        if not common:
            raise ValueError("No common validation step")
        step = max(common)
        values = {f"{v}_{p}": s[step] for (v, p), s in series.items()}
        old_best = min(values["v2_rl"], values["v2_ea"])
        new_best = min(values["v3_rl"], values["v3_ea"])
        paired.append({
            "seed": seed, "last_common_validation_step": step, **values,
            "v2_best": old_best, "v3_best": new_best,
            "best_cost_reduction_fraction": 1 - new_best / old_best,
            "ea_cost_reduction_fraction": 1 - values["v3_ea"] / values["v2_ea"],
        })
        ax.axvspan(0, 512 * 24, color="#999999", alpha=0.12)
        ax.set(title=f"Seed {seed}", xlabel="Environment steps",
               xlim=(0, before["requested_steps"]))
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].set_ylabel("Operating cost, excluding request penalties (CU/day)")
    fig.suptitle("Same-budget GAVDAC + EA | six fixed validation days", fontsize=15, y=0.98)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=4, frameon=False)
    steps = ", ".join(f"{s:,}" for s in sorted({p["last_common_validation_step"] for p in paired}))
    fig.text(0.5, 0.80, f"Shaded region: replay warmup. Last common validation: {steps} steps.",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=0.71, bottom=0.15, left=0.09, right=0.98, wspace=0.10)
    fig.savefig(out / "learning_curves.png", dpi=170)
    plt.close(fig)
    payload = {
        "before": [str(p.resolve()) for p in args.before],
        "after": [str(p.resolve()) for p in args.after], "paired": paired,
        "mean_best_cost_reduction_fraction": float(np.mean([
            p["best_cost_reduction_fraction"] for p in paired
        ])),
        "seed_count": len(seeds),
        "scope": "short paired runs; package comparison, not isolated ablations or convergence proof",
    }
    with (out / "comparison.json").open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"COMPARISON={out}")


if __name__ == "__main__":
    main()
