"""Plot individual batteries from frozen PPO tuning endpoints."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "terminal_ref": "PPO：期末惩罚对照",
    "market_q16": "PPO：电价势函数 + 库存目标",
    "deadline_q16": "PPO：密集库存目标",
    "deadline_q64": "PPO：强库存目标",
}
COLORS = ["#777777", "#20875b", "#226e9b", "#8952a5"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=Path(
        "results/oracle_milp/battery_activity_diagnosis_20261017_a",
    ))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=int, default=766)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    experiment = args.experiment
    manifest = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((experiment / "analysis/summary.json").read_text(encoding="utf-8"))
    if args.seed not in manifest["seeds"] or args.row not in manifest["validation_rows"]:
        raise ValueError("Plot a declared validation day and an evaluated seed")
    args.output.mkdir(parents=True, exist_ok=False)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if not font.exists():
        raise FileNotFoundError("Chinese plot font is required")
    plt.rcParams.update({
        "font.family": FontProperties(fname=str(font)).get_name(),
        "axes.unicode_minus": False, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    config = manifest["env_args"]
    modes = manifest["modes"]
    initial = config["initial_soc"]
    frames = [pd.read_csv(
        experiment / "runs" / f"{mode}_seed{args.seed}" / "validation_trajectories.csv",
    ) for mode in modes]
    labels = [LABELS.get(mode, mode) for mode in modes]
    oracle = pd.read_csv(args.oracle / "trajectories.csv")
    frames.append(oracle)
    labels.append("MILP：完美预测 / 严格日末闭合")
    colors = COLORS[:len(modes)] + ["#b3422c"]
    figure = plt.figure(figsize=(14, 9), layout="constrained")
    grid = figure.add_gridspec(3, 3, height_ratios=[0.6, 1.3, 1.3])
    tariff = figure.add_subplot(grid[0, :])
    first = oracle[(oracle.row_idx == args.row) & (oracle.agent == "agent1")]
    tariff.stairs(first.price, np.arange(25), baseline=None, color="#9c7c15", linewidth=2)
    tariff.set(title="公开电价", ylabel="场景 CU/kWh", xlim=(0, 24),
               xticks=np.arange(0, 25, 3))
    tariff.grid(alpha=0.18)
    for index, param in enumerate(config["data"]):
        agent = param["agent_name"]
        soc_axis = figure.add_subplot(grid[1, index])
        power_axis = figure.add_subplot(grid[2, index])
        for frame, label, color in zip(frames, labels, colors):
            data = frame[(frame.row_idx == args.row) & (frame.agent == agent)].sort_values("col_idx")
            if len(data) != 24:
                raise ValueError("Incomplete daily trajectory")
            soc_axis.plot(np.arange(25), [initial, *data.soc], color=color,
                          label=label, linewidth=1.8)
            power_axis.stairs(data.pb, np.arange(25), baseline=None, color=color, linewidth=1.8)
        soc_axis.axhspan(initial - 0.03, initial + 0.03, color="#777777", alpha=0.08)
        soc_axis.axhline(initial, color="#aaaaaa", linewidth=0.8, linestyle="--")
        soc_axis.set(title=f"{agent}：{param['battery_cap']:.0f} kWh",
                     ylabel="SOC", ylim=(config["soc_min"] - 0.03, config["soc_max"] + 0.04))
        power_axis.axhline(0, color="#aaaaaa", linewidth=0.8)
        power_axis.set(title="正为放电，负为充电", ylabel="场景 kW",
                       ylim=(-param["battery_charge_max"] * 1.12,
                             param["battery_discharge_max"] * 1.12))
        for axis in (soc_axis, power_axis):
            axis.set(xlabel="小时", xlim=(0, 24), xticks=[0, 6, 12, 18, 24])
            axis.grid(alpha=0.18)
        if index == 0:
            soc_axis.legend(fontsize=8, loc="upper right")
    figure.suptitle(
        f"PPO 最终策略与日循环参考 | 验证日 {args.row}，预先指定种子 {args.seed}", fontsize=15,
    )
    figure.savefig(args.output / f"battery_row_{args.row}_seed{args.seed}.png", dpi=150)
    plt.close(figure)

    for mode, frame in zip(modes, frames):
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained",
                                    sharex=True, sharey=True)
        for axis, row in zip(axes.flat, manifest["validation_rows"]):
            for index, param in enumerate(config["data"]):
                data = frame[(frame.row_idx == row) & (frame.agent == param["agent_name"])]
                axis.plot(np.arange(25), [initial, *data.soc], label=param["agent_name"],
                          color=["#226e9b", "#b3422c", "#20875b"][index])
            axis.axhspan(initial - 0.03, initial + 0.03, color="#777777", alpha=0.08)
            axis.axhline(initial, color="#aaaaaa", linewidth=0.8, linestyle="--")
            axis.set(title=f"验证日 {row}", xlabel="小时", ylabel="SOC", xlim=(0, 24),
                     ylim=(0.17, 0.84), xticks=[0, 6, 12, 18, 24])
            axis.grid(alpha=0.18)
        axes.flat[0].legend(fontsize=9)
        figure.suptitle(
            f"{LABELS.get(mode, mode)} | 种子 {args.seed}，全部验证日，逐电池", fontsize=15,
        )
        figure.savefig(args.output / f"all_days_{mode}_seed{args.seed}.png", dpi=150)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    seeds = manifest["seeds"]
    for index, seed in enumerate(seeds):
        selected = [next(r for r in summary["runs"] if r["seed"] == seed and r["mode"] == mode)
                    for mode in modes]
        positions = np.arange(len(modes)) + (index - (len(seeds) - 1) / 2) * 0.08
        for axis, values in zip(axes, (
            [r["validation"]["mean_full_cost"] for r in selected],
            [r["acceptance"]["max_terminal_soc_error"] for r in selected],
        )):
            axis.plot(positions, values, color="#bbbbbb", linewidth=0.8)
            for i, value in enumerate(values):
                axis.scatter(positions[i], value, color=COLORS[i], s=35)
                axis.annotate(str(seed), (positions[i], value), xytext=(5, 6),
                              textcoords="offset points", fontsize=9)
    oracle_summary = json.loads((args.oracle / "summary.json").read_text(encoding="utf-8"))
    axes[0].axhline(oracle_summary["mean_full_cost"], color="#b3422c", linestyle="--",
                    label="日循环 MILP 均值")
    axes[0].set_ylabel("实际运行成本（CU/天）")
    axes[0].legend(fontsize=9)
    axes[1].axhline(0.03, color="#b3422c", linestyle="--", label="预定误差上限")
    axes[1].set_ylabel("所有验证日、所有电池的最大日末 SOC 误差")
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.set_xticks(range(len(modes)), [LABELS.get(m, m) for m in modes])
        axis.margins(x=0.2, y=0.18)
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle("配对种子最终结果：成本与库存分开评价，数字为种子编号", fontsize=14)
    figure.savefig(args.output / "paired_results.png", dpi=150)
    plt.close(figure)
    print(f"FIGURES={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
