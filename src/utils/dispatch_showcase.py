"""Explicit best-case validation displays, separate from aggregate evaluation."""

import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from controllers.cqmix_controller import CQMixMAC
from runners.episode_runner import EpisodeRunner
from utils.policy_evaluation import fixed_rows


class _NoMetrics:
    def log_stat(self, *args):
        pass


def day_metrics(records, env, initial_soc, tolerance):
    if len(records) != env.n_agents * 24:
        raise ValueError("A showcase day requires 24 records per agent")
    errors, spans, active = [], [], []
    for i, agent in enumerate(env.possible_agents):
        data = sorted((r for r in records if r["agent"] == agent), key=lambda r: r["col_idx"])
        if [r["col_idx"] for r in data] != list(range(24)):
            raise ValueError("Missing or duplicate hourly showcase records")
        errors.append(abs(data[-1]["soc"] - initial_soc))
        spans.append(max([initial_soc] + [r["soc"] for r in data])
                     - min([initial_soc] + [r["soc"] for r in data]))
        active.append(
            sum(r["pb"] < -0.01 * env.params["battery_charge_max"][i] for r in data) >= 2
            and sum(r["pb"] > 0.01 * env.params["battery_discharge_max"][i] for r in data) >= 2
        )
    cost = sum(r["full_cost"] for r in records)
    baseline = sum(r["baseline_cost"] for r in records)
    if baseline <= 0:
        raise ValueError("Showcase baseline cost must be positive")
    end_soc = np.array([
        next(r["soc"] for r in records if r["agent"] == agent and r["col_idx"] == 23)
        for agent in env.possible_agents
    ])
    price = env.data_dict["price"][env.train_indices].max()
    # A common display score corrects initial-energy depletion, not the reward.
    inventory_cost = float(np.sum(
        np.abs(end_soc - initial_soc) * env.params["battery_caps"]
        * 1.25 * (price + env.params["cycle_cost"]) / env.params["raw_ch"],
    ))
    balance = max(abs(
        r["pv"] + r["pd"] + r["pg"] + max(r["pb"], 0) + r["peer_import"]
        - r["load"] - max(-r["pb"], 0) - r["peer_export"] - r["spill"]
    ) for r in records)
    qualified = (
        max(errors) <= tolerance and all(active) and cost + inventory_cost < baseline
        and balance <= 1e-5
    )
    return {
        "row_idx": int(records[0]["row_idx"]), "qualified": bool(qualified),
        "operating_cost": float(cost), "baseline_cost": float(baseline),
        "cost_saving_ratio": float(1 - cost / baseline),
        "inventory_adjusted_saving_ratio": float(1 - (cost + inventory_cost) / baseline),
        "inventory_adjustment_cu": inventory_cost,
        "max_terminal_soc_error": float(max(errors)),
        "min_soc_span": float(min(spans)),
        "bidirectional_battery_fraction": float(np.mean(active)),
        "charge_kwh": float(sum(max(-r["pb"], 0) for r in records)),
        "discharge_kwh": float(sum(max(r["pb"], 0) for r in records)),
        "generator_kwh": float(sum(r["pd"] for r in records)),
        "grid_kwh": float(sum(r["pg"] for r in records)),
        "peer_sent_kwh": float(sum(r["peer_export"] for r in records)),
        "max_balance_error": float(balance),
    }


def select_days(metrics, count):
    if isinstance(count, bool) or int(count) != count or count < 1 or not metrics:
        raise ValueError("Showcase needs candidates and a positive integer day count")
    ranked = sorted(metrics, key=lambda d: (
        not d["qualified"], -d["inventory_adjusted_saving_ratio"],
        d["max_terminal_soc_error"], d["row_idx"],
    ))
    # If no day passes, expose diagnostics; never relabel a poor day as excellent.
    qualified = [d for d in ranked if d["qualified"]]
    return (qualified or ranked)[:int(count)]


def _write_csv(path, records):
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_day(records, metadata, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(exist_ok=False)
    hours, boundaries = np.arange(24), np.arange(25)
    agents = sorted({r["agent"] for r in records})
    groups = {a: sorted((r for r in records if r["agent"] == a), key=lambda r: r["col_idx"])
              for a in agents}
    hourly = lambda key: np.array([sum(r[key] for r in records if r["col_idx"] == h)
                                   for h in hours])
    tag = "QUALIFIED BEST CASE" if metadata["qualified"] else "DIAGNOSTIC: NOT QUALIFIED"
    title = (
        f"{tag} | validation row {metadata['row_idx']} | {metadata['policy_source']} "
        f"checkpoint {metadata['checkpoint_step']} | selected, not typical"
    )
    images = {}

    def finish(fig, name):
        fig.suptitle(title, fontsize=11)
        path = output / f"{name}.png"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        images[name] = path

    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    ax.stairs(hourly("load"), boundaries, baseline=None, label="Load", color="#226e9b")
    ax.stairs(hourly("pv"), boundaries, baseline=None, label="Available PV", color="#aa861b")
    tariff = ax.twinx()
    tariff.stairs([r["price"] for r in groups[agents[0]]], boundaries,
                  baseline=None, color="#b3422c", label="Tariff", linestyle="--")
    ax.set(xlabel="Hour", ylabel="Scenario kW", xlim=(0, 24))
    tariff.set_ylabel("Tariff (scenario CU/kWh)")
    ax.legend(loc="upper left")
    tariff.legend(loc="upper right")
    ax.grid(alpha=0.18)
    finish(fig, "inputs")

    fig, axes = plt.subplots(len(agents), 1, figsize=(12, 3 * len(agents)),
                             layout="constrained", squeeze=False)
    supply = (("pv", "PV", "#aa861b"), ("pd", "Generator", "#b3422c"),
              ("pg", "Grid", "#777777"), ("discharge", "Battery discharge", "#20875b"),
              ("peer_import", "Peer received", "#8952a5"))
    sinks = (("load", "Load", "#226e9b"), ("charge", "Battery charge", "#55ad98"),
             ("peer_export", "Peer sent", "#c19bd1"), ("spill", "Spill", "#c9c9c9"))
    for ax, agent in zip(axes.flat, agents):
        data = groups[agent]
        positive, negative = np.zeros(24), np.zeros(24)
        for key, label, color in supply + sinks:
            values = np.array([
                max(r["pb"], 0) if key == "discharge" else max(-r["pb"], 0)
                if key == "charge" else r[key] for r in data
            ])
            source = (key, label, color) in supply
            bottom = positive if source else negative
            ax.bar(hours + 0.5, values if source else -values, bottom=bottom,
                   width=0.9, color=color, label=label)
            if source:
                positive += values
            else:
                negative -= values
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set(title=f"{agent} | supply (+), demand and disposal (-)",
               xlabel="Hour", ylabel="Scenario kW", xlim=(0, 24))
        ax.grid(axis="y", alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="outside lower center", fontsize=8)
    finish(fig, "dispatch")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained", sharex=True)
    for agent in agents:
        data = groups[agent]
        axes[0].plot(boundaries, [data[0]["soc_init"]] + [r["soc"] for r in data], label=agent)
        axes[1].stairs([r["pb"] for r in data], boundaries, baseline=None, label=agent)
    axes[0].axhline(metadata["initial_soc"], color="#777777", linestyle="--",
                    linewidth=0.8, label="Initial inventory")
    axes[0].set(ylabel="SOC", ylim=(metadata["soc_min"] - 0.03, metadata["soc_max"] + 0.03))
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(xlabel="Hour", ylabel="Battery kW (+ discharge / - charge)", xlim=(0, 24))
    for ax in axes:
        ax.legend(ncol=4, fontsize=9)
        ax.grid(alpha=0.18)
    finish(fig, "battery")

    fig, axes = plt.subplots(len(agents), 1, figsize=(11, 2.4 * len(agents)),
                             layout="constrained", squeeze=False)
    for ax, agent in zip(axes.flat, agents):
        data = groups[agent]
        for key, label in (("peer_import", "Receive"), ("peer_export", "Send"),
                           ("exchange_loss", "Loss")):
            ax.stairs([r[key] for r in data], boundaries, baseline=None, label=label)
        ax.set(title=agent, ylabel="Scenario kW", xlabel="Hour", xlim=(0, 24))
        ax.legend(ncol=3, fontsize=9)
        ax.grid(alpha=0.18)
    finish(fig, "exchange")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    components = [
        hourly("buy_cost"), hourly("gen_cost"), -hourly("env_reward"),
        hourly("bess_cost"), hourly("exchange_cost"), -hourly("punishment"),
    ]
    names = ["Grid", "Generation", "Emission proxy", "Battery degradation", "Transfer", "Penalty"]
    bottom = np.zeros(24)
    for values, name in zip(components, names):
        axes[0].bar(hours + 0.5, values, bottom=bottom, width=0.9, label=name)
        bottom += values
    axes[0].set(xlabel="Hour", ylabel="Operating cost (CU/hour)", xlim=(0, 24))
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(boundaries, np.r_[0, np.cumsum(hourly("baseline_cost"))],
                 label="No-control baseline")
    axes[1].plot(boundaries, np.r_[0, np.cumsum(hourly("full_cost"))],
                 label="Actual operating cost")
    axes[1].set(xlabel="Hour", ylabel="Cumulative cost (CU)", xlim=(0, 24))
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.18)
    finish(fig, "costs")
    return images


class DispatchShowcase:
    def __init__(self, root, logger, args):
        self.root, self.logger, self.args = Path(root), logger, args
        self.count = int(args.showcase_days)
        self.tolerance = float(args.showcase_soc_tolerance)
        if (self.count <= 0
                or not np.isfinite(self.tolerance) or self.tolerance < 0):
            raise ValueError("Invalid dispatch showcase settings")
        self.published = {}

    def publish_final(self, runner, mac, policy_store, step):
        for source, best in policy_store.best.items():
            if self.published.get(source) == best["checkpoint"]:
                continue
            self._publish(runner, mac, source, best, step)
            self.published[source] = best["checkpoint"]

    def _publish(self, runner, mac, source, best, step):
        directory = self.root / source / f"checkpoint_{best['checkpoint_step']}_logged_{step}"
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint = Path(best["checkpoint"])
        # A learner's GRU state may retain a non-leaf autograd graph. Construct
        # an inference controller without copying it or consuming training RNG.
        with torch.random.fork_rng(devices=[]):
            clone = CQMixMAC(runner.batch.scheme, runner.batch.groups, runner.args)
        clone.load_models(str(checkpoint))
        args = deepcopy(runner.args)
        args.env_args.update(result_dir=str(directory), run_id="display", log_trajectories=False)
        display = EpisodeRunner(args, _NoMetrics())
        display.new_batch = runner.new_batch
        rows = fixed_rows(
            display.env, "validation", self.args.showcase_candidate_rows,
            len(display.env.validation_indices),
        )
        all_records, metrics = [], []
        try:
            for row in rows:
                display.run(clone, test_mode=True, test_options={
                    "source": "validation", "row_idx": row,
                    "initial_soc": self.args.evaluation_initial_soc,
                    "trajectory_context": {"policy_source": source,
                                           "checkpoint_step": best["checkpoint_step"]},
                })
                records = [dict(r) for r in display.env.ep_buffer]
                all_records.extend(records)
                metrics.append(day_metrics(
                    records, display.env, self.args.evaluation_initial_soc, self.tolerance,
                ))
            selected = select_days(metrics, self.count)
            report = {
                "policy_source": source, "checkpoint_step": best["checkpoint_step"],
                "logged_step": step, "checkpoint": str(checkpoint.resolve()),
                "policy_sha256": best["policy_sha256"], "candidate_rows": rows,
                "initial_soc": self.args.evaluation_initial_soc,
                "soc_min": display.env.soc_min, "soc_max": display.env.soc_max,
                "terminal_soc_tolerance": self.tolerance,
                "scope": "Selected validation best cases, not typical days or test-set performance",
                "qualification": "Each battery terminal SOC within tolerance; >=2 effective charge/discharge hours; positive inventory-adjusted saving",
                "ranking": "Qualified first; descending inventory-adjusted saving vs no-control baseline",
                "qualified_count": sum(m["qualified"] for m in metrics),
                "candidates": metrics, "selected": selected,
            }
            _write_csv(directory / "candidate_metrics.csv", metrics)
            _write_csv(directory / "candidate_trajectories.csv", all_records)
            images = {}
            for rank, metric in enumerate(selected, 1):
                records = [r for r in all_records if r["row_idx"] == metric["row_idx"]]
                images[rank] = plot_day(
                    records, {**report, **metric}, directory / f"rank_{rank}_row_{metric['row_idx']}",
                )
            (directory / "selection.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
            shutil.copytree(checkpoint, directory / "checkpoint")
            self.logger.log_dispatch_report(directory, report, images, step)
        finally:
            display.close_env()
