"""Physical-model calibration and perfect-foresight dispatch with replay checks."""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import pdist, squareform

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_current_model_milp import LinearModel, quadratic_epigraph
from direct_main import load_yaml, recursive_dict_update
from envs.maenv import MGEnv

VERSION = "gavdac_dispatch_v3"
COLORS = ["#176f85", "#b33e65", "#49813e", "#8952a5", "#b77815", "#3d60b3"]


def solve_network(load, pv, price, params, limits, efficiency, fee, initial_soc,
                  soc_min=0.2, soc_max=0.8, terminal_soc=None, segments=32,
                  battery_enabled=True):
    """Joint dispatch on a uniform-loss, finite-capacity community bus."""
    load, pv, price = [np.asarray(a, dtype=float) for a in (load, pv, price)]
    limits = np.asarray(limits, dtype=float)
    if not (
        load.ndim == 2 and load.shape == pv.shape
        and price.shape == (len(load),) and load.shape[1] == len(params)
        and limits.shape == (len(params),)
        and all(np.isfinite(a).all() for a in (load, pv, price, limits))
        and (load >= 0).all() and (pv >= 0).all() and (price > 0).all()
        and (limits >= 0).all() and 0 < efficiency <= 1 and 0 <= fee < price.min() * efficiency
        and segments >= 2 and soc_min <= initial_soc <= soc_max
    ):
        raise ValueError("Invalid physical dispatch inputs")
    if terminal_soc is not None and not soc_min <= terminal_soc <= soc_max:
        raise ValueError("Invalid terminal SOC")
    model = LinearModel()
    all_states, all_decisions = [], []
    transfer_rows = [dict() for _ in range(len(load))]
    error_bound = 0.0
    for a, p in enumerate(params):
        cap, pc, pdmax = p["battery_cap"], p["battery_charge_max"], p["battery_discharge_max"]
        if not battery_enabled:
            pc = pdmax = 0.0
        dg, eta_c, eta_d = p["dg_max"], p["raw_ch"], p["raw_dis"]
        soc = [model.var(soc_min, soc_max) for _ in range(len(load) + 1)]
        model.row({soc[0]: 1}, initial_soc, initial_soc)
        if terminal_soc is not None:
            model.row({soc[-1]: 1}, terminal_soc, terminal_soc)
        decisions = []
        for t in range(len(load)):
            generation = model.var(0, dg, cost=p["env_param"])
            charge = model.var(0, pc, cost=p["cycle_cost"])
            discharge = model.var(0, pdmax, cost=p["cycle_cost"])
            mode = model.var(0, 1, integer=True)
            model.row({charge: 1, mode: -pc}, high=0)
            model.row({discharge: 1, mode: pdmax}, high=pdmax)
            model.row({
                soc[t + 1]: cap, soc[t]: -cap,
                charge: -eta_c, discharge: 1 / eta_d,
            }, 0, 0)
            imported = model.var(0, load[t, a] + pc, cost=price[t])
            spill = model.var(0, pv[t, a] + dg + pdmax)
            receive = model.var(0, limits[a])
            send = model.var(0, limits[a], cost=fee)
            deficit = model.var(0, 1, integer=True)
            model.row({receive: 1, deficit: -limits[a]}, high=0)
            model.row({imported: 1, deficit: -(load[t, a] + pc)}, high=0)
            model.row({send: 1, deficit: limits[a]}, high=limits[a])
            model.row({spill: 1, deficit: pv[t, a] + dg + pdmax},
                      high=pv[t, a] + dg + pdmax)
            model.row({
                imported: 1, generation: 1, discharge: 1, charge: -1, spill: -1,
                receive: 1, send: -1,
            }, load[t, a] - pv[t, a], load[t, a] - pv[t, a])
            transfer_rows[t].update({receive: 1, send: -efficiency})
            gen_max = p["costa"] * dg**2 + p["costb"] * dg + p["costc"]
            gen_cost = model.var(0, gen_max + 1, cost=1)
            quadratic_epigraph(
                model, generation, gen_cost, p["costa"], p["costb"], p["costc"],
                np.linspace(0, dg, segments + 1),
            )
            decisions.append((generation, charge, discharge, imported, spill, receive, send))
        all_states.append(soc)
        all_decisions.append(decisions)
        error_bound += len(load) * p["costa"] * (dg / segments)**2 / 4
    for entries in transfer_rows:
        model.row(entries, 0, 0)
    result = model.solve(gap=1e-7, limit=60)
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"Physical MILP failed: {result.message}")
    schedules = []
    for decisions, soc in zip(all_decisions, all_states):
        schedule = {
            key: np.array([result.x[d[i]] for d in decisions]).tolist()
            for i, key in enumerate(("pd", "charge", "discharge", "pg", "spill",
                                     "peer_import", "peer_export"))
        }
        schedule["soc"] = result.x[soc].tolist()
        schedules.append(schedule)
    return {
        "status": int(result.status), "mip_gap": float(result.mip_gap),
        "objective": float(result.fun), "dual_bound": float(result.mip_dual_bound),
        "interpolation_error_bound": float(error_bound),
        "exact_cost_lower_bound": float(result.mip_dual_bound) - error_bound,
        "schedules": schedules,
    }


def solve_agent(load, pv, price, p, initial_soc, soc_min=0.2, soc_max=0.8,
                terminal_soc=None, segments=32, battery_enabled=True):
    result = solve_network(
        np.asarray(load)[:, None], np.asarray(pv)[:, None], price, [p], [0], 1, 0,
        initial_soc, soc_min, soc_max, terminal_soc, segments, battery_enabled,
    )
    result["schedule"] = result.pop("schedules")[0]
    return result


def representative_days(env, source, count=6):
    rows = getattr(env, f"{source}_indices")
    features = np.concatenate([
        env.raw_data[k][rows] / env.raw_data[k][rows].std()
        for k in ("load", "pv", "price")
    ], axis=1)
    labels = cut_tree(linkage(features, method="ward"), n_clusters=[count]).ravel()
    distances = squareform(pdist(features, metric="sqeuclidean"))
    selected = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        center = members[np.argmin(distances[np.ix_(members, members)].sum(axis=1))]
        selected.append({"row_idx": int(rows[center]), "member_count": len(members)})
    return sorted(selected, key=lambda r: r["row_idx"])


def solve_day(env, config, row, source, initial, battery_enabled=True, exchange_enabled=True):
    if not exchange_enabled:
        isolated = deepcopy(config)
        for p in isolated["data"]:
            p["exchange_max"] = 0
        env = make_env(isolated, env.render_dir)
    solution = solve_network(
        env.agent_data["load"][row], env.agent_data["pv"][row], env.data_dict["price"][row],
        config["data"], env.params["exchange_max"], env.exchange_efficiency, env.exchange_fee,
        initial, env.soc_min, env.soc_max, terminal_soc=initial, battery_enabled=battery_enabled,
    )
    env.reset_test(source=source, row_idx=row, initial_soc=initial)
    for t in range(24):
        actions = np.zeros((env.n_agents, 2))
        for a, (p, s) in enumerate(zip(config["data"], solution["schedules"])):
            actions[a, 0] = s["pd"][t] / p["dg_max"] if p["dg_max"] else 0
        physical_pb = np.array([
            s["discharge"][t] - s["charge"][t] for s in solution["schedules"]
        ])
        actions[:, 1] = env.battery_action_from_power(physical_pb, env.params["socs"])
        if np.max(np.abs(actions)) > 1 + 1e-7:
            raise RuntimeError("MILP reconstructed action outside bounds")
        env.step(np.clip(actions, env.action_lows, env.action_highs))
    records = [dict(r) for r in env.ep_buffer]
    max_error = 0.0
    for r in records:
        a, t = env.possible_agents.index(r["agent"]), r["col_idx"]
        s = solution["schedules"][a]
        max_error = max(
            max_error, abs(r["soc"] - s["soc"][t + 1]),
            abs(r["pb"] - (s["discharge"][t] - s["charge"][t])),
        )
    # Uniform price/loss/fee makes optimal routing degenerate. Compare team
    # grid/spill/transfer totals; runtime uses deterministic pro-rata allocation.
    for t in range(24):
        for key in ("pg", "spill", "peer_import", "peer_export"):
            max_error = max(max_error, abs(
                sum(r[key] for r in records if r["col_idx"] == t)
                - sum(s[key][t] for s in solution["schedules"]),
            ))
    if max_error > 1e-5:
        raise RuntimeError(f"Physical environment/MILP replay mismatch {max_error}")
    exact = sum(r["full_cost"] for r in records)
    pwl = solution["objective"]
    bound = solution["interpolation_error_bound"]
    if not -1e-5 <= pwl - exact <= bound + 1e-5:
        raise RuntimeError("Exact cost outside PWL interpolation certificate")
    metrics = []
    for a, p in enumerate(config["data"]):
        data = [r for r in records if r["agent"] == env.possible_agents[a]]
        states = [initial] + [r["soc"] for r in data]
        discharged = sum(max(r["pb"], 0) for r in data)
        charged = sum(max(-r["pb"], 0) for r in data)
        load = sum(r["load"] for r in data)
        pv = sum(r["pv"] for r in data)
        spill = sum(r["spill"] for r in data)
        metrics.append({
            "row_idx": row, "agent": env.possible_agents[a],
            "soc_span_fraction": (max(states) - min(states)) / (env.soc_max - env.soc_min),
            "discharge_use_fraction": discharged / (
                (env.soc_max - env.soc_min) * p["battery_cap"] * p["raw_dis"]
            ),
            "charged_energy": charged, "discharged_energy": discharged,
            "charge_hours": sum(r["pb"] < -0.01 * p["battery_charge_max"] for r in data),
            "discharge_hours": sum(r["pb"] > 0.01 * p["battery_discharge_max"] for r in data),
            "generator_hours": sum(r["pd"] > 0.05 * p["dg_max"] for r in data),
            "generator_fraction": sum(r["pd"] for r in data) / load,
            "grid_fraction": sum(r["pg"] for r in data) / load,
            "pv_fraction": pv / load,
            "pv_utilization": max(pv - spill, 0) / pv if pv > 0 else 1,
            "terminal_soc_error": abs(data[-1]["soc"] - initial),
            "full_cost": sum(r["full_cost"] for r in data),
            "peer_sent_energy": sum(r["peer_export"] for r in data),
            "peer_received_energy": sum(r["peer_import"] for r in data),
        })
    return {
        "row_idx": row, "full_cost": exact, "pwl_objective": pwl,
        "generator_fraction": sum(r["pd"] for r in records) / sum(r["load"] for r in records),
        "grid_fraction": sum(r["pg"] for r in records) / sum(r["load"] for r in records),
        "transfer_fraction": sum(r["peer_import"] for r in records) / sum(r["load"] for r in records),
        "interpolation_error_bound": bound, "max_replay_error": max_error,
        "max_mip_gap": solution["mip_gap"],
        "all_solved": solution["status"] == 0,
        "peer_sent_energy": sum(r["peer_export"] for r in records),
        "exchange_loss_energy": sum(r["exchange_loss"] for r in records),
        "agents": metrics, "solutions": [solution], "records": records,
    }


def acceptance(results, require_resource_share=False):
    agents = [a for r in results for a in r["agents"]]
    tests = {
        "all_solved": all(r["all_solved"] for r in results),
        "soc_span_at_least_90pct_usable": min(a["soc_span_fraction"] for a in agents) >= 0.9,
        "discharge_at_least_90pct_usable": min(a["discharge_use_fraction"] for a in agents) >= 0.9,
        "meaningful_charge_and_discharge": all(
            a["charge_hours"] >= 2 and a["discharge_hours"] >= 2 for a in agents
        ),
        "generator_active_not_full_time": all(4 <= a["generator_hours"] < 24 for a in agents),
        "local_generators_and_grid_meaningful": all(
            a["generator_fraction"] >= 0.05 and a["grid_fraction"] >= 0.05 for a in agents
        ),
        "team_generation_and_grid_meaningful": all(
            0.1 <= r["generator_fraction"] <= 0.75 and r["grid_fraction"] >= 0.1
            for r in results
        ),
        "available_pv_used": min(a["pv_utilization"] for a in agents) >= 0.9,
        "cyclic_soc": max(a["terminal_soc_error"] for a in agents) < 1e-6,
    }
    # The installation-design target is validation-only. Weather on test days
    # may reduce available solar; test acceptance checks its actual utilization.
    if require_resource_share:
        tests["mean_pv_fraction_at_least_20pct"] = np.mean(
            [a["pv_fraction"] for a in agents],
        ) >= 0.2
    tests = {key: bool(value) for key, value in tests.items()}
    return {"passed": all(tests.values()), "checks": tests}


def comparison(env, config, row, source, initial):
    joint = solve_day(env, config, row, source, initial)
    isolated = solve_day(env, config, row, source, initial, exchange_enabled=False)
    no_battery = solve_day(env, config, row, source, initial, battery_enabled=False)
    joint.update(
        isolated_cost=isolated["full_cost"], no_battery_cost=no_battery["full_cost"],
        cooperation_saving_fraction=1 - joint["full_cost"] / isolated["full_cost"],
        battery_cost_saving_fraction=1 - joint["full_cost"] / no_battery["full_cost"],
        ablations_solved=isolated["all_solved"] and no_battery["all_solved"],
        ablation_max_gap=max(isolated["max_mip_gap"], no_battery["max_mip_gap"]),
        ablation_max_replay_error=max(isolated["max_replay_error"], no_battery["max_replay_error"]),
        cooperation_saving_lower_fraction=(
            1 - joint["full_cost"] / isolated["solutions"][0]["exact_cost_lower_bound"]
        ),
        battery_saving_lower_fraction=(
            1 - joint["full_cost"] / no_battery["solutions"][0]["exact_cost_lower_bound"]
        ),
        ablations={
            name: {k: v for k, v in result.items() if k != "records"}
            for name, result in (("isolated", isolated), ("no_battery", no_battery))
        },
    )
    return joint


def cooperation_acceptance(results):
    verdict = acceptance(results)
    verdict["checks"].update({
        "cooperation_saves_at_least_half_pct_each_day": all(
            r["cooperation_saving_lower_fraction"] >= 0.005 for r in results
        ),
        "mean_cooperation_saving_at_least_1pct": np.mean(
            [r["cooperation_saving_fraction"] for r in results],
        ) >= 0.01,
        "battery_saves_at_least_1pct_each_day": all(
            r["battery_saving_lower_fraction"] >= 0.01 for r in results
        ),
        "transfer_at_least_2pct_of_load_each_day": all(r["transfer_fraction"] >= 0.02 for r in results),
        "ablations_solved": all(r["ablations_solved"] for r in results),
    })
    verdict["checks"] = {k: bool(v) for k, v in verdict["checks"].items()}
    verdict["passed"] = all(verdict["checks"].values())
    return verdict


def plot_results(out, records, rows):
    frame = pd.DataFrame(records)
    hours = np.arange(24)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")
    for ax, row in zip(axes.flat, rows):
        hourly = frame[frame.row_idx == row].groupby("col_idx")
        for key, label, color in (
            ("load", "Load", "#3d60b3"), ("pv", "PV available", "#b77815"),
            ("pd", "Generator", "#b33e65"), ("pg", "Grid import", "#777777"),
            ("pb", "Battery (+ discharge)", "#176f85"),
        ):
            ax.stairs(hourly[key].sum(), np.arange(25), label=label, color=color,
                      baseline=None, linewidth=1.5)
        ax.axhline(0, linewidth=0.6, color="#aaaaaa")
        ax.set(title=f"Representative row {row}", xlabel="Hour", ylabel="Scenario power (kW)",
               xlim=(0, 24))
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(alpha=0.18)
    axes.flat[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Physical-model perfect foresight | no device-activity bonuses", fontsize=16)
    fig.savefig(out / "representative_dispatch.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True, sharey=True,
                             layout="constrained")
    for ax, row in zip(axes.flat, rows):
        for i, (agent, data) in enumerate(frame[frame.row_idx == row].groupby("agent")):
            ax.plot(np.arange(25), [data.soc_init.iloc[0]] + data.soc.tolist(),
                    color=COLORS[i], marker=".", markersize=3, label=agent)
        ax.axhline(0.2, color="#999999", linewidth=0.6, linestyle="--")
        ax.axhline(0.8, color="#999999", linewidth=0.6, linestyle="--")
        ax.set(title=f"Row {row}", ylabel="SOC", xlabel="Hour boundary",
               ylim=(0.15, 0.85), xlim=(0, 24))
        ax.grid(alpha=0.18)
    axes.flat[0].legend(ncol=3, fontsize=9)
    fig.suptitle("All three batteries | daily terminal SOC equals initial SOC", fontsize=16)
    fig.savefig(out / "representative_soc.png", dpi=160)
    plt.close(fig)


def plot_cooperation(out, records, results, rows, config):
    frame = pd.DataFrame(records)
    roles = {p["agent_name"]: p["role"].replace("_", " ") for p in config["data"]}
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
    for ax, row in zip(axes.flat, rows):
        for i, (agent, data) in enumerate(frame[frame.row_idx == row].groupby("agent")):
            ax.stairs(data.peer_import - data.peer_export, np.arange(25),
                      baseline=None, label=f"{agent}: {roles[agent]}", color=COLORS[i])
        ax.axhline(0, color="#777777", linewidth=0.7)
        ax.set(title=f"Representative row {row}", xlabel="Hour",
               ylabel="Peer power: + receive / - send (kW)", xlim=(0, 24))
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(alpha=0.18)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Finite, lossy exchange | policy still controls generation and storage", fontsize=15)
    fig.savefig(out / "representative_exchange.png", dpi=160)
    plt.close(fig)
    selected = max(rows, key=lambda r: (
        frame[frame.row_idx == r].pv.sum() / frame[frame.row_idx == r].load.sum()
    ))
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), layout="constrained")
    for a, (agent, data) in enumerate(frame[frame.row_idx == selected].groupby("agent")):
        ax = axes[a, 0]
        for key, label, color in (("load", "Load", "#3d60b3"), ("pv", "PV available", "#b77815"),
                                  ("pd", "Generator", "#b33e65"), ("pg", "Grid", "#777777")):
            ax.stairs(data[key], np.arange(25), baseline=None, label=label, color=color)
        ax.set(title=f"{agent}: {roles[agent]}", ylabel="Power (kW)")
        if a == 0:
            ax.legend(fontsize=8, ncol=2)
        ax = axes[a, 1]
        ax.stairs(data.pb, np.arange(25), baseline=None, color=COLORS[0])
        ax.axhline(0, color="#777777", linewidth=0.7)
        ax.set(title="Battery: + discharge / - charge", ylabel="Battery power (kW)")
        ax = axes[a, 2]
        for key, label, color in (("peer_import", "Receive", "#49813e"),
                                  ("peer_export", "Send", "#8952a5")):
            ax.stairs(data[key], np.arange(25), baseline=None, label=label, color=color)
        ax.set(title="Community exchange", ylabel="Link power (kW)")
        if a == 0:
            ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set(xlabel="Hour", xlim=(0, 24), xticks=[0, 6, 12, 18, 24])
        ax.grid(alpha=0.18)
    fig.suptitle(f"Three distinct roles | high-solar representative row {selected}", fontsize=16)
    fig.savefig(out / "agent_dispatch.png", dpi=160)
    plt.close(fig)
    days = pd.DataFrame(results).sort_values("row_idx")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout="constrained")
    for key, label, color in (("isolated_cost", "Links off (reoptimized)", "#777777"),
                              ("no_battery_cost", "Batteries off (reoptimized)", "#b33e65"),
                              ("full_cost", "Joint operation", "#176f85")):
        axes[0].plot(days.row_idx, days[key], label=label, color=color, linewidth=1.2)
    axes[0].set(ylabel="Full operating cost (CU/day)")
    axes[0].legend(fontsize=9)
    for key, label, color in (("cooperation_saving_fraction", "Cooperation benefit", "#8952a5"),
                              ("battery_cost_saving_fraction", "Battery benefit", "#176f85")):
        axes[1].plot(days.row_idx, 100 * days[key], label=label, color=color, linewidth=1.2)
    axes[1].set(xlabel="Source day row (not a calendar date)", ylabel="Full-cost reduction (%)")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.suptitle("Same inputs, equipment and cyclic SOC | only links or batteries removed", fontsize=15)
    fig.savefig(out / "cooperation_gains.png", dpi=160)
    plt.close(fig)


def make_env(config, out):
    args = deepcopy(config)
    args.update(result_dir=str(out), log_trajectories=False, run_id="replay")
    return MGEnv(args, "physical_milp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-test-days", action="store_true")
    parser.add_argument("--all-days", action="store_true")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-id")
    args = parser.parse_args()
    config = recursive_dict_update(
        load_yaml(str(ROOT / "src/config/default.yaml")),
        load_yaml(str(ROOT / "src/config/envs/microgrid.yaml")),
    )
    name = args.output_id or (
        "cooperation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    )
    out = ROOT / "results/oracle_milp" / name
    out.mkdir(parents=True, exist_ok=False)
    original = deepcopy(config["env_args"])
    env = make_env(original, out)
    initial = float(config["evaluation_initial_soc"])
    if args.all_test_days and args.split != "test":
        raise ValueError("--all-test-days requires the test split")
    representatives = representative_days(env, args.split)
    rows = (getattr(env, f"{args.split}_indices").tolist() if args.all_days or args.all_test_days
            else [r["row_idx"] for r in representatives])
    results, records = [], []
    for i, row in enumerate(rows):
        result = comparison(env, original, row, args.split, initial)
        results.append({k: v for k, v in result.items() if k != "records"})
        records.extend(result["records"])
        if i % 20 == 0 or i == len(rows) - 1:
            print(f"Completed {i + 1}/{len(rows)} days", flush=True)
    verdict = cooperation_acceptance(results)
    pd.DataFrame(records).to_csv(out / "trajectories.csv", index=False)
    pd.DataFrame([a for r in results for a in r["agents"]]).to_csv(out / "agent_metrics.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("solutions", "agents", "ablations")}
                  for r in results]).to_csv(out / "day_metrics.csv", index=False)
    with (out / "config_snapshot.yaml").open("x", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    plot_results(out, records, [r["row_idx"] for r in representatives])
    plot_cooperation(out, records, results, [r["row_idx"] for r in representatives], original)
    summary = {
        "model_version": VERSION, "days": rows, "representatives": representatives,
        "split": args.split, "joint_milps": 3 * len(rows),
        "acceptance": verdict, "daily_terminal_soc": initial,
        "terminal_scope": "MILP cyclic-day reference; training environment continues across days",
        "units": "scenario kW/kWh/CU, assumptions not verified source metadata",
        "mean_full_cost": float(np.mean([r["full_cost"] for r in results])),
        "mean_no_battery_cost": float(np.mean([r["no_battery_cost"] for r in results])),
        "mean_isolated_cost": float(np.mean([r["isolated_cost"] for r in results])),
        "mean_cooperation_saving_fraction": float(np.mean([r["cooperation_saving_fraction"] for r in results])),
        "min_cooperation_saving_fraction": min(r["cooperation_saving_fraction"] for r in results),
        "min_certified_cooperation_saving_fraction": min(r["cooperation_saving_lower_fraction"] for r in results),
        "mean_battery_cost_saving_fraction": float(np.mean([r["battery_cost_saving_fraction"] for r in results])),
        "min_battery_cost_saving_fraction": min(r["battery_cost_saving_fraction"] for r in results),
        "min_certified_battery_saving_fraction": min(r["battery_saving_lower_fraction"] for r in results),
        "mean_transfer_fraction": float(np.mean([r["transfer_fraction"] for r in results])),
        "min_transfer_fraction": min(r["transfer_fraction"] for r in results),
        "max_replay_error": max(r["max_replay_error"] for r in results),
        "max_ablation_replay_error": max(r["ablation_max_replay_error"] for r in results),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src/envs/maenv.py", ROOT / "src/config/envs/microgrid.yaml",
                      Path(__file__), *sorted((ROOT / "src/envs/data").glob("*.csv")))
        },
    }
    for filename, payload in (("summary.json", summary), ("solver_results.json", results)):
        with (out / filename).open("x", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)
    print(f"RESULTS={out}", flush=True)
    if not verdict["passed"]:
        raise SystemExit("Acceptance failed; inspect the saved diagnostics")


if __name__ == "__main__":
    main()
