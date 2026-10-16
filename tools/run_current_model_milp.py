"""Perfect foresight economic dispatch for maenv_sep_v1, with explicit PWL errors."""

import argparse
import csv
import hashlib
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
import scipy
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from direct_main import load_yaml, recursive_dict_update
from envs.maenv import MGEnv
from utils.policy_evaluation import fixed_rows


class LinearModel:
    def __init__(self):
        self.lower, self.upper, self.cost, self.integer = [], [], [], []
        self.rows, self.row_lower, self.row_upper = [], [], []

    def var(self, low, high, cost=0, integer=False):
        index = len(self.lower)
        self.lower.append(low)
        self.upper.append(high)
        self.cost.append(cost)
        self.integer.append(int(integer))
        return index

    def row(self, entries, low=-np.inf, high=np.inf):
        self.rows.append(entries)
        self.row_lower.append(low)
        self.row_upper.append(high)

    def conditional(self, entries, active, low=-np.inf, high=np.inf):
        minimum = sum(v * (self.lower[k] if v >= 0 else self.upper[k])
                      for k, v in entries.items())
        maximum = sum(v * (self.upper[k] if v >= 0 else self.lower[k])
                      for k, v in entries.items())
        if np.isfinite(low):
            relax = max(0, low - minimum)
            self.row({**entries, active: -relax}, low=low - relax)
        if np.isfinite(high):
            relax = max(0, maximum - high)
            self.row({**entries, active: relax}, high=high + relax)

    def solve(self, gap, limit):
        rr, cc, vv = [], [], []
        for row, entries in enumerate(self.rows):
            for col, value in entries.items():
                rr.append(row)
                cc.append(col)
                vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(self.rows), len(self.lower)))
        return milp(
            np.array(self.cost), integrality=np.array(self.integer),
            bounds=Bounds(self.lower, self.upper),
            constraints=LinearConstraint(
                matrix.tocsc(), np.array(self.row_lower), np.array(self.row_upper),
            ),
            options={"mip_rel_gap": gap, "time_limit": limit},
        )


def quadratic_epigraph(model, value, result, a, b, c, knots, active=None):
    for left, right in zip(knots[:-1], knots[1:]):
        slope = a * (left + right) + b
        intercept = c - a * left * right
        entries = {result: 1, value: -slope}
        if active is None:
            model.row(entries, low=intercept)
        else:
            entries[active] = -intercept
            model.row(entries, low=0)


def solve_agent(net, prices, params, initial_soc, segments=40, gap=1e-4,
                time_limit=30, positive_power=1e-3, projection_guard=1e-6):
    """Five disjunctive transitions retain reference projection and actual pb sign."""
    net, prices = np.asarray(net, dtype=float), np.asarray(prices, dtype=float)
    cap = params["battery_cap"]
    eta_ch, eta_dis = params["raw_ch"], params["raw_dis"]
    dg = params["dg_max"]
    if segments < 2 or positive_power <= 0 or projection_guard <= 0:
        raise ValueError("Invalid approximation settings")
    if not 0.2 <= initial_soc <= 0.8:
        raise ValueError("initial_soc must be in [0.2, 0.8]")
    if any(params[k] < 0 for k in ("costa", "bessa", "costb", "bessb", "costc", "bessc")):
        raise ValueError("This convex PWL formulation requires nonnegative cost coefficients")
    model = LinearModel()
    soc = [model.var(0.2, 0.8) for _ in range(len(net) + 1)]
    model.row({soc[0]: 1}, initial_soc, initial_soc)
    decisions = []
    generation_knots = np.linspace(0, dg, segments + 1)
    x_low, x_high = -0.4 * cap, 3.4 * cap
    battery_knots = np.linspace(x_low, x_high, segments + 1)
    for t, (demand, price) in enumerate(zip(net, prices)):
        pd = model.var(0, dg)
        q = model.var(-cap, cap)
        pb = model.var(-cap, cap)
        modes = [model.var(0, 1, integer=True) for _ in range(5)]
        model.row(dict.fromkeys(modes, 1), 1, 1)
        charge, discharge, low_pos, low_zero, high = modes
        for active, eta, q_low, q_high in (
            (charge, eta_ch, -cap, 0), (discharge, eta_dis, positive_power, cap),
        ):
            model.conditional({q: 1}, active, q_low, q_high)
            model.conditional({pb: 1, q: -1}, active, 0, 0)
            model.conditional(
                {soc[t + 1]: cap, soc[t]: -cap, q: eta}, active, 0, 0,
            )
        for active in (low_pos, low_zero):
            model.conditional({q: 1}, active, 0, cap)
            model.conditional({soc[t + 1]: 1}, active, 0.2, 0.2)
            model.conditional({pb: 1, soc[t]: -cap}, active, -0.2 * cap, -0.2 * cap)
            model.conditional(
                {soc[t]: cap, q: -eta_dis}, active,
                high=(0.2 - projection_guard) * cap,
            )
        model.conditional({pb: 1}, low_pos, low=positive_power)
        model.conditional({soc[t]: 1}, low_zero, 0.2, 0.2)
        model.conditional({q: 1}, high, -cap, 0)
        model.conditional({soc[t + 1]: 1}, high, 0.8, 0.8)
        model.conditional({pb: 1, soc[t]: -cap}, high, -0.8 * cap, -0.8 * cap)
        model.conditional(
            {soc[t]: cap, q: -eta_ch}, high, low=(0.8 + projection_guard) * cap,
        )

        # Import is exactly max(net - pd - pb, 0), even at negative prices.
        pg_low, pg_high = demand - dg - cap, demand + cap
        pg = model.var(pg_low, pg_high)
        imported = model.var(0, max(pg_high, 0), cost=price)
        grid_mode = model.var(0, 1, integer=True)
        model.row({pg: 1, pd: 1, pb: 1}, demand, demand)
        model.row({imported: 1, pg: -1}, low=0)
        model.row({imported: 1, grid_mode: -max(pg_high, 0)}, high=0)
        relax = max(-pg_low, 0)
        model.row({imported: 1, pg: -1, grid_mode: relax}, high=relax)

        gen_max = params["costa"] * dg**2 + params["costb"] * dg + params["costc"]
        gen = model.var(0, max(gen_max, 1) * 2, cost=0.5)
        quadratic_epigraph(
            model, pd, gen, params["costa"], params["costb"], params["costc"],
            generation_knots,
        )
        positive = model.var(0, 1, integer=True)
        nonpositive = model.var(0, 1, integer=True)
        model.row({positive: 1, discharge: -1, low_pos: -1}, 0, 0)
        model.row({positive: 1, nonpositive: 1}, 1, 1)
        x0 = model.var(min(x_low, 0), x_high)
        x1 = model.var(min(x_low, 0), x_high)
        model.row(
            {x0: 1, x1: 1, pb: -1, soc[t + 1]: 3 * cap}, 3 * cap, 3 * cap,
        )
        for value, active, weight in ((x0, nonpositive, 1), (x1, positive, 0.3)):
            model.row({value: 1, active: -x_low}, low=0)
            model.row({value: 1, active: -x_high}, high=0)
            cost_max = max(
                params["bessa"] * x**2 + params["bessb"] * x + params["bessc"]
                for x in (x_low, x_high)
            )
            cost = model.var(-2 * abs(cost_max) - 1, 2 * abs(cost_max) + 1, cost=weight)
            quadratic_epigraph(
                model, value, cost, params["bessa"], params["bessb"], params["bessc"],
                battery_knots, active,
            )
        decisions.append((pd, q, pb, pg, imported, modes))
    started = time.perf_counter()
    result = model.solve(gap, time_limit)
    seconds = time.perf_counter() - started
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"MILP has no usable incumbent: {result.message}")
    residuals = []
    for entries, lo, hi in zip(model.rows, model.row_lower, model.row_upper):
        val = sum(result.x[i] * v for i, v in entries.items())
        residuals.append(max(lo - val, val - hi, 0))
    residuals.extend(np.maximum(np.array(model.lower) - result.x, 0))
    residuals.extend(np.maximum(result.x - np.array(model.upper), 0))
    residuals.extend(abs(result.x[i] - round(result.x[i]))
                     for i, integer in enumerate(model.integer) if integer)
    if max(residuals) > 2e-5:
        raise RuntimeError(f"MILP incumbent constraint violation: {max(residuals)}")
    schedule = {
        "pd": [float(result.x[d[0]]) for d in decisions],
        "requested_pb": [float(result.x[d[1]]) for d in decisions],
        "pb": [float(result.x[d[2]]) for d in decisions],
        "pg": [float(result.x[d[3]]) for d in decisions],
        "soc": result.x[soc].tolist(),
        "mode": [int(np.argmax(result.x[d[5]])) for d in decisions],
    }
    error_bound = len(net) * (
        0.5 * params["costa"] * (dg / segments)**2 / 4
        + params["bessa"] * ((x_high - x_low) / segments)**2 / 4
    )
    return {
        "status": int(result.status), "message": result.message,
        "pwl_objective": float(result.fun),
        "pwl_dual_bound": float(result.mip_dual_bound),
        "mip_gap": float(result.mip_gap), "solve_seconds": seconds,
        "max_constraint_violation": float(max(residuals)),
        "quadratic_interpolation_error_bound": error_bound,
        "restricted_domain_exact_cost_lower_bound": float(result.mip_dual_bound) - error_bound,
        "schedule": schedule,
    }


def replay_request(before, target, requested, mode, cap, eta_ch, eta_dis):
    # A normal transition exactly at a SOC boundary can cross the strict
    # projection test due to solver arithmetic. Nudge inward, never change modes.
    if mode in (0, 1):
        eta = eta_ch if mode == 0 else eta_dis
        target = np.clip(target, 0.2 + 1e-10, 0.8 - 1e-10)
        requested = (before - target) * cap / eta
        if mode == 0:
            requested = min(requested, 0)
    return requested


def replay_day(env, row, solutions, initial_soc):
    env.reset_test(source="test", row_idx=row, initial_soc=initial_soc)
    corrections = []
    for t in range(24):
        actions = np.zeros((env.n_agents, 2), dtype=float)
        for a, result in enumerate(solutions):
            p, schedule = env.params, result["schedule"]
            cap = p["battery_caps"][a]
            before = env.params["socs"][a]
            q, mode = schedule["requested_pb"][t], schedule["mode"][t]
            q = replay_request(
                before, schedule["soc"][t + 1], q, mode, cap,
                p["raw_ch"][a], p["raw_dis"][a],
            )
            actions[a] = [max(schedule["pd"][t], 0) / p["dg_max"][a], q / cap]
            corrections.append(abs(q - schedule["requested_pb"][t]))
        if np.any(actions < env.action_lows - 1e-7) or np.any(actions > env.action_highs + 1e-7):
            raise RuntimeError("Reconstructed actions violate configured bounds")
        env.step(actions)
    records = [dict(record) for record in env.ep_buffer]
    max_soc, max_pb, max_grid = 0.0, 0.0, 0.0
    for record in records:
        a, t = env.possible_agents.index(record["agent"]), record["col_idx"]
        s = solutions[a]["schedule"]
        max_soc = max(max_soc, abs(record["soc"] - s["soc"][t + 1]))
        max_pb = max(max_pb, abs(record["pb"] - s["pb"][t]))
        max_grid = max(max_grid, abs(record["pg"] - s["pg"][t]))
        record["economic_objective"] = record["buy_cost"] + 0.5 * record["gen_cost"] + record["bess_cost"]
        record["solver_mode"] = s["mode"][t]
    if max_soc > 2e-6 or max_pb > 2e-4 or max_grid > 2e-4:
        raise RuntimeError(f"Environment replay mismatch: soc={max_soc}, pb={max_pb}, grid={max_grid}")
    return records, {
        "max_soc_error": max_soc, "max_pb_error": max_pb, "max_grid_error": max_grid,
        "max_requested_power_numerical_correction": max(corrections),
    }


def baseline_day(env, row, initial_soc):
    env.reset_test(source="test", row_idx=row, initial_soc=initial_soc)
    for _ in range(24):
        env.step(np.zeros((env.n_agents, 2)))
    return [
        {**r, "economic_objective": r["buy_cost"] + 0.5 * r["gen_cost"] + r["bess_cost"]}
        for r in env.ep_buffer
    ]


def write_csv(path, records):
    with path.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_day(path, row, records, baseline, agents):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#168078", "#b54465", "#4065ae"]
    first = [r for r in records if r["agent"] == agents[0]]
    hours = np.arange(24)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")
    ax = axes[0, 0]
    ax.plot(hours, [r["load_raw"] for r in first], label="Raw load", color="#4065ae")
    ax.plot(hours, [r["pv_raw"] for r in first], label="Raw PV", color="#d5a12a")
    ax.set(title="Source data (CSV units unverified)", ylabel="Raw input value")
    ax.legend()
    ax = axes[0, 1]
    for key, label, color in (("load", "Load", "#4065ae"), ("pv", "PV", "#168078"),
                               ("price", "Price", "#b54465")):
        ax.plot(hours, [r[key] for r in first], label=label, color=color)
    ax.axhline(0, color="#999999", linewidth=0.6)
    ax.set(title="Inputs actually used by the model", ylabel="Standardized value")
    ax.legend(ncol=3)
    for a, agent in enumerate(agents):
        data = [r for r in records if r["agent"] == agent]
        ax = axes[1, 0]
        ax.plot(np.arange(25), [data[0]["soc_init"]] + [r["soc"] for r in data],
                label=agent, color=colors[a])
        axes[1, 1].plot(hours, [r["pg"] for r in data], label=agent, color=colors[a])
    axes[1, 0].set(title="SOC (no terminal target)", ylim=(0.18, 0.82), ylabel="SOC")
    axes[1, 1].set(title="Grid exchange: positive = import", ylabel="Model power")
    axes[1, 0].legend()
    axes[1, 1].legend()
    for group, label, color in ((records, "Optimized", "#168078"), (baseline, "Zero-action baseline", "#b54465")):
        daily = [sum(r["economic_objective"] for r in group if r["col_idx"] == t) for t in hours]
        axes[2, 0].plot(hours, np.cumsum(daily), label=label, color=color)
    axes[2, 0].set(title="Cumulative economic objective", ylabel="Model cost (not currency)")
    axes[2, 0].legend()
    keys = ("buy_cost", "gen_cost", "bess_cost")
    for i, (group, label, color) in enumerate(((records, "Optimized", "#168078"), (baseline, "Baseline", "#b54465"))):
        values = [sum(r[key] * (0.5 if key == "gen_cost" else 1) for r in group) for key in keys]
        axes[2, 1].bar(np.arange(3) + (i - 0.5) * 0.35, values, width=0.35, label=label, color=color)
    axes[2, 1].set_xticks(np.arange(3), ["Grid", "0.5 x generation", "Battery"])
    axes[2, 1].set(title="Economic cost components", ylabel="Model cost (not currency)")
    axes[2, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    for ax in axes[:, 0]:
        ax.set_xlabel("Hour index")
    axes[0, 1].set_xlabel("Hour index")
    axes[1, 1].set_xlabel("Hour index")
    fig.suptitle(f"Perfect foresight | fixed test row {row} (zero-based) | PWL MILP", fontsize=15)
    fig.savefig(path / f"row_{row}_overview.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout="constrained")
    for ax, agent in zip(axes, agents):
        data = [r for r in records if r["agent"] == agent]
        for key, label, color in (("pd", "Generation", "#4065ae"), ("pb", "Battery (+ discharge)", "#168078"),
                                  ("pg", "Grid (+ import)", "#b54465")):
            ax.step(hours, [r[key] for r in data], where="mid", label=label, color=color)
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.set(title=agent, ylabel="Model power")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Hour index")
    fig.suptitle(f"Dispatch | row {row} | normalized inputs, not physical energy units", fontsize=14)
    fig.savefig(path / f"row_{row}_dispatch.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--segments", type=int, default=40)
    parser.add_argument("--time-limit", type=float, default=30)
    parser.add_argument("--gap", type=float, default=1e-4)
    parser.add_argument("--output-id")
    args = parser.parse_args()
    config = load_yaml(str(ROOT / "src/config/default.yaml"))
    config = recursive_dict_update(config, load_yaml(str(ROOT / "src/config/envs/microgrid.yaml")))
    output_id = args.output_id or (
        "current_model_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid4().hex[:8]
    )
    out = ROOT / "results/oracle_milp" / output_id
    out.mkdir(parents=True, exist_ok=False)
    env_config = deepcopy(config["env_args"])
    env_config.update(result_dir=str(out), run_id="replay", log_trajectories=False)
    env = MGEnv(env_config, "perfect_foresight")
    rows = fixed_rows(env, "test", config["typical_day_rows"], args.days)
    initial = config["evaluation_initial_soc"]
    all_records, all_baselines, summaries, all_solutions = [], [], [], {}
    with (out / "config_snapshot.yaml").open("x", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    print(f"OUTPUT_DIR={out}", flush=True)
    for row in rows:
        solutions = []
        for a, params in enumerate(config["env_args"]["data"]):
            print(f"Solving row={row}, agent={a + 1}", flush=True)
            solution = solve_agent(
                env.data_dict["load_pv"][row], env.data_dict["price"][row], params,
                initial, args.segments, args.gap, args.time_limit,
            )
            solutions.append(solution)
        records, replay = replay_day(env, row, solutions, initial)
        baseline = baseline_day(env, row, initial)
        cost = sum(r["economic_objective"] for r in records)
        pwl = sum(s["pwl_objective"] for s in solutions)
        error_bound = sum(s["quadratic_interpolation_error_bound"] for s in solutions)
        if pwl - cost < -0.01 or pwl - cost > error_bound + 0.01:
            raise RuntimeError("Economic cost falls outside PWL error certificate")
        base_cost = sum(r["economic_objective"] for r in baseline)
        summary = {
            "row_idx": row, "model_economic_cost": cost, "zero_action_cost": base_cost,
            "economic_cost_reduction": base_cost - cost,
            "economic_cost_reduction_fraction": (base_cost - cost) / abs(base_cost),
            "pwl_objective": pwl, "pwl_minus_replay_cost": pwl - cost,
            "pwl_interpolation_error_bound": error_bound,
            "max_agent_mip_gap": max(s["mip_gap"] for s in solutions),
            "all_agents_optimal_within_requested_gap": all(s["status"] == 0 for s in solutions),
            "restricted_domain_lower_bound": sum(s["restricted_domain_exact_cost_lower_bound"] for s in solutions),
            "max_soc_replay_error": replay["max_soc_error"],
            "max_power_replay_error": replay["max_pb_error"],
            "grid_import_cost": sum(r["buy_cost"] for r in records),
            "half_generation_cost": sum(0.5 * r["gen_cost"] for r in records),
            "battery_cost": sum(r["bess_cost"] for r in records),
            "omitted_emission_cost": sum(-r["env_reward"] for r in records),
            "omitted_action_penalty": sum(-r["punishment"] for r in records),
            "projection_events": sum(r["battery_action_clip"] > 1e-5 for r in records),
            "tiny_discharge_hours": sum(0 < r["pb"] < 0.01 for r in records),
            "final_soc": [r["soc"] for r in records if r["col_idx"] == 23],
            "solve_seconds": sum(s["solve_seconds"] for s in solutions),
        }
        summaries.append(summary)
        all_solutions[str(row)] = {"agents": solutions, "replay_validation": replay}
        all_records.extend(records)
        all_baselines.extend(baseline)
        write_csv(out / f"row_{row}_trajectory.csv", records)
        plot_day(out, row, records, baseline, env.possible_agents)
        print(json.dumps(summary), flush=True)
    write_csv(out / "typical_day_trajectories.csv", all_records)
    write_csv(out / "zero_action_baseline.csv", all_baselines)
    write_csv(out / "day_summary.csv", [
        {**s, "final_soc": json.dumps(s["final_soc"])} for s in summaries
    ])
    with (out / "solver_results.json").open("x", encoding="utf-8") as file:
        json.dump(all_solutions, file, indent=2, allow_nan=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": env.battery_model_version, "dispatch_solved": True,
        "objective": "sum(grid purchase + 0.5 * generation + battery cost)",
        "objective_excludes": ["emissions", "action punishment", "RL reward scaling"],
        "rows": rows, "row_index_base": 0, "initial_soc_each_day": initial,
        "day_coupling": "independent fixed-day evaluation", "terminal_soc_constraint": None,
        "selection": "existing fixed test-day default, not seasonal representatives",
        "segments": args.segments, "positive_discharge_minimum": 1e-3,
        "strict_projection_soc_guard": 1e-6,
        "certificate_scope": "PWL MILP restricted by positive-power floor and strict-projection guard; not global original-model optimum",
        "units": "standardized model costs and powers; not currency or verified kWh",
        "pathologies": [
            "negative standardized prices enter settlement",
            "pb>0 immediately discounts all battery cost to 30 percent",
            "legacy projected SOC and actual battery power are not efficiency-consistent",
        ],
        "scipy_version": scipy.__version__, "solver": "scipy.optimize.milp / HiGHS",
        "requested_mip_gap": args.gap, "time_limit_per_agent": args.time_limit,
        "mean_optimized_model_cost": float(np.mean([s["model_economic_cost"] for s in summaries])),
        "mean_baseline_model_cost": float(np.mean([s["zero_action_cost"] for s in summaries])),
        "all_optimal_within_requested_gap": all(s["all_agents_optimal_within_requested_gap"] for s in summaries),
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src/envs/data").glob("*.csv")
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src/envs/maenv.py", Path(__file__))
        },
        "days": summaries,
    }
    with (out / "manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
    report = [
        "# Perfect Foresight Economic Dispatch", "",
        "No training, reward, source CSV, or existing artifact was modified.",
        f"Rows (zero based): {rows}. Each day starts at SOC {initial}; no terminal target.",
        "The eight consecutive fixed test days are not seasonal representative days.", "",
        "## Formulation",
        "Current standardized inputs, independent per-agent grid settlement,",
        "legacy charge/discharge SOC equations and saturation projection are retained.",
        "Only economic cost is optimized: grid + 0.5 generation + battery.",
        "Emission costs and requested-action penalties are reported but excluded.",
        f"Quadratics use {args.segments} chord-interpolation segments; errors are bounded.",
        "Strict positive-power branches use pb >= 0.001; strict SOC projections use a 1e-6 guard.",
        "All certificates apply only to this restricted domain, not the original discontinuous model.",
        "A time-limited incumbent is explicitly distinguished from a certified PWL optimum.", "",
        "## Interpretation",
        "Costs are standardized model values, not money. A pb just above zero reduces",
        "the entire battery-cost polynomial to 30%; tiny discharge can therefore dominate",
        "the optimum. Saturation and negative normalized prices also produce nonphysical incentives.",
        "These are properties of the requested model, not solver corrections.", "",
        "## Outputs",
        "- typical_day_trajectories.csv: all agents and hours, raw and standardized inputs.",
        "- zero_action_baseline.csv: same starting SOC and dates, all actions zero.",
        "- day_summary.csv: replay cost, interpolation errors, gaps, omitted penalties.",
        "- solver_results.json: every solver result and decision trajectory.",
        "- row_*_overview.png / row_*_dispatch.png: 16 figures.",
    ]
    (out / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"COMPLETED={out}", flush=True)


if __name__ == "__main__":
    main()
