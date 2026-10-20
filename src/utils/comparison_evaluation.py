"""Isolated continuous-inventory evaluation with independent trajectory accounting."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from controllers.cqmix_controller import CQMixMAC
from runners.episode_runner import EpisodeRunner


class _QuietLogger:
    def log_stat(self, *args):
        pass


def make_evaluation_runner(config, policy=None):
    """Construct a separate runner; never borrow a live training environment."""
    config = deepcopy(config)
    config.update(device="cpu", use_cuda=False, batch_size_run=1)
    config["env_args"]["log_trajectories"] = False
    args = SimpleNamespace(**config)
    runner = EpisodeRunner(args, _QuietLogger())
    info = runner.get_env_info()
    for key in ("n_agents", "n_actions", "state_shape", "obs_shape", "action_spaces"):
        setattr(args, key, info[key])
    scheme = {
        "state": {"vshape": info["state_shape"]},
        "obs": {"vshape": info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (info["n_actions"],), "group": "agents", "dtype": torch.float32},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": info["n_agents"]}
    if policy is not None:
        mac = policy
    elif args.name == "gavdac_mappo":
        from modules.mappo import MAPPOController
        mac = MAPPOController(scheme, groups, args)
    else:
        mac = CQMixMAC(scheme, groups, args)
    if policy is None:
        mac.agent.eval()
    runner.setup(scheme, groups, {}, mac)
    return runner, mac


def summarize_trajectories(records, env, rows, initial_soc, inventory_mode):
    if inventory_mode not in {"continuous", "daily_reset"}:
        raise ValueError("Unknown inventory mode")
    if len(records) != len(rows) * env.episode_limit * env.n_agents:
        raise ValueError("Unexpected trajectory length")
    initial = np.broadcast_to(np.asarray(initial_soc, dtype=float), (env.n_agents,)).copy()
    if not (np.isfinite(initial).all()
            and ((initial >= env.soc_min) & (initial <= env.soc_max)).all()):
        raise ValueError("Invalid initial inventory")
    lookup = {}
    for record in records:
        identity = (record["row_idx"], record["col_idx"], record["agent"])
        if identity in lookup:
            raise ValueError("Duplicate trajectory record")
        if not all(np.isfinite(value) for value in record.values()
                   if isinstance(value, (int, float, np.number))):
            raise ValueError("Non-finite trajectory value")
        lookup[identity] = record
    p, dt = env.params, env.time_step_hours
    totals = {key: 0.0 for key in (
        "economic_cost", "full_cost", "buy_cost", "gen_cost", "bess_cost", "exchange_cost",
        "emission_cost", "penalty_cost", "charge_kwh", "discharge_kwh", "grid_kwh",
        "generator_kwh", "peer_sent_kwh", "peer_received_kwh", "spill_kwh",
    )}
    maxima = {"balance": 0.0, "soc_transition": 0.0, "cost": 0.0, "exchange_balance": 0.0}
    end = initial.copy()
    days = []
    for row in rows:
        before = end.copy() if inventory_mode == "continuous" else initial.copy()
        start = before.copy()
        costs = {"economic_cost": 0.0, "full_cost": 0.0}
        for hour in range(env.episode_limit):
            exported, imported = 0.0, 0.0
            for i, agent in enumerate(env.possible_agents):
                identity = (row, hour, agent)
                if identity not in lookup:
                    raise ValueError("Missing trajectory record")
                r = lookup[identity]
                if abs(r["soc_init"] - before[i]) > 1e-10:
                    raise ValueError("Inventory discontinuity")
                if not (
                    env.soc_min - 1e-10 <= r["soc"] <= env.soc_max + 1e-10
                    and -p["battery_charge_max"][i] - 1e-8 <= r["pb"]
                    <= p["battery_discharge_max"][i] + 1e-8
                    and -1e-8 <= r["pd"] <= p["dg_max"][i] + 1e-8
                    and min(r["pg"], r["spill"], r["peer_import"], r["peer_export"]) >= -1e-8
                    and max(r["peer_import"], r["peer_export"]) <= p["exchange_max"][i] + 1e-8
                ):
                    raise ValueError("Physical equipment bound violated")
                expected_load, expected_pv = (env.agent_data[key][row, hour, i]
                                              for key in ("load", "pv"))
                if (abs(r["load"] - expected_load) > 1e-8
                        or abs(r["pv"] - expected_pv) > 1e-8
                        or abs(r["price"] - env.data_dict["price"][row, hour]) > 1e-10):
                    raise ValueError("Trajectory exogenous data changed")
                balance = (
                    r["pv"] + r["pd"] + r["pg"] + r["pb"] + r["peer_import"]
                    - r["load"] - r["peer_export"] - r["spill"]
                )
                energy = (r["pb"] / p["raw_dis"][i] if r["pb"] >= 0
                          else r["pb"] * p["raw_ch"][i])
                soc_error = r["soc"] - (before[i] - energy * dt / p["battery_caps"][i])
                independent = {
                    "buy_cost": r["pg"] * r["price"] * dt,
                    "gen_cost": (p["costa"][i] * r["pd"]**2
                                 + p["costb"][i] * r["pd"] + p["costc"][i]) * dt,
                    "bess_cost": p["cycle_cost"][i] * abs(r["pb"]) * dt,
                    "exchange_cost": r["peer_export"] * env.exchange_fee * dt,
                    "emission_cost": p["env_param"][i] * r["pd"] * dt,
                    "penalty_cost": p["battery_punishment_params"][i]
                    * abs(r["pb_requested"] - r["pb"]) * dt,
                }
                independent["economic_cost"] = sum(independent[key] for key in
                                                     ("buy_cost", "gen_cost", "bess_cost",
                                                      "exchange_cost"))
                independent["full_cost"] = (independent["economic_cost"]
                                             + independent["emission_cost"]
                                             + independent["penalty_cost"])
                errors = [abs(independent[key] - r[key]) for key in (
                    "buy_cost", "gen_cost", "bess_cost", "exchange_cost",
                    "economic_cost", "full_cost",
                )]
                errors += [abs(independent["emission_cost"] + r["env_reward"]),
                           abs(independent["penalty_cost"] + r["punishment"])]
                maxima["balance"] = max(maxima["balance"], abs(balance))
                maxima["soc_transition"] = max(maxima["soc_transition"], abs(soc_error))
                maxima["cost"] = max(maxima["cost"], *errors)
                if abs(balance) > 1e-8 or abs(soc_error) > 1e-10 or max(errors) > 1e-8:
                    raise ValueError("Trajectory physical transition or cost check failed")
                for key, value in independent.items():
                    totals[key] += value
                for key in costs:
                    costs[key] += independent[key]
                energy_terms = {
                    "charge_kwh": max(-r["pb"], 0), "discharge_kwh": max(r["pb"], 0),
                    "grid_kwh": r["pg"], "generator_kwh": r["pd"],
                    "peer_sent_kwh": r["peer_export"], "peer_received_kwh": r["peer_import"],
                    "spill_kwh": r["spill"],
                }
                for key, value in energy_terms.items():
                    totals[key] += value * dt
                exported += r["peer_export"]
                imported += r["peer_import"]
                before[i] = r["soc"]
            exchange_error = abs(imported - exported * env.exchange_efficiency)
            maxima["exchange_balance"] = max(maxima["exchange_balance"], exchange_error)
            if exchange_error > 1e-8:
                raise ValueError("Community exchange balance violated")
        end = before.copy()
        days.append({"row_idx": row, **costs, "initial_soc": start.tolist(),
                     "terminal_soc": end.tolist()})
    return {
        "inventory_mode": inventory_mode, "rows": rows, "days": days, "totals": totals,
        "mean_economic_cost": totals["economic_cost"] / len(rows),
        "mean_full_cost": totals["full_cost"] / len(rows),
        "worst_day_full_cost": max(day["full_cost"] for day in days),
        "initial_soc": initial.tolist(), "terminal_soc": end.tolist(),
        "max_errors": maxima, "physical_checks_passed": True,
        "boundary": "SOC carried; policy hidden state and previous action reset daily"
        if inventory_mode == "continuous" else "SOC and policy state reset daily",
    }


def evaluate_inventory(runner, mac, rows, initial_soc, source, inventory_mode,
                       policy_source="rl", checkpoint_step=0):
    """Evaluate on a dedicated runner, using the ordinary policy/batch contract."""
    rows = list(rows)
    if (not rows or any(isinstance(row, bool) or not isinstance(row, (int, np.integer))
                        for row in rows)
            or rows != sorted(set(rows))
            or any(row not in getattr(runner.env, f"{source}_indices") for row in rows)):
        raise ValueError("Evaluation rows must be ordered, unique and inside their split")
    if inventory_mode not in {"continuous", "daily_reset"}:
        raise ValueError("Unknown inventory mode")
    if inventory_mode == "continuous" and any(b != a + 1 for a, b in zip(rows, rows[1:])):
        raise ValueError("Continuous evaluation requires adjacent rows")
    initial = np.broadcast_to(np.asarray(initial_soc, dtype=float),
                              (runner.env.n_agents,)).copy()
    end = initial.copy()
    records, returns = [], []
    steps_before = runner.t_env
    with torch.no_grad():
        for row in rows:
            soc = end if inventory_mode == "continuous" else initial
            _, value = runner.run(
                mac, test_mode=True, test_options={
                    "source": source, "row_idx": row, "initial_soc": soc.copy(),
                    "trajectory_context": {
                        "policy_source": policy_source, "checkpoint_step": checkpoint_step,
                    },
                },
            )
            if not np.isfinite(value):
                raise ValueError("Non-finite evaluation return")
            records.extend(deepcopy(runner.env.ep_buffer))
            returns.append(float(value))
            end = runner.env.params["socs"].copy()
    if runner.t_env != steps_before:
        raise ValueError("Evaluation consumed training interactions")
    summary = summarize_trajectories(records, runner.env, rows, initial, inventory_mode)
    summary.update(returns=returns, mean_return=float(np.mean(returns)),
                   policy_source=policy_source, checkpoint_step=checkpoint_step)
    return summary, records
