import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envs.maenv import MGEnv  # noqa: E402


def simulate_episode(x, env, row_idx, initial_socs):
    actions = x.reshape(env.episode_limit, env.n_agents, 2)
    socs = initial_socs.astype(float).copy()
    totals = dict.fromkeys((
        "episode_return", "grid_purchase_cost", "baseline_grid_purchase_cost",
        "generate_cost", "bess_cost", "env_cost", "mean_soc",
    ), 0.0)
    for t in range(env.episode_limit):
        result = env.dispatch(actions[t], socs, (row_idx, t))
        socs = result["soc"]
        totals["episode_return"] += float(np.mean(result["reward"].astype(np.float32)))
        totals["grid_purchase_cost"] += float(result["buy_cost"].sum())
        totals["baseline_grid_purchase_cost"] += float(
            np.maximum(result["load"] - result["pv"], 0).sum()
            * result["price"] * env.time_step_hours,
        )
        totals["generate_cost"] += float(result["gen_cost"].sum())
        totals["bess_cost"] += float(result["bess_cost"].sum())
        totals["env_cost"] += float(-result["env_reward"].sum())
        totals["mean_soc"] += float(socs.mean()) / env.episode_limit
    totals["grid_purchase_saving"] = (
        totals["baseline_grid_purchase_cost"] - totals["grid_purchase_cost"]
    )
    baseline = totals["baseline_grid_purchase_cost"]
    totals["grid_purchase_saving_ratio"] = totals["grid_purchase_saving"] / baseline if baseline > 1e-9 else 0
    totals.update(curtailment_penalty=0.0, soc_reserve_penalty=0.0)
    return totals


def objective_fn(x, env, row_idx, initial_socs):
    metrics = simulate_episode(x, env, row_idx, initial_socs)
    return -metrics["episode_return"]


def make_initial_guess(env, row_idx, initial_socs, mode):
    t_horizon = env.episode_limit
    n_agents = env.n_agents
    dg_max = env.params["dg_max"]
    load_pv = env.agent_data["load"][row_idx] - env.agent_data["pv"][row_idx]
    price = env.data_dict["price"][row_idx]

    x0 = np.zeros((t_horizon, n_agents, 2), dtype=np.float64)
    if mode == "gen_cover":
        positive_net = np.maximum(load_pv, 0.0)
        pd = np.zeros_like(positive_net)
        np.divide(positive_net, dg_max[None, :], out=pd, where=dg_max[None, :] > 0)
        pd = np.clip(pd, 0.0, 1.0)
        x0[:, :, 0] = pd
    elif mode == "price_shift":
        median_price = float(np.median(price))
        positive_net = np.maximum(load_pv, 0.0)
        pd = np.zeros_like(positive_net)
        np.divide(positive_net, dg_max[None, :], out=pd, where=dg_max[None, :] > 0)
        pd = np.clip(pd, 0.0, 1.0)
        x0[:, :, 0] = pd
        x0[:, :, 1] = np.where(price[:, None] >= median_price, 0.7, -0.7)
    return x0.reshape(-1)


def optimize_day(env, row_idx, initial_socs, maxiter):
    bounds = list(zip(
        np.tile(env.action_lows.flatten(), env.episode_limit),
        np.tile(env.action_highs.flatten(), env.episode_limit),
    ))

    candidates = ["zeros", "gen_cover", "price_shift"]
    best = None
    best_res = None
    for mode in candidates:
        x0 = make_initial_guess(env, row_idx, initial_socs, mode)
        res = minimize(
            objective_fn,
            x0,
            args=(env, row_idx, initial_socs),
            method="L-BFGS-B",
            bounds=bounds,
            # The reference rounds team rewards to float32.
            options={"maxiter": maxiter, "ftol": 1e-6, "maxfun": 50000, "eps": 1e-4},
        )
        metrics = simulate_episode(res.x, env, row_idx, initial_socs)
        if best is None or metrics["episode_return"] > best["episode_return"]:
            best = metrics
            best_res = res

    best["solver_success"] = bool(best_res.success)
    best["solver_message"] = str(best_res.message)
    best["solver_fun"] = float(best_res.fun)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="microgrid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-days", type=int, default=20)
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--initial-soc", type=float, default=0.35)
    parser.add_argument("--maxiter", type=int, default=120)
    args = parser.parse_args()

    config_path = ROOT / "src" / "config" / "envs" / f"{args.env_config}.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_args = cfg["env_args"]
    env_args["seed"] = args.seed
    env_args["result_dir"] = str(ROOT / "results")
    env = MGEnv(env_args, algo_name="perfect_prediction_oracle")

    if args.split == "test":
        row_indices = np.sort(env.test_indices)
    elif args.split == "train":
        row_indices = np.sort(env.train_indices)
    else:
        row_indices = np.arange(env.data_shape[0])
    row_indices = row_indices[: args.num_days]

    initial_socs = np.full(env.n_agents, args.initial_soc, dtype=float)
    results = []
    for idx, row_idx in enumerate(row_indices, start=1):
        print(f"[{idx}/{len(row_indices)}] optimizing row {row_idx} ...", flush=True)
        metrics = optimize_day(env, int(row_idx), initial_socs, args.maxiter)
        metrics["row_idx"] = int(row_idx)
        results.append(metrics)
        print(
            f"  return={metrics['episode_return']:.4f} "
            f"grid_cost={metrics['grid_purchase_cost']:.2f} "
            f"saving_ratio={metrics['grid_purchase_saving_ratio']:.4f}",
            flush=True,
        )

    summary = {
        "model_version": env.battery_model_version,
        "solver_kind": "local_numerical_search_not_a_global_bound",
        "seed": args.seed,
        "split": args.split,
        "num_days": len(results),
        "initial_soc": args.initial_soc,
        "episode_return_mean": float(np.mean([r["episode_return"] for r in results])),
        "grid_purchase_cost_mean": float(
            np.mean([r["grid_purchase_cost"] for r in results])
        ),
        "baseline_grid_purchase_cost_mean": float(
            np.mean([r["baseline_grid_purchase_cost"] for r in results])
        ),
        "grid_purchase_saving_mean": float(
            np.mean([r["grid_purchase_saving"] for r in results])
        ),
        "grid_purchase_saving_ratio_mean": float(
            np.mean([r["grid_purchase_saving_ratio"] for r in results])
        ),
        "generate_cost_mean": float(np.mean([r["generate_cost"] for r in results])),
        "bess_cost_mean": float(np.mean([r["bess_cost"] for r in results])),
        "env_cost_mean": float(np.mean([r["env_cost"] for r in results])),
        "curtailment_penalty_mean": float(
            np.mean([r["curtailment_penalty"] for r in results])
        ),
        "soc_reserve_penalty_mean": float(
            np.mean([r["soc_reserve_penalty"] for r in results])
        ),
        "mean_soc_mean": float(np.mean([r["mean_soc"] for r in results])),
    }

    out_dir = ROOT / "results" / "oracle_perfect_prediction" / (
        env.battery_model_version + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid4().hex[:6]
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    stem = f"{args.env_config}_{args.split}_{len(results)}d_seed{args.seed}"
    with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / f"{stem}_episodes.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
