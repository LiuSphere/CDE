"""Joint-horizon economic MILP with continuous-inventory environment replay."""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import yaml

from direct_main import load_yaml, recursive_dict_update
from tools.run_physical_microgrid_milp import make_env, solve_network
from utils.comparison_evaluation import summarize_trajectories


def peak_memory_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def solve_continuous(env, config, rows, source="validation", initial=0.35,
                     segments=32, time_limit=60, mip_gap=1e-7, diagnostics_path=None):
    rows = list(rows)
    allowed = getattr(env, f"{source}_indices")
    if (not rows or any(isinstance(r, bool) or not isinstance(r, (int, np.integer))
                        or r not in allowed for r in rows)
            or any(b != a + 1 for a, b in zip(rows, rows[1:]))):
        raise ValueError("Joint horizon requires adjacent ordered rows within the split")
    started = time.perf_counter()
    solution = solve_network(
        env.agent_data["load"][rows].reshape(-1, env.n_agents),
        env.agent_data["pv"][rows].reshape(-1, env.n_agents),
        env.data_dict["price"][rows].reshape(-1), config["data"],
        env.params["exchange_max"], env.exchange_efficiency, env.exchange_fee,
        initial, env.soc_min, env.soc_max, terminal_soc=None, segments=segments,
        objective="economic_cost", time_limit=time_limit, mip_gap=mip_gap,
        diagnostics_path=diagnostics_path,
    )
    solve_seconds = time.perf_counter() - started
    records, end = [], np.full(env.n_agents, initial)
    max_error = 0.0
    for day, row in enumerate(rows):
        env.reset_test(source=source, row_idx=row, initial_soc=end.copy())
        for hour in range(env.episode_limit):
            t = day * env.episode_limit + hour
            actions = np.zeros((env.n_agents, 2))
            for a, p in enumerate(config["data"]):
                actions[a, 0] = (solution["schedules"][a]["pd"][t] / p["dg_max"]
                                 if p["dg_max"] else 0)
            power = np.array([s["discharge"][t] - s["charge"][t]
                              for s in solution["schedules"]])
            actions[:, 1] = env.battery_action_from_power(power, env.params["socs"])
            if (not np.isfinite(actions).all()
                    or (actions < env.action_lows - 1e-7).any()
                    or (actions > env.action_highs + 1e-7).any()):
                raise ValueError("MILP reconstructed action outside bounds")
            env.step(np.clip(actions, env.action_lows, env.action_highs))
        day_records = [dict(r) for r in env.ep_buffer]
        records.extend(day_records)
        end = env.params["socs"].copy()
        for hour in range(env.episode_limit):
            t = day * env.episode_limit + hour
            current = [r for r in day_records if r["col_idx"] == hour]
            for a, agent in enumerate(env.possible_agents):
                r = next(r for r in current if r["agent"] == agent)
                s = solution["schedules"][a]
                max_error = max(max_error, abs(r["soc"] - s["soc"][t + 1]),
                                abs(r["pb"] - s["discharge"][t] + s["charge"][t]),
                                abs(r["pd"] - s["pd"][t]))
            # Uniform tariffs and transfer fees allow different optimal routing.
            for key in ("pg", "spill", "peer_import", "peer_export"):
                max_error = max(max_error, abs(
                    sum(r[key] for r in current)
                    - sum(s[key][t] for s in solution["schedules"])))
    summary = summarize_trajectories(records, env, rows, initial, "continuous")
    exact = summary["totals"]["economic_cost"]
    interpolation = solution["objective"] - exact
    certificate_passed = bool(
        -1e-5 <= interpolation <= solution["interpolation_error_bound"] + 1e-5)
    accepted = bool(solution["status"] == 0 and max_error <= 1e-5 and certificate_passed)
    summary.update(
        objective_name="economic_cost", terminal_constraint="free within equipment bounds",
        solve_seconds=solve_seconds, max_schedule_replay_error=max_error,
        interpolation_difference=interpolation, certificate_passed=certificate_passed,
        certified_economic_suboptimality_upper=max(
            0.0, exact - solution["exact_cost_lower_bound"]),
        acceptance_passed=accepted,
        boundary="Joint perfect-foresight horizon; physical SOC carried between rows",
    )
    return summary, solution, records


def solve_inventory(env, config, rows, source="validation", initial=0.35,
                    segments=32, time_limit=60, mip_gap=1e-7, mode="continuous",
                    diagnostics_dir=None):
    if mode == "continuous":
        return solve_continuous(
            env, config, rows, source, initial, segments, time_limit, mip_gap,
            Path(diagnostics_dir) / "solver_diagnostics.json" if diagnostics_dir else None)
    if mode != "daily_reset":
        raise ValueError("Unknown MILP inventory mode")
    rows = list(rows)
    if not rows or rows != sorted(set(rows)):
        raise ValueError("Daily-reset rows must be ordered and unique")
    records, summaries, solutions = [], [], []
    for row in rows:
        summary, solution, day_records = solve_continuous(
            env, config, [row], source, initial, segments, time_limit, mip_gap,
            Path(diagnostics_dir) / f"solver_row_{row}.json" if diagnostics_dir else None)
        records.extend(day_records)
        summaries.append(summary)
        solutions.append(solution)
    summary = summarize_trajectories(records, env, rows, initial, "daily_reset")
    summary.update(
        objective_name="economic_cost", terminal_constraint="free each day within device bounds",
        solve_seconds=sum(s["solve_seconds"] for s in summaries),
        max_schedule_replay_error=max(s["max_schedule_replay_error"] for s in summaries),
        interpolation_difference=sum(s["interpolation_difference"] for s in summaries),
        certificate_passed=all(s["certificate_passed"] for s in summaries),
        certified_economic_suboptimality_upper=sum(
            s["certified_economic_suboptimality_upper"] for s in summaries),
        acceptance_passed=all(s["acceptance_passed"] for s in summaries),
        boundary="Independent daily perfect foresight; initial SOC reset, free daily terminal SOC")
    solution = {
        "status": max(s["status"] for s in solutions),
        "mip_gap": max(s["mip_gap"] for s in solutions),
        "objective_name": "economic_cost", "joint_milps": len(solutions),
        "objective": sum(s["objective"] for s in solutions),
        "dual_bound": sum(s["dual_bound"] for s in solutions),
        "interpolation_error_bound": sum(s["interpolation_error_bound"] for s in solutions),
        "exact_cost_lower_bound": sum(s["exact_cost_lower_bound"] for s in solutions),
        "daily_solutions": solutions,
    }
    return summary, solution, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=list(range(731, 738)))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--retrospective-holdout", action="store_true")
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--mip-gap", type=float, default=1e-7)
    parser.add_argument("--mode", choices=["continuous", "daily_reset"], default="continuous")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.retrospective_holdout:
        raise ValueError("Holdout requires explicit retrospective evaluation authorization")
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / "results").resolve()) or out == ROOT / "results":
        raise ValueError("Output must be a new directory below results/")
    out.mkdir(parents=True, exist_ok=False)
    config = recursive_dict_update(
        load_yaml(ROOT / "src/config/default.yaml"),
        load_yaml(ROOT / "src/config/envs/microgrid.yaml"))
    (out / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = make_env(config["env_args"], out)
    with (out / "attempt.json").open("x", encoding="utf-8") as file:
        json.dump({
            "rows": args.rows, "split": args.split, "mode": args.mode,
            "time_limit_seconds": args.time_limit, "segments": args.segments,
            "mip_gap": args.mip_gap, "initial_soc": config["evaluation_initial_soc"],
            "terminal_soc": "free within equipment bounds", "objective": "economic_cost",
            "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "solver_entry_sha256": hashlib.sha256(
                (ROOT / "tools/run_physical_microgrid_milp.py").read_bytes()).hexdigest(),
        }, file, indent=2, allow_nan=False)
    summary, solution, records = solve_inventory(
        env, config["env_args"], args.rows, args.split, config["evaluation_initial_soc"],
        args.segments, args.time_limit, args.mip_gap, args.mode, out)
    summary.update(
        split=args.split, hardware=platform.platform(), scipy_version=scipy.__version__,
        process_peak_working_set_bytes=peak_memory_bytes(),
        memory_scope="Whole process working set, not solver-only allocation",
        thread_environment={key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        source_sha256={
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(
                list((ROOT / "src").rglob("*.py"))
                + list((ROOT / "src/config").rglob("*.yaml"))
                + list((ROOT / "src/envs/data").glob("*.csv"))
                + [Path(__file__).resolve(), ROOT / "tools/run_physical_microgrid_milp.py",
                   ROOT / "tools/run_current_model_milp.py"]))},
        information="Perfect actual load/PV/tariffs across the complete joint horizon",
        bound_scope="Economic cost only; full_cost reported but not optimized or bounded",
    )
    with (out / "trajectories.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for name, payload in (("summary.json", summary), ("solver.json", solution)):
        with (out / name).open("x", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
    print(json.dumps({"output": str(out), "accepted": summary["acceptance_passed"],
                      "solve_seconds": summary["solve_seconds"],
                      "status": solution["status"], "gap": solution["mip_gap"]}), flush=True)
    if not summary["acceptance_passed"]:
        raise SystemExit("Joint continuous MILP acceptance failed; evidence preserved")


if __name__ == "__main__":
    main()
