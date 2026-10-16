"""Gymnasium interface around the unedited maenv_sep.py MyEnv implementation."""

import builtins
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pandas import read_csv


def snapshot_reference(source, config, output):
    source, config, output = Path(source), Path(config), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [source, config] + [source.parent / f"{key}.csv" for key in ("price", "load", "pv")]
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Reference bundle filenames must be unique")
    hashes = {}
    for path in paths:
        target = output / path.name
        shutil.copyfile(path, target)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference snapshot mismatch: {path}")
        hashes[path.name] = expected
    (output / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return hashes


def load_reference(bundle):
    bundle = Path(bundle)
    spec = importlib.util.spec_from_file_location("sb3_reference_maenv_sep", bundle / "maenv_sep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def reference_open(path, *args, **kwargs):
        # Only redirect the reference's hardcoded config path, scoped to this
        # module. Do not change cwd, process builtins, or the reference file.
        if str(path).replace("\\", "/") == "./mgenv/env/env_config.json":
            path = bundle / "env_config.json"
        return builtins.open(path, *args, **kwargs)

    module.open = reference_open
    return module


class ReferenceMicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, bundle, record=False):
        self.bundle = Path(bundle)
        self.config = json.loads((self.bundle / "env_config.json").read_text(encoding="utf-8"))
        self.agent_config = self.config["data"]
        self.n_agents = len(self.agent_config)
        self.agents = [a["agent_name"] for a in self.agent_config]
        module = load_reference(self.bundle)
        self.cores = [module.MyEnv(i, multi_target=False, test_mode=record)
                      for i in range(self.n_agents)]
        self.record = record
        if record:
            # Use the native action/cost recorder but keep all file output under
            # this run. The reference otherwise writes ./test_log on day end.
            for core in self.cores:
                core.data_save = lambda: None
        self.raw = {
            name: np.array(read_csv(self.bundle / f"{name}.csv", header=None))
            for name in ("price", "load", "pv")
        }
        self.action_lows = np.array([a["low_action"] for a in self.agent_config], dtype=np.float32)
        self.action_highs = np.array([a["high_action"] for a in self.agent_config], dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(self.n_agents * 2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.tile(np.array([-np.inf, -np.inf, -np.inf, 0.2], dtype=np.float32), self.n_agents),
            high=np.tile(np.array([np.inf, np.inf, np.inf, 0.8], dtype=np.float32), self.n_agents),
            dtype=np.float32,
        )
        self.steps_in_day = 0
        self.records = []
        self.started = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if options:
            if not self.record:
                raise ValueError("Fixed-day options are evaluation-only")
            row = options["row_idx"]
            initial = np.broadcast_to(np.asarray(options["initial_soc"], dtype=float), (self.n_agents,))
            if (
                isinstance(row, bool) or int(row) != row
                or not 0 <= row < self.cores[0].rows
                or not np.all(np.isfinite(initial) & (initial >= 0.2) & (initial <= 0.8))
            ):
                raise ValueError("Invalid fixed day or initial SOC")
            for i, core in enumerate(self.cores):
                core._MyEnv__node = [int(row), 0]
                core._MyEnv__soc = float(initial[i])
        elif not self.started or seed is not None:
            for core in self.cores:
                core._MyEnv__node = [0, 0]
                core._MyEnv__soc = 0.2
        self.started = True
        self.steps_in_day = 0
        self.records = []
        if self.record:
            for core in self.cores:
                for name in ("action_data", "observation_data", "punishment_data",
                             "cost_data", "reward_data"):
                    getattr(core, name).clear()
        obs = np.concatenate([core.reset()[0] for core in self.cores]).astype(np.float32)
        return obs, {}

    def scale_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("PPO action must be a finite flat joint action")
        if np.any(action < -1 - 1e-6) or np.any(action > 1 + 1e-6):
            raise ValueError("PPO action is outside its Box space")
        return self.action_lows + 0.5 * (action.reshape(self.n_agents, 2) + 1) * (
            self.action_highs - self.action_lows
        )

    def step(self, action):
        if self.steps_in_day >= 24:
            raise RuntimeError("Reset after the 24-step truncation")
        physical = self.scale_action(action)
        index = tuple(self.cores[0]._MyEnv__node)
        before = [float(core._MyEnv__soc) for core in self.cores]
        returned = [core.step(physical[i]) for i, core in enumerate(self.cores)]
        obs = np.concatenate([r[0] for r in returned]).astype(np.float32)
        rewards = np.array([r[1] for r in returned], dtype=np.float32)
        reward = float(rewards.mean())
        if not np.isfinite(obs).all() or not np.isfinite(reward):
            raise FloatingPointError("Non-finite reference environment result")
        self.steps_in_day += 1
        if self.record:
            for i, core in enumerate(self.cores):
                pd, pb, pg = core.action_data[-1]
                gen_cost, bess_cost, buy_cost = core.cost_data[-1]
                penalty = core.punishment_data[-1][0]
                state = core.observation_data[-1]
                env_reward = -pd * self.agent_config[i]["env_param"]
                self.records.append({
                    "row_idx": index[0], "col_idx": index[1], "agent": self.agents[i],
                    "soc_init": before[i], "soc": float(state[3]),
                    "action_pd": float(physical[i, 0]), "action_pb": float(physical[i, 1]),
                    "pd": float(pd), "pb": float(pb), "pg": float(pg),
                    "gen_cost": float(gen_cost), "bess_cost": float(bess_cost),
                    "buy_cost": float(buy_cost), "env_reward": float(env_reward),
                    "punishment": float(penalty), "reward": float(rewards[i]),
                    "team_reward": reward,
                    "price": float(state[0]), "load": float(state[1]), "pv": float(state[2]),
                    "price_raw": float(self.raw["price"][index]),
                    "load_raw": float(self.raw["load"][index]),
                    "pv_raw": float(self.raw["pv"][index]),
                    "economic_cost": float(buy_cost + 0.5 * gen_cost + bess_cost),
                })
        info = {
            "row_idx": index[0],
            "mean_soc": float(np.mean([core._MyEnv__soc for core in self.cores])),
        }
        return obs, reward, False, self.steps_in_day == 24, info


def evaluate_days(model, env, rows, initial_soc):
    returns, records = [], []
    for row in rows:
        obs, _ = env.reset(options={"row_idx": row, "initial_soc": initial_soc})
        value = 0.0
        for _ in range(24):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            value += reward
        if terminated or not truncated or not np.isfinite(value):
            raise RuntimeError("Invalid fixed-day evaluation")
        returns.append(value)
        records.extend(env.records)
    return {
        "rows": list(rows), "returns": returns, "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }, records


def write_records(path, records):
    with Path(path).open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
