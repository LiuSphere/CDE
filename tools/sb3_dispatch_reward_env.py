"""Experimental SB3 interface and inventory rewards for the current MGEnv."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

REWARD_MODES = ("current", "terminal", "potential")
TUNING_REWARD_MODES = ("deadline",)
VALIDATION_ROWS = [740, 766, 808, 862, 880, 904]


def load_environment_class(source_root):
    path = Path(source_root) / "src/envs/maenv.py"
    spec = importlib.util.spec_from_file_location("ppo_reward_microgrid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MGEnv


class DispatchRewardEnv(gym.Env):
    """Joint PPO policy; independent finite 24-hour days, not an IPPO baseline."""

    metadata = {"render_modes": []}

    def __init__(self, env_class, config, reward_mode, output, *, gamma=0.99,
                 source="train", inventory_weight=1.0, inventory_quadratic=0.0,
                 deadline_power=2.0, deadline_start_hour=0,
                 future_price_potential=False, deficit_only_potential=False):
        if reward_mode not in REWARD_MODES + TUNING_REWARD_MODES:
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        if source not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {source}")
        if not np.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("Reward discount must match a valid PPO discount")
        if (not np.isfinite([inventory_weight, inventory_quadratic, deadline_power,
                             deadline_start_hour]).all()
                or inventory_weight <= 0 or inventory_quadratic < 0 or deadline_power <= 0
                or not 0 <= deadline_start_hour < 24
                or not isinstance(future_price_potential, bool)
                or not isinstance(deficit_only_potential, bool)):
            raise ValueError("Invalid inventory reward coefficients")
        self.inventory_weight = float(inventory_weight)
        self.inventory_quadratic = float(inventory_quadratic)
        self.deadline_power = float(deadline_power)
        self.deadline_start_hour = float(deadline_start_hour)
        self.future_price_potential = future_price_potential
        self.deficit_only_potential = deficit_only_potential
        settings = deepcopy(config)
        settings.update(result_dir=str(output), log_trajectories=False)
        self.core = env_class(settings, "sb3_reward")
        self.reward_mode, self.gamma, self.source = reward_mode, gamma, source
        self.initial_soc = float(config["initial_soc"])
        self.capacities = self.core.params["battery_caps"].copy()
        train_prices = self.core.data_dict["price"][self.core.train_indices]
        self.inventory_price = float(train_prices.mean()) * self.core.params["raw_dis"]
        self.published_tariff = self.core.data_dict["price"][self.core.train_indices[0]].copy()
        self.terminal_price = (
            1.25 * (float(train_prices.max()) + self.core.params["cycle_cost"])
            / self.core.params["raw_ch"]
        )
        self.action_space = spaces.Box(
            -1, 1, shape=(self.core.n_agents * 2,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            np.concatenate([self.core.spaces[a]["observation"].low
                            for a in self.core.possible_agents]),
            np.concatenate([self.core.spaces[a]["observation"].high
                            for a in self.core.possible_agents]),
            dtype=np.float32,
        )
        self.steps_in_day = 24
        self.reward_strength = 1.0

    def potential(self, soc):
        if self.reward_mode == "deadline":
            urgency = np.clip(
                (self.steps_in_day - self.deadline_start_hour)
                / (self.core.episode_limit - self.deadline_start_hour), 0, 1,
            )**self.deadline_power
            inventory_soc = np.minimum(soc, self.initial_soc) if self.deficit_only_potential else soc
            value = -urgency * self.objective_inventory_cost(inventory_soc)
            if self.future_price_potential and self.steps_in_day < self.core.episode_limit:
                future_price = float(self.published_tariff[self.steps_in_day:].max())
                value += float(np.dot(
                    (np.asarray(soc) - self.initial_soc) * self.capacities,
                    self.core.params["raw_dis"] * future_price,
                ))
            return float(value)
        energy_deviation = (np.asarray(soc) - self.initial_soc) * self.capacities
        return float(np.dot(energy_deviation, self.inventory_price))

    def objective_inventory_cost(self, soc):
        energy_error = (np.asarray(soc) - self.initial_soc) * self.capacities
        usable_energy = (self.core.soc_max - self.core.soc_min) * self.capacities
        return (
            self.inventory_weight * self.terminal_inventory_cost(soc)
            + self.inventory_quadratic * float(np.dot(
                energy_error**2 / usable_energy, self.terminal_price,
            ))
        )

    def terminal_inventory_cost(self, soc):
        error_kwh = np.abs(np.asarray(soc) - self.initial_soc) * self.capacities
        return float(np.dot(error_kwh, self.terminal_price))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"row_idx"}:
            raise ValueError("Only row_idx may be supplied; initial SOC is fixed")
        rows = getattr(self.core, f"{self.source}_indices")
        row = options.get("row_idx")
        if row is None:
            row = int(self.np_random.choice(rows))
        if isinstance(row, bool) or int(row) != row or row not in rows:
            raise ValueError(f"Row {row} is not in the {self.source} split")
        if self.source == "train":
            obs = self.core.reset(options={
                "row_idx": int(row), "initial_soc": [self.initial_soc] * self.core.n_agents,
            })
        else:
            obs = self.core.reset_test(
                source=self.source, row_idx=int(row), initial_soc=self.initial_soc,
            )
        self.row = int(row)
        self.steps_in_day = 0
        self.episode_return = 0.0
        return np.asarray(obs, dtype=np.float32).flatten(), {"row_idx": self.row}

    def step(self, action):
        if self.steps_in_day >= 24:
            raise RuntimeError("Reset after the finite dispatch day")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Expected a finite flat joint action in [-1, 1]")
        native = action.reshape(self.core.n_agents, 2).copy()
        native[:, 0] = (native[:, 0] + 1) / 2
        old_potential = self.potential(self.core.params["socs"])
        raw_reward, done, info = self.core.step(native)
        self.steps_in_day += 1
        if done != (self.steps_in_day == 24):
            raise RuntimeError("Unexpected physical environment day boundary")
        terminal_cost = self.terminal_inventory_cost(self.core.params["socs"]) if done else 0.0
        shaping = 0.0
        if self.reward_mode in ("potential", "deadline"):
            # A true terminal has zero potential, so no terminal inventory bonus
            # can be harvested. Discounted shaping sums to -Phi(initial) = 0.
            next_potential = 0.0 if done else self.potential(self.core.params["socs"])
            shaping = self.gamma * next_potential - old_potential
        penalty = (self.objective_inventory_cost(self.core.params["socs"])
                   if done and self.reward_mode != "current" else 0.0)
        scale = self.core.reward_scale / self.core.n_agents
        reward = float(raw_reward + self.reward_strength * scale * (shaping - penalty))
        self.episode_return += reward
        info.update(
            row_idx=self.row, raw_reward=raw_reward, shaping_cu=shaping,
            terminal_inventory_cost=terminal_cost, training_inventory_cost=penalty,
        )
        if done:
            info["day"] = self.metrics()
        return self.core.get_state(), reward, done, False, info

    def metrics(self):
        if self.steps_in_day != 24:
            raise RuntimeError("Daily metrics require a completed episode")
        records, p = self.core.ep_buffer, self.core.params
        by_agent = []
        for i, agent in enumerate(self.core.possible_agents):
            data = [r for r in records if r["agent"] == agent]
            states = [self.initial_soc] + [r["soc"] for r in data]
            charge = sum(max(-r["pb"], 0.0) for r in data)
            discharge = sum(max(r["pb"], 0.0) for r in data)
            charge_hours = int(sum(r["pb"] < -0.01 * p["battery_charge_max"][i] for r in data))
            discharge_hours = int(sum(r["pb"] > 0.01 * p["battery_discharge_max"][i] for r in data))
            by_agent.append({
                "agent": agent, "charge_kwh": charge, "discharge_kwh": discharge,
                "charge_hours": charge_hours, "discharge_hours": discharge_hours,
                "terminal_soc": float(states[-1]),
                "terminal_soc_error": abs(states[-1] - self.initial_soc),
                "soc_span": max(states) - min(states),
                "bidirectional": charge_hours >= 2 and discharge_hours >= 2,
                "generator_hours": int(sum(r["pd"] > 0.05 * p["dg_max"][i] for r in data)),
            })
        full_cost = float(self.core._stats["full_cost"])
        boundary = self.terminal_inventory_cost(self.core.params["socs"])
        return {
            "row_idx": self.row, "full_cost": full_cost,
            "terminal_inventory_cost": boundary, "inventory_adjusted_cost": full_cost + boundary,
            "training_inventory_cost": self.objective_inventory_cost(self.core.params["socs"]),
            "return": self.episode_return,
            "charge_kwh": sum(a["charge_kwh"] for a in by_agent),
            "discharge_kwh": sum(a["discharge_kwh"] for a in by_agent),
            "terminal_error_kwh": float(np.dot(
                np.abs(self.core.params["socs"] - self.initial_soc), self.capacities,
            )),
            "mean_terminal_soc_error": float(np.mean([a["terminal_soc_error"] for a in by_agent])),
            "max_terminal_soc_error": max(a["terminal_soc_error"] for a in by_agent),
            "bidirectional_agent_fraction": float(np.mean([a["bidirectional"] for a in by_agent])),
            "mean_soc_span": float(np.mean([a["soc_span"] for a in by_agent])),
            "mean_generator_hours": float(np.mean([a["generator_hours"] for a in by_agent])),
            "generator_kwh": self.core._stats["generator_energy"],
            "grid_kwh": self.core._stats["grid_import_energy"],
            "peer_kwh": self.core._stats["peer_sent_energy"],
            "agents": by_agent,
        }

    def close(self):
        self.core.close()


def evaluate(model, env, rows, *, collect_records=False):
    days, records = [], []
    for row in rows:
        obs, _ = env.reset(options={"row_idx": row})
        for _ in range(24):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
        if not terminated or truncated:
            raise RuntimeError("Expected a true finite-horizon terminal")
        days.append(info["day"])
        if collect_records:
            records.extend(dict(r) for r in env.core.ep_buffer)
    if not days:
        raise ValueError("Cannot evaluate an empty day set")
    keys = [k for k, value in days[0].items()
            if k != "row_idx" and isinstance(value, (float, int))]
    return {
        "rows": list(rows), "days": days,
        **{f"mean_{k}": float(np.mean([d[k] for d in days])) for k in keys},
    }, records
