"""Dimensionally consistent, CPU-resident microgrid dispatch environment."""

import csv
import functools
import pathlib

import numpy as np
from gymnasium import spaces
from pandas import read_csv
from pettingzoo import ParallelEnv


class MGEnv(ParallelEnv):
    metadata = {"render_modes": ["human", None], "name": "MGEnv_facmac_dispatch_v3"}

    def __init__(self, config, algo_name, render_mode="human"):
        if config.get("model_version") != "facmac_dispatch_v3":
            raise ValueError("Use facmac_dispatch_v3; legacy models require their saved code")
        self.battery_model_version = config["model_version"]
        self.render_mode = render_mode
        self.time_step_hours = float(config["time_step_hours"])
        self.soc_min, self.soc_max = float(config["soc_min"]), float(config["soc_max"])
        self.initial_soc = float(config["initial_soc"])
        self.reward_scale = float(config["reward_scale"])
        self.exchange_efficiency = float(config["exchange_efficiency"])
        self.exchange_fee = float(config["exchange_fee"])
        if not (0 < self.exchange_efficiency <= 1 and np.isfinite(self.exchange_fee)
                and self.exchange_fee >= 0):
            raise ValueError("Invalid exchange efficiency or fee")
        self.observation_price_scale = float(config["observation_price_scale"])
        self.observation_power_scale = float(config["observation_power_scale"])
        self.forecast_horizon = int(config["forecast_horizon"])
        self.forecast_scale_bounds = np.asarray(config["forecast_scale_bounds"], dtype=float)
        if not (
            self.forecast_horizon == config["forecast_horizon"]
            and 1 <= self.forecast_horizon <= 24
            and self.forecast_scale_bounds.shape == (2,)
            and np.isfinite(self.forecast_scale_bounds).all()
            and 0 < self.forecast_scale_bounds[0] <= 1 <= self.forecast_scale_bounds[1]
        ):
            raise ValueError("Invalid causal forecast settings")
        if not (
            np.isfinite([self.time_step_hours, self.soc_min, self.soc_max,
                         self.initial_soc, self.reward_scale,
                         self.observation_price_scale, self.observation_power_scale]).all()
            and self.time_step_hours == 1
            and 0 <= self.soc_min < self.soc_max <= 1
            and self.soc_min <= self.initial_soc <= self.soc_max
            and min(self.reward_scale, self.observation_price_scale,
                    self.observation_power_scale) > 0
        ):
            raise ValueError("Invalid hourly physical-model settings")
        self.n_agents = len(config["data"])
        self.possible_agents = [f"agent{i + 1}" for i in range(self.n_agents)]
        agents_config = {a["agent_name"]: a for a in config["data"]}
        self.params = self.params_init(agents_config)
        self.action_lows = np.tile([0.0, -1.0], (self.n_agents, 1)).astype(np.float32)
        self.action_highs = np.ones((self.n_agents, 2), dtype=np.float32)
        self.n_actions = 2
        self.obs_size = 20
        obs_low = np.zeros(self.obs_size, dtype=np.float32)
        obs_low[3], obs_low[4:6], obs_low[19] = self.soc_min, -1, -np.inf
        obs_high = np.full(self.obs_size, np.inf, dtype=np.float32)
        obs_high[3], obs_high[4:6] = self.soc_max, 1
        self.spaces = {
            agent: {
                "observation": spaces.Box(
                    obs_low.copy(), obs_high.copy(),
                ),
                "action": spaces.Box(self.action_lows[i], self.action_highs[i]),
            }
            for i, agent in enumerate(self.possible_agents)
        }
        data_path = pathlib.Path(__file__).parent / "data"
        self.raw_data = {
            k: np.array(read_csv(data_path / f"{k}.csv", header=None), dtype=float)
            for k in ("price", "load", "pv")
        }
        scales = {k: float(config[f"{k}_scale"]) for k in self.raw_data}
        if not all(np.isfinite(v) and v > 0 for v in scales.values()):
            raise ValueError("Data unit/installation scales must be finite and positive")
        self.data_dict = {k: a * scales[k] for k, a in self.raw_data.items()}
        if not all(np.isfinite(a).all() and np.all(a >= 0) for a in self.data_dict.values()):
            raise ValueError("Physical inputs must be finite and nonnegative")
        if not np.all(self.data_dict["price"] > 0):
            raise ValueError("This no-export scenario requires positive purchase prices")
        if self.exchange_fee >= self.data_dict["price"].min() * self.exchange_efficiency:
            raise ValueError("Exchange fee must be below avoided grid purchase cost")
        self.data_shape = self.data_dict["price"].shape
        if any(a.shape != self.data_shape for a in self.data_dict.values()):
            raise ValueError("Price, load and PV tables must have identical shapes")
        self.data_dict["load_pv"] = self.data_dict["load"] - self.data_dict["pv"]
        self.agent_data = {}
        for key in ("load", "pv"):
            hour = np.arange(self.data_shape[1])[:, None]
            weights = self.params[f"{key}_share"] * (
                1 + self.params[f"{key}_profile_amplitude"] * np.cos(
                    2 * np.pi * (hour - self.params[f"{key}_profile_peak_hour"]) / 24,
                )
            )
            weights /= weights.sum(axis=1, keepdims=True)
            self.agent_data[key] = self.data_dict[key][:, :, None] * weights[None, :, :]
        self.episode_limit = self.data_shape[1]
        if self.episode_limit != 24 or self.data_shape[0] < 733:
            raise ValueError("The experiment requires 24-hour days and nonempty held-out splits")
        self.train_indices = np.arange(731)
        self.validation_indices, self.test_indices = np.array_split(
            np.arange(731, self.data_shape[0]), 2,
        )
        if not np.all(self.data_dict["price"] == self.data_dict["price"][0]):
            raise ValueError("Published tariff context requires this scenario's periodic daily tariff")
        self.forecast_templates = {
            key: self.agent_data[key][self.train_indices].mean(axis=0)
            for key in ("load", "pv")
        }
        self.seed = config.get("seed")
        self.rng = np.random.default_rng(self.seed)
        self.idx = [0, 0]
        self._training_state = None
        self.is_testing = False
        self.test_source = "train"
        self.trajectory_context = {}
        self.log_trajectories = bool(config.get("log_trajectories", False))
        self.log_trajectories_mode = config.get("log_trajectories_mode", "test")
        self.render_dir = (
            pathlib.Path(config["result_dir"]) / "microgrid_trajs"
            / str(algo_name) / str(config.get("run_id", "no_run_id"))
        )
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_size = 10_000
        self.ep_counter = 0
        self._start_episode()
        self.state = self._observations()

    def params_init(self, agents):
        fields = {
            "dg_max": "dg_max", "battery_caps": "battery_cap",
            "battery_charge_max": "battery_charge_max",
            "battery_discharge_max": "battery_discharge_max",
            "battery_punishment_params": "battery_punishment_param",
            "raw_ch": "raw_ch", "raw_dis": "raw_dis",
            "cycle_cost": "cycle_cost", "costa": "costa", "costb": "costb",
            "costc": "costc", "env_param": "env_param", "exchange_max": "exchange_max",
        }
        for key in ("load", "pv"):
            for suffix in ("share", "profile_amplitude", "profile_peak_hour"):
                fields[f"{key}_{suffix}"] = f"{key}_{suffix}"
        params = {
            k: np.array([agents[a][field] for a in self.possible_agents], dtype=float)
            for k, field in fields.items()
        }
        if not all(np.isfinite(a).all() and (a >= 0).all() for a in params.values()):
            raise ValueError("Equipment and cost parameters must be finite and nonnegative")
        if not (
            (params["battery_caps"] > 0).all()
            and (params["raw_ch"] > 0).all() and (params["raw_ch"] <= 1).all()
            and (params["raw_dis"] > 0).all() and (params["raw_dis"] <= 1).all()
            and all(
                (params[f"{key}_share"] > 0).all()
                and np.isclose(params[f"{key}_share"].sum(), 1, rtol=0, atol=1e-10)
                and (params[f"{key}_profile_amplitude"] < 1).all()
                and (params[f"{key}_profile_peak_hour"] < 24).all()
                for key in ("load", "pv")
            )
        ):
            raise ValueError("Invalid capacity, efficiency or load/PV allocation")
        params["socs"] = np.full(self.n_agents, self.initial_soc)
        return params

    def _observations(self):
        index = tuple(self.idx)
        load, pv = [self.agent_data[key][index] for key in ("load", "pv")]
        stored = ((self.params["socs"] - self.soc_min) * self.params["battery_caps"]
                  * self.params["raw_dis"])
        phase = 2 * np.pi * self.idx[1] / 24
        lower, upper = self.battery_power_limits(self.params["socs"])
        hours = (self.idx[1] + np.arange(1, self.forecast_horizon + 1)) % 24
        tariff = self.data_dict["price"][0, hours]
        forecasts = {}
        for key, current in (("load", load), ("pv", pv)):
            template = self.forecast_templates[key]
            # Nighttime PV uses train-only climatology, not tomorrow's weather.
            ratio = np.ones(self.n_agents)
            np.divide(current, template[self.idx[1]], out=ratio,
                      where=template[self.idx[1]] > 1e-8)
            ratio = np.clip(ratio, *self.forecast_scale_bounds)
            forecasts[key] = template[hours].mean(axis=0) * ratio
        net_forecast = forecasts["load"] - forecasts["pv"]
        return np.column_stack((
            np.full(self.n_agents, self.data_dict["price"][index] / self.observation_price_scale),
            load / self.observation_power_scale,
            pv / self.observation_power_scale,
            self.params["socs"],
            np.full(self.n_agents, np.sin(phase)), np.full(self.n_agents, np.cos(phase)),
            (load.sum() - load) / self.observation_power_scale,
            (pv.sum() - pv) / self.observation_power_scale,
            (stored.sum() - stored) / (4 * self.observation_power_scale),
            self.params["dg_max"] / self.observation_power_scale,
            self.params["battery_caps"] / (4 * self.observation_power_scale),
            self.params["exchange_max"] / self.observation_power_scale,
            -lower / self.observation_power_scale,
            upper / self.observation_power_scale,
            np.full(self.n_agents, tariff[0] / self.observation_price_scale),
            np.full(self.n_agents, tariff.mean() / self.observation_price_scale),
            np.full(self.n_agents, tariff.max() / self.observation_price_scale),
            forecasts["load"] / self.observation_power_scale,
            forecasts["pv"] / self.observation_power_scale,
            (net_forecast.sum() - net_forecast) / self.observation_power_scale,
        )).astype(np.float32)

    def get_obs(self):
        return [obs.copy() for obs in self.state]

    def get_state(self):
        return self.state.flatten()

    def get_env_info(self):
        return {
            "n_agents": self.n_agents, "n_actions": 2, "obs_shape": self.obs_size,
            "state_shape": self.obs_size * self.n_agents, "episode_limit": self.episode_limit,
            "action_spaces": [self.spaces[a]["action"] for a in self.possible_agents],
            "actions_dtype": np.float32, "normalise_actions": False,
        }

    @functools.cache
    def observation_space(self, agent):
        return self.spaces[f"agent{agent}"]["observation"]

    @functools.cache
    def action_space(self, agent):
        return self.spaces[f"agent{agent}"]["action"]

    def _start_episode(self):
        self.ep_buffer = []
        self._ep_steps = 0
        self._stats = {}

    def reset(self, seed=None, options=None):
        if options is not None:
            if set(options) != {"row_idx", "initial_soc"}:
                raise ValueError("Training reset requires row_idx and initial_soc")
            row = options["row_idx"]
            socs = np.asarray(options["initial_soc"], dtype=float)
            if (isinstance(row, bool) or int(row) != row or row not in self.train_indices
                    or socs.shape != (self.n_agents,) or not np.isfinite(socs).all()
                    or not ((socs >= self.soc_min) & (socs <= self.soc_max)).all()):
                raise ValueError("Training scenario must use training rows and feasible SOCs")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._training_state is not None:
            self.idx, self.params["socs"] = self._training_state
            self._training_state = None
        if options is not None:
            self.idx, self.params["socs"] = [int(row), 0], socs.copy()
        self.is_testing, self.test_source = False, "train"
        self.trajectory_context = {}
        self.ep_counter += 1
        self._start_episode()
        self.state = self._observations()
        return self.get_obs()

    def training_scenario(self):
        """Capture a common EA start, including a pending evaluation restore."""
        index, socs = (self._training_state if self._training_state is not None
                       else (self.idx, self.params["socs"]))
        if index[1] != 0 or index[0] not in self.train_indices:
            raise ValueError("Capture an EA scenario at a training-day boundary")
        return {"row_idx": int(index[0]), "initial_soc": socs.copy().tolist()}

    def reset_all(self, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        self._training_state = None
        self.idx = [int(rng.choice(self.train_indices)), 0]
        self.params["socs"] = np.full(self.n_agents, self.initial_soc)
        return self.reset()

    def reset_test(self, seed=None, source="test", row_idx=None, initial_soc=None,
                   trajectory_context=None):
        subsets = {
            "train": self.train_indices, "validation": self.validation_indices,
            "test": self.test_indices, "all": np.arange(self.data_shape[0]),
        }
        if source not in subsets:
            raise ValueError(f"Unknown evaluation source: {source}")
        indices = subsets[source]
        if row_idx is not None:
            if isinstance(row_idx, bool) or int(row_idx) != row_idx or row_idx not in indices:
                raise ValueError(f"row_idx {row_idx} is not in the {source} split")
            row = int(row_idx)
        elif self.is_testing and self.test_source == source:
            row = self.idx[0]
        else:
            row = int(indices[0])
        socs = self.params["socs"].copy()
        if initial_soc is not None:
            socs = np.broadcast_to(np.asarray(initial_soc, dtype=float), (self.n_agents,)).copy()
        if not np.all(np.isfinite(socs) & (socs >= self.soc_min) & (socs <= self.soc_max)):
            raise ValueError("initial_soc is outside the configured bounds")
        if not self.is_testing:
            self._training_state = (self.idx.copy(), self.params["socs"].copy())
        self.idx, self.params["socs"] = [row, 0], socs
        self.is_testing, self.test_source = True, source
        self.trajectory_context = dict(trajectory_context or {})
        self.ep_counter += 1
        self._start_episode()
        self.state = self._observations()
        return self.get_obs()

    def battery_power_limits(self, socs):
        p, dt = self.params, self.time_step_hours
        socs = np.asarray(socs, dtype=float)
        if (socs.shape != (self.n_agents,) or not np.isfinite(socs).all()
                or (socs < self.soc_min - 1e-10).any()
                or (socs > self.soc_max + 1e-10).any()):
            raise ValueError("Invalid SOCs for battery power limits")
        upper = np.minimum(
            p["battery_discharge_max"],
            np.maximum(socs - self.soc_min, 0) * p["battery_caps"] * p["raw_dis"] / dt,
        )
        lower = -np.minimum(
            p["battery_charge_max"],
            np.maximum(self.soc_max - socs, 0) * p["battery_caps"] / (p["raw_ch"] * dt),
        )
        return lower, upper

    def battery_action_from_power(self, power, socs):
        """Encode physical schedules in the same feasible-fraction Box actions."""
        power = np.asarray(power, dtype=float)
        lower, upper = self.battery_power_limits(socs)
        if (power.shape != (self.n_agents,) or not np.isfinite(power).all()
                or (power < lower - 1e-6).any() or (power > upper + 1e-6).any()):
            raise ValueError("Battery power is outside its feasible interval")
        power = np.clip(power, lower, upper)
        limits = np.where(power >= 0, upper, -lower)
        action = np.zeros(self.n_agents)
        np.divide(power, limits, out=action, where=limits > 0)
        return action

    def _clear_exchange(self, net):
        supply = np.minimum(np.maximum(-net, 0), self.params["exchange_max"])
        demand = np.minimum(np.maximum(net, 0), self.params["exchange_max"])
        sent = min(supply.sum(), demand.sum() / self.exchange_efficiency)
        exported = supply * (sent / supply.sum()) if supply.sum() > 0 else supply
        imported = demand * (sent * self.exchange_efficiency / demand.sum()) if demand.sum() > 0 else demand
        return np.maximum(net - imported, 0), np.maximum(-net - exported, 0), imported, exported

    def dispatch(self, actions, socs, index):
        """Pure one-step physical transition, shared by rollout and local search."""
        actions, socs = np.asarray(actions, dtype=float), np.asarray(socs, dtype=float)
        if actions.shape != (self.n_agents, 2) or socs.shape != (self.n_agents,):
            raise ValueError("Invalid action or SOC shape")
        if not (
            np.isfinite(actions).all() and np.isfinite(socs).all()
            and (actions >= self.action_lows - 1e-7).all()
            and (actions <= self.action_highs + 1e-7).all()
            and (socs >= self.soc_min - 1e-10).all()
            and (socs <= self.soc_max + 1e-10).all()
        ):
            raise ValueError("Actions/SOC must be finite and inside their bounds")
        actions = np.clip(actions, self.action_lows, self.action_highs)
        p, dt = self.params, self.time_step_hours
        cap, eta_c, eta_d = p["battery_caps"], p["raw_ch"], p["raw_dis"]
        pd = actions[:, 0] * p["dg_max"]
        lower, upper = self.battery_power_limits(socs)
        requested = actions[:, 1] * np.where(actions[:, 1] >= 0, upper, -lower)
        pb = requested
        next_soc = np.clip(
            socs - np.where(pb >= 0, pb / eta_d, pb * eta_c) * dt / cap,
            self.soc_min, self.soc_max,
        )
        soc_requested = socs - np.where(
            requested >= 0, requested / eta_d, requested * eta_c,
        ) * dt / cap
        load, pv = [self.agent_data[key][index] for key in ("load", "pv")]
        net_grid = load - pv - pd - pb
        # Uniform-loss community bus: route only residual surplus to deficits.
        # Pro-rata clearing has no forecast and never changes policy decisions.
        pg, spill, imported, exported = self._clear_exchange(net_grid)
        price = float(self.data_dict["price"][index])
        baseline_pg, _, _, baseline_export = self._clear_exchange(load - pv)
        baseline_cost = (baseline_pg * price + baseline_export * self.exchange_fee + p["costc"]) * dt
        gen = (p["costa"] * pd**2 + p["costb"] * pd + p["costc"]) * dt
        bess = p["cycle_cost"] * np.abs(pb) * dt
        buy = pg * price * dt
        exchange_cost = exported * self.exchange_fee * dt
        emission = p["env_param"] * pd * dt
        clip = np.abs(requested - pb)
        penalty = p["battery_punishment_params"] * clip * dt
        economic = buy + gen + bess + exchange_cost
        full_cost = economic + emission + penalty
        reward = (baseline_cost - full_cost) * self.reward_scale
        return {
            "load": load, "pv": pv, "price": price, "pd": pd, "pb": pb, "pg": pg,
            "spill": spill, "net_grid_request": net_grid, "soc": next_soc,
            "peer_export": exported, "peer_import": imported,
            "exchange_loss": exported * (1 - self.exchange_efficiency),
            "exchange_cost": exchange_cost,
            "pb_requested": requested, "soc_requested": soc_requested,
            "battery_action_clip": clip, "gen_cost": gen, "bess_cost": bess,
            "buy_cost": buy, "economic_cost": economic, "economic_reward": -economic,
            "eco_reward": -economic, "env_reward": -emission, "punishment": -penalty,
            "reward": reward, "full_cost": full_cost, "baseline_cost": baseline_cost,
            "cost_reward": -full_cost * self.reward_scale,
        }

    def step(self, actions):
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions).reshape(self.n_agents, 2)
        index, before = tuple(self.idx), self.params["socs"].copy()
        result = self.dispatch(actions, before, index)
        self.params["socs"] = result["soc"]
        team_reward = float(np.mean(result["reward"].astype(np.float32)))
        dt = self.time_step_hours
        baseline = np.maximum(result["load"] - result["pv"], 0) * result["price"] * dt
        saving = baseline - result["buy_cost"]
        values = {
            "grid_purchase_cost": result["buy_cost"].sum(),
            "baseline_grid_purchase_cost": baseline.sum(), "grid_purchase_saving": saving.sum(),
            "mean_soc": result["soc"].mean(), "economic_cost": result["economic_cost"].sum(),
            "economic_reward": result["economic_reward"].sum(),
            "full_cost": result["full_cost"].sum(), "generate_cost_reward": -result["gen_cost"].sum(),
            "bess_cost_reward": -result["bess_cost"].sum(),
            "buy_cost_reward": -result["buy_cost"].sum(),
            "battery_punishment_reward": result["punishment"].sum(),
            "battery_action_clip": result["battery_action_clip"].sum(),
            "soc_violation": np.sum(np.maximum(self.soc_min - result["soc_requested"], 0)
                                    + np.maximum(result["soc_requested"] - self.soc_max, 0)),
            "battery_charge_energy": np.maximum(-result["pb"], 0).sum() * dt,
            "battery_discharge_energy": np.maximum(result["pb"], 0).sum() * dt,
            "grid_import_energy": result["pg"].sum() * dt,
            "generator_energy": result["pd"].sum() * dt, "spill_energy": result["spill"].sum() * dt,
            "peer_sent_energy": result["peer_export"].sum() * dt,
            "peer_received_energy": result["peer_import"].sum() * dt,
            "exchange_loss_energy": result["exchange_loss"].sum() * dt,
            "exchange_cost": result["exchange_cost"].sum(),
            "baseline_full_cost": result["baseline_cost"].sum(),
            "full_cost_saving": (result["baseline_cost"] - result["full_cost"]).sum(),
        }
        for key, value in values.items():
            self._stats[key] = self._stats.get(key, 0.0) + float(value)
        for i, agent in enumerate(self.possible_agents):
            record = {
                "episode": self.ep_counter, "agent": agent,
                "policy_source": self.trajectory_context.get("policy_source", "unspecified"),
                "checkpoint_step": self.trajectory_context.get("checkpoint_step", ""),
                "battery_model_version": self.battery_model_version, "time_step_hours": dt,
                "row_idx": index[0], "col_idx": index[1], "soc_init": float(before[i]),
                "action_pd": float(actions[i, 0]), "action_pb": float(actions[i, 1]),
                "team_reward": team_reward, "pg_total": float(result["pg"].sum()),
                "baseline_grid_purchase_cost": float(baseline[i]),
                "grid_purchase_saving": float(saving[i]),
            }
            record.update({k: float(v if np.ndim(v) == 0 else v[i]) for k, v in result.items()})
            record.update({f"{k}_raw": float(v[index]) for k, v in self.raw_data.items()})
            self.ep_buffer.append(record)
        self._ep_steps += 1
        self.idx[1] += 1
        terminated = self.idx[1] == self.episode_limit
        if terminated:
            indices = (np.arange(self.data_shape[0]) if self.test_source == "all"
                       else getattr(self, f"{self.test_source}_indices"))
            row = self.idx[0] + 1
            self.idx = [row if row <= indices[-1] else int(indices[0]), 0]
            write_ok = self.log_trajectories and (
                self.log_trajectories_mode == "all"
                or (self.log_trajectories_mode == "test" and self.is_testing)
                or (self.log_trajectories_mode == "train" and not self.is_testing)
            )
            if write_ok:
                path = self._current_csv_path()
                exists = path.exists()
                with path.open("a" if exists else "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=list(self.ep_buffer[0]))
                    if not exists:
                        writer.writeheader()
                    writer.writerows(self.ep_buffer)
        self.state = self._observations()
        info = dict(self._stats)
        info["mean_soc"] /= self._ep_steps
        base = info["baseline_grid_purchase_cost"]
        info["grid_purchase_saving_ratio"] = info["grid_purchase_saving"] / base if base > 1e-9 else 0
        info["episode_limit"] = terminated
        return team_reward, terminated, info

    def _current_csv_path(self):
        start = ((max(self.ep_counter, 1) - 1) // self.bucket_size) * self.bucket_size + 1
        return self.render_dir / f"episodes_{self.test_source}_{start:06d}-{start + self.bucket_size - 1:06d}.csv"

    def render(self, mode="human"):
        self.render_mode = mode

    def close(self):
        pass
