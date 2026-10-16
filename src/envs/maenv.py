import csv
import functools
import pathlib

import numpy as np
from gymnasium import spaces
from pandas import read_csv
from pettingzoo import ParallelEnv


def standardize_data(data: np.ndarray) -> np.ndarray:
    std = float(np.sqrt(data.var()))
    if std < 1e-8:
        std = 1.0
    return ((data - float(data.mean())) / std).astype(np.float32)


class MGEnv(ParallelEnv):
    """The maenv_sep.py single-objective model with the FACMAC rollout API."""

    metadata = {"render_modes": ["human", None], "name": "MGEnv_maenv_sep_v1"}

    def __init__(self, config, algo_name, render_mode="human"):
        self.render_mode = render_mode
        self.battery_model_version = "maenv_sep_v1"
        self.time_step_hours = float(config.get("time_step_hours", 1.0))
        if self.time_step_hours != 1.0:
            raise ValueError("maenv_sep uses a fixed one-hour time step")
        self.n_agents = len(config["data"])
        self.possible_agents = [f"agent{i + 1}" for i in range(self.n_agents)]
        agents_config = {agent.pop("agent_name"): agent for agent in config["data"]}
        self.params = self.params_init(agents_config)
        self.action_lows = np.array(
            [agents_config[a]["low_action"] for a in self.possible_agents],
            dtype=np.float32,
        )
        self.action_highs = np.array(
            [agents_config[a]["high_action"] for a in self.possible_agents],
            dtype=np.float32,
        )
        self.n_actions = 2
        self.spaces = {
            agent: {
                "observation": spaces.Box(
                    low=np.array(agents_config[agent]["low_state"], dtype=np.float32),
                    high=np.array(agents_config[agent]["high_state"], dtype=np.float32),
                    shape=(4,),
                    dtype=np.float32,
                ),
                "action": spaces.Box(
                    low=self.action_lows[i],
                    high=self.action_highs[i],
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
            for i, agent in enumerate(self.possible_agents)
        }

        data_path = pathlib.Path(__file__).parent / "data"
        self.raw_data = {
            name: np.array(read_csv(data_path / f"{name}.csv", header=None))
            for name in ("price", "load", "pv")
        }
        self.data_dict = {
            name: standardize_data(data) for name, data in self.raw_data.items()
        }
        self.data_dict["load_pv"] = self.data_dict["load"] - self.data_dict["pv"]
        self.data_shape = self.data_dict["price"].shape
        self.episode_limit = self.data_shape[1]
        if self.episode_limit != 24:
            raise ValueError("maenv_sep requires 24 columns per day")

        # Preserve the reference's training rows 0..730. Split the remaining
        # days for the project's disjoint checkpoint-selection and test sets.
        self.train_indices = np.arange(min(731, self.data_shape[0]))
        heldout = np.arange(len(self.train_indices), self.data_shape[0])
        self.validation_indices, self.test_indices = np.array_split(heldout, 2)
        self.seed = config.get("seed")
        self.rng = np.random.default_rng(self.seed)
        self.idx = [0, 0]
        self.state = self._observations()
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

    def params_init(self, agents_config):
        names = {
            "dg_max": "dg_max",
            "battery_caps": "battery_cap",
            "battery_punishment_params": "battery_punishment_param",
            "raw_ch": "raw_ch",
            "raw_dis": "raw_dis",
            "bessa": "bessa",
            "bessb": "bessb",
            "bessc": "bessc",
            "costa": "costa",
            "costb": "costb",
            "costc": "costc",
            "env_param": "env_param",
        }
        params = {
            key: np.array(
                [agents_config[a][field] for a in self.possible_agents], dtype=float,
            )
            for key, field in names.items()
        }
        # These derived ratings retain checkpoint metadata compatibility; the
        # reference maps battery actions to capacity, not separate power limits.
        params["battery_charge_max"] = params["battery_caps"].copy()
        params["battery_discharge_max"] = params["battery_caps"].copy()
        params["socs"] = np.full(self.n_agents, 0.2, dtype=float)
        return params

    def _observations(self):
        index = tuple(self.idx)
        return np.array(
            [
                [
                    self.data_dict["price"][index],
                    self.data_dict["load"][index],
                    self.data_dict["pv"][index],
                    soc,
                ]
                for soc in self.params["socs"]
            ],
            dtype=np.float32,
        )

    def get_obs(self):
        return [obs.copy() for obs in self.state]

    def get_state(self):
        return self.state.flatten()

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "obs_shape": 4,
            "state_shape": 4 * self.n_agents,
            "episode_limit": self.episode_limit,
            "action_spaces": [self.spaces[a]["action"] for a in self.possible_agents],
            "actions_dtype": np.float32,
            "normalise_actions": False,
        }

    @functools.cache
    def observation_space(self, agent: int):
        return self.spaces[f"agent{agent}"]["observation"]

    @functools.cache
    def action_space(self, agent: int):
        return self.spaces[f"agent{agent}"]["action"]

    def _start_episode(self):
        self.ep_buffer = []
        self._ep_steps = 0
        self._stats = dict.fromkeys(
            (
                "total_power_loss", "grid_purchase_cost",
                "baseline_grid_purchase_cost", "grid_purchase_saving",
                "soc_violation", "mean_soc", "economic_reward", "economic_cost",
                "buy_cost_reward", "generate_cost_reward", "bess_cost_reward",
                "battery_punishment_reward", "battery_action_clip",
            ),
            0.0,
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._training_state is not None:
            self.idx, self.params["socs"] = self._training_state
            self._training_state = None
        self.is_testing = False
        self.test_source = "train"
        self.trajectory_context = {}
        self.ep_counter += 1
        self._start_episode()
        self.state = self._observations()
        return self.get_obs()

    def reset_all(self, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        self._training_state = None
        self.idx = [int(rng.choice(self.train_indices)), 0]
        self.params["socs"] = np.full(self.n_agents, 0.2, dtype=float)
        return self.reset()

    def reset_test(
        self, seed=None, source="test", row_idx=None, initial_soc=None,
        trajectory_context=None,
    ):
        subsets = {
            "train": self.train_indices, "validation": self.validation_indices,
            "test": self.test_indices, "all": np.arange(self.data_shape[0]),
        }
        if source not in subsets:
            raise ValueError(f"Unknown evaluation source: {source}")
        indices = subsets[source]
        if row_idx is not None:
            if (
                isinstance(row_idx, bool) or int(row_idx) != row_idx
                or row_idx not in indices
            ):
                raise ValueError(f"row_idx {row_idx} is not in the {source} split")
            row_index = int(row_idx)
        elif self.is_testing and self.test_source == source:
            row_index = self.idx[0]
        else:
            row_index = int(indices[0])
        socs = self.params["socs"].copy()
        if initial_soc is not None:
            socs = np.broadcast_to(
                np.asarray(initial_soc, dtype=float), (self.n_agents,),
            ).copy()
            if not np.all(np.isfinite(socs) & (socs >= 0.2) & (socs <= 0.8)):
                raise ValueError("initial_soc must be finite and in [0.2, 0.8]")
        if not self.is_testing:
            self._training_state = (self.idx.copy(), self.params["socs"].copy())
        self.idx = [row_index, 0]
        self.params["socs"] = socs
        self.is_testing = True
        self.test_source = source
        self.trajectory_context = dict(trajectory_context or {})
        self.ep_counter += 1
        self._start_episode()
        self.state = self._observations()
        return self.get_obs()

    def step(self, actions):
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()
        actions = np.asarray(actions).reshape(self.n_agents, self.n_actions)
        index = tuple(self.idx)
        price = self.data_dict["price"][index]
        load = self.data_dict["load"][index]
        pv = self.data_dict["pv"][index]
        last_socs = self.params["socs"].copy()
        caps = self.params["battery_caps"]
        pds = actions[:, 0] * self.params["dg_max"]
        requested = actions[:, 1] * caps
        punishment = -self.params["battery_punishment_params"] * (
            np.maximum(requested - last_socs * caps, 0)
            + np.maximum(-(1 - last_socs) * caps - requested, 0)
        )
        socs_raw = last_socs + np.where(
            requested < 0,
            -self.params["raw_ch"] * requested / caps,
            -self.params["raw_dis"] * requested / caps,
        )
        pbs = requested.copy()
        low, high = socs_raw < 0.2, socs_raw > 0.8
        pbs[low] = (last_socs[low] - 0.2) * caps[low]
        pbs[high] = (last_socs[high] - 0.8) * caps[high]
        socs = np.clip(socs_raw, 0.2, 0.8)
        self.params["socs"] = socs
        pg = load - pv - pds - pbs
        x = pbs + 3 * caps * (1 - socs)
        bess_cost = -(
            self.params["bessa"] * x**2 + self.params["bessb"] * x
            + self.params["bessc"]
        ) * np.where(pbs > 0, 0.3, 1.0)
        generate_cost = -(
            self.params["costa"] * pds**2 + self.params["costb"] * pds
            + self.params["costc"]
        )
        buy_cost = np.where(pg > 0, -pg * price, 0.0)
        economic_reward = 0.5 * generate_cost + bess_cost + buy_cost
        env_reward = -pds * self.params["env_param"]
        reward = (economic_reward + env_reward + punishment) * 0.1
        team_reward = float(np.mean(reward.astype(np.float32)))
        # maenv_sep returns the current hour's inputs with the updated SOC,
        # before advancing its clock. Keep that observation timing unchanged.
        self.state = self._observations()

        baseline_cost = max(float(load - pv), 0.0) * float(price)
        grid_cost = -buy_cost
        saving = baseline_cost - grid_cost
        action_clip = np.abs(requested - pbs)
        self._ep_steps += 1
        values = {
            "total_power_loss": np.sum(np.abs(pg)),
            "grid_purchase_cost": np.sum(grid_cost),
            "baseline_grid_purchase_cost": baseline_cost * self.n_agents,
            "grid_purchase_saving": np.sum(saving),
            "soc_violation": np.sum(
                np.maximum(0.2 - socs_raw, 0) + np.maximum(socs_raw - 0.8, 0),
            ),
            "mean_soc": np.mean(socs),
            "economic_reward": np.sum(economic_reward),
            "economic_cost": np.sum(-economic_reward),
            "buy_cost_reward": np.sum(buy_cost),
            "generate_cost_reward": np.sum(0.5 * generate_cost),
            "bess_cost_reward": np.sum(bess_cost),
            "battery_punishment_reward": np.sum(punishment),
            "battery_action_clip": np.sum(action_clip),
        }
        for key, value in values.items():
            self._stats[key] += float(value)
        for i, agent in enumerate(self.possible_agents):
            self.ep_buffer.append({
                "episode": self.ep_counter,
                "policy_source": self.trajectory_context.get("policy_source", "unspecified"),
                "checkpoint_step": self.trajectory_context.get("checkpoint_step", ""),
                "battery_model_version": self.battery_model_version,
                "time_step_hours": self.time_step_hours,
                "agent": agent, "row_idx": self.idx[0], "col_idx": self.idx[1],
                "price": float(price), "load": float(load), "pv": float(pv),
                "load_sd": float(load), "pv_sd": float(pv),
                "price_raw": float(self.raw_data["price"][index]),
                "load_raw": float(self.raw_data["load"][index]),
                "pv_raw": float(self.raw_data["pv"][index]),
                "soc_init": float(last_socs[i]), "soc": float(socs[i]),
                "action_pd": float(actions[i, 0]), "action_pb": float(actions[i, 1]),
                "pd": float(pds[i]), "pb": float(pbs[i]),
                "pb_requested": float(requested[i]), "soc_requested": float(socs_raw[i]),
                "battery_action_clip": float(action_clip[i]),
                "pg": float(pg[i]), "pg_total": float(np.sum(pg)),
                "gen_cost": float(-generate_cost[i]), "bess_cost": float(-bess_cost[i]),
                "buy_cost": float(grid_cost[i]),
                "economic_reward": float(economic_reward[i]),
                "economic_cost": float(-economic_reward[i]),
                "baseline_grid_purchase_cost": float(baseline_cost),
                "grid_purchase_saving": float(saving[i]),
                "eco_reward": float(economic_reward[i]),
                "env_reward": float(env_reward[i]), "punishment": float(punishment[i]),
                "reward": float(reward[i]), "team_reward": team_reward,
            })

        self.idx[1] += 1
        terminated = self.idx[1] == self.episode_limit
        if terminated:
            indices = (
                getattr(self, f"{self.test_source}_indices")
                if self.test_source != "all" else np.arange(self.data_shape[0])
            )
            next_row = self.idx[0] + 1
            self.idx = [next_row if next_row <= indices[-1] else int(indices[0]), 0]
            write_ok = self.log_trajectories and (
                self.log_trajectories_mode == "all"
                or (self.log_trajectories_mode == "test" and self.is_testing)
                or (self.log_trajectories_mode == "train" and not self.is_testing)
            )
            if write_ok and self.ep_buffer:
                path = self._current_csv_path()
                exists = path.exists()
                with path.open(
                    "a" if exists else "w", newline="", encoding="utf-8",
                ) as file:
                    writer = csv.DictWriter(file, fieldnames=list(self.ep_buffer[0]))
                    if not exists:
                        writer.writeheader()
                    writer.writerows(self.ep_buffer)

        info = dict(self._stats)
        info["mean_soc"] /= max(self._ep_steps, 1)
        baseline = info["baseline_grid_purchase_cost"]
        info["grid_purchase_saving_ratio"] = (
            info["grid_purchase_saving"] / baseline if abs(baseline) > 1e-6 else 0.0
        )
        # Daily rollout boundaries truncate a continuing process, not the model.
        info["episode_limit"] = terminated
        return team_reward, terminated, info

    def _current_csv_path(self):
        start = ((max(self.ep_counter, 1) - 1) // self.bucket_size) * self.bucket_size + 1
        end = start + self.bucket_size - 1
        return self.render_dir / f"episodes_{self.test_source}_{start:06d}-{end:06d}.csv"

    def render(self, mode="human"):
        self.render_mode = mode

    def close(self):
        pass
