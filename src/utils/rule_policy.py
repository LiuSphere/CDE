"""Frozen causal tariff/SOC baseline implementing the runner policy interface."""

import numpy as np
import torch


class TariffSOCPolicy:
    def __init__(self, env, settings):
        self.name = settings["name"]
        low, high = settings["low_price_quantile"], settings["high_price_quantile"]
        self.reserve = float(settings["reserve_soc"])
        self.target = float(settings["charge_target_soc"])
        if not (0 <= low < high <= 1
                and env.soc_min <= self.reserve < self.target <= env.soc_max):
            raise ValueError("Invalid rule thresholds or SOC bands")
        prices = env.data_dict["price"][env.train_indices]
        self.low_price, self.high_price = np.quantile(prices, [low, high])
        self.price_scale = env.observation_price_scale
        self.price_tolerance = 1e-7
        self.power_scale = env.observation_power_scale
        self.dt = env.time_step_hours
        self.params = {key: value.copy() for key, value in env.params.items()
                       if key != "socs"}

    def init_hidden(self, batch_size):
        if batch_size != 1:
            raise ValueError("Rule evaluation requires batch_size=1")

    def select_actions(self, ep_batch, t_ep, t_env, test_mode=False):
        if not test_mode:
            raise ValueError("The rule policy is evaluation-only")
        obs = ep_batch["obs"][0, t_ep].cpu().numpy().astype(float)
        return torch.as_tensor(self.actions(obs), dtype=torch.float32).unsqueeze(0)

    def actions(self, obs):
        obs = np.asarray(obs, dtype=float)
        p = self.params
        if obs.shape != (len(p["dg_max"]), 20) or not np.isfinite(obs).all():
            raise ValueError("Rule requires finite physical-model observations")
        price = obs[:, 0] * self.price_scale
        net = (obs[:, 1] - obs[:, 2]) * self.power_scale
        community_net = net + (obs[:, 6] - obs[:, 7]) * self.power_scale
        soc = obs[:, 3]
        charge_limit = obs[:, 12] * self.power_scale
        discharge_limit = obs[:, 13] * self.power_scale
        forecast_price = obs[:, 15] * self.price_scale
        forecast_net = (obs[:, 17] - obs[:, 18]) * self.power_scale

        charge_room = np.minimum(
            charge_limit,
            np.maximum(self.target - soc, 0) * p["battery_caps"] / (p["raw_ch"] * self.dt),
        )
        discharge_room = np.minimum(
            discharge_limit,
            np.maximum(soc - self.reserve, 0) * p["battery_caps"] * p["raw_dis"] / self.dt,
        )
        battery = -np.minimum(np.maximum(-net, 0), charge_room)
        profitable_charge = (
            forecast_price * p["raw_ch"] * p["raw_dis"]
            > price + p["cycle_cost"] * (1 + p["raw_ch"] * p["raw_dis"])
        )
        grid_charge = (
            (price <= self.low_price + self.price_tolerance)
            & profitable_charge & (forecast_net > 0)
        )
        battery = np.where(grid_charge, -charge_room, battery)
        discharge = (price >= self.high_price - self.price_tolerance) & (net > 0)
        battery = np.where(discharge, np.minimum(net, discharge_room), battery)

        marginal_budget = price - p["costb"] - p["env_param"]
        generation = np.where(
            marginal_budget > 0,
            np.divide(marginal_budget, 2 * p["costa"],
                      out=p["dg_max"].copy(), where=p["costa"] > 0),
            0,
        )
        # Generation can serve local demand and current community deficits, not future actuals.
        demand = np.maximum(net - battery, 0) + np.minimum(
            np.maximum(community_net - net, 0), p["exchange_max"],
        )
        generation = np.minimum(np.maximum(generation, 0), np.minimum(p["dg_max"], demand))
        limits = np.where(battery >= 0, discharge_limit, charge_limit)
        fraction = np.divide(battery, limits, out=np.zeros_like(battery), where=limits > 0)
        return np.column_stack((
            np.divide(generation, p["dg_max"], out=np.zeros_like(generation),
                      where=p["dg_max"] > 0),
            np.clip(fraction, -1, 1),
        )).astype(np.float32)

    def specification(self):
        return {
            "name": self.name,
            "low_price": float(self.low_price),
            "high_price": float(self.high_price),
            "reserve_soc": self.reserve,
            "charge_target_soc": self.target,
            "price_comparison_tolerance": self.price_tolerance,
            "threshold_source": "training tariff only",
            "forecast_source": "current observation causal 4-hour features",
        }
