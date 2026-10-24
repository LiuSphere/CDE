"""H08: initialize only the local Critic output weights and bias smaller."""

import json
import math
from pathlib import Path

import torch

from learners.gavdac_learner import GAVDACLearner
from modules.critic import GAVDACCritic


def small_init_enabled(args):
    value = getattr(args, "critic_small_output_init", False)
    if type(value) is not bool:
        raise ValueError("critic_small_output_init must be boolean")
    return value


def _check_tensor(value):
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError("H08 requires CPU float32 tensors")
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite H08 tensor")


def _check_parameters(parameters, gradients=False):
    for parameter in parameters:
        _check_tensor(parameter)
        if gradients and parameter.grad is not None:
            _check_tensor(parameter.grad)


class SmallInitGAVDACCritic(GAVDACCritic):
    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        # Preserve the original construction order and RNG draws.
        factor = 0.003 * math.sqrt(self.fc3.in_features)
        with torch.no_grad():
            self.fc3.weight.mul_(factor)
            self.fc3.bias.mul_(factor)


class SmallInitGAVDACLearner(GAVDACLearner):
    def __init__(self, mac, scheme, logger, args):
        if not small_init_enabled(args):
            raise ValueError("H08 learner requires critic_small_output_init=true")
        super().__init__(mac, scheme, logger, args)
        self.critic_steps = 0
        self.actor_steps = 0
        self.target_steps = 0
        self.finite_updates = 0
        _check_parameters(self.agent_params + self.critic_params)
        self._check_targets()
        self.critic_optimiser.register_step_post_hook(self._critic_step_completed)
        self.agent_optimiser.register_step_post_hook(self._actor_step_completed)

    def _make_critic(self, scheme, args):
        return SmallInitGAVDACCritic(scheme, args)

    def _critic_step_completed(self, optimizer, args, kwargs):
        _check_parameters(self.critic_params, gradients=True)
        self.critic_steps += 1

    def _actor_step_completed(self, optimizer, args, kwargs):
        _check_parameters(self.agent_params, gradients=True)
        self.actor_steps += 1

    def _check_targets(self):
        _check_parameters(self.target_mac.parameters())
        _check_parameters(self.target_critic.parameters())
        if self.mixer is not None:
            _check_parameters(self.target_mixer.parameters())

    def _critic_loss(self, batch, denominator=None):
        policy, inputs, loss = super()._critic_loss(batch, denominator)
        _check_tensor(loss)
        return policy, inputs, loss

    def _actor_loss(self, batch, actor_policy, critic_inputs):
        loss = super()._actor_loss(batch, actor_policy, critic_inputs)
        _check_tensor(loss)
        return loss

    def _update_targets(self):
        super()._update_targets()
        self._check_targets()
        self.target_steps += 1

    def _update_targets_soft(self, tau):
        super()._update_targets_soft(tau)
        self._check_targets()
        self.target_steps += 1

    def train(self, batch, t_env, episode_num):
        before = (self.critic_steps, self.actor_steps, self.target_steps)
        super().train(batch, t_env, episode_num)
        after = (self.critic_steps, self.actor_steps, self.target_steps)
        if after != tuple(count + 1 for count in before):
            raise RuntimeError("H08 requires one Critic, Actor and target step per update")
        self.finite_updates += 1

    def diagnostics(self):
        return {
            "critic_small_output_init": True,
            "initial_output_bound": 0.003,
            "initial_output_factor": 0.003 * math.sqrt(self.critic.fc3.in_features),
            "critic_steps": self.critic_steps,
            "actor_steps": self.actor_steps,
            "target_steps": self.target_steps,
            "finite_updates": self.finite_updates,
            "exact_resume_supported": False,
        }

    def save_models(self, path):
        super().save_models(path)
        with (Path(path) / "h08_diagnostics.json").open("x", encoding="utf-8") as file:
            json.dump(self.diagnostics(), file, indent=2)

    def load_models(self, path):
        record = json.loads((Path(path) / "h08_diagnostics.json").read_text(encoding="utf-8"))
        for key in ("critic_small_output_init", "initial_output_bound",
                    "initial_output_factor", "exact_resume_supported"):
            if type(record[key]) is not type(self.diagnostics()[key]) or record[key] != self.diagnostics()[key]:
                raise ValueError(f"Invalid H08 diagnostic identity: {key}")
        counts = [record[key] for key in
                  ("critic_steps", "actor_steps", "target_steps", "finite_updates")]
        if any(type(count) is not int or count < 0 for count in counts) or len(set(counts)) != 1:
            raise ValueError("Invalid H08 diagnostic counts")
        # Baseline files omit Critic/replay/RNG; these counts are not an exact resume.
        super().load_models(path)
        _check_parameters(self.agent_params + self.critic_params)
        for key in ("critic_steps", "actor_steps", "target_steps", "finite_updates"):
            setattr(self, key, record[key])
