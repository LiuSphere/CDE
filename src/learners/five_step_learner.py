"""Opt-in within-episode five-step TD targets; original Actor/loss/cadence."""

import json
import math
from pathlib import Path

import torch

from learners.gavdac_learner import GAVDACLearner


def five_step_enabled(args):
    value = getattr(args, "five_step_critic", False)
    if type(value) is not bool:
        raise ValueError("five_step_critic must be boolean")
    return value


def _check_tensor(value):
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError("H05 requires CPU float32")
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite H05 tensor")


@torch.no_grad()
def n_step_targets(rewards, terminated, mask, target_vals, gamma, n_steps=5):
    if type(n_steps) is not int or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Invalid target discount")
    if (target_vals.ndim != 3 or target_vals.shape[1] < 1 or
            any(value.shape != (*target_vals.shape[:2], 1) for value in (rewards, terminated, mask))):
        raise ValueError("Targets require B,T,C and reward/termination/mask B,T,1")
    for value in (rewards, terminated, mask, target_vals):
        _check_tensor(value)
    if any(not torch.all((value == 0) | (value == 1)) for value in (terminated, mask)):
        raise ValueError("Termination and valid masks must be binary")
    length = target_vals.shape[1]
    starts = torch.arange(length)
    active = mask.bool()
    result = torch.zeros_like(target_vals)
    # All starting times advance together, stopping each return at its first boundary.
    for offset in range(min(n_steps, length)):
        positions = starts + offset
        indices = positions.clamp(max=length - 1)
        valid = (positions < length)[None, :, None] & mask[:, indices].bool()
        active = active & valid
        result.add_((gamma ** offset) * torch.where(active, rewards[:, indices], 0))
        continuing = active & ~terminated[:, indices].bool()
        next_indices = (positions + 1).clamp(max=length - 1)
        next_valid = ((positions + 1) < length)[None, :, None] & mask[:, next_indices].bool()
        stopping = continuing & ((offset == n_steps - 1) | ~next_valid)
        result.add_((gamma ** (offset + 1)) * torch.where(stopping, target_vals[:, indices], 0))
        active = continuing & next_valid
    _check_tensor(result)
    return result


class FiveStepGAVDACLearner(GAVDACLearner):
    uses_population_training = False

    def __init__(self, mac, scheme, logger, args):
        self.five_step_critic = five_step_enabled(args)
        if getattr(args, "use_cuda", False):
            raise ValueError("H05 requires CPU float32")
        super().__init__(mac, scheme, logger, args)
        self.critic_optimizer_steps = 0
        self.actor_optimizer_steps = 0
        self.target_updates = 0
        self.finite_update_checks = 0
        self.critic_optimiser.register_step_post_hook(self._critic_stepped)
        self.agent_optimiser.register_step_post_hook(self._actor_stepped)
        self._check_parameters(self.agent_params + self.critic_params)

    @staticmethod
    def _check_parameters(parameters):
        for param in parameters:
            _check_tensor(param)
            if param.grad is not None:
                _check_tensor(param.grad)

    def _critic_stepped(self, optimizer, arguments, keywords):
        self.critic_optimizer_steps += 1
        self._check_parameters(self.critic_params)

    def _actor_stepped(self, optimizer, arguments, keywords):
        self.actor_optimizer_steps += 1
        self._check_parameters(self.agent_params)

    def _build_td_targets(self, rewards, terminated, mask, target_vals):
        if not self.five_step_critic:
            return super()._build_td_targets(rewards, terminated, mask, target_vals)
        return n_step_targets(rewards, terminated, mask, target_vals, self.args.gamma)

    def _critic_loss(self, batch, denominator=None):
        result = super()._critic_loss(batch, denominator)
        _check_tensor(result[2])
        return result

    def _actor_loss(self, batch, actor_policy, critic_inputs):
        loss = super()._actor_loss(batch, actor_policy, critic_inputs)
        _check_tensor(loss)
        return loss

    def _targets_updated(self):
        self.target_updates += 1
        parameters = list(self.target_mac.parameters()) + list(self.target_critic.parameters())
        if self.mixer is not None:
            parameters += list(self.target_mixer.parameters())
        self._check_parameters(parameters)

    def _update_targets_soft(self, tau):
        super()._update_targets_soft(tau)
        self._targets_updated()

    def _update_targets(self):
        super()._update_targets()
        self._targets_updated()

    def train(self, batch, t_env, episode_num):
        super().train(batch, t_env, episode_num)
        if not self.critic_optimizer_steps == self.actor_optimizer_steps == self.target_updates:
            raise RuntimeError("H05 optimizer/target cadence mismatch")
        self.finite_update_checks += 1

    def diagnostics(self):
        return {
            "five_step_critic": self.five_step_critic, "target_horizon": 5,
            "critic_optimizer_steps": self.critic_optimizer_steps,
            "actor_optimizer_steps": self.actor_optimizer_steps, "target_updates": self.target_updates,
            "finite_update_checks": self.finite_update_checks, "exact_resume_supported": False,
        }

    def save_models(self, path):
        super().save_models(path)
        (Path(path) / "h05_diagnostics.json").write_text(
            json.dumps(self.diagnostics(), indent=2, allow_nan=False), encoding="utf-8",
        )

    def load_models(self, path):
        saved = json.loads((Path(path) / "h05_diagnostics.json").read_text(encoding="utf-8"))
        for key in ("five_step_critic", "target_horizon", "exact_resume_supported"):
            if saved[key] != self.diagnostics()[key]:
                raise ValueError("H05 diagnostic configuration mismatch")
        keys = ("critic_optimizer_steps", "actor_optimizer_steps", "target_updates", "finite_update_checks")
        if any(type(saved[key]) is not int or saved[key] < 0 for key in keys) or len({saved[k] for k in keys}) != 1:
            raise ValueError("Invalid H05 successful-update counters")
        super().load_models(path)
        for key in keys:
            setattr(self, key, saved[key])
