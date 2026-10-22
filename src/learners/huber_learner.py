"""Opt-in scaled unit-Huber TD regression; all policy/target rules are inherited."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from learners.gavdac_learner import GAVDACLearner


def huber_enabled(args):
    value = getattr(args, "scaled_huber_critic", False)
    if type(value) is not bool:
        raise ValueError("scaled_huber_critic must be boolean")
    return value


class HuberGAVDACLearner(GAVDACLearner):
    uses_population_training = False

    def __init__(self, mac, scheme, logger, args):
        self.scaled_huber_critic = huber_enabled(args)
        if getattr(args, "use_cuda", False):
            raise ValueError("H04 requires CPU float32")
        super().__init__(mac, scheme, logger, args)
        self.critic_optimizer_steps = 0
        self.actor_optimizer_steps = 0
        self.target_updates = 0
        self.finite_update_checks = 0
        self.critic_optimiser.register_step_post_hook(self._critic_stepped)
        self.agent_optimiser.register_step_post_hook(self._actor_stepped)
        self._check_parameters(self.agent_params + self.critic_params)

    @staticmethod
    def _check_tensor(value):
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise ValueError("H04 requires CPU float32")
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite H04 tensor")

    def _check_parameters(self, parameters):
        for param in parameters:
            self._check_tensor(param)
            if param.grad is not None:
                self._check_tensor(param.grad)

    def _critic_stepped(self, optimizer, arguments, keywords):
        self.critic_optimizer_steps += 1
        self._check_parameters(self.critic_params)

    def _actor_stepped(self, optimizer, arguments, keywords):
        self.actor_optimizer_steps += 1
        self._check_parameters(self.agent_params)

    def _td_error_loss(self, masked_td_error, denominator):
        if not self.scaled_huber_critic:
            return super()._td_error_loss(masked_td_error, denominator)
        self._check_tensor(masked_td_error)
        if not torch.isfinite(torch.as_tensor(denominator)).all() or denominator <= 0:
            raise ValueError("H04 requires a finite positive valid-transition count")
        loss = 2 * F.huber_loss(
            masked_td_error, torch.zeros_like(masked_td_error), reduction="none", delta=1.0,
        ).sum() / denominator
        self._check_tensor(loss)
        return loss

    def _actor_loss(self, batch, actor_policy, critic_inputs):
        loss = super()._actor_loss(batch, actor_policy, critic_inputs)
        self._check_tensor(loss)
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
            raise RuntimeError("H04 optimizer/target cadence mismatch")
        self.finite_update_checks += 1

    def diagnostics(self):
        return {
            "scaled_huber_critic": self.scaled_huber_critic, "huber_delta": 1.0,
            "critic_optimizer_steps": self.critic_optimizer_steps,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "target_updates": self.target_updates, "finite_update_checks": self.finite_update_checks,
            "exact_resume_supported": False,
        }

    def save_models(self, path):
        super().save_models(path)
        (Path(path) / "h04_diagnostics.json").write_text(
            json.dumps(self.diagnostics(), indent=2, allow_nan=False), encoding="utf-8",
        )

    def load_models(self, path):
        saved = json.loads((Path(path) / "h04_diagnostics.json").read_text(encoding="utf-8"))
        expected = self.diagnostics()
        for key in ("scaled_huber_critic", "huber_delta", "exact_resume_supported"):
            if saved[key] != expected[key]:
                raise ValueError("H04 diagnostic configuration mismatch")
        keys = ("critic_optimizer_steps", "actor_optimizer_steps", "target_updates", "finite_update_checks")
        if any(type(saved[key]) is not int or saved[key] < 0 for key in keys) or len({saved[k] for k in keys}) != 1:
            raise ValueError("Invalid H04 successful-update counters")
        super().load_models(path)
        for key in keys:
            setattr(self, key, saved[key])
