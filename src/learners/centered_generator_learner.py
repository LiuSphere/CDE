"""Explicit opt-in H10 learner; baseline optimization arithmetic stays inherited."""

import json
from pathlib import Path

import torch

from learners.gavdac_learner import GAVDACLearner
from modules.centered_generator_critic import CenteredGeneratorGAVDACCritic


def validate_center_generator_flag(value=False):
    if not isinstance(value, bool):
        raise ValueError("critic_center_generator_action must be a boolean")
    return value


class CenteredGeneratorGAVDACLearner(GAVDACLearner):
    uses_population_training = False

    def __init__(self, mac, scheme, logger, args):
        if not validate_center_generator_flag(
            getattr(args, "critic_center_generator_action", False)
        ):
            raise ValueError("H10 learner requires critic_center_generator_action:true")
        super().__init__(mac, scheme, logger, args)
        self.critic_steps = 0
        self.actor_steps = 0
        self.target_steps = 0
        self.finite_updates = 0
        self.train_updates = 0
        self.critic_phase_guards = 0
        self.actor_phase_guards = 0
        self.critic_optimiser.register_step_post_hook(self._critic_post_step)
        self.agent_optimiser.register_step_post_hook(self._actor_post_step)

    def _make_critic(self, scheme, args):
        return CenteredGeneratorGAVDACCritic(scheme, args)

    @staticmethod
    def _finite_tensors(tensors):
        for value in tensors:
            if (value.device.type != "cpu" or value.dtype != torch.float32
                    or not torch.isfinite(value).all()):
                raise ValueError("H10 actual model/gradient/optimizer must be finite CPU float32")

    def _check_optimizer(self, optimizer, params):
        self._finite_tensors(params)
        self._finite_tensors(param.grad for param in params if param.grad is not None)
        self._finite_tensors(
            value for state in optimizer.state.values() for value in state.values()
            if isinstance(value, torch.Tensor)
        )

    def _critic_post_step(self, optimizer, unused_args, unused_kwargs):
        self._check_optimizer(optimizer, self.critic_params)
        self.critic_steps += 1

    def _actor_post_step(self, optimizer, unused_args, unused_kwargs):
        self._check_optimizer(optimizer, self.agent_params)
        self.actor_steps += 1

    def _critic_loss(self, batch, denominator=None):
        before = (self.critic.centered_forward_calls, self.target_critic.centered_forward_calls)
        result = super()._critic_loss(batch, denominator)
        after = (self.critic.centered_forward_calls, self.target_critic.centered_forward_calls)
        if after != (before[0] + 1, before[1] + 1):
            raise RuntimeError("H10 replay/target Critic paths must each center exactly once")
        self.critic_phase_guards += 1
        return result

    def _actor_loss(self, batch, actor_policy, critic_inputs):
        before = self.critic.centered_forward_calls
        loss = super()._actor_loss(batch, actor_policy, critic_inputs)
        if self.critic.centered_forward_calls != before + 1:
            raise RuntimeError("H10 Actor Critic path must center exactly once")
        self.actor_phase_guards += 1
        return loss

    def _check_targets(self):
        self._finite_tensors(self.target_mac.parameters())
        self._finite_tensors(self.target_critic.parameters())
        if self.mixer is not None:
            self._finite_tensors(self.target_mixer.parameters())
        self.target_steps += 1

    def _update_targets(self):
        super()._update_targets()
        self._check_targets()

    def _update_targets_soft(self, tau):
        super()._update_targets_soft(tau)
        self._check_targets()

    def train(self, batch, t_env, episode_num):
        before = (self.critic_steps, self.actor_steps, self.target_steps)
        super().train(batch, t_env, episode_num)
        if (self.critic_steps, self.actor_steps, self.target_steps) != tuple(
            value + 1 for value in before
        ):
            raise RuntimeError("H10 requires one Critic, Actor and target update per train")
        self.finite_updates += 1
        self.train_updates += 1

    def h10_diagnostics(self):
        self._check_optimizer(self.critic_optimiser, self.critic_params)
        self._check_optimizer(self.agent_optimiser, self.agent_params)
        for module in (self.target_mac.agent, self.target_critic, self.target_mixer):
            self._finite_tensors(module.parameters())
        return {
            "critic_center_generator_action": True,
            "critic_action_map": "[2*a0-1,a1]",
            "critic_steps": self.critic_steps, "actor_steps": self.actor_steps,
            "target_steps": self.target_steps, "finite_updates": self.finite_updates,
            "train_updates": self.train_updates,
            "parent_critic_phase_guards": self.critic_phase_guards,
            "parent_actor_phase_guards": self.actor_phase_guards,
            "mapping_guard_scope": "Loss guards execute in parent serial or workers; parent counters do not count worker calls",
            "learner_workers": getattr(self.args, "learner_workers", 0),
            "exact_resume_supported": False,
        }

    def save_models(self, path):
        super().save_models(path)
        with (Path(path) / "h10_diagnostics.json").open("x", encoding="utf-8") as stream:
            json.dump(self.h10_diagnostics(), stream, indent=2, allow_nan=False)
