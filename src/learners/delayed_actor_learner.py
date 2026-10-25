"""Reconstructed H03 parent for H11; targets still update on every call."""

import json
from pathlib import Path

import torch

from learners.gavdac_learner import GAVDACLearner


def validate_update_intervals(args):
    actor = getattr(args, "actor_update_interval", 1)
    target = getattr(args, "target_update_interval", 1)
    for name, value in (("actor_update_interval", actor),
                        ("target_update_interval", target)):
        if type(value) is not int or value not in (1, 2):
            raise ValueError(f"{name} must be integer 1 or 2")
    if target == 2 and actor != 2:
        raise ValueError("target interval 2 requires actor interval 2")
    return actor, target


class DelayedActorGAVDACLearner(GAVDACLearner):
    target_update_interval = 1
    diagnostics_filename = "h11_parent_diagnostics.json"
    lineage_role = "H11_reconstructed_H03_parent"

    def __init__(self, mac, scheme, logger, args):
        actor, target = validate_update_intervals(args)
        if actor != 2 or target != self.target_update_interval:
            raise ValueError("Learner class does not match registered actor/target intervals")
        super().__init__(mac, scheme, logger, args)
        self.actor_update_interval = actor
        self.target_update_interval = target
        self.critic_steps = 0
        self.actor_steps = 0
        self.target_updates = 0
        self.finite_checks = 0
        self.optimizer_finite_checks = {"critic": 0, "actor": 0}
        self.critic_optimiser.register_step_post_hook(self._critic_post_step)
        self.agent_optimiser.register_step_post_hook(self._actor_post_step)

    def _post_step(self, phase, params):
        if any(p.dtype != torch.float32 or p.device.type != "cpu"
               or not torch.isfinite(p).all()
               or (p.grad is not None and (
                   p.grad.dtype != torch.float32 or p.grad.device.type != "cpu"
                   or not torch.isfinite(p.grad).all()))
               for p in params):
            raise ValueError(f"Non-finite or non-CPU-float32 H11 parent {phase} update")
        self.optimizer_finite_checks[phase] += 1

    def _critic_post_step(self, optimizer, args, kwargs):
        self._post_step("critic", self.critic_params)

    def _actor_post_step(self, optimizer, args, kwargs):
        self._post_step("actor", self.agent_params)

    def _check_finite(self, loss, params):
        if not torch.isfinite(loss).all() or any(
            param.grad is not None and not torch.isfinite(param.grad).all()
            for param in params
        ):
            raise ValueError("Non-finite H11 parent loss or gradient")

    def train(self, batch, t_env, episode_num):
        if getattr(self.args, "learner_workers", 0) and self.gradient_pool is None:
            from learners.gradient_pool import GradientPool
            self.gradient_pool = GradientPool(self, self.scheme)
        self.critic_optimiser.zero_grad(set_to_none=True)
        if self.gradient_pool is None:
            actor_policy, critic_inputs, loss = self._critic_loss(batch)
            loss.backward()
        else:
            loss = self.gradient_pool.backward(self, batch, "critic")
        self._check_finite(loss, self.critic_params)
        torch.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_steps += 1
        pg_loss = None

        if self.critic_steps % self.actor_update_interval == 0:
            self._set_critic_grad(False)
            try:
                self.agent_optimiser.zero_grad(set_to_none=True)
                if self.gradient_pool is None:
                    pg_loss = self._actor_loss(batch, actor_policy, critic_inputs)
                    pg_loss.backward()
                else:
                    pg_loss = self.gradient_pool.backward(self, batch, "actor")
            finally:
                self._set_critic_grad(True)
            self._check_finite(pg_loss, self.agent_params)
            torch.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()
            self.actor_steps += 1
        elif self.gradient_pool is not None:
            self.gradient_pool.discard_actor(batch)
        else:
            self.mac.hidden_states = self.mac.hidden_states.detach()
            del actor_policy, critic_inputs

        self._target_phase()
        modules = [self.mac.agent, self.target_mac.agent, self.critic, self.target_critic]
        if self.mixer is not None:
            modules.extend([self.mixer, self.target_mixer])
        if any(param.dtype != torch.float32 or param.device.type != "cpu"
               or not torch.isfinite(param).all()
               for module in modules for param in module.parameters()):
            raise ValueError("Non-finite or non-CPU-float32 H11 parent updated weight")
        self.finite_checks += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss/critic_loss", loss.item(), t_env)
            if pg_loss is not None:
                self.logger.log_stat("loss/policy_loss", pg_loss.item(), t_env)
            self.log_stats_t = t_env

    def _target_phase(self):
        # This is the original H03 block, exposed for the separately committed child.
        mode = getattr(self.args, "target_update_mode", "hard")
        if mode == "hard":
            self._update_targets()
        elif mode in ("soft", "exponential_moving_average"):
            self._update_targets_soft(getattr(self.args, "target_update_tau", 0.001))
        else:
            raise ValueError(f"unknown target update mode: {mode}")
        self.target_updates += 1

    def write_diagnostics(self, path, **context):
        payload = {
            "lineage_role": self.lineage_role,
            "actor_update_interval": self.actor_update_interval,
            "target_update_interval": self.target_update_interval,
            "target_update_tau": getattr(self.args, "target_update_tau", 0.001),
            "critic_optimizer_steps": self.critic_steps,
            "actor_optimizer_steps": self.actor_steps,
            "target_updates": self.target_updates,
            "finite_update_checks": self.finite_checks,
            "optimizer_post_step_finite_checks": self.optimizer_finite_checks,
            "train_updates": self.finite_checks,
            "exact_resume_supported": False,
            **context,
        }
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def save_models(self, path):
        super().save_models(path)
        self.write_diagnostics(Path(path) / self.diagnostics_filename)
