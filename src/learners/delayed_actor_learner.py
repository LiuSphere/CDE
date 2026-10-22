"""H03 opt-in Actor cadence; Critic and target cadence remain unchanged."""

import json
from pathlib import Path

import torch

from learners.gavdac_learner import GAVDACLearner


class DelayedActorGAVDACLearner(GAVDACLearner):
    def __init__(self, mac, scheme, logger, args):
        interval = getattr(args, "actor_update_interval", 1)
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("actor_update_interval must be a positive integer")
        super().__init__(mac, scheme, logger, args)
        self.actor_update_interval = interval
        self.critic_steps = 0
        self.actor_steps = 0
        self.target_updates = 0
        self.finite_checks = 0

    def _check_finite(self, loss, params):
        if not torch.isfinite(loss).all() or any(
            param.grad is not None and not torch.isfinite(param.grad).all()
            for param in params
        ):
            raise ValueError("Non-finite H03 loss or gradient")

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

        mode = getattr(self.args, "target_update_mode", "hard")
        if mode == "hard":
            self._update_targets()
        elif mode in ("soft", "exponential_moving_average"):
            self._update_targets_soft(getattr(self.args, "target_update_tau", 0.001))
        else:
            raise ValueError(f"unknown target update mode: {mode}")
        self.target_updates += 1
        modules = [self.mac.agent, self.target_mac.agent, self.critic, self.target_critic]
        if self.mixer is not None:
            modules.extend([self.mixer, self.target_mixer])
        if any(not torch.isfinite(param).all()
               for module in modules for param in module.parameters()):
            raise ValueError("Non-finite H03 updated weight")
        self.finite_checks += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss/critic_loss", loss.item(), t_env)
            # A skipped Actor phase has no loss, not a synthetic zero.
            if pg_loss is not None:
                self.logger.log_stat("loss/policy_loss", pg_loss.item(), t_env)
            self.log_stats_t = t_env

    def write_diagnostics(self, path, **context):
        payload = {
            "actor_update_interval": self.actor_update_interval,
            "critic_optimizer_steps": self.critic_steps,
            "actor_optimizer_steps": self.actor_steps,
            "target_updates": self.target_updates,
            "finite_update_checks": self.finite_checks,
            "exact_resume_supported": False,
            **context,
        }
        with Path(path).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def save_models(self, path):
        super().save_models(path)
        self.write_diagnostics(Path(path) / "h03_diagnostics.json")
