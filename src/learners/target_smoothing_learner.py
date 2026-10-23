"""Opt-in target-action smoothing, with one private parent draw per train batch."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from components.episode_buffer import EpisodeBatch
from learners.gavdac_learner import GAVDACLearner

NOISE_FIELD = "_h07_target_epsilon"


def target_policy_smoothing_enabled(args):
    value = getattr(args, "target_policy_smoothing", False)
    if type(value) is not bool:
        raise ValueError("target_policy_smoothing must be boolean")
    return value


def _check_tensor(value):
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError("H07 requires CPU float32")
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite H07 tensor")


class TargetSmoothedGAVDACLearner(GAVDACLearner):
    uses_population_training = False

    def __init__(self, mac, scheme, logger, args):
        self.target_policy_smoothing = target_policy_smoothing_enabled(args)
        if getattr(args, "use_cuda", False):
            raise ValueError("H07 requires CPU float32")
        if type(getattr(args, "seed", None)) is not int or args.seed < 0:
            raise ValueError("H07 requires a nonnegative integer seed")
        super().__init__(mac, scheme, logger, args)
        self.noise_generator = (
            torch.Generator(device="cpu").manual_seed(args.seed) if self.target_policy_smoothing else None
        )
        self.critic_optimizer_steps = 0
        self.actor_optimizer_steps = 0
        self.target_updates = 0
        self.finite_update_checks = 0
        self.target_noise_batches = 0
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

    def _prepare_batch(self, batch):
        if NOISE_FIELD in batch.data.transition_data or NOISE_FIELD in batch.data.episode_data:
            raise ValueError("H07 private noise field already exists")
        actions = batch["actions"]
        _check_tensor(actions)
        if actions.shape != (batch.batch_size, batch.max_seq_length, self.n_agents, self.n_actions):
            raise ValueError("H07 actions must have B x T x N x A shape")
        noise = torch.randn(actions.shape, generator=self.noise_generator, dtype=torch.float32)
        noise.mul_(0.2).clamp_(-0.5, 0.5)
        self.target_noise_batches += 1
        # New dictionary only; neither replay nor its existing tensors are modified.
        data = SimpleNamespace(
            transition_data={**batch.data.transition_data, NOISE_FIELD: noise},
            episode_data=batch.data.episode_data,
        )
        return EpisodeBatch(batch.scheme, batch.groups, batch.batch_size, batch.max_seq_length,
                            data=data, device=batch.device, out_device=batch.out_device)

    def _target_actions(self, batch, target_policy):
        actions = super()._target_actions(batch, target_policy)
        if not self.target_policy_smoothing:
            return actions
        _check_tensor(actions)
        if NOISE_FIELD not in batch.data.transition_data:
            raise ValueError("H07 target actions require parent-generated noise")
        noise = batch[NOISE_FIELD]
        _check_tensor(noise)
        if noise.shape != actions.shape or (noise.abs() > 0.5).any():
            raise ValueError("H07 noise shape or clipped range invalid")
        low, high = self.target_mac._get_action_bounds(actions)
        _check_tensor(low)
        _check_tensor(high)
        if (high <= low).any():
            raise ValueError("H07 requires positive finite Box ranges")
        smoothed = actions + noise * ((high - low) * 0.5)
        result = self.target_mac._clamp_actions_to_space(smoothed)
        _check_tensor(result)
        return result

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
        params = list(self.target_mac.parameters()) + list(self.target_critic.parameters())
        if self.mixer is not None:
            params += list(self.target_mixer.parameters())
        self._check_parameters(params)

    def _update_targets_soft(self, tau):
        super()._update_targets_soft(tau)
        self._targets_updated()

    def _update_targets(self):
        super()._update_targets()
        self._targets_updated()

    def train(self, batch, t_env, episode_num):
        prepared = self._prepare_batch(batch) if self.target_policy_smoothing else batch
        super().train(prepared, t_env, episode_num)
        if not self.critic_optimizer_steps == self.actor_optimizer_steps == self.target_updates:
            raise RuntimeError("H07 optimizer/target cadence mismatch")
        self.finite_update_checks += 1
        if self.target_policy_smoothing and self.target_noise_batches != self.finite_update_checks:
            raise RuntimeError("H07 must draw once per complete parent update")

    def diagnostics(self):
        return {
            "target_policy_smoothing": self.target_policy_smoothing,
            "normalized_target_noise_std": 0.2, "normalized_target_noise_clip": 0.5,
            "target_noise_seed": self.args.seed, "target_noise_batches": self.target_noise_batches,
            "critic_optimizer_steps": self.critic_optimizer_steps,
            "actor_optimizer_steps": self.actor_optimizer_steps, "target_updates": self.target_updates,
            "finite_update_checks": self.finite_update_checks, "exact_resume_supported": False,
        }

    def save_models(self, path):
        super().save_models(path)
        (Path(path) / "h07_diagnostics.json").write_text(
            json.dumps(self.diagnostics(), indent=2, allow_nan=False), encoding="utf-8")

    def load_models(self, path):
        saved = json.loads((Path(path) / "h07_diagnostics.json").read_text(encoding="utf-8"))
        for key in ("target_policy_smoothing", "normalized_target_noise_std",
                    "normalized_target_noise_clip", "target_noise_seed", "exact_resume_supported"):
            if saved[key] != self.diagnostics()[key]:
                raise ValueError("H07 diagnostic configuration mismatch")
        keys = ("critic_optimizer_steps", "actor_optimizer_steps", "target_updates", "finite_update_checks")
        if any(type(saved[key]) is not int or saved[key] < 0 for key in keys) or len({saved[k] for k in keys}) != 1:
            raise ValueError("Invalid H07 successful-update counters")
        if type(saved["target_noise_batches"]) is not int or saved["target_noise_batches"] != (
            saved["finite_update_checks"] if self.target_policy_smoothing else 0
        ):
            raise ValueError("Invalid H07 noise-batch counter")
        super().load_models(path)
        for key in (*keys, "target_noise_batches"):
            setattr(self, key, saved[key])
