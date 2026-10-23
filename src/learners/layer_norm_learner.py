"""Opt-in per-example Critic hidden LayerNorm; original targets and Actor path."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from learners.gavdac_learner import GAVDACLearner
from modules.critic import GAVDACCritic


def critic_layer_norm_enabled(args):
    value = getattr(args, "critic_layer_norm", False)
    if type(value) is not bool:
        raise ValueError("critic_layer_norm must be boolean")
    return value


def _check_tensor(value):
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError("H06 requires CPU float32")
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite H06 tensor")


class LayerNormGAVDACCritic(GAVDACCritic):
    def __init__(self, scheme, args):
        super().__init__(scheme, args)
        self.norm1 = nn.LayerNorm(args.rnn_hidden_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(args.rnn_hidden_dim, eps=1e-5)

    def forward(self, inputs, actions, hidden_state=None):
        _check_tensor(inputs)
        if actions is not None:
            _check_tensor(actions)
            inputs = torch.cat([
                inputs.view(-1, self.input_shape - self.n_actions),
                actions.contiguous().view(-1, self.n_actions),
            ], dim=-1)
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_shape:
            raise ValueError("Critic input must be flattened observations and actions")
        x = F.relu(self.norm1(self.fc1(inputs)))
        x = F.relu(self.norm2(self.fc2(x)))
        result = self.fc3(x)
        _check_tensor(result)
        return result, hidden_state


class LayerNormGAVDACLearner(GAVDACLearner):
    uses_population_training = False

    def __init__(self, mac, scheme, logger, args):
        self.critic_layer_norm = critic_layer_norm_enabled(args)
        if getattr(args, "use_cuda", False):
            raise ValueError("H06 requires CPU float32")
        super().__init__(mac, scheme, logger, args)
        self.critic_optimizer_steps = 0
        self.actor_optimizer_steps = 0
        self.target_updates = 0
        self.finite_update_checks = 0
        self.critic_optimiser.register_step_post_hook(self._critic_stepped)
        self.agent_optimiser.register_step_post_hook(self._actor_stepped)
        self._check_parameters(self.agent_params + self.critic_params)

    def _make_critic(self, scheme, args):
        if self.critic_layer_norm:
            return LayerNormGAVDACCritic(scheme, args)
        return super()._make_critic(scheme, args)

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
        super().train(batch, t_env, episode_num)
        if not self.critic_optimizer_steps == self.actor_optimizer_steps == self.target_updates:
            raise RuntimeError("H06 optimizer/target cadence mismatch")
        self.finite_update_checks += 1

    def diagnostics(self):
        return {
            "critic_layer_norm": self.critic_layer_norm, "layer_norm_eps": 1e-5,
            "layer_norm_hidden_layers": 2,
            "critic_optimizer_steps": self.critic_optimizer_steps,
            "actor_optimizer_steps": self.actor_optimizer_steps, "target_updates": self.target_updates,
            "finite_update_checks": self.finite_update_checks, "exact_resume_supported": False,
        }

    def save_models(self, path):
        super().save_models(path)
        (Path(path) / "h06_diagnostics.json").write_text(
            json.dumps(self.diagnostics(), indent=2, allow_nan=False), encoding="utf-8")

    def load_models(self, path):
        saved = json.loads((Path(path) / "h06_diagnostics.json").read_text(encoding="utf-8"))
        for key in ("critic_layer_norm", "layer_norm_eps", "layer_norm_hidden_layers", "exact_resume_supported"):
            if saved[key] != self.diagnostics()[key]:
                raise ValueError("H06 diagnostic configuration mismatch")
        keys = ("critic_optimizer_steps", "actor_optimizer_steps", "target_updates", "finite_update_checks")
        if any(type(saved[key]) is not int or saved[key] < 0 for key in keys) or len({saved[k] for k in keys}) != 1:
            raise ValueError("Invalid H06 successful-update counters")
        super().load_models(path)
        for key in keys:
            setattr(self, key, saved[key])
