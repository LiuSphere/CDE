"""Opt-in H02: clipped targets from two independently learned team critics."""

import copy
import json
from pathlib import Path

import torch as th
from torch.optim import Adam, RMSprop

from learners.gavdac_learner import GAVDACLearner
from modules.critic import GAVDACCritic
from modules.graph_mixer import GraphDec


class ClippedDoubleGAVDACLearner(GAVDACLearner):
    uses_population_training = False

    def __init__(self, mac, scheme, logger, args):
        if not getattr(args, "clipped_double_critic", False):
            raise ValueError("The H02 learner requires clipped_double_critic=True")
        if getattr(args, "use_cuda", False):
            raise ValueError("H02 is a CPU float32 experiment")
        if getattr(args, "learner_workers", 0):
            raise ValueError("H02 learner sharding is not supported; use learner_workers=0")
        super().__init__(mac, scheme, logger, args)
        self.critic2 = GAVDACCritic(scheme, args)
        self.mixer2 = GraphDec(args) if self.mixer is not None else None
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.target_mixer2 = copy.deepcopy(self.mixer2)
        self.critic2_params = list(self.critic2.parameters())
        if self.mixer2 is not None:
            self.critic2_params += list(self.mixer2.parameters())
        if getattr(args, "optimizer", "rmsprop") == "adam":
            self.critic2_optimiser = Adam(
                self.critic2_params, lr=args.critic_lr,
                eps=getattr(args, "optimizer_epsilon", 10e-8),
            )
        else:
            self.critic2_optimiser = RMSprop(
                self.critic2_params, lr=args.critic_lr,
                alpha=args.optim_alpha, eps=args.optim_eps,
            )
        self.optimizer_steps = {"actor": 0, "critic1": 0, "critic2": 0}
        self.finite_update_checks = 0
        self.last_update_metrics = {}

    def _team_values(self, critic, mixer, inputs, actions, states, obs, hidden, batch):
        critic.init_hidden(batch.batch_size)
        q, critic.hidden_states = critic(inputs, actions, critic.hidden_states)
        if mixer is not None:
            return self._mix_q_values(mixer, q, states, agent_obs=obs, hidden_states=hidden)
        return q.view(batch.batch_size, actions.shape[1], self.n_agents)

    @th.no_grad()
    def _target_team_values(self, batch, target_actions, target_hidden):
        inputs = self._build_inputs_sequence(batch, start=1)
        values = [
            self._team_values(
                critic, mixer, inputs, target_actions[:, 1:],
                batch["state"][:, 1:], batch["obs"][:, 1:],
                target_hidden[:, 1:], batch,
            )
            for critic, mixer in (
                (self.target_critic, self.target_mixer),
                (self.target_critic2, self.target_mixer2),
            )
        ]
        return th.minimum(values[0], values[1])

    @staticmethod
    def _critic_loss(q, targets, mask):
        expanded = mask.expand_as(q)
        return (((targets - q) * expanded) ** 2).sum() / expanded.sum()

    def _step(self, loss, optimiser, params, name):
        if not th.isfinite(loss):
            raise FloatingPointError(f"H02 non-finite {name} loss")
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip, error_if_nonfinite=True)
        optimiser.step()
        if any(not th.isfinite(p).all() for p in params):
            raise FloatingPointError(f"H02 non-finite {name} parameter")
        self.optimizer_steps[name] += 1

    def train(self, batch, t_env, episode_num):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        with th.no_grad():
            target_policy = self.target_mac.forward_sequence(batch)
            target_actions = self.target_mac._clamp_actions_to_space(target_policy["actions"])
            target_q = self._target_team_values(
                batch, target_actions, target_policy["hidden_states"],
            )
            targets = rewards.expand_as(target_q) + self.args.gamma * (
                1 - terminated.expand_as(target_q)
            ) * target_q

        # Retain the Actor graph across both critic steps; the critics only
        # receive detached context, preserving the baseline's gradient boundary.
        policy = self.mac.forward_sequence(batch, end=-1)
        inputs = self._build_inputs_sequence(batch, end=-1)
        losses = []
        for critic, mixer, optimiser, params, name in (
            (self.critic, self.mixer, self.critic_optimiser, self.critic_params, "critic1"),
            (self.critic2, self.mixer2, self.critic2_optimiser, self.critic2_params, "critic2"),
        ):
            q = self._team_values(
                critic, mixer, inputs, actions.detach(), batch["state"][:, :-1],
                batch["obs"][:, :-1], policy["hidden_states"].detach(), batch,
            )
            loss = self._critic_loss(q, targets, mask)
            self._step(loss, optimiser, params, name)
            losses.append(loss.item())

        self._set_critic_grad(False)
        try:
            policy_q = self._team_values(
                self.critic, self.mixer, inputs, policy["actions"],
                batch["state"][:, :-1], batch["obs"][:, :-1],
                policy["hidden_states"], batch,
            )
            policy_loss = -policy_q.mean() + policy["raw_actions"].square().mean() * 1e-3
            self._step(policy_loss, self.agent_optimiser, self.agent_params, "actor")
        finally:
            self._set_critic_grad(True)
        mode = getattr(self.args, "target_update_mode", "hard")
        if mode == "hard":
            self._update_targets()
        elif mode in ("soft", "exponential_moving_average"):
            self._update_targets_soft(getattr(self.args, "target_update_tau", 0.001))
        else:
            raise ValueError(f"Unknown H02 target update mode: {mode}")
        self.finite_update_checks += 1
        self.last_update_metrics = {
            "critic1_loss": losses[0], "critic2_loss": losses[1],
            "policy_loss": policy_loss.item(),
        }
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for name, value in (
                ("loss/critic_loss", losses[0]), ("loss/critic2_loss", losses[1]),
                ("loss/policy_loss", policy_loss.item()),
            ):
                self.logger.log_stat(name, value, t_env)
            self.log_stats_t = t_env

    def _set_critic_grad(self, enabled):
        super()._set_critic_grad(enabled)
        for param in self.critic2_params:
            param.requires_grad_(enabled)

    def _update_targets(self):
        super()._update_targets()
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        if self.mixer2 is not None:
            self.target_mixer2.load_state_dict(self.mixer2.state_dict())

    def _update_targets_soft(self, tau):
        super()._update_targets_soft(tau)
        with th.no_grad():
            for target, online in zip(
                self.target_critic2.parameters(), self.critic2.parameters(), strict=True,
            ):
                target.lerp_(online, tau)
            if self.mixer2 is not None:
                for target, online in zip(
                    self.target_mixer2.parameters(), self.mixer2.parameters(), strict=True,
                ):
                    target.lerp_(online, tau)

    def completion_record(self, joint_steps, episodes, ea_cycles):
        return {
            "algorithm": "H02 clipped double mixed-Q targets",
            "actual_joint_interactions": joint_steps,
            "agent_transition_samples": joint_steps * self.n_agents,
            "budget_overshoot": joint_steps - self.args.t_max,
            "outer_episodes": episodes, "ea_cycles": ea_cycles,
            "optimizer_steps": dict(self.optimizer_steps),
            "total_optimizer_steps": sum(self.optimizer_steps.values()),
            "finite_update_checks": self.finite_update_checks,
            "last_update_metrics": dict(self.last_update_metrics),
            "parameter_counts": {
                "actor": sum(p.numel() for p in self.agent_params),
                "critic1_including_mixer": sum(p.numel() for p in self.critic_params),
                "critic2_including_mixer": sum(p.numel() for p in self.critic2_params),
            },
            "device": "cpu", "dtype": "float32",
            "resume_limitation": "Learner state only; no environment/replay/RNG exact-resume claim",
        }

    def save_models(self, path):
        super().save_models(path)
        th.save({
            "critic1": self.critic.state_dict(), "critic2": self.critic2.state_dict(),
            "mixer2": self.mixer2.state_dict() if self.mixer2 is not None else None,
            "target_actor": self.target_mac.agent.state_dict(),
            "target_critic1": self.target_critic.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "target_mixer1": self.target_mixer.state_dict() if self.mixer is not None else None,
            "target_mixer2": self.target_mixer2.state_dict() if self.mixer2 is not None else None,
            "critic1_optimizer": self.critic_optimiser.state_dict(),
            "critic2_optimizer": self.critic2_optimiser.state_dict(),
            "optimizer_steps": self.optimizer_steps,
            "finite_update_checks": self.finite_update_checks,
            "last_update_metrics": self.last_update_metrics,
        }, Path(path) / "h02_learner.th")

    def load_models(self, path):
        state = th.load(Path(path) / "h02_learner.th", map_location="cpu", weights_only=True)
        super().load_models(path)
        for module, name in (
            (self.critic, "critic1"), (self.critic2, "critic2"),
            (self.target_mac.agent, "target_actor"),
            (self.target_critic, "target_critic1"), (self.target_critic2, "target_critic2"),
            (self.mixer2, "mixer2"), (self.target_mixer if self.mixer is not None else None, "target_mixer1"),
            (self.target_mixer2, "target_mixer2"),
        ):
            if module is not None:
                module.load_state_dict(state[name])
        self.critic_optimiser.load_state_dict(state["critic1_optimizer"])
        self.critic2_optimiser.load_state_dict(state["critic2_optimizer"])
        self.optimizer_steps = dict(state["optimizer_steps"])
        self.finite_update_checks = state["finite_update_checks"]
        self.last_update_metrics = dict(state["last_update_metrics"])

    def write_completion(self, directory, joint_steps, episodes, ea_cycles):
        with (Path(directory) / "completion_checks.json").open("x", encoding="utf-8") as file:
            json.dump(self.completion_record(joint_steps, episodes, ea_cycles),
                      file, indent=2, allow_nan=False)

    def cuda(self, device="cuda:0"):
        raise ValueError("H02 is a CPU float32 experiment")
