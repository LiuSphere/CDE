import copy

import torch as th
from torch.optim import Adam, RMSprop

from components.episode_buffer import EpisodeBatch
from modules.critic import GAVDACCritic
from modules.graph_mixer import GraphDec


class GAVDACLearner:
    def __init__(self, mac, scheme, logger, args):
        signed = getattr(args, "signed_mixer_hyperweights", False)
        if type(signed) is not bool:
            raise ValueError("signed_mixer_hyperweights must be boolean")
        if signed and not getattr(self, "signed_mixer_hyperweights", False):
            raise ValueError("signed_mixer_hyperweights requires its explicit H14 learner")
        huber = getattr(args, "scaled_huber_critic", False)
        if type(huber) is not bool:
            raise ValueError("scaled_huber_critic must be boolean")
        if huber and not getattr(self, "scaled_huber_critic", False):
            raise ValueError("scaled_huber_critic requires its explicit H12 learner")
        workers = getattr(args, "learner_workers", 0)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError("learner_workers must be a nonnegative integer")
        self.scheme = scheme
        self.gradient_pool = None
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = GAVDACCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.mixer = None
        if self.args.n_agents > 1:
            self.mixer = GraphDec(args)
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(
                params=self.agent_params,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(
                params=self.agent_params,
                lr=args.lr,
                eps=getattr(args, "optimizer_epsilon", 10e-8),
            )
        else:
            raise Exception(
                "unknown optimizer {}".format(
                    getattr(self.args, "optimizer", "rmsprop")
                )
            )

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(
                params=self.critic_params,
                lr=args.critic_lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(
                params=self.critic_params,
                lr=args.critic_lr,
                eps=getattr(args, "optimizer_epsilon", 10e-8),
            )
        else:
            raise Exception(
                "unknown optimizer {}".format(
                    getattr(self.args, "optimizer", "rmsprop")
                )
            )

        self.log_stats_t = -self.args.learner_log_interval - 1

    def _critic_loss(self, batch, denominator=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Train the critic batched
        with th.no_grad():
            target_policy = self.target_mac.forward_sequence(batch)
            target_actions = self.target_mac._clamp_actions_to_space(
                target_policy["actions"]
            )

        # Actor weights do not change during the critic update. Reuse its history,
        # detaching only the critic's view so its loss cannot enter the actor graph.
        actor_policy = self.mac.forward_sequence(batch, end=-1)
        critic_inputs = self._build_inputs_sequence(batch, end=-1)

        self.critic.init_hidden(batch.batch_size)
        critic_out, self.critic.hidden_states = self.critic(
            critic_inputs,
            actions.detach(),
            self.critic.hidden_states,
        )
        if self.mixer is not None:
            q_taken = self._mix_q_values(
                self.mixer,
                critic_out,
                batch["state"][:, :-1],
                agent_obs=batch["obs"][:, :-1],
                hidden_states=actor_policy["hidden_states"].detach(),
            )
        else:
            q_taken = critic_out.view(
                batch.batch_size, batch.max_seq_length - 1, self.n_agents
            )

        with th.no_grad():
            self.target_critic.init_hidden(batch.batch_size)
            target_critic_out, self.target_critic.hidden_states = self.target_critic(
                self._build_inputs_sequence(batch, start=1),
                target_actions[:, 1:],
                self.target_critic.hidden_states,
            )
            if self.mixer is not None:
                target_vals = self._mix_q_values(
                    self.target_mixer,
                    target_critic_out,
                    batch["state"][:, 1:],
                    agent_obs=batch["obs"][:, 1:],
                    hidden_states=target_policy["hidden_states"][:, 1:],
                )
            else:
                target_vals = target_critic_out.view(
                    batch.batch_size, batch.max_seq_length - 1, self.n_agents
                )

        targets = (
            rewards.expand_as(target_vals)
            + self.args.gamma * (1 - terminated.expand_as(target_vals)) * target_vals
        )
        td_error = targets.detach() - q_taken
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = self._td_error_loss(masked_td_error,
            mask.sum() if denominator is None else denominator
        )
        return actor_policy, critic_inputs, loss

    def _td_error_loss(self, masked_td_error, denominator):
        return (masked_td_error**2).sum() / denominator

    def _actor_loss(self, batch, actor_policy, critic_inputs):
        self.critic.init_hidden(batch.batch_size)
        chosen_action_qvals, self.critic.hidden_states = self.critic(
            critic_inputs,
            actor_policy["actions"],
            self.critic.hidden_states,
        )
        if self.mixer is not None:
            chosen_action_qvals = self._mix_q_values(
                self.mixer,
                chosen_action_qvals,
                batch["state"][:, :-1],
                agent_obs=batch["obs"][:, :-1],
                hidden_states=actor_policy["hidden_states"],
            )
        else:
            chosen_action_qvals = chosen_action_qvals.view(
                batch.batch_size, batch.max_seq_length - 1, self.n_agents
            )
        return (
            -chosen_action_qvals.mean()
            + actor_policy["raw_actions"].square().mean() * 1e-3
        )

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        if getattr(self.args, "learner_workers", 0) and self.gradient_pool is None:
            from learners.gradient_pool import GradientPool
            self.gradient_pool = GradientPool(self, self.scheme)
        self.critic_optimiser.zero_grad(set_to_none=True)
        if self.gradient_pool is None:
            actor_policy, critic_inputs, loss = self._critic_loss(batch)
            loss.backward()
        else:
            loss = self.gradient_pool.backward(self, batch, "critic")
        th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        # All critic shards finish and the parent steps before any actor gradient.
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
        th.nn.utils.clip_grad_norm_(
            self.agent_params, self.args.grad_norm_clip
        )
        self.agent_optimiser.step()

        if getattr(self.args, "target_update_mode", "hard") == "hard":
            self._update_targets()
        elif getattr(self.args, "target_update_mode", "hard") in [
            "soft",
            "exponential_moving_average",
        ]:
            self._update_targets_soft(
                tau=getattr(self.args, "target_update_tau", 0.001)
            )
        else:
            raise Exception(
                "unknown target update mode: {}!".format(
                    getattr(self.args, "target_update_mode", "hard")
                )
            )

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(
                "loss/critic_loss", loss.item(), t_env
            )
            self.logger.log_stat(
                "loss/policy_loss", pg_loss.item(), t_env
            )
            self.log_stats_t = t_env

    def close(self):
        if self.gradient_pool is not None:
            self.gradient_pool.close()
            self.gradient_pool = None

    def _update_targets_soft(self, tau):
        with th.no_grad():
            for target_param, param in zip(
                self.target_mac.parameters(), self.mac.parameters()
            ):
                target_param.lerp_(param, tau)

            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.lerp_(param, tau)

            if self.mixer is not None:
                for target_param, param in zip(
                    self.target_mixer.parameters(), self.mixer.parameters()
                ):
                    target_param.lerp_(param, tau)

    def _mix_q_values(
        self,
        mixer,
        agent_qs,
        states,
        agent_obs=None,
        hidden_states=None,
        team_rewards=None,
    ):
        if self.mixer is not None:
            q_tot, _, _ = mixer(
                agent_qs,
                states,
                agent_obs=agent_obs,
                team_rewards=team_rewards,
                hidden_states=hidden_states,
            )
            return q_tot
        return mixer(agent_qs, states)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        if self.args.recurrent_critic:
            # The individual Q conditions on the global action-observation history and individual action
            inputs.append(
                batch["obs"][:, t]
                .repeat(1, self.args.n_agents, 1)
                .view(bs, self.args.n_agents, -1)
            )
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(
                        th.zeros_like(
                            batch["actions"][:, t]
                            .repeat(1, self.args.n_agents, 1)
                            .view(bs, self.args.n_agents, -1)
                        )
                    )
                else:
                    inputs.append(
                        batch["actions"][:, t - 1]
                        .repeat(1, self.args.n_agents, 1)
                        .view(bs, self.args.n_agents, -1)
                    )
        else:
            inputs.append(batch["obs"][:, t])

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _build_inputs_sequence(self, batch, start=0, end=None):
        if self.args.recurrent_critic:
            stop = batch.max_seq_length if end is None else end % batch.max_seq_length
            inputs = [
                self._build_inputs(batch, t).view(
                    batch.batch_size, self.n_agents, -1
                )
                for t in range(start, stop)
            ]
            return th.stack(inputs, dim=1).reshape(-1, inputs[0].shape[-1])
        return batch["obs"][:, start:end].reshape(-1, batch["obs"].shape[-1])

    def _set_critic_grad(self, enabled):
        for param in self.critic_params:
            param.requires_grad_(enabled)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self, device="cuda:0"):
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.critic.cuda(device=device)
        self.target_critic.cuda(device=device)
        if self.mixer is not None:
            self.mixer.cuda(device=device)
            self.target_mixer.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.agent_optimiser.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage)
            )
        self.agent_optimiser.load_state_dict(
            th.load(f"{path}/opt.th", map_location=lambda storage, loc: storage)
        )
