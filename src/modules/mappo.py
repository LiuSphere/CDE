"""Shared local Gaussian actor and centralized value policy using SB3 PPO."""

from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

from envs.maenv import MGEnv


class LocalFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space["local"].shape[0])

    def forward(self, observations):
        return observations["local"]


class CentralFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space["central"].shape[0])

    def forward(self, observations):
        return observations["central"]


class SplitMLP(nn.Module):
    def __init__(self, local_dim, central_dim, width):
        super().__init__()
        self.latent_dim_pi = self.latent_dim_vf = width
        self.policy_net = nn.Sequential(
            nn.Linear(local_dim, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.value_net = nn.Sequential(
            nn.Linear(central_dim, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class CentralizedPolicy(ActorCriticPolicy):
    """No centralized tensor can reach the actor feature extractor."""

    def __init__(self, observation_space, action_space, lr_schedule, width=64, **kwargs):
        self.width = width
        super().__init__(
            observation_space, action_space, lr_schedule,
            features_extractor_class=LocalFeatures, share_features_extractor=False,
            net_arch=[], activation_fn=nn.Tanh, **kwargs)

    def _build_mlp_extractor(self):
        self.vf_features_extractor = CentralFeatures(self.observation_space)
        self.mlp_extractor = SplitMLP(
            self.observation_space["local"].shape[0],
            self.observation_space["central"].shape[0], self.width)


def observation_space(info, last_action=True, agent_id=True):
    local_dim = (info["obs_shape"] + (info["n_actions"] if last_action else 0)
                 + (info["n_agents"] if agent_id else 0))
    return spaces.Dict({
        "local": spaces.Box(-np.inf, np.inf, (local_dim,), dtype=np.float32),
        "central": spaces.Box(-np.inf, np.inf, (info["state_shape"],), dtype=np.float32),
    })


def policy_inputs(observations, state, previous, last_action=True, agent_id=True):
    observations = np.asarray(observations, dtype=np.float32)
    n = len(observations)
    local = [observations]
    if last_action:
        local.append(np.asarray(previous, dtype=np.float32))
    if agent_id:
        local.append(np.eye(n, dtype=np.float32))
    return {"local": np.concatenate(local, axis=-1),
            "central": np.broadcast_to(np.asarray(state, dtype=np.float32),
                                        (n, len(state))).copy()}


class MicrogridAgentVecEnv(VecEnv):
    """Three agent trajectories in ONE joint world, not three world replicas."""

    def __init__(self, config):
        self.config = deepcopy(config)
        self.env = MGEnv(deepcopy(config["env_args"]), config["name"])
        info = self.env.get_env_info()
        self.n_agents = info["n_agents"]
        self.last_action = config["obs_last_action"]
        self.agent_id = config["obs_agent_id"]
        self.previous = np.zeros((self.n_agents, info["n_actions"]), dtype=np.float32)
        self.pending_actions = None
        self.joint_steps = 0
        self.episodes = 0
        self.completed_stats = []
        self.render_mode = None
        for space in info["action_spaces"]:
            if (not isinstance(space, spaces.Box)
                    or not np.array_equal(space.low, info["action_spaces"][0].low)
                    or not np.array_equal(space.high, info["action_spaces"][0].high)):
                raise ValueError("MAPPO requires common continuous Box bounds")
        super().__init__(
            self.n_agents, observation_space(info, self.last_action, self.agent_id),
            info["action_spaces"][0])

    def _observation(self):
        return policy_inputs(self.env.get_obs(), self.env.get_state(), self.previous,
                             self.last_action, self.agent_id)

    def reset(self):
        # Only the first SB3 vector reset seeds the joint environment. Automatic
        # daily resets retain the same environment inventory/data progression.
        self.env.reset(seed=self._seeds[0])
        self._reset_seeds()
        self._reset_options()
        self.previous.fill(0)
        return self._observation()

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self.previous.shape or not np.isfinite(actions).all():
            raise ValueError("Invalid joint MAPPO action")
        self.pending_actions = actions.copy()

    def step_wait(self):
        if self.pending_actions is None:
            raise RuntimeError("step_async must precede step_wait")
        reward, done, info = self.env.step(self.pending_actions)
        self.previous = self.pending_actions.copy()
        self.pending_actions = None
        self.joint_steps += 1
        observation = self._observation()
        infos = [dict(info) for _ in range(self.n_agents)]
        if done:
            self.episodes += 1
            self.completed_stats.append(dict(info))
            for i in range(self.n_agents):
                infos[i].update(
                    terminal_observation={key: value[i].copy()
                                          for key, value in observation.items()},
                    **{"TimeLimit.truncated": bool(info["episode_limit"])})
            self.env.reset()
            self.previous.fill(0)
            observation = self._observation()
        return (observation, np.full(self.n_agents, reward, dtype=np.float32),
                np.full(self.n_agents, done, dtype=bool), infos)

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        if indices is not None:
            raise ValueError("Joint environment attributes cannot vary by agent")
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("Independent per-agent environment methods are not supported")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))


class MAPPOController:
    """EpisodeRunner-compatible deterministic inference adapter."""

    def __init__(self, scheme, groups, args, policy=None):
        self.args = args
        self.n_agents = args.n_agents
        self.agent = policy if policy is not None else CentralizedPolicy(
            observation_space(
                dict(obs_shape=args.obs_shape, n_actions=args.n_actions,
                     n_agents=args.n_agents, state_shape=args.state_shape),
                args.obs_last_action, args.obs_agent_id),
            args.action_spaces[0], lambda _: args.mappo_learning_rate,
            width=args.mappo_width, log_std_init=args.mappo_log_std_init)
        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = None

    @torch.no_grad()
    def select_actions(self, batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if not test_mode:
            raise ValueError("MAPPO training uses its on-policy vector environment")
        obs = batch["obs"][bs, t_ep]
        previous = (torch.zeros_like(batch["actions"][bs, 0]) if t_ep == 0
                    else batch["actions"][bs, t_ep - 1])
        parts = [obs]
        if self.args.obs_last_action:
            parts.append(previous)
        if self.args.obs_agent_id:
            parts.append(torch.eye(self.n_agents).expand(obs.shape[0], -1, -1))
        local = torch.cat(parts, dim=-1).reshape(-1, sum(p.shape[-1] for p in parts))
        central = batch["state"][bs, t_ep].repeat_interleave(self.n_agents, dim=0)
        distribution = self.agent.get_distribution({"local": local, "central": central})
        actions = distribution.mode().reshape(obs.shape[0], self.n_agents, -1)
        low = torch.as_tensor(self.args.action_spaces[0].low)
        high = torch.as_tensor(self.args.action_spaces[0].high)
        return actions.clamp(low, high)

    def save_models(self, path):
        torch.save(self.agent.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        self.agent.load_state_dict(torch.load(
            f"{path}/agent.th", map_location="cpu", weights_only=True))
        self.agent.eval()
