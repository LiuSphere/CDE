# MAPPO Baseline

This is an independent feed-forward parameter-sharing MAPPO baseline:
local Gaussian actor, centralized state-value critic, common team reward,
on-policy GAE and clipped PPO updates. It is not a renamed GAVDAC learner,
an independent-local-critic IPPO, or the retired desktop SB3 environment.

Primary references:

- Yu et al., NeurIPS2022, "The Surprising Effectiveness of PPO in Cooperative,
  Multi-Agent Games", https://arxiv.org/abs/2103.01955 .
- PPO/GAE/distribution core: installed stable-baselines3 2.9.0, pinned by
  uv.lock; official source https://github.com/DLR-RM/stable-baselines3/tree/v2.9.0 .

## Frozen Development Budget

Before any C3 training or holdout evaluation:

- One fixed configuration, no hyperparameter sweep or performance-based restart.
- Unit/integration checks using synthetic inputs and at most10000joint
  environment interactions per integration process; all are development only.
- One0.5Mjoint-interaction engineering run, rounded to complete on-policy
  rollout boundaries. Weights never initialize any formal seed.
- Failed checks retained; a material algorithm/interface/fairness problem pauses
  automation for user input instead of expanding the tuning budget.
- Only after acceptance, freeze code/config and launch three fresh6Mjoint-step
  formal seeds on their assigned original hosts.

## Information And Updates

Actor sees current local observation, own previous executed action, agent ID.
Critic sees current joint state only. Dimensions and bounds derive from MGEnv.
No agent sees future realized data; existing causal forecast features unchanged.
Actor and critic share parameters across agents, not with each other.
Feed-forward two64-unit Tanh layers replace recurrence for this independent
algorithm comparison; no claim to isolate architecture or match capacity.
Previous action resets daily, physical SOC follows MGEnv progression.

One joint environment is exposed as one trajectory per agent to the SB3
on-policy buffer: all agents step simultaneously with the same team reward.
Three agent samples are NOT three independent environment interactions.
SB3's sample counter must be divided by n_agents for budget/evaluation/ETA.
The PPO likelihood is per-agent, summed over the two action coordinates,
not a product ratio across all agents.

Use the existing SB3 diagonal Gaussian with raw sampled-action log probabilities,
then clip only the executed action to its Box bounds, as SB3 PPO does.
Rollout retains raw samples for consistent importance ratios. Deterministic
evaluation clips the Gaussian mean; critic input cannot reach that mean.
No invented tanh correction for an unclipped Gaussian, no silent distribution
fallback. Daily episode_limit is a truncation: terminal observation preserved,
SB3 bootstraps V(terminal), and GAE does not leak across the reset boundary.

YAML is authoritative for fixed numeric options. Initial choice:240joint steps
(ten days) per rollout, three agent trajectories, five full-batch PPO epochs,
no independent minibatch mixing and no value/observation/reward normalization.
The lack of value normalization and recurrent memory are disclosed departures
from some published MAPPO variants; this is not an exact reproduction of their
benchmark settings. Adam, advantage normalization and clipped value updates
are supplied by the existing core. No environment/reward changes.

## Acceptance

Tests must prove actor invariance to critic-only information and nonzero
critic dependence, finite and consistent sampled-action log probabilities,
ratio1before update, correct common reward and joint counters, truncation
bootstrapping/reset, previous-action and action-bounds agreement with the
EpisodeRunner, learner parameter updates, CPU float32, deterministic replay,
strict checkpoint round-trip and both continuous/daily-reset accounting.
Fixed validation selection rows731-738, initialSOC0.35, same10000joint-step
selection interval; no holdout until all actors/solver/rule protocols frozen.
Record actual joint steps, agent samples, updates, episodes, eval steps,
wall time and source/config/data identities. Inference bundle may contain
critic weights but evaluation actions never consume critic information.
