# VMAS FACMAC-EA corrected-metric tuning history (seed5 onward)

## Scope and authority

- This is the primary tuning record for `facmac_ea_lr_mid_seed5`, `facmac_ea_lr_mid_seed6`, and all later VMAS runs.
- Seed3/4 results are excluded from tuning decisions because the two runs did not use one consistent VMAS termination/truncation implementation. Do not use their reward or success values to select later hyperparameters.
- `experiment_tuning_history/vmas_facmac_ea.md` is a secondary source for old pre-correction runs and provisional learning-rate/action-noise priors only.
- Every eligible entry must identify the metric semantics, code revision, W&B run ID, configuration, seed, final `t_env`, and equal-budget metrics.

## Corrected VMAS metric contract

- `success_rate` is `float(terminated[0])`; it is success only when VMAS reports task termination.
- `episode_limit` is true only for pure truncation (`truncated=True` and `terminated=False`). A simultaneous task termination takes precedence over the time-limit flag.
- The runner ends an episode on `terminated or truncated`, while success analysis keeps the two meanings separate.
- Never compare corrected and pre-correction `success_rate` values directly.

## Tuning protocol

- First reproduce a known configuration with the corrected wrapper before changing hyperparameters.
- Keep environment, algorithm structure, training budget, replay settings, EA schedule, and seed fixed when establishing a corrected baseline.
- Compare `success_rate` with best/RL/EA rewards, `episode_limit`, contact rate, package displacement, and late-window variability.
- Use equal `t_env` budgets and multiple seeds before declaring a configuration better.
- Change one tuning dimension at a time. Do not infer a success improvement from reward alone.

## Seed5/6 starting point

- The generated configurations use `lr=2e-4`, `critic_lr=7e-5`, and `act_noise=0.05`; only the seed differs.
- Remote 715: `facmac_ea_lr_mid_seed5` (`seed=5`).
- Local: `facmac_ea_lr_mid_seed6` (`seed=6`).
- The corrected source and these configs are in commit `97fb00bcb8b99f0116688b1987bbcbbfb498f112`; future comparisons must verify the actual run used the intended code revision.
- These runs are not complete until W&B reports an explicit finished state and their metrics are recorded below.

## Future run log

Add one dated section per completed seed5/6-or-later comparison. Record:

- W&B run ID, run name, configuration, seed, code revision, and metric semantics.
- Final and equal-budget late-window `t_env`, best/RL/EA rewards, and variability.
- `success_rate`, pure-truncation `episode_limit`, contact rate, and package displacement.
- Which run was selected, why, remaining uncertainty, and the next configuration changes.

No eligible seed5-or-later training result has been recorded yet.
