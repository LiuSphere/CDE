# FACMAC Experiment Status

Updated 2026-10-18. The user selected v03 as the default and stopped the
cross-version tuning campaign. No heartbeat or automatic training remains active.

- v03 seed0 reached 6,000,144 steps; original final reporting failed from a
  Windows long path. The explicit recovery at `results/ftrec03b` is retained.
- v03 seed1 completed normally at `results/ft03s1a`.
- v03 seed2 was stopped by the user at `results/ft03s2a`; it is incomplete.
- No confirmed three-seed result or algorithmic upper limit is claimed.
- Non-v03 local result artifacts were deleted by user request.

The current retained-run manifest is
`results/facmac_tuning/campaign_20261017_a/state.json`.
The paper comparison plan is `docs/paper_experiment_plan.md`.

The old protocol is preserved as historical text at
`docs/archive/facmac_tuning_campaign.md`; its queue, commands and old result
paths must not be interpreted as current instructions.
