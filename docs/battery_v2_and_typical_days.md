# Corrected Battery Model and Fixed-Day Policy Evaluation

Historical version note: this document describes `efficiency_balance_v2`.
The active branch now uses `heterogeneous_exchange_v2`; see
`docs/heterogeneous_microgrid_validation.md`. Statements below about "new runs"
apply only to that historical implementation, not the active branch.

## Scope and Version

New runs use `efficiency_balance_v2`. This is a new simulator version, not a
reproduction of run `4fkujsew`. Existing checkpoints, trajectories and CSV inputs
are not modified. The old paper's numbers remain historical and must not be
relabelled as corrected-model results.

The generator, grid and storage cost expressions are unchanged. The battery
constraint penalty now applies to the magnitude of the corrected power request,
using the same penalty coefficient as before. This necessarily changes rewards
when the requested battery action is infeasible.

Simulation conventions are explicit:

- `env_args.time_step_hours: 1.0`.
- Each agent has `battery_cap: 200.0`.
- Separate `battery_charge_max: 200.0` and `battery_discharge_max: 200.0`.
- `raw_ch` and `raw_dis` are charge and discharge efficiencies, both 0.95.
- Admissible SOC remains 0.2 to 0.8.

These values are modelling assumptions, not verified physical metadata for the
source CSV files. Ratings and capacity have different dimensions even when their
numerical values coincide. Do not label input data with kW/kWh without provenance.
Non-hourly steps are supported by the battery transition; the legacy monetary
cost expressions have not been redesigned for arbitrary step durations.

## Energy Balance

Battery power is measured on the grid side. Positive power is discharge;
negative power is charge. For capacity E, time interval dt and efficiencies:

```text
discharging: SOC_next = SOC - P * dt / (eta_dis * E)
charging:    SOC_next = SOC - eta_ch * P * dt / E    (P < 0)

maximum feasible discharge = (SOC - SOC_min) * E * eta_dis / dt
maximum feasible charge    = (SOC_max - SOC) * E / (eta_ch * dt)
```

The normalised action first maps to the separate rated charge/discharge power.
Power is then limited to what the current SOC permits. SOC is calculated from
that executed power using the same efficiency equation. Final SOC clipping only
removes numerical roundoff. Requested SOC and action-correction magnitude remain
logged for diagnosing infeasible policy requests.

## Selection Versus Final Evaluation

With a seeded row permutation, the first 80% remain training rows. The former
20% holdout is divided into separate validation and test subsets. Normalisation
is still the existing full-dataset normalisation; this change does not claim to
resolve preprocessing leakage.

Each periodic microgrid evaluation uses the same validation rows and initial SOC
for the RL actor and EA elite. Default selection uses the first 8 sorted
validation row indices, and `evaluation_initial_soc: 0.35`. Row indices are
zero-based CSV indices, not calendar dates.

- `validation_rows: []`: choose the default fixed validation rows.
- `typical_day_rows: []`: choose the first `fixed_eval_days` sorted test rows.
- Explicit row lists must belong to the relevant split and have no duplicates.
- Evaluation disables exploration and does not consume the training environment RNG.
- Best RL and EA inference policies are saved separately; EA saving starts only
  after at least one evolution cycle.
- Best means highest fixed-validation mean return, not lowest monetary cost.

Each improved candidate is immutable:

```text
results/models/<unique_token>/best_policies/
  source/                   exact source/configuration snapshot
  best_rl.json              pointer and metadata for the best RL policy
  best_ea.json              pointer and metadata for the best EA policy
  rl/<training_step>/        agent.th, config.yaml, metadata.json
  ea/<training_step>/        agent.th, config.yaml, metadata.json
```

The best-policy pointer updates only within the newly created run directory.
Previously saved candidates remain intact. Checkpoints contain policy weights,
not a complete resumable EA population or replay buffer.
`save_model=False` disables both regular and best-policy persistence.

Metadata includes validation rows/returns, initial SOC, policy source, training
step, battery version, source hashes, data hashes and the policy hash.
Final evaluation loads the saved configuration automatically and rejects
changed source, data, environment settings or policy weights.

## New Training

Only local disabled-W&B smoke tests have been run during implementation.
An example full-run command, not started automatically:

```powershell
uv run python src/direct_main.py --config facmac_ea --env-config microgrid `
  --override wandb_mode=disabled `
  --override unique_token=facmac_ea_battery_v2_seed0 `
  --override t_max=3000000
```

For the non-EA comparison, use `--config facmac_no_ea` and a different
`unique_token`, retaining the same seed, environment and fixed-day settings.
Never reuse a run/output id containing existing artifacts.

## Export Typical Days Without Retraining

Read the best EA pointer and run inference-only evaluation:

```powershell
$best = Get-Content results/models/facmac_ea_battery_v2_seed0/best_policies/best_ea.json -Raw | ConvertFrom-Json
uv run python src/direct_main.py --config facmac_ea --env-config microgrid `
  --override "policy_checkpoint=$($best.checkpoint)" `
  --override evaluate=True `
  --override wandb_mode=disabled `
  --override evaluation_output_id=facmac_ea_battery_v2_seed0_typical
```

Use `best_rl.json` for the matched non-EA run. By default both use the same first
8 sorted test rows when their seeds and splits match. The CLI exports the rows
in the current invocation; it does not select the most favourable day afterwards.
For a documented explicit test subset, add `--override "typical_day_rows=[...]"`.

Outputs:

```text
results/microgrid_trajs/<algorithm_name>/<evaluation_output_id>/
  episodes_test_*.csv
  evaluation.json
```

CSV records contain policy source, checkpoint step, battery model version,
time interval, row/column indices, initial/current SOC, requested/executed battery
power and correction magnitude, plus the existing dispatch and cost fields.
`evaluation.json` contains the test returns and the selection-validation record.
Do not call a smoke-test policy the final best experimental policy.

The existing oracle tools implement the older storage/cost formulations. They
have not been validated against v2 and must not be used as an equivalent physical
baseline without a separate update and validation.

## Checks

```powershell
uv run python tests/test_microgrid_battery.py
uv run python tests/test_policy_evaluation.py
uv run python tests/test_corrected_training.py
```

The last check uses the real entry point, crosses 512-episode warmup, asserts
actual learner updates, and completes EA cycles with generation counts 1 and 3.
It also exercises the non-EA condition and checks that no evolution is performed.

Implementation verification on 2026-10-14:

- Full unittest discovery: 11 tests passed.
- Generation 1: 8 real learner updates, 11 evolution calls.
- Generation 3: 8 real learner updates, 33 evolution calls.
- Non-EA: 5 real learner updates, no evolution calls.
- Persistent smoke run: `battery_v2_smoke_20261014_b`, 12,500-step budget.
- Saved EA candidate: `best_policies/ea/12144`, selected on validation rows 4 and 9.
- Inference export: `battery_v2_fixed_test_20261014_b`, test rows 1 and 3,
  initial SOC 0.35, 144 trajectory rows (2 days x 24 steps x 3 agents).
- Validation logs: `results/battery_v2_all_tests_20261014.log` and
  `results/battery_v2_fixed_test_20261014_b.log`.

No three-million-step formal experiment was started.
