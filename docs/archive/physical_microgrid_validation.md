# Physical Microgrid Model and Validation

> Historical physical-v1 results. The active environment is now
> `facmac_dispatch_v3`; see `facmac_environment_v3.md`.
> Commands below describe the earlier tool revision, not the current CLI.

## Provenance and Limits

Branch: `codex/microgrid-physical-env`.
Pre-change reference: tag `maenv-sep-v1-pre-physical-20261017`.
New environment version: `physical_dispatch_v1`.

The source CSV files are unchanged. kW, kWh and CU (scenario currency units)
are explicit modelling assumptions, not verified source metadata or a real
plant calibration. `price_scale=0.01` is a declared scenario conversion,
not a claim that the source tariff was definitely cents per kWh.
`pv_scale=4` represents an installation-size scenario applied at load time.

The original failure was reproduced on row 927: a battery discharged only
0.023 model units over the day, or 0.0383% of its usable energy.
The old high-SOC polynomial and discontinuous 70% positive-discharge discount
are removed. Physical power enters settlement; scaling is observation-only.
Charge efficiency multiplies stored energy; discharge efficiency divides
energy removed. Power ratings are separate from energy capacity.
Idle storage is free, while throughput degradation and losses are positive.

References for the modelling principles, not exact scenario parameter values:

- PyPSA storage energy balance and cyclic storage:
  https://docs.pypsa.org/v1.4.0/user-guide/optimization/storage/
- SAM cycle degradation penalties:
  https://samrepo.nlr.gov/help/battery_dispatch_fom.html
- NREL 2024 ATB, separate storage power/energy ratings, 2-10 hour durations,
  approximately one cycle/day, and its 85% round-trip reference assumption:
  https://atb.nrel.gov/electricity/2024/2023/utility-scale_battery_storage

Our scenario uses 4-hour nameplate duration, a 60% usable SOC window, and
0.93 efficiency on each side (86.49% round trip). The degradation coefficient,
tariff conversion, generator costs and emission-cost proxy are scenario inputs;
they are not estimated from those references.

## Calibration

Command:

```powershell
uv run --no-sync .venv/Scripts/python.exe -B tools/run_physical_microgrid_milp.py `
  --calibrate --output-id <new-unused-id>
```

Ward clustering followed by within-cluster squared-distance medoids selects
validation rows 740, 766, 808, 862, 880 and 904. The 182 validation rows are
disjoint from all test rows. Clustering uses input profiles, not policy reward.
Price profiles are identical between days, so distances reflect load and PV.

Only PV installation scale changes between the three physical candidates:

| PV Scale | Decision | Reason |
| --- | --- | --- |
| 1 | Reject | Mean validation solar contribution is below the 20% design target |
| 4 | Accept | All battery, generator, grid, solar-utilization and cyclic checks pass |
| 6 | Reject | Some validation days spill more than 10% of available solar |

Calibration record:
`results/oracle_milp/physical_calibration_20261017_b/calibration.json`.
An earlier `_a` attempt stopped on a JSON serialization error before producing
a completed calibration; it is not used as successful evidence.

The 20% solar-contribution installation-design target is validation-only.
Test weather is not forced to deliver a fixed resource share. Actual
availability utilization, all storage checks, generator/grid participation,
and battery economic benefit remain hard checks on every test day/agent.

## Locked-Parameter Evaluation

```powershell
uv run --no-sync .venv/Scripts/python.exe -B tools/run_physical_microgrid_milp.py `
  --all-test-days --output-id <new-unused-id>
```

Completed output: `results/oracle_milp/physical_all_test_20261017_a/`.
All 182 days, rows 913-1094, were solved with and without batteries.
Both cases re-optimize generators and grid import. There are 1,092
agent/day MILPs and 13,104 optimized replay records.
All solver results reached the requested gap, and every acceptance check passed.

| Metric Across 546 Agent-Days | Minimum | Mean | Maximum |
| --- | ---: | ---: | ---: |
| Fraction of usable SOC span covered | 1.00 | 1.00 | 1.00 |
| Discharge / usable AC-side energy | 1.00 | 1.3503 | 2.00 |
| Meaningful charge hours/day | 3 | 6.403 | 10 |
| Meaningful discharge hours/day | 3 | 4.973 | 7 |
| Generator active hours/day | 15 | 16 | 17 |
| Generator energy / load | 26.94% | 32.30% | 41.33% |
| Grid import / load | 39.18% | 48.97% | 67.71% |
| Available PV / load | 4.93% | 19.51% | 30.84% |
| Available PV utilized | 100% | 100% | 100% |

The discharge ratio uses the 20%-80% usable energy window times discharge
efficiency. It is not the number of full nameplate-capacity cycles.
Generator activity requires output above 5% of its rating; battery activity
requires power above 1% of its corresponding rating.

Daily full-cost reduction relative to the re-optimized no-battery case:
minimum 3.58%, mean of daily ratios 5.92%, maximum 9.12%.
Mean scenario full cost is 1984.5278 CU/day with storage and
2104.0864 CU/day without storage. These are scenario costs, not measured bills.

Maximum solver/replay discrepancy is below 5.5e-11.
The maximum observed power-balance residual is below 1.2e-13.
SOC, charge/discharge power limits, cyclic terminal SOC and PWL error
certificates all pass. Numerical clipping is below 5.4e-11 power units.

MILP minimizes purchase + generation + throughput + emission cost.
Feasible actions have zero request penalty, matching the environment's full
reward apart from its documented scalar scaling and float32 averaging.
The reference enforces terminal SOC equal to initial SOC (0.35), so it does
not borrow free initial energy. The training environment instead continues
SOC across days; this cyclic reference is not claimed to be a global upper
bound on arbitrary non-cyclic RL rewards.

## Software Checks

- Physical regression tests were run red before changing the environment.
- Thirty-two focused unit tests pass, including physical conservation, invalid
  inputs, saturation, policy checkpoint/replay and historical solver tests.
- A flat-price cyclic MILP test confirms that wasteful battery cycling is
  not rewarded.
- A real representative-day regression enforces device participation and
  storage economic benefit; a prohibitive-degradation counterexample confirms
  that the solver does not force uneconomic storage activity.
- The real training entry point crosses replay warmup and completes learner
  updates for EA generations 1, 3 and the no-EA comparison.
- Integration log: `results/physical_training_checks_20261017_a.log`.
- Observations/actions/state remain 4/2/12 for three agents, CPU-resident.
- No long-run RL/PPO retraining was performed; previous trained strategies
  are not results for this model version.

## Outputs and Reproduction

The evaluation directory contains representative dispatch/SOC PNGs, all
trajectories, per-agent and per-day metrics, the re-optimized no-battery
results, exact configuration snapshot, solver diagnostics and source/data
hashes. Output directories are exclusive and cannot overwrite previous runs.

The original sep MILP now rejects the physical environment explicitly.
The local numerical-search tool shares the pure environment transition and
remains a local search, not a certified optimum. The separate SB3 reference
tool still runs its historical external reference snapshot, not this new
physical environment.
