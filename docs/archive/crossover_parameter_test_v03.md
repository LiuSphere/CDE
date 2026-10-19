# Reduced Crossover Parameter Test

Historical experiment record: the user selected the crossover-free
`gavdac_ea_main` on 2026-10-20. The 0.25 candidate is not the selected method.
Seed0 completed; seed1 was not started and its continuation was cancelled.
Heartbeat `0-25` was deleted; all experiment artifacts remain unchanged.
The protocol below records the original experiment, not permission to resume.

Authorized on 2026-10-19: change parameters to reduce possible crossover harm
and test the change. This is a new exploratory parameter test, not an automatic
resumption or alteration of the frozen paper matrix.

## Candidate

- Parent: `facmac_ea_tune_v03_long`.
- Candidate: `facmac_v03_crossover025`.
- Only behavioral parameter change: `crossover_prob: 1.0 -> 0.25`.
- The production crossover is extra whole-shared-actor clone recombination,
  not parameter mixing. Lowering its probability retains more distinct parents
  before mutation without changing elite protection or mutation.
- The candidate value is a hypothesis, not an estimated optimum. No change to
  environment, reward, inputs, network, RNG protocol or learning intensity.
- v03 remains the entry-point default. Existing YAMLs stay unchanged.

## Preflight

`tests/test_crossover_tuning.py` loads the actual merged entry configuration,
asserts that only name/probability differ, and exercises the actual SSNE epoch.
With fixed distinct child parents, mutation disabled and 100 evenly spaced
gate draws, identical child pairs are 0, 25 and 100 for probabilities 0, .25
and 1 respectively. This is a controlled mechanism check, not a claim about
the final training population's measured diversity.

The real CPU smoke crosses the original 512-episode replay warmup, checks
finite model parameters, and requires both learner updates and EA cycles.
Logs and frozen source/config/input hashes are under `results/cx25check/`.

## Training And Evaluation

Run candidate seeds0 and1 sequentially for 6M each, including EA
rollouts, using the existing CPU/online aggregate-only W&B protocol. Keep
checkpoint validation rows731-738 and all six candidate validation rows
740,766,808,862,880,904. Draw representative trajectories only after normal
training completion. Do not examine the retrospective holdout to tune.

Use the most recent parent seed0 `results/v03r0a` (raw six-day mean full_cost
1931.878822), not the best historical seed0. The no-crossover seed0 `pnc0a`
is an endpoint reference, not a reason to select a favorable parent.

After completion, verify budget, normal shutdown, W&B finish and final reports.
Run the existing complete analyzer:

```powershell
uv run python tools/analyze_gavdac_tuning.py --run-dir results/cx25s0b --output results/cx25check/cx25s0b_analysis.json
uv run python tools/analyze_gavdac_tuning.py --run-dir results/cx25s1a --output results/cx25check/cx25s1a_analysis.json
```

Compare validation curves, all six daily costs, monetary economic_cost
separately from full_cost, worst day, physical SOC/power/balance constraints,
and battery charging/discharging. Treat the absolute terminal-SOC inventory
correction as a diagnostic, not real salvage or the primary economic objective.
A >=1% raw mean-cost improvement without degraded constraint quality is an
exploratory positive signal, not proof. Do not select per-day RL/EA winners.
Use the validation-selected RL actor as primary and EA separately.

Two seeds remain exploratory: a promising candidate needs matching
seed2 and continuous-SOC evaluation before promotion. Existing Python,
fastrand and Gym sampling are not fully seeded; no perfectly paired claim.

## Existing Paused Work

The user explicitly stopped old experiments after the initial pause.
`pnc1a` and its descendants have been terminated with identity checks. All
artifacts remain, but no in-memory resume is possible and it is not complete.
The old paper heartbeat stays PAUSED and its queue must not resume.

The initial candidate attempt cx25s0a failed W&B initialization before training,
and its evidence is preserved. After user-authorized cleanup, the minimal
online probe cxwb02 successfully initialized/logged/finished in 15.485 seconds.
Use new directories cx25s0b and cx25s1a; do not overwrite the failure.
Run one computing training at a time. Separate finite heartbeat `0-25`
inspects completion near ETA and launches only the second authorized seed.
The old paper heartbeat remains PAUSED. Do not add further
seeds or tuning rounds. Stop and report any new startup failure; no unlimited
retries or automatic offline fallback.
