# Historical FACMAC + EA Parameter Tuning Campaign

Archived on 2026-10-18. This protocol is no longer active and does not authorize
new runs. The user selected v03, stopped seed2 and deleted the heartbeat.
Non-v03 artifacts were deleted by request. Some paths below are historical
references only; removed configurations are available in the backup/Git history.

## Authorization And Scope

The user authorized autonomous local parameter tuning, sequential training runs
and thread wakeups on 2026-10-17. Use `uv`, CPU and `src/direct_main.py`.
Do not change environment parameters, rewards, source CSVs, algorithm structure
or update-intensity controls as part of this campaign. No commits, tags, pushes,
new threads, paid services, or unrelated automations are authorized.
Keep W&B aggregate-only during training; dispatch plots only at normal completion.
Keep each version as a new YAML file, never rewrite a version already run.
Never overwrite or delete an earlier run.
Use short new run directories/tokens, e.g. `results/ft04s0a` / `ft04s0a`.
The first 6M run completed its budget but failed final reporting because repeated
long run names exceeded Windows path limits. It was recovered using identical
source/data and saved actors at `results/ftrec03b`, resumed W&B ID s9lv09ch.
Preserve the failure evidence. If a future report-only failure needs recovery,
use `tools/recover_facmac_showcase.py` with a new short output directory; never
silently declare the original process successful. The analysis tool accepts
`--recovery <recovery.json>` only with the explicit verified error and hashes.

## Baseline Evidence

Baseline: `facmac_ea_compact_v2`, seed 0, 3,000,144 actual environment steps,
run `results/facmac_ea_compact_v2_final_only_20261017_a`.
Exact evaluation returns are paired from stdout with evaluation steps in stderr.
Console "Recent Stats" returns are smoothed and must not replace exact returns.

- RL best validation return 16.77205 at 2,973,504 steps, within the final 1%.
  The last 0.5M-step mean was 14.18 versus 13.82 in the preceding 0.5M.
  This is not a stable convergence plateau, nor evidence of numerical divergence.
- EA best return 19.80377 at 50,304 steps, with later validation regressions.
  The early 0.5M-step standard deviation was 5.50; final 0.5M was 1.01.
- On all six fixed validation candidate days, RL cost was 2067.07 CU/day,
  ordinary savings 15.63%, inventory-adjusted savings 13.10%.
  EA cost was 2123.13 CU/day, ordinary savings 13.34%, adjusted 9.83%.
  Neither source had a day meeting the showcase cyclic-SOC qualification.
- Matching perfect-foresight cyclic MILP mean cost was 1832.77 CU/day,
  savings 25.19%. This is a reference, not a fair information-matched optimum.
- Baseline saved EA best actor has `rnn.weight_ih` max absolute weight 1847.75
  versus 4.12 for the RL best actor, and RMS 24.83 versus 0.344. This strengthens
  the excessive-mutation hypothesis, but is not proof of causal degradation,
  saturation or numeric divergence. Retained evidence:
  `results/facmac_tuning/campaign_20261017_a/baseline_weight_diagnostics.json`.

Version v03 changes only `t_max` from 3M to 6M. It restarts from scratch:
current checkpoints do not contain the full critic/replay/RNG/EA training state.
Seed 0 alone is not deterministic replay because Python/fastrand/Gym sampling
are not all seeded by the normal entry point. Do not claim a perfectly paired
comparison or patch this behavior silently.

## Experiment Decisions

State and run manifests live in `results/facmac_tuning/campaign_20261017_a/`.
Read `state.json` on every wakeup. Record conclusions and next experiment
before launching it. Only one campaign training run may be active at a time.
Preserve the original 8-day checkpoint-selection validation set and all six
showcase candidate days for longitudinal comparison. Never tune on test data.

1. First isolate insufficient horizon with v03, 6M steps. Compare its first 3M
   with the baseline and its second 3M with its own first half. An improving tail
   is not convergence. Inspect exact rewards, cost metrics, constraint behavior,
   and final-vs-best regression; a falling actor loss alone is not success.
2. If actor stability remains limiting, create a separate version changing only
   actor learning rate (e.g. 3e-4 -> 1e-4), leaving critic_lr and other controls
   fixed to the selected parent. Try 5e-5 as a distinct version if warranted.
3. If EA diversity/destructive mutation remains limiting, use a separate
   mutation-family experiment. Current `frac: 1` touches every matrix entry;
   `prob_reset_and_sup: .05` includes large perturbations/resets. Test lower
   mutated fraction and then lower reset probability separately. Keep the shared
   policy evolved exactly once, with the same generation/update schedule.
4. Explore Gaussian noise or target smoothing only when rollout/learning
   evidence points there, each in a new file. Do not change multiple unrelated
   parameter families at once.
5. Confirm promising settings and their parent with at least seeds 0, 1, 2 at
   equal budget before declaring an improvement. Sequential runs are preferred
   so throughput/ETA stay interpretable. Keep policy size and update intensity
   fixed unless explicit user approval expands scope.

Primary comparisons: exact validation reward curves, all-six-day mean total
cost and inventory-adjusted saving, worst-day cost, SOC/activity qualifications.
Report RL and EA separately and compare matched sources, not the best source
chosen after seeing each day's outcome. A checkpoint chosen on 8 validation days
can rank differently on the six candidate days; retain and report this mismatch.
The inventory correction is a diagnostic, not a reward or true terminal value.
Training carries SOC across days and does not impose cyclic daily inventory.
Separate actual constraint violations (`soc_violation`, power/balance limits)
from the showcase's extra terminal-inventory qualification. Failing the latter
alone is not a violated environment constraint or evidence of nonconvergence.
Do not chase cyclic-SOC qualification by silently changing the reward or making
it a hard selection gate that the training task never specified. Report raw
economics, inventory correction, actual feasibility and battery activity
separately. Residual daily depletion/excess can still explain why one-day cost
comparisons differ from sustained operating performance.

No fixed budget of runs can prove an intrinsic algorithmic ceiling. Operational
stop criterion: at least three relevant parameter families fail to improve
the confirmed incumbent mean validation cost by 1% at equal budget across three
seeds, without reducing constraint/activity quality; tail statistics no longer
show useful improvement. Interpret variance honestly; do not overclaim
significance from three seeds. Document what was tried and suspected structural
limits, stop further training, and disable the campaign automation. Ask before
structural/reward/environment changes. If evidence still shows useful learning,
continue with a justified horizon experiment rather than declaring a ceiling.

## Wakeup Protocol

Use one thread heartbeat, not a standalone new conversation. Update this
heartbeat rather than creating duplicates. Inspect only the exact campaign
process and logs. Use the launch PID, command token, process start time and W&B
state to avoid PID reuse mistakes.

- If training has completed, analyze its retained logs with
  `uv run python tools/analyze_facmac_tuning.py --run-dir <run> --output <new.json>`.
  Check W&B completion and final local artifacts, update the state and decision
  notes, choose the next justified version/confirmation seed and start it.
- If remaining time is less than about 10 minutes, wait in bounded intervals
  until normal completion, checking for errors and adjusting ETA. Do not treat
  a failed process as successful completion.
- If remaining time exceeds 10 minutes, update the same heartbeat to wake near
  the newly estimated completion time and end the turn. Estimate from recent
  step/time progress, not only the original runtime. Do not continuously monitor
  a multi-hour run or start a duplicate.
- If still active but ETA unavailable, inspect whether it is computing,
  evaluating or uploading. Schedule a cautious later check; never restart or
  hide failures automatically.
- Recheck remaining time while waiting. If it grows past 10 minutes, reschedule.
- After launching a run, verify bounded startup, record its command, PID, paths,
  parameters, expected completion and W&B ID, and schedule the next wakeup.
- Keep notifications quiet for unchanged/nonactionable states; notify on
  completed analysis, next-round launch, failure, stopping criterion, or a user
  decision. Local scheduled work needs this machine and the app available.

## Analysis Command

```powershell
uv run python tools/analyze_facmac_tuning.py `
  --run-dir results/facmac_ea_compact_v2_final_only_20261017_a `
  --output results/facmac_tuning/campaign_20261017_a/baseline_analysis.json
```
