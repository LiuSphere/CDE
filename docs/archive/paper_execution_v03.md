# Paper Experiment Execution

Historical execution protocol for the original crossover-enabled v03 matrix.
The old campaign was stopped by the user. The currently selected main method
is crossover-free `gavdac_ea_main`; see README.md. This document does not
authorize resuming the old queue or relabelling it as the new main method.

Authorized on 2026-10-18: continue the experiments in paper_experiment_plan.md.
This supersedes the previous stopped campaign only for the fixed paper matrix,
not unrestricted parameter tuning. No automatic commits, pushes or deletions.

## Training Matrix

Keep v03 fixed. New configs facmac_v03_no_ea, facmac_v03_no_mutation and
facmac_v03_no_crossover differ only by name and the declared ablation flag.
Train each at 6M counted training environment steps for seeds0/1/2, sequentially.
Use short, new tokens. Main method always reports RL validation-selected Actor;
EA Actor is separate supplementary evidence, not a per-day selection oracle.
W&B online, aggregate-only, final plots after normal completion.

Use the most recent completed main seed0 v03r0a, not the best historical seed0;
retain seed1 ft03s1a, and rerun incomplete main seed2 from scratch. This reuse is
provisional until saved configs, environment/data hashes and the narrow code
change below have been audited. Never count ft03s2a as completed.
If reuse fails audit, repeat affected full-method seeds at equal budget.

First queue: pure FACMAC seed0, no-mutation seed0, no-crossover seed0, then the
same three for seed1, then seed2, then full v03 seed2.
MAPPO seed0/1/2 is a pending second phase, not implemented or launched yet.
User authorization includes completing this planned independent comparator;
keep separate explicit MAPPO modules and an explicit branch in direct_main,
not a registry or a replacement of FACMAC. Use primary MAPPO documentation
and a proven on-policy implementation/library when compatible; SB3 centralized
joint-action PPO must not be relabelled MAPPO. Check recurrent/local Actor
information and centralized critic contracts before training.

## Preflight Findings

- EpisodeRunner increments t_env for every non-test rollout, including each EA
  population rollout and RL rollout. Fixed validation does not increment it.
- No-EA naturally has no population rollout, EA updates or policy injection.
  Record actual interactions, updates and wall time; not identical compute.
- Entry seeds NumPy/PyTorch and MGEnv's own Generator, but not Python random,
  fastrand or Gym action-space sampling. Preserve this protocol for this batch
  to avoid silently changing v03. Call these independent seed-labelled runs,
  not deterministic paired experiments. Do not change seed behavior mid-batch.
- A deterministic red test exposed that agent-level clone recombination ignored
  crossover_prob=0. Fixed only that gate. p=1 retains the original RNG call
  sequence and operations. This is required for a real no-crossover ablation;
  it does not promote a changed v03 main algorithm.
- All three ablations cross replay warmup and learner updates in real-entry
  smoke tests. EA ablations complete EA cycles; no-EA completes learner updates
  with zero EA calls. The no-mutation operator has zero calls; no-crossover's
  clone-recombination gate is checked deterministically, including p=1 RNG draws.

## Evaluation And Leakage

The fixed train split is rows0-730, validation731-912, test913-1094.
Checkpoint validation remains first8 validation days; showcase candidates remain
740,766,808,862,880,904. No test results select checkpoints or tune this batch.
Old physical validation documents explicitly report solving all182 test days
and prior environment diagnostics on test data. Therefore the existing test
split cannot honestly be described as historically untouched independent data.

Proceed with training without looking at new test performance. Freeze the
existing182-day holdout as a retrospective benchmark. True independent external
validation requires new, previously unused load/PV data from the user; do not
invent that data or quietly relabel validation days. Clearly disclose the
historical reuse if reporting the retrospective benchmark. This limitation
does not block the training/ablation queue.

Before final evaluation implement the common continuous-SOC test protocol,
rules baseline and matched no-daily-cyclic MILP. Select representative days from
exogenous load/PV only before seeing comparative test performance. Retain initial
and final inventories, raw cost, inventory diagnostic, physical feasibility,
worst day, all-day means and per-seed results. Do not use the six validation
showcase days as the paper's independent test evidence.

## Persistence And Safety

State: results/paper20261018/state.json. Read it on every wakeup along with this
document, plan and AGENTS.md. Old facmac_tuning state is stopped historical data;
do not revive its parameter search. One training process at a time.
Launch via installed uv binary and src/direct_main.py with online W&B.
Log exact token, PID, creation time, command, stdout/stderr, source/data/config
hashes and W&B ID in each launch.json. Verify actual learner and EA (when enabled)
before scheduling next wakeup. No changes to running jobs' source/config.

ETA >10min: reschedule same heartbeat near completion and end turn.
ETA <10min: bounded wait with error/newETA checks; if >10min again, reschedule.
No ETA: identify initialization/computation/upload stage, cautiously recheck,
never assume failure or restart from observation timeout. No killing runs or
offline fallback without explicit user approval.
After completion verify shutdown, W&B finished, expected final images and all
metrics; use tools/analyze_gavdac_tuning.py with a new analysis path. No-EA has
only RL reports; EA eval0 is a placeholder, not an independently trained policy.
Save actual constraint checks and compare all6 validation days, not best day.
Schedule next fixed queued run only after safe analysis, without parameter tuning.

After FACMAC queue finishes implement/test MAPPO and evaluation before further
training; keep source changes and compatibility checks recorded. A new module
changes global checkpoint source hashes, so old policies must be evaluated using
their verified saved source snapshots plus identical data, not by disabling
integrity checks. Retain original checkpoints/metadata. Do not rewrite their
hashes or silently claim full-trajectory resume.

Notify only completed results, next run, failure, needed decision or final summary.
At final completion delete the heartbeat. Machine and app must remain running.
