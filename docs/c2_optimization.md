# C2 Single-Hypothesis Optimization

Authorized2026-10-21 by the user, separate from the immutable cmp20 campaign.
The user approved freezing C3 and common experimental conditions, optimizing
only C2 (including structural design), one hypothesis per round, committing
each code change and reverting a failed hypothesis.

## Frozen Comparison

- Keep the existing C3 MAPPO seeds0/1/2, selected Actors and recorded costs.
  Never retrain, weaken or retune C3 to make C2 win.
- Same environment, rewards, equipment, source CSVs, actor observation boundary,
  train/validation/retained rows and per-run6Mjoint-interaction budget.
  EA rollouts count toward C2's budget. No checkpoint weight transfer.
- Same original hosts:707slot0seed0,707slot1seed1,local slot0seed2.
  CPU float32; OMP/MKL/OPENBLAS1thread. Check identities and exclusive slots.
- A fourth auxiliary slot is authorized on arch715 via SSH alias715, without
  migrating H01or adding a seed. See [execution hosts](c2_execution_hosts.md).
- Validation-only checkpoint selection remains rows731-738 every10000steps.
  Freeze the RL Actor before exploratory retained comparison.
- Main comparison is continuous SOC over rows913-1094, initialSOC0.35,
  free finalSOC; daily hidden state/previous action resets unchanged.
  Daily-reset results and EA Actors are supplementary, never winner-selected.
- Primary metric is whole-period economic_cost, not full_cost or a showcase.
  Report SOC, costs, physical checks, actual interactions, updates and time.
- Freeze C3 using the original evaluation manifests and accepted trajectory
  hashes. Preserve cmp20, including its unapproved MILP acceptance decision.

The3% target means1-cost(C2)/cost(C3)>=0.03 for EACH seed label and for
the ratio of three-seed mean costs. It does not mean a3percentage-point
increase in savings versus the rule. Seed labels do not imply matched RNG.

Repeated iteration after observing the retained comparison makes this an
EXPLORATORY benchmark-development campaign. Do not call the final selected
round an untouched test, a statistical significance result or universal
superiority. New independent data are needed for an independent confirmation.
Success is an optimization target, not a promised attainable outcome.

## Round Lifecycle

1. Register exactly one hypothesis, its single intervention, mechanical checks,
   fixed configuration and success/failure rule before editing algorithm code.
2. Implement it as an opt-in variant. Preserve baseline behavior by default.
   No simultaneous learning-rate, network-width, reward or EA schedule search.
3. Test gradients/contracts, run a real smoke crossing warmup, a learner update
   and an EA cycle; verify CPU float32, finite losses and no crossover.
4. Inspect scoped diffs/author/branch/remote. Commit and normally push the
   verified code before launching three fresh6Mseed runs with new tokens.
   Store that exact implementation commit in the round record.
5. Accept all three complete training runs and original-host/source evaluations.
   Do not change active source, judge from early losses or replace a bad seed.
6. If any seed or the mean misses3%, mark the round failed, preserve all outputs
   and use git revert on ONLY that round's implementation commit; normal push.
   No reset/amend/force, no deletion, and no rollback of unrelated user work.
   Register the next single hypothesis only after this round is closed.
7. Stop on all target conditions, or pause and ask on correctness/provenance,
   permissions, resource failures, ambiguous rollback conflicts or absence of
   a defensible next intervention. Do not fabricate success or keep retrying
   an identical configuration to select a lucky run.

One round's budget is three fresh6Mjoint-step runs plus necessary explicitly
recorded engineering tests and evaluation; there is no hidden parameter grid.
Use the per-host exclusive-slot supervisor and same heartbeat with phase-aware ETA.
When earliest remaining ETA>10minutes, reschedule near it and end the turn.
No new automated training wave until tests and the current hypothesis freeze pass.

## H01: Action-Only Actor Gradient Through the Mixer

Hypothesis: allowing Actor parameters to change the critic's attention context
directly during policy optimization creates a non-action gradient route.
Holding that context fixed while preserving the action route will improve
economic dispatch sufficiently to meet the frozen three-seed3% target.
This is a testable conjecture, not a diagnosed performance cause.

Single intervention: in the Actor loss only, detach actor_policy.hidden_states
before passing it to the mixer. Keep action gradients, actor recurrence,
critic/mixer training, targets, parameter count, optimizers and schedules intact.
The mixer forward value must remain identical for fixed inputs and parameters.
The default gavdac_ea_main and C3 behavior must remain unchanged.

Variant: gavdac_c2_h01, identical entry-merged main settings apart from name
and actor_mixer_context_grad:false. Missing flag retains original true behavior.

Mechanical gates:
- detached/non-detached context yields identical forward Q;
- H01 blocks only the direct context gradient, retaining nonzero action and
  recurrent-actor parameter gradients and finite complete updates;
- baseline default equals explicit true, including parameter updates;
- real variant smoke crosses warmup and completes EA and learner updates.

Economic gate: all three seed-label continuous costs and their mean must be
at least3%below the frozen C3 costs. Otherwise revert H01; no post-hoc threshold,
Actor, seed, window or SOC-boundary change. Passing gradient tests alone does
not validate the economic hypothesis.

Primary theoretical motivation: deterministic policy improvement differentiates
the action-value with respect to actions and the policy with respect to its
parameters. See Silver et al., ICML2014, proceedings.mlr.press/v32/silver14.html.
This does not establish that H01 must improve this implementation.

## Records

Version-controlled per-hypothesis evidence and conclusions live in
[the hypothesis ledger](c2_hypotheses/README.md), one file per round.
Record both successful and failed hypotheses, separating engineering checks
from economic acceptance. Append final results and actual revert commits;
never remove a failed record when its implementation is reverted.

Detailed state, hypotheses, commits, source/data/config/checkpoint identities,
failures, reversions, run acceptance and scheduling belong under
results/c2opt21/. They do not overwrite results/cmp20/state.json.
The original P4 solver acceptance question remains paused and unresolved.
