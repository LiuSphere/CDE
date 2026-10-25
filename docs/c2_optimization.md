# C2 Single-Hypothesis Optimization

Authorized2026-10-21 by the user, separate from the immutable cmp20 campaign.
The user approved freezing C3 and common experimental conditions, optimizing
only C2 (including structural design), one hypothesis per round, committing
each code change and preserving an explicit rollback destination. The
Beijing2026-10-25 lineage amendment below permits incremental descendants
of promising variants instead of mandatory return to the original baseline.

## Frozen Comparison

- Keep the existing C3 MAPPO seeds0/1/2, selected Actors and recorded costs.
  Never retrain, weaken or retune C3 to make C2 win.
- Same environment, rewards, equipment, source CSVs, actor observation boundary,
  train/validation/retained rows and per-run6Mjoint-interaction budget.
  EA rollouts count toward C2's budget. No checkpoint weight transfer.
- Same original hosts:707slot0seed0,707slot1seed1,local slot0seed2.
  CPU float32; OMP/MKL/OPENBLAS1thread. Check identities and exclusive slots.
- The fourth auxiliary slot on arch715 was disabled then explicitly
  re-enabled by the user on2026-10-25. Future auxiliary work requires
  fresh-wave readiness checks; current H11's frozen three-seed protocol
  and original formal seed hosts are unchanged. No fourth formal seed.
  See [execution hosts](c2_execution_hosts.md).
- Validation-only checkpoint selection remains rows731-738 every10000steps.
  Freeze the RL Actor before exploratory retained comparison.
- Main comparison is continuous SOC over rows913-1094, initialSOC0.35,
  free finalSOC; daily hidden state/previous action resets unchanged.
  Daily-reset results and EA Actors are supplementary, never winner-selected.
- Primary metric is whole-period economic_cost, not full_cost or a showcase.
  Report SOC, costs, physical checks, actual interactions, updates and time.
- Freeze C3 using the original evaluation manifests and accepted trajectory
  hashes. Preserve cmp20, including its unapproved MILP acceptance decision.

Current user clarification, Beijing2026-10-25:

- EACH seed must have strictly lower economic_cost than its corresponding
  frozen C3 PPO/MAPPO seed: cost(C2,seed)<cost(C3,seed).
- The ratio of three-seed mean costs must save at least3%:
  1-mean(cost(C2))/mean(cost(C3))>=0.03.
- There is NO per-seed3% requirement. A seed saving1% is acceptable if all
  seeds beat C3 and the ratio-of-means gate passes. Equality with C3 fails
  the strict per-seed improvement condition.

The comparator remains the existing frozen C3 MAPPO; "PPO" does not replace
it with another implementation, Actor or evaluation. This is a cost
reduction, not a3percentage-point increase versus the rule, nor an arithmetic
mean of per-seed percentage savings. Seed labels do not imply matched RNG.

H01-H10 were registered and closed under the older EACH-seed3% plus
ratio-of-means3% rule. Preserve their immutable protocols/results/archives
and original closure decisions. A separately labelled new-rule reassessment
does not itself retrain, re-evaluate, restore an algorithm or rewrite history.
The current authoritative gate is state.current_economic_gate; historical
evaluation/closed-state candidate_maximum_cost fields are not current
per-seed thresholds. New rounds must freeze independent guards for this
clarified rule rather than execute retired H01-H10 freeze mains.

Repeated iteration after observing the retained comparison makes this an
EXPLORATORY benchmark-development campaign. Do not call the final selected
round an untouched test, a statistical significance result or universal
superiority. New independent data are needed for an independent confirmation.
Success is an optimization target, not a promised attainable outcome.

## Round Lifecycle

1. Select and record an exact parent version, then register one incremental
   hypothesis relative to that parent, mechanical checks, fixed configuration,
   final-target and parent-retention criteria before editing algorithm code.
2. Implement it as an opt-in variant. Preserve baseline behavior by default.
   No simultaneous learning-rate, network-width, reward or EA schedule search.
3. Test gradients/contracts, run a real smoke crossing warmup, a learner update
   and an EA cycle; verify CPU float32, finite losses and no crossover.
4. Inspect scoped diffs/author/branch/remote. Commit and normally push the
   verified code before launching three fresh6Mseed runs with new tokens.
   Store that exact implementation commit in the round record.
5. Accept all three complete training runs and original-host/source evaluations.
   Do not change active source, judge from early losses or replace a bad seed.
6. Report the fixed C3 final target separately from incremental progress.
   Missing the final target need not discard a defensible improvement:
   after all evidence is accepted, explicitly retain it as an exploratory
   parent if its preregistered retention criteria pass. Otherwise normally
   revert ONLY the child's registered delta commits to its exact parent.
   Preserve all outputs; no reset/amend/force, deletion or unrelated rollback.
   Close and record this decision before registering another child.
7. Stop on all target conditions, or pause and ask on correctness/provenance,
   permissions, resource failures, ambiguous rollback conflicts or absence of
   a defensible next intervention. Do not fabricate success or keep retrying
   an identical configuration to select a lucky run.

One round's budget is three fresh6Mjoint-step runs plus necessary explicitly
recorded engineering tests and evaluation; there is no hidden parameter grid.
Use the per-host exclusive-slot supervisor and same heartbeat with phase-aware ETA.
When earliest remaining ETA>10minutes, reschedule near it and end the turn.
No new automated training wave until tests and the current hypothesis freeze pass.

## Lineage Amendment: 2026-10-25

The user explicitly permits improving a promising tested hypothesis further,
revisiting an earlier mechanism with a new defensible change, and building
on a retained variant rather than always returning to the original C2 root.
Only one new incremental intervention is active at a time. The complete
inherited mechanism list is frozen alongside that delta; an accumulated
multi-mechanism variant must not be presented as a root-isolated ablation.

Keep three distinct references: the actual parent, the original C2 root,
and the unchanged frozen C3 PPO/MAPPO. Report per-seed costs and ratio-of-means
against all available accepted references, including regressions; do not
attribute the total accumulated gain to the latest delta. A historical
same-seed label is not paired RNG, and a numerical improvement on reused
retained data is not independent evidence of causality or significance.

Before implementation, each new node must record:

- Unique node/round ID and parent ID, exact parent code commit or complete
  reproducible code recipe, configuration, immutable protocol/result hashes,
  qualification scope and selection rationale.
- Original-root identity, all inherited mechanisms, the sole new delta,
  primary basis, risks, mechanical tests and accumulated compute.
- New implementation/delta commits, new source/config/data/tool/dependency
  identities, parent equivalence checks, and exact rollback destination.
- Fixed C3 final gate, a separate preregistered criterion for retaining or
  replacing the parent, and the three-seed fresh-run budget.
- Every host/slot/token/creation identity, validation-selected Actor and
  evaluation artifact, all seed outcomes, parent/root/C3 differences, and
  the final keep/reject/revisit decision with its evidence.

Code inheritance is NOT checkpoint weight transfer or exact resume. Each
formal child still uses exactly three fresh6Mphysical-joint runs including
EA, the original seeds/hosts, validation-only Actor selection and frozen
evaluation/inventory boundaries. No extra economic pilot, grid, seed
replacement, lucky retry or early-stop-based promotion. A failed seed
cannot be hidden by choosing a different Actor, day or metric.

A parent may miss the final3% gate yet be useful for further exploration if
its training/evaluation/provenance and engineering checks are accepted and
the retention rationale is explicit. Passing the final C3 gate and being a
promising exploratory parent are different statuses. Do not silently promote
a parent from engineering-only or incomplete/invalid evidence.

H01-H10 remain closed with their original commits, results and reversions.
Revisiting one is a NEW registered lineage node with new outputs, not a
reopened old run. The earlier blanket ban on reusing a tested economic
mechanism is superseded only to this extent: a chosen parent/descendant
must be explicitly reconstructed and qualified as new opt-in code. Never
execute old successful freeze/launch/accept/copy mains, reuse old tokens,
restore a known correctness failure or silently enable unrelated retired
flags. New guards must enumerate the exact inherited mechanisms and reject
all others. Active host source remains immutable.

Use `state.research_lineage_policy` and the node records indexed by
`results/c2opt21/lineage_registry_a.json`. New node artifacts are immutable;
index/status changes retain incremental history. Distinguish working
checkout, selected parent and active child. At this amendment the working
code is the restored root, no parent/child is selected, and H03 is only a
review candidate because all three historical seeds beat C3 while mean
saving is1.3118%. That observation does not establish reproducible
improvement or authorize restoring its code/runs. The same gavdac remains
paused; this policy update does not automatically launch or resume anything.

## H01: Action-Only Actor Gradient Through the Mixer

Historical registration below retains its original pre-clarification gate.
For future rounds use the current clarified gate above.

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

## Execution Amendment: 2026-10-22

After all H02training finished, the user requested multiprocessing for future
rounds. This supersedes the earlier instruction not to adopt the independent
execution branch, but does not change H02's actual serial execution identity,
C3, physical-interaction budgets, optimizer-update rules or economic gates.
Do not retrospectively relabel old results or rerun them merely for this change.

Verify the integration commit and serial/parallel output, gradient and complete
update checks before use. Freeze the new exact execution source/configuration,
rollout-worker and learner-worker counts per host, and process/thread/RAM
budgets across concurrent slots before any new formal launch. Qualify the
chosen settings on both hosts; do not assume that four experiment slots mean
four processes, or that one local timing screen guarantees remote speedup.
Use multiple processes within a run, each with one compute thread.

Rollout-only exact equality and tolerance-based sharded-gradient equivalence
are distinct. Float32 reduction differences do not establish long-horizon
policy/economic identity. Report this execution amendment when comparing to
the frozen C3, and preserve original baseline/protected file identities rather
than overwriting them with the new execution hashes.
