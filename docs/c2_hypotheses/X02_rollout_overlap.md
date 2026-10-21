# X02: Overlap The EA Cycle's Common-Start RL Rollout

Registered 2026-10-21 after X01 closed, before implementation. Isolated
execution branch only; not H02, no economic tuning or new formal runs.

## Hypothesis And Intervention

On EA cycles, population evaluation and the one exploring RL rollout use the
same explicit training row/SOC and frozen policies. Evolution and learning wait
for all of them. Therefore, the parent can execute that one RL rollout in an
isolated runner while the persistent workers evaluate the population.

Commit all resulting population episodes first, then the RL episode, in the
original logical order. Preserve exploration in the parent with the same
`t_env` offset, recurrent reset, random draw order, episode counters, SOC
restore, ordered CSV/statistics and replay. Do not parallelize causally
dependent ordinary RL episodes or overlap learner/evolution updates.

This single intervention does not include record-payload trimming, shared
memory, a new network, extra samples or a different update schedule.

## Reference And Gates

- Reference: accepted X01 code `d129a94`, archived before edits at
  `results/mp_iter_ref_b/`; archive `results/mp_iter_ref_b.tar` SHA256
  `5b829605085568d161ecadf5e5a5e9baf5542ee73a7632b36222638e068beb64`.
- Fixed main config, CPU float32, one thread/process, 4/4 workers,
  57,600 joint steps and identical update counts per case.
- Two paired reference/candidate repetitions in AB/BA order.
- Engineering: all tests pass; exact real episodes, inventory, hidden states,
  statistics, CSV order and captured random states; an exact short full-training
  trace match against X01. Profiling must not double-count the nested parent
  rollout. Parent commits no partial successful group after a worker error.
- Practical: aggregate startup-inclusive throughput improves at least 2%;
  candidate faster in both pairs. This is a local practical threshold, not
  statistical significance or a guaranteed remote/6M-step speedup.
- On failure preserve this record/raw artifacts, normally revert the isolated
  implementation commit and verify restoration before another hypothesis.
  Keep the shared benchmark infrastructure and accepted X01.

## Failure Boundary

A fatal worker failure can occur after the overlapped RL rollout has physically
executed and consumed parent exploration draws, even though no group results
are committed. Fail the run and preserve diagnostics; never continue, reuse
those speculative samples or claim exact resume after such an error.

## State

Registered; implementation and evidence pending.
