# C2 Hypothesis Ledger

Version-controlled records for the exploratory campaign in
[c2_optimization.md](../c2_optimization.md).
Raw artifacts and live slot/ETA state stay in `results/c2opt21/`.
The original `results/cmp20/` campaign remains separate and preserved.

| Hypothesis | Intervention | Engineering | Economic Gate | State |
| --- | --- | --- | --- | --- |
| [H01](H01.md) | Fix mixer context during Actor optimization | Passed | Pending | Three seeds running |

## Isolated Execution Rounds

These `X` records concern execution cost only, not the C2 economic hypothesis
queue. They neither start H02 nor modify/deploy to frozen H01 hosts. They have
their own correctness and timing gates; economic superiority is not assessed.

| Record | Intervention | State |
| --- | --- | --- |
| [X01](X01_execution_cache.md) | Affine shards reuse Actor history and inputs | Accepted: +7.14% versus previous 4/4 execution |
| [X02](X02_rollout_overlap.md) | Overlap the EA cycle's one common-start RL rollout | Accepted: +3.24% versus X01 |

## Record Rules

- One file per hypothesis, registered before algorithm changes. At most one
  active hypothesis; no simultaneous interventions or parameter grid.
- Record the causal conjecture, single change, frozen comparator, success/
  failure criteria, exact commits, configuration, tokens, host/slot allocation,
  validation evidence, per-seed/mean results and rollback verification.
- Engineering checks and economic acceptance are different gates. A working
  gradient change is not evidence that the cost target has been met.
- Append dated observations and final decisions. Do not rewrite the original
  hypothesis or threshold after seeing results. Preserve failed records too.
- On failure, revert the explicitly recorded implementation commits in reverse
  order; preserve this ledger, protocol and raw artifacts. Record revert SHAs,
  baseline-restoration checks and normal push before opening the next round.
- On success, report all three seeds and limitations. Reused retained data
  remain exploratory, not independent superiority evidence.

## Scheduling

Reuse the same `gavdac` heartbeat. One local, two SSH707 and one arch715 slot
are authorized; readiness is checked separately. See
[execution hosts](../c2_execution_hosts.md). H01still has only three seeds.
Check every active slot and schedule at the earliest estimated completion plus
a small buffer, not the end of the whole batch.
If remaining time exceeds10minutes, reschedule and end the turn. Otherwise
wait at most10minutes total and re-check all slots.

A freed slot can accept/evaluate its completed seed. Start another hypothesis
only after the current round is closed and any required rollback is verified;
do not modify frozen source while any seed from the current round is active.
Progress, ETA and acceptance are read from the live campaign state, not this
index. Update this index and the hypothesis record when the phase changes.
