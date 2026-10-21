# X03: Match Rollout Worker Width To The Five-Member Population

Registered 2026-10-21 after accepted X02, before measurement. Resource allocation
only; no implementation, network, objective, update schedule or sample changes.

## Hypothesis

Five rollout workers can execute the five deterministic population policies
in one wave instead of the 4+1 waves of four workers. Retain four learner
workers, where the previous screen showed a useful resource/time tradeoff.
This may reduce total execution time enough to justify one more process.

## Gate And Stopping Rule

- Frozen implementation: accepted X02 `3d364cb`, plus evidence-only commits.
- Compare `0/0`, `4/4`, `5/4` on current immutable source, 57,600 joint steps
  and identical update counts per case, two alternating-order repetitions.
- Accept 5/4 only if aggregate startup-inclusive throughput exceeds 4/4 by at
  least 2%, with improvement in both repetitions. Otherwise retain 4/4 and
  record this resource hypothesis as not meeting the gate. No implementation
  revert is needed because this round changes no code/default.
- All timing artifacts must have unchanged raw input hashes, actual interactions
  and update counts. Summarize direct final-versus-serial gains, not a sum of
  earlier screen percentages.
- After this check, stop this local execution round if accepted correctness
  gates remain satisfied. Do not expand into changed learning equations,
  asynchronous updates, extra training or endless worker-count scans. No claim
  of global optimality, perfection, remote speedup or economic superiority.

## State

Registered; measurements pending.
