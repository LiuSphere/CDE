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

Resource hypothesis not accepted on 2026-10-21. The additional worker did not
meet the original aggregate 2% practical threshold. Retain 4/4; no code/default
was changed and no implementation revert is needed.

`results/x03_width_a/summary.json` contains the two alternating repetitions.
4/4 total seconds were 109.195/109.604; 5/4 were 107.564/107.396. Both improved,
but aggregate throughput gain is only approximately 1.79%, below the gate.
This is not evidence that five workers are universally inferior.

`results/x03_memory_snapshots_a.json` records Windows process-tree snapshots:
4/4 had 10 processes, working-set sum 5897.8 MiB, private committed sum
9407.7 MiB; 5/4 had 11 processes, 6420.0 MiB and 10333.9 MiB respectively.
These are instantaneous, non-peak sums; shared resident pages can be counted
more than once and committed private memory is not resident physical RAM.
Do not treat these as a proved maximum or per-host allocation guarantee.

Stop this bounded execution iteration after the direct serial comparison is
reported. Accepted X01/X02, tests and frozen-original-run boundaries remain.

## Direct Final Comparison

All six cases completed with 57,600 joint interactions and 945 learner updates;
the benchmark's raw source/config/data/tool hashes remained unchanged.

| Rollout / Learner | Mean Total s | Mean Post-Warmup s | Steady Update ms |
| --- | ---: | ---: | ---: |
| 0 / 0 | 134.266 | 119.551 | 86.826 |
| 4 / 4 | 109.400 | 90.775 | 58.555 |
| 5 / 4 | 107.480 | 88.818 | 58.345 |

Recommended 4/4 versus serial: total throughput +22.73%, elapsed time -18.52%,
post-warmup throughput +31.70%, steady-update throughput +48.28%. These are
different metrics; none is a sum of the earlier X01/X02 screen percentages.
Two repetitions on this Windows host are descriptive, not a statistical,
remote-host, 6M-step, convergence or dispatch-economics guarantee.

The local execution objective has reached its declared stopping point:
accepted correctness gates, repeated measured gains and a rejected marginal
resource increase. This does not establish global optimality or perfection.
