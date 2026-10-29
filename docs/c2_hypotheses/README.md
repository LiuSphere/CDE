# C2 Hypothesis Ledger

Version-controlled records for the exploratory campaign in
[c2_optimization.md](../c2_optimization.md).
Raw artifacts and live slot/ETA state stay in `results/c2opt21/`.
The original `results/cmp20/` campaign remains separate and preserved.

## Current Gate

User clarified Beijing2026-10-25: every seed must strictly beat its
corresponding frozen C3 PPO/MAPPO economic_cost; only the ratio of three-seed
mean costs must save at least3%. There is no per-seed3% requirement.
See [current protocol](../c2_optimization.md#frozen-comparison).
H01-H10 rows below retain their historical registered decisions and
per-seed3% counts. NEW `results/c2opt21/gate_reassessment_a.json` separately
applies the clarified rule to original recorded costs, without rerunning or
rewriting any frozen artifact. None passes the mean3% condition, so their
closed/reverted state is unchanged. That clarification alone did not resume
gavdac; the later explicit user "continue" authorizes the H11 work below.

## Experiment Lineage

Beijing2026-10-25 user authorization permits descendants of promising
variants and new attempts based on previously tested mechanisms, not a
mandatory root reset after every round. One new delta per round; record all
inherited mechanisms, exact parent/delta/rollback identities, complete seed
results and keep/reject rationale. Final C3 success is distinct from
exploratory parent retention. Compare against parent, original C2 and fixed
C3 without attributing accumulated gains to the last delta.

Rules: [lineage amendment](../c2_optimization.md#lineage-amendment-2026-10-25).
Machine index: `results/c2opt21/lineage_registry_a.json`; future node records
are new immutable artifacts, with index/status changes recorded incrementally.
At the lineage amendment working checkout=C2_ROOT and no parent was chosen.
After explicit resume, H11 selected H03 as its historical exploratory parent.
All new engineering,three-seed training and primary evaluations were accepted,
but both economic rules failed. The registered child-only normal revert
restores qualifiedparent4e3aa00,not C2_ROOT;the parent and all evidence remain.
NEW306restoration tests passed. This does not establish causal improvement.
H01-H11 below remain historical closed nodes; revisit only through a newly
registered/qualified descendant, never old tokens or successful mains.
Samegavdac continues actual next-single-hypothesis review; no new formal
training before preregistration,engineering acceptance and immutable freeze.
H12registers a new child of the qualified H03parent:Actor2/targets1 inherited,
scaledunit-Huber Critic regression the only delta. It revisits H04's loss
mechanism on a different recorded parent,not an old run/config restoration.

| Hypothesis | Intervention | Engineering | Economic Gate | State |
| --- | --- | --- | --- | --- |
| [H01](H01.md) | Fix mixer context during Actor optimization | Passed | Failed all3seed gates and mean | Closed; reverted69d60ef/62b1bfa;146tests passed |
| [H02](H02.md) | Clipped double mixed-Q targets | All training/evaluation accepted | Only seed2 meets3%; mean saving0.9007%; target failed | Closed; reverted859f242;153restoration tests passed |
| [H03](H03.md) | Actor optimizer every second Critic step; targets unchanged | All training and primary evaluations accepted | Onlyseed2meets3%; mean saving1.3118%; target failed | Closed; reverted45c40fd;177restoration tests passed |
| [H04](H04.md) | Scaled unit-Huber Critic TD regression; local MSEscale retained | All3training/primaryevaluations accepted | Onlyseed2passes3%; mean saving1.2008%; targetfailed | Closed; reverted249a127;185restoration tests passed |
| [H05](H05.md) | Fixed five-step within-episode Critic targets | All3training and original-host primaryevaluations accepted | All3miss3%; ratio-of-means saving-9.5328%; targetfailed | Closed; revertedbdab135;195restoration tests passed |
| [H06](H06.md) | Per-sample LayerNorm at both Critic hidden preactivations | All3training/primaryevaluations accepted | All3miss3%; mean saving0.8320%; targetfailed | Closed; revertedc462786;207restoration tests passed |
| [H07](H07.md) | Fixed clipped normalized target-action noise, original single Critic | All3training/original-host primaryevaluations accepted | All3miss3%;ratio-of-means saving-0.3511%;targetfailed | Closed;revertedc43fd12;223restoration tests passed |
| [H08](H08.md) | Small initialization of only local Critic fc3 weight and bias | All3training/original-host primaryevaluations accepted | All3miss3%;ratio-of-means saving-0.4992%;targetfailed | Closed;reverted496e125;239restoration tests passed |
| [H09](H09.md) | Remove only Actor normalized-action L2 penalty | All3training/originalhostprimaryevaluations accepted | Onlyseed2passes3%;ratio-of-meanssaving-2.9197%;targetfailed | Closed;revertedbdbba2f;255restoration tests passed |
| [H10](H10.md) | Center only generator-fraction action input inside Critic | All3training/originalhostprimary accepted | Onlyseed2passes3%;ratio-of-meanssaving0.9429%;targetfailed | Closed;reverted9f368c7;273restoration tests passed;nextreview paused |
| [H11](H11.md) | H03 parent Actor2; ONLY newdelta alltargets every2Criticsteps | All3training/primaryaccepted;24focused+306full restoration pass | 2/3strictseed gates;ratio-of-meanssaving-1.1150%;final+retention fail | Closed;child-only revert89b0954 to qualifiedparent4e3aa00;ledger/support/results preserved |
| [H12](H12.md) | H03 parent Actor2/targets1; ONLY newdelta scaledunit-Huber Critic TD regression | All3originaltraining/182dayprimary accepted;independentfreeze/raw/finite/physics checks pass | All3strictly<C3;mean343315.2311,saving1.34171%fails3%;retentionmean104.0436belowH03passes;seed2parentregression418.6091 | Closed retained_target_not_met;retain implementationde78e9c/support8883790;no economicrevert;nextchild requires newregistration |
| [H13](H13.md) | Retained H12 parent; ONLY newdelta per-example Actor fc1 pre-ReLU LayerNorm | Original-host engineering/26supportguards/153rawfreeze accepted; seed2 complete training and182day primary accepted | Seed2cost343001.4691,3.24051%belowC3,645.7126belowH12; no three-seed final/retention decision | Two original7076M runs active at runtime6eafc099; localfree; child9d16922 rollback only to retained H12runtime8883790 after complete results if retentionfails |

## Isolated Execution Rounds

These `X` records concern execution cost only, not the C2 economic hypothesis
queue. They neither start H02 nor modify/deploy to frozen H01 hosts. They have
their own correctness and timing gates; economic superiority is not assessed.
After H02closure, the user authorized adopting this execution infrastructure.
Main integration `9f03926` passed172local/168per-remote tests and four exclusive
2/2smokes; see [qualified execution](../c2_execution_hosts.md#qualified-multiprocessing-execution).
This adoption does not alter or reopen the closed H01/H02results.

| Record | Intervention | State |
| --- | --- | --- |
| [X01](X01_execution_cache.md) | Affine shards reuse Actor history and inputs | Accepted: +7.14% versus previous 4/4 execution |
| [X02](X02_rollout_overlap.md) | Overlap the EA cycle's one common-start RL rollout | Accepted: +3.24% versus X01 |
| [X03](X03_worker_width.md) | Match rollout width to the five-member population | Not accepted: +1.79% below 2% gate; retain 4/4 |
| [X04](X04_shared_transport.md) | Reuse shared learner weight/batch/gradient buffers | Accepted at 2/2: +3.41%; 4/4 resource-unqualified |
| [X05](X05_bound_state_sync.md) | Bind model/shared state pairs once | Not accepted: initial timing gate failed; restored X04 |
| [X06](X06_flat_gradient_reduce.md) | Reduce/check gradients as a flat phase vector | Not accepted: +1.31% below screen gate; restored X04 |

## Record Rules

- One file per new hypothesis/node, registered before algorithm changes.
  At most one active child; one new intervention relative to its exact
  recorded parent, no simultaneous new interventions or parameter grid.
- Record the causal conjecture, single change, frozen comparator, success/
  failure criteria, exact commits, configuration, tokens, host/slot allocation,
  validation evidence, per-seed/mean results and rollback verification.
- Engineering checks and economic acceptance are different gates. A working
  gradient change is not evidence that the cost target has been met.
- Append dated observations and final decisions. Do not rewrite the original
  hypothesis or threshold after seeing results. Preserve failed records too.
- Record fixed-C3 target status separately from incremental parent retention.
  Retain a promising node only with complete accepted evidence and its
  preregistered rationale/criterion. Otherwise revert child delta commits
  in reverse order to the exact recorded parent, not necessarily C2_ROOT.
  Preserve ledger/protocol/raw artifacts and record restoration checks,
  revert SHAs and normal push before opening the next child.
- On success, report all three seeds and limitations. Reused retained data
  remain exploratory, not independent superiority evidence.

## Scheduling

Reuse the same `gavdac` heartbeat. The user re-enabled arch715 on2026-10-25:
authorized capacity is one local,two SSH707 and one715auxiliary slot.
715requires fresh-wave readiness before new work; current H11 remains
exactly three frozen formal seeds on local/707. Readiness is checked separately. See
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
