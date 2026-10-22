# X06: Reduce Shared Gradients As One Flat Vector

Registered 2026-10-22 before implementation. The last candidate in this bounded
follow-up targets parent reduction overhead, not learning equations.

## Hypothesis And Intervention

X04 allocates one shared flat gradient storage per worker and phase, but the
parent clones/adds/checks each parameter separately. Reduce each phase's flat
storage in the same worker order, check finiteness once, then assign disjoint
parameter views. Preserve `None` gradients with explicit participation flags.
Workers zero unused slots so stale values cannot affect the flat check.
The parent reduction must own its storage: optimizer clipping must never
modify the shared worker output buffers.

## Gates

- Reference is accepted X04 under `results/x05_ref_a/`; after X05 rollback,
  both learner source and tests match this snapshot's SHA256 exactly.
- Same fixed 2/2 setting, CPU float32 and one thread/process. Focused
  weights/gradients/optimizer/target equivalence and exact predecessor training
  trace including critic gradients; all final regression tests pass.
- Screen: one 28,800-interaction pair, steady-update throughput gain >=2%.
- Acceptance: two alternating pairs, each case 57,600 interactions and 945
  updates; both total times faster, aggregate total throughput gain >=2%.
- No source edits or concurrent benchmarks during measurement.
- On failure restore X04 manually, preserve candidate source and evidence.
  No formal source, default setting, or running process is changed.
- After this round, finish final regression and documentation. More intrusive
  model/compiler/learning changes are outside this bounded execution follow-up.

## State

Not accepted. The predefined screen gate failed; implementation and
candidate-only tests restored to X04.

- `results/x06_red_a.log`: both mechanism-specific tests failed before the
  intervention (20 vector finite checks instead of 2; unused slots retained
  NaNs), establishing a red-capable structural check.
- `results/x06_focused_a.log`: all 7 focused tests passed in 34.839 seconds.
- `results/x06_trace_candidate_a.json`: exact predecessor trace versus
  `x05_trace_ref_a.json`, including 101 replay episodes, 48 updates and critic
  gradient digests.
- `results/x06_screen_a/`: each case 28,800 interactions and 345 updates.
  Reference 62.431 seconds, candidate 61.637. Total throughput +1.29%,
  steady-update throughput +1.31%, below the >=2% steady screen gate.
- This is an unaccepted marginal gain, not proof that flat reduction is
  universally unhelpful. Do not retain it based on an easier post-hoc gate.
- Candidate preserved under `results/x06_rejected_a/`. Restore SHA256 checks
  and final regression close the round. Shared diagnostic tools remain.
- Final byte-for-byte restoration and 170-test regression are recorded in
  [X04](X04_shared_transport.md#final-restoration-and-regression).
