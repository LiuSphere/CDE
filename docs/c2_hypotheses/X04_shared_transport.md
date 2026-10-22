# X04: Reuse Shared Learner Transport Buffers

Registered 2026-10-22 before implementation, on the isolated execution branch.
This does not authorize deployment, resume, or changes to frozen H02 sources.

## Hypothesis And Single Intervention

The synchronous learner copies NumPy weights and batch shards into executor
messages and returns copied NumPy gradients every phase. Reusable CPU shared
buffers should reduce serialization and allocation without changing equations.
Keep worker-local parameters, Actor graph lifetime, fixed shard identities,
ordered gradient addition, parent-only clipping/steps and target updates.
Do not skip target synchronization: the current soft targets change each step.

## Frozen Reference And Gates

- Reference `f08eb686833653157f596ae83e0c94ede03fd8c2`, archived before edits
  under `results/x04_ref_a/`; tar SHA256
  `0176c10debc3c008647037e9fc77577dc267ed0715bdbd3abecceafda1ded4efe69`.
- Initial unmodified learner: all 3 focused tests passed in 21.809 seconds.
- Diagnostic `results/x04_profile_a/`: actual 28,800-interaction, 345-update
  run with 2/2 workers; cProfile perturbs timings, so it is not acceptance
  evidence. Worker profiles identify network compute, backward and repeated
  state loading; parent wait time is not a direct measurement of IPC cost.
- Correctness: full state/gradient/optimizer/target float32 equivalence at
  `rtol=1e-4, atol=1e-5`, plus exact predecessor trace when applicable.
  Cover variable batch/sequence sizes, fewer episodes than workers, unused
  gradients, phase misuse, retained graph lifetime and failure propagation.
- Timing: same `gavdac_ea_main`, one thread/process, fixed worker counts and
  57,600 joint interactions / 945 updates. Two alternating reference/candidate
  repetitions; no source edits or concurrent benchmarks during measurement.
  Require both pairs faster and aggregate total throughput gain >= 2%.
- Resource gate: preserve the existing active local H02 run; begin with
  lower-width diagnosis when memory is constrained. Record the tested width;
  do not generalize a reduced-width result to 4/4 or remote hosts.
- Failed implementation is restored from its task-local frozen source,
  preserving diagnostics, tests where applicable and all raw evidence.
  No commit or push without explicit approval.

## State

Accepted locally at 2/2 on 2026-10-22; 4/4 remains resource-unqualified.

### Initial Engineering And Resource Checks

- `results/x04_focused_a.log`: 5 focused tests passed in 30.834 seconds.
- `results/x04_trace_ref_a.json` / `x04_trace_candidate_a.json`: identical
  101 replay episodes, 48 full-update state digests and captured parent RNG,
  with 2 rollout / 2 learner workers on both source versions.
- `results/x04_pair_a/`: deliberately interrupted during the first 4/4 case
  when free physical memory fell to about 600 MiB. No timing verdict; no
  formal process was stopped. All task-created Python processes exited.
- `results/x04_pair_b/`: fixed 2/2 comparison, with the same original step,
  update and alternating-pair gates. Lower-width capacity is a limitation,
  not evidence of 4/4 deployment qualification.
- The initial parent's broad cProfile report contains distorted nested/thread
  cumulative times; do not derive IPC fractions from it. Worker profiles and
  timing CSVs are separate diagnostics. The reusable profiler subsequently
  uses parent phase wall-clock timers and worker-local cProfile only.

### Paired Timing

`results/x04_pair_b/` completed all four 57,600-interaction / 945-update cases.
Reference total seconds 126.564 / 128.255; candidate 122.478 / 123.946.
Both paired throughput gains were positive (3.34% / 3.48%); aggregate total
throughput gain 3.41%, post-warmup 2.98%, steady updates 8.79%.
Mean first update increased from 4.948 to 7.806 seconds due to shared-buffer
setup. Startup is included in the total gate. The timing gate passes at 2/2;
`results/x04_full_a.log`: all 170 tests passed in 315.821 seconds.
No 4/4 or remote-host speedup is established.

### Final Restoration And Regression

X05 and X06 were screened separately and rejected by their original gates.
Both candidate sources and test variants remain in their rejected snapshots.
After restoration, SHA256 matched the accepted X04 snapshot exactly:

- `src/learners/gradient_pool.py`:
  `44d8e70d3695784e268ada80c0757911d401cd4f0cff2648f841a95d1521e6bc`
- `tests/test_parallel_learner.py`:
  `35574916416be3ee3ae8dc60c99f61c44a4055d48d76809370b9c4ed716271cb`

`results/x04_trace_final_a.json` matched the frozen X04 trace in
`x05_trace_ref_a.json` exactly, including critic-gradient digests, all 101 replay
episodes, 48 complete update state digests and captured parent RNG.
`results/x04_profile_b/` exercised the revised standalone profiler through
14,400 interactions and 45 updates with profiles from both learner workers.
`results/x04_full_final_a.log`: all 170 final tests passed in 301.275 seconds.
No task-created Python processes remained afterwards; the original local H02
process was still present. Main retained only its pre-existing manuscript edits.

This closes the bounded follow-up after two subsequent candidates missed
their practical screen gates. Shared execution defaults are unchanged; no
commit, push or deployment has been performed.
