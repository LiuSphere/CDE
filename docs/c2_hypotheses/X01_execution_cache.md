# X01: Affine Gradient Workers Reuse Actor History

Registered 2026-10-21 before implementation on the isolated
`codex/multiprocess-rollouts` execution branch. This is not H02, an economic
algorithm intervention, or a claim of better dispatch. Do not change or restart
the active H01 runs.

## Hypothesis

The current synchronous sharder recomputes Actor history and sends the same
batch again in the Actor phase. Affining each episode shard to one persistent
worker permits retaining that history until the parent critic step completes.
Sending only updated critic/mixer weights in the second phase should improve
controlled training throughput without changing the learning equations.

## Single Intervention

Use one single-worker executor per fixed shard. Cache that shard's batch,
Actor graph and critic inputs between the two gradient phases. Do not reload
Actor weights between these phases or step any worker optimizer.
Keep the parent aggregation/clipping/critic-then-actor/target order unchanged.

## Frozen Reference And Gates

- Reference: commit `6b35268`, archived before edits under
  `results/mp_iter_ref_a/`, archive `results/mp_iter_ref_a.tar`, SHA256
  `128c7682fc651ab82970ea8db9dae5675135a2461413abecceafda1ded4efe69`.
- Same `gavdac_ea_main`, CPU float32, one thread/process, 4 rollout and 4 learner
  workers, 57,600 joint interactions per case, same update count.
- Compare reference/candidate in alternating AB/BA order for two repetitions.
  No concurrent benchmark cases or source edits during measurement.
- Engineering gate: complete update weights/gradients/optimizers/targets at
  `rtol=1e-4, atol=1e-5`, exact deterministic rollout checks, real warmup/EA
  integration, all tests pass, wrong phase/batch order fails explicitly.
- Practical gate: aggregate paired startup-inclusive training throughput
  improves at least 5%, with candidate faster in both paired repetitions.
  This small local benchmark is not statistical or remote-host proof.
- Failure: preserve artifacts and this record, normally revert the isolated
  implementation commit before another execution hypothesis. Shared benchmark
  infrastructure and evidence documents are not part of the intervention.

## State

Accepted for execution use on 2026-10-21. No economic results used.

Implementation: `d129a94`. Shared ABBA benchmark infrastructure: `de8d97e`
(not a hypothesis rollback commit). Registration: `4119327`.

- `results/x01_focused_a.log`: 8 tests passed in 55.238 seconds.
- `results/x01_full_a.log`: all 165 tests passed in 288.004 seconds.
- `results/x01_pair_a/`: two AB/BA pairs, each case 57,600 joint interactions
  and 945 updates. Reference total seconds 120.509/120.575; candidate
  113.110/111.910. Each paired throughput gain 6.54%/7.74%, aggregate 7.14%;
  post-warmup gain 9.46%, steady-update gain 18.05%.
- `results/x01_trace_reference_a.json` and `x01_trace_candidate_a.json`:
  exact full-training trace match for 101 replay episodes and 48 complete
  updates, including recorded parameter/gradient/optimizer/target digests and
  captured Python/NumPy/PyTorch parent RNG state. This is not proof of strict
  long-horizon reproducibility or a claim about unrecorded RNG sources.

Both pairs improved and exceeded the original 5% aggregate gate. Keep the
implementation. No rollback required. Local controlled execution evidence
does not establish remote-host gains or improved dispatch economics.
