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

Registered; implementation and measurements pending. No economic results used.
