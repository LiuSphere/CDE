# FACMAC Runtime Optimization

Date: 2026-10-17
Branch: `codex/facmac-performance-optimization`
Baseline revision: `09e8348f4356c7d4da2546d73b8ecace5d6ee274`

The subsequent learner execution redesign is documented in
[FACMAC Sequence Execution](facmac_sequence_execution.md). The measurements
below describe the first, inference/replay-only round.

## Scope

This is a runtime-efficiency change, not a change to the learning objective.
The existing uncommitted environment and evaluation changes were retained in
both baseline and optimized runs. The baseline revision identifies the original
controller and replay implementations, not the entire dirty working tree.

No rewards, environment constraints, data CSVs, learning rates, batch sizes,
update frequencies, EA scheduling, network architecture, or CPU placement were
changed. No claim is made about improved convergence or final economic savings.

## Findings And Changes

1. `CQMixMAC.select_actions` detached actions but retained recurrent hidden
   states with autograd history. Disable autograd for this inference method.
   The learner's differentiable `forward` path remains unchanged.
2. Full-batch action selection reconstructed an `EpisodeBatch` on every call.
   Use the existing batch for the default full slice. Explicit subsets still
   use the original slicing behavior, including during initial random actions.
3. Replay sampling copied advanced-indexed samples again and allocated shared
   storage despite the single-process training path. Advanced indexing already
   provides independent storage. Keep one clone for the full-buffer slice,
   which otherwise aliases replay, and remove the redundant work.
4. The timing harness did not seed all random sources used by EA and initial
   actions. Seed Python, NumPy, PyTorch, fastrand PCG32, and action spaces in the
   harness only. Record replay sampling, actual training steps, wall clock, and
   throughput. Console and CSV use the same interval, excluding CSV flushing.

Regression tests initially failed on the unwanted graph, full-batch slicing,
and shared-memory allocation, then passed after the production changes.

## Measurements

Configuration: `facmac_ea` + the current `microgrid` working-tree configuration,
CPU, PyTorch 2.11.0+cu128, six PyTorch threads, seed 0. Baseline and optimized
training runs were not run concurrently. The first pair also launched a
lightweight focused-test process; the second pair had no concurrent test
workloads started by this task.

The prescribed 12,144-training-step baseline completed **zero learner updates**:
512 episodes of 24 steps require at least 12,288 training steps for warmup.
Consequently, the main comparison uses 12,720 training steps, with 10 learner
updates and 11 EA generations per run. There are also 16 evaluation episodes.

Two runs per implementation were measured. The following values come from the
CSV interval, before CSV serialization. Throughput is total training steps
divided by total measured wall time across the two runs.

| Metric | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| Mean wall clock | 20.907 s | 19.982 s | -4.4% |
| Training steps / second | 608.4 | 636.6 | +4.6% |
| Mean rollout time, training and evaluation | 10.575 s | 9.569 s | -9.5% |
| Mean action selection time | 4.379 s | 3.506 s | -19.9% |
| Mean replay sampling time | 0.0450 s | 0.0123 s | -72.7% |
| Mean learner update time, ten updates | 3.724 s | 3.866 s | +3.8% |

Phase timers are nested and must not be added together. Action selection also
includes target-policy selection during learner updates. Learner execution
itself is not faster in this measurement. Startup, evaluation, and warmup
account for much of this short run; two repetitions do not establish a precise
speedup or predict steady-state, learning-heavy training throughput.

Raw comparison files:

- `results/facmac_perf_baseline_seeded_20261017_a.csv`
- `results/facmac_perf_baseline_seeded_20261017_b.csv`
- `results/facmac_perf_optimized_seeded_20261017_a.csv`
- `results/facmac_perf_optimized_seeded_20261017_b.csv`

Initial exploratory timing files without complete seeding are not included in
the comparison.

## Correctness Checks

- `uv run python -m unittest discover -s tests -p "test_*.py"`: 52 tests passed.
- Training integration crossed replay warmup, checked finite CPU gradients,
  and exercised EA generation counts 1 and 3, as well as no-EA training.
- Six focused regression tests cover exact deterministic actions and recurrent
  hidden states, preserved actor gradients, Gaussian exploration, full-batch
  selection, independent replay samples, episode-constant fields, and buffer
  wraparound.
- An additional differential run loaded the original controller and replay
  classes from the baseline revision into the current training pipeline.
  At seed 0, all 546 rollout returns and all 10 learner updates matched exactly,
  with zero tolerance for online and target network parameters and both
  optimizer states.
- The differential result is recorded in
  `results/facmac_perf_equivalence_20261017_a.json`.

Repeat the measured optimized workload with a new output filename:

```powershell
uv run python tests/test_training_timing.py --config gavdac_ea --timesteps 12720 --output results/<new-file>.csv
```

## Follow-Up Candidates

These are code-inspection candidates, not measured optimizations delivered here:

- Profile a substantially longer, post-warmup workload before changing the
  learner. Its online policy currently runs separately for critic hidden
  features and actor optimization. Reusing that sequence needs careful
  gradient-lifetime and complete-update equivalence checks.
- Check the graph mixer's unused per-node reward allocation computation when
  only total Q is requested, retaining checkpoint and optional-output behavior.
- Evaluate any learning-quality changes separately across multiple seeds,
  fixed validation days, and full economic cost. A short throughput benchmark
  cannot establish better policies.
