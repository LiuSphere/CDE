# GAVDAC Sequence Execution

This file retains its historical name for existing documentation links.
The execution design remains relevant to the current CPU implementation.
Benchmark figures and validation counts below are historical measurements from
2026-10-17, not current-main performance guarantees or current test-suite totals.
Captured comparison sources and generated timing files may not be available
in a cleaned checkout; use current-source benchmarks when those inputs are absent.

## Scope

This second optimization round changes execution structure, not the model's
parameters, learning objective, or training intensity. All model operations
remain float32 on CPU. No mixed precision, quantization, reduced batches,
shortened recurrent histories, or skipped updates were introduced.

Existing environment, evaluation, data, and unrelated working-tree changes
were retained. The starting controller, policy, learner, and graph modules
were captured under `results/facmac_arch_reference_20261017/` before editing.
Those generated reference files are local artifacts, not new production paths.

## Spawned Rollout Execution

The `codex/multiprocess-rollouts` branch adds opt-in
`--override rollout_workers=N` (default `0`, serial). This is per-training-run
capacity, separate from the experiment supervisor's exclusive slot count.
Do not change a running campaign's frozen source or resume an old checkpoint
under this source. Qualify the execution variant before a new formal campaign.

Persistent CPU workers execute independent EA population policies on their
explicit common training scenario, and independent fixed validation days on
explicit rows and initial inventories. Each process has one PyTorch compute
thread; initialization uses `spawn` on both Windows and Linux. The pool is
reused, with a startup barrier so very short initial jobs do not all execute
on the first worker. Errors propagate; there is no silent serial fallback
after worker failure. Shutdown is performed on normal and exceptional exits.

Only the parent owns replay insertion, training counters, statistics, trajectory
files, learner updates, evolution, checkpoint selection and W&B. Worker results
are committed in original population/day order, regardless of completion order.
The parent restores final environment inventory and the pending training state
after validation, episode counters and recurrent hidden states. Workers never
explore: exploration RNG consumption remains in the parent, and dependent RL
episodes and critic-before-actor updates retain their original ordering.

X02 overlaps the EA cycle's single common-start exploring RL rollout in a
separate parent-side runner while population workers execute. The main Actor's
exploration RNG is used at the original post-population `t_env` offset. The
parent commits population episodes, then the RL episode; only then can learning
or evolution proceed. This does not parallelize dependent ordinary RL episodes.
The helper never writes CSVs; ordered parent commits own all trajectory output.
A fatal population failure may follow a physically executed speculative RL
rollout: fail the run, do not reuse those samples or claim exact resume.
See [X02 evidence](c2_hypotheses/X02_rollout_overlap.md): exact short training
trace match, 168 tests passed, two paired local screens improved total
throughput a further 3.24% versus X01, not an economic advantage.

This rollout path does not implement asynchronous learning, larger rollout
batches or additional training interactions. More CPU usage is not
itself evidence of a speedup. The five-member population exposes at most five
simultaneous population jobs; eight validation days expose at most eight.
Increasing workers beyond available independent jobs cannot accelerate them,
while model learning still has sequential dependencies.

Run the equivalence checks and alternating-order real training benchmark:

```powershell
uv run python -m unittest discover -s tests -p test_parallel_rollouts.py
uv run python tools/benchmark_parallel_rollouts.py --workers 0 2 4 8 --timesteps 28800 --repeats 2 --output results/<new-benchmark-directory>
```

The tests compare exact real rollout tensors, episode returns, inventory,
training restore, hidden states, ordered statistics and CSVs. A seeded complete
training comparison includes every replay episode, per-update actor/critic/
target weights, gradients, optimizer state, update step/episode and parent RNG.
Worker failures must surface without partially committing the group's parent
inventory or counters. These short-run checks do not prove identical long-run
economic performance.

### Synchronous Learner Sharding

The separate opt-in `learner_workers` setting partitions the already-sampled
batch on its episode dimension, never the recurrent time dimension. Worker
processes compute gradients without stepping an optimizer. The parent sums
critic gradients normalized by the original global valid-transition count,
clips once and performs exactly one original critic optimizer step.
Only then are updated critic weights sent for the actor phase. Actor shard
means are weighted by episode count, including the original unmasked policy
regularizer; parent aggregation, clipping and one actor optimizer step follow.
Target updates happen once in the parent, after both optimizers.

The X01 iteration uses one single-worker executor per affine shard, retaining
its Actor graph and batch across the parent critic step. The second phase sends
only updated critic/mixer weights. Actor parameters are not reloaded while its
graph is live. Explicit phase/round/batch and worker-identity checks prevent
stale-cache reuse. The cache is released after each Actor gradient phase.
See [X01 evidence](c2_hypotheses/X01_execution_cache.md): exact reference trace
match over 48 complete updates, all 165 tests passed, two paired 57,600-step
benchmarks improved total throughput by 7.14% versus the initial 4/4 sharder.
This additional gain is not directly combinable with timings from a different
screen, and is not a long-run or economic advantage claim.

No extra sampling, learning steps, target updates or environment interactions
are introduced. Shard reductions and smaller GEMMs can change float32 rounding,
so this mode requires tolerance-based full-update checks rather than a claim
of bitwise or long-horizon policy identity. Rollout-only mode is tested for
exact equality. Each learner worker and rollout worker is a separate process;
budget their combined memory and CPU use across concurrent experiment slots.

The benchmark logs both startup-inclusive wall clock and post-warmup wall clock,
actual joint interactions and learner-update counts. It alternates candidate
order across repetitions; use a fresh results directory. Instrumented timers
overlap and the worker subprocess internals are not included in the parent's
single-episode profiler counters. Measure whole training time, not the sum of
those counters. Preserve the existing frozen original results separately.

Explicit pairs avoid running an unnecessary Cartesian grid:

```powershell
uv run python tools/benchmark_parallel_rollouts.py --setting 0 0 --setting 4 4 --setting 4 8 --timesteps 57600 --repeats 1 --output results/<new-benchmark-directory>
```

Each pair is `(rollout_workers, learner_workers)`. Every process is assigned one
compute thread. For direct launches, set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
and `OPENBLAS_NUM_THREADS` to `1` before starting Python, and pass the selected
counts using `--override rollout_workers=N --override learner_workers=M`.
Do not multiply these counts by every experimental slot without budgeting RAM
and CPU for all concurrently active runs.
The exclusive-slot supervisor accepts `--rollout-workers N --learner-workers M`
and records both counts in `launch.json`. Defaults add no overrides to its
original command. Nonzero settings require `--threads 1` and are rejected for
MAPPO and frozen evaluation queues; they cannot silently change C3 or an
already-frozen evaluation manifest. Preserve the existing `--detach`,
credential-file, new-token and per-host slot requirements.

The profiler disables checkpoint saving, suppresses repeated evaluation/logging
and holds those settings identical between candidates. Its results are
controlled training benchmarks, not measured durations of the complete formal
campaign. `manifest.json` records raw source/config/data/tool hashes, platform
and thread settings; changing the inputs mid-benchmark raises an error.
`summary.json` includes individual runs, first-update startup, steady mean
update time and actual update/interaction budgets. `comparison.json` separates
startup-inclusive, post-warmup and steady-update throughput. Do not substitute
one for another or extrapolate a short noisy sample into a guaranteed speedup.

Validation artifacts in this worktree:

- `results/mp_full_a.log`: all 161 tests passed in 277.234 seconds.
- `results/mp_focused_b.log`: final seven spawned-rollout/sharded-learner tests
  passed in 53.284 seconds after the startup-cleanup-only adjustment.
- `results/mp_full_b.log`: final 163-test suite, including supervised parallel
  launch and comparator/evaluation guards, passed in 273.089 seconds.
- `results/mpbench_a/`: initial rollout-only alternating-order screen.
- `results/mpbench_b/`: combined rollout/learner first grid screen.
- `results/mpbench_c/`: final longer 57,600-step, 0/0 versus 4/4 versus 4/8
  screen with frozen actual inputs. One repetition per setting is descriptive,
  not a statistical performance guarantee.

Final Windows screen: every setting performed 57,600 joint interactions and
945 learner updates, CPU float32, one compute thread per process. An existing
local H01 run remained active on another core. No running formal source was
modified. Total time includes process startup/shutdown. Post-warmup time includes
the first learner update's pool startup and remaining shutdown, but not the
earlier rollout-pool startup. Steady update time excludes the first update.

| Rollout / Learner Workers | Total Seconds | Post-Warmup Seconds | Steady Update ms |
| --- | ---: | ---: | ---: |
| 0 / 0 | 135.399 | 120.489 | 87.378 |
| 4 / 4 | 122.178 | 103.361 | 69.950 |
| 4 / 8 | 122.735 | 103.778 | 66.730 |

Relative to serial, 4/4 increased startup-inclusive throughput by 10.8%,
post-warmup throughput by 16.6%, and steady-update throughput by 24.9%.
4/8 increased steady-update throughput by 30.9%, but did not improve this
short run's total time over 4/4. Prefer 4/4 as the measured low-resource starting
point, not as a proved global optimum. These are one-repeat local controlled
benchmarks, not demonstrated 6M-step or remote-host speedups, convergence
improvements or economic advantages. The long-run source/selection protocol
still requires its own qualification before use.

## Execution Changes

### Policy Sequence Interface

`CQMixMAC.forward_sequence(batch, end=None)` owns sequence input preparation
and starts a fresh recurrent history at timestep zero. Its outputs are:

- `actions`: scaled policy actions, shape `(batch, time, agents, actions)`.
- `raw_actions`: the original tanh outputs used by policy regularization.
- `hidden_states`: shape `(batch, time, agents, hidden)`.

Agent IDs and previous replay actions are assembled once for the whole
sequence. `RNNAgent.forward_sequence` batches the input and output linear
layers across time, but still calls the existing GRUCell sequentially.
The single-step rollout interface and every checkpoint parameter name remain
unchanged.

### Learner Reuse

For the three-agent microgrid's 25-state episodes, the old learner ran the online actor for 25 critic
feature states and another 25 actor states. It now computes the 24 used online
states once. The target actor still computes all 25 states for bootstrapping.
Total recurrent forward steps per update decrease from 75 to 49; this is not
a claim of a corresponding percentage reduction in complete update time.

The actor sequence is prepared before the critic update, while actor weights
are still unchanged. Critic features use a detached view of its hidden states.
The actor loss is then evaluated against the newly updated critic, preserving
the original critic-before-actor optimization order. Critic/mixer parameters
are frozen during actor optimization and restored in a `finally` block.
Critic observation inputs are also prepared once and reused.

The actor graph consequently remains live across critic optimization.
Peak process memory was not quantified, so memory-neutral execution is not
claimed.

### Graph Readout

When no team rewards are requested, the graph mixer no longer computes unused
per-agent reward fractions. The heads and their checkpoint weights remain
present, and explicit local-reward requests retain their output.

For sigmoid-constrained graph weights, constant scaling moves from the large
per-sample weight matrix to the smaller matrix product:

```text
X @ (sigmoid(W) * scale)  ->  (X @ sigmoid(W)) * scale
```

This preserves the real-number expression while reducing elementwise work and
intermediate storage. Float32 rounding can differ. The `abs`, `clamp`, and
unconstrained cases retain their original scaling behavior.

## Numerical Checks

The new focused tests check sequence outputs and gradients against timestep
execution, prefix and noncontiguous batches, all input-feature combinations,
and complete updates with Adam/RMSprop, soft/hard targets, and one/three agents.
Graph contractions and gradients are compared against the original scaled
weight expression. Optional local rewards and total-Q gradients are checked.

An independent run of the captured implementation and the optimized
implementation used identical seeded real training data and initial networks:

| Check | Result |
| --- | ---: |
| Training environment steps | 14,400 |
| Rollouts compared, including evaluation | 616 |
| Learner updates compared | 45 |
| Maximum network parameter absolute difference | 5.22e-8 |
| Maximum episode-return absolute difference | 2.53e-6 |
| Networks, gradients, and optimizer states | Passed `rtol=1e-4`, `atol=1e-5` at every update |

The independent fixed-batch comparison also passed after 14 complete updates,
with a maximum parameter difference of 8.75e-7. Synthetic and real-trajectory
experiments have different inputs, so these are separate checks.

`uv run python -m unittest discover -s tests -p "test_*.py"` passed all 64 tests.
The integration tests cross replay warmup, check finite CPU gradients, exercise
EA generation counts 1 and 3, and cover no-EA training. Checkpoint inference and
environment contract tests also pass.

Evidence:

- `results/facmac_arch_real_equivalence_20261017_a.json`
- `results/facmac_arch_tests_20261017_a.log`
- `results/facmac_arch_paired_scaled_threads6_20261017_a.json`

These are short-run equivalence checks, not proof of identical long-horizon
policies or unchanged multi-seed final economic performance.

## Performance

Machine conditions: PyTorch 2.11.0+cu128 executing on CPU, six PyTorch threads.
Other user-owned experiments were already running and were not stopped.
Consequently, absolute timings fluctuate.

### Interleaved Learner Benchmark

The new benchmark can load the captured implementation beside the current
one, initialize identical weights, and alternate measurement order. This
reduces dependence on unrelated machine-load changes between separate runs.
It also checks complete final network, gradient, optimizer, and loss states.

At batch 512 and 25 states, after two warmup updates and 12 measured updates:

| Metric | Captured Implementation | Optimized |
| --- | ---: | ---: |
| Mean update seconds | 0.4033 | 0.3223 |
| Updates / second | 2.479 | 3.103 |
| Update throughput gain | | +25.1% |

An initial separate-process comparison looked slower; the paired comparison
and real training measurements did not reproduce that result. Unmatched
process timings under the concurrent workload were not used to select the
implementation. A single-thread experiment was substantially slower in
absolute terms, so the production thread behavior was not changed.

### Real Training Loop

Each 14,400-step run performs 45 learner updates and 18 EA generations. The
second matched pair includes explicit loop and post-warmup timers:

| Interval | Baseline Seconds | Optimized Seconds | Throughput Gain |
| --- | ---: | ---: | ---: |
| Whole run | 40.049 | 37.290 | +7.4% |
| Training loop | 33.399 | 30.620 | +9.1% |
| First learner update through loop end | 22.100 | 19.265 | +14.7% |
| Sum of 45 learner updates | 19.939 | 17.112 | +16.5% |

The training-loop timer starts at the first training rollout and ends at
environment close, including evaluation inside the loop but excluding initial
construction and final logger/thread shutdown. The post-warmup timer covers
2,016 new environment steps, from the first update at step 12,384 through step
14,400. Its first update time is included.

The first whole-run pair improved only about 1.8%, emphasizing the effect of
initialization and machine-load variation. Neither pair establishes a precise
long-run throughput gain. The fixed 12,144-step harness still cannot measure
learner performance because it completes zero updates before warmup.

Phase timers overlap. In particular, target inference now uses sequence
execution instead of `select_actions`; its time remains in `learner_update`
but is no longer included in the `action_selection` timer. Do not interpret
that action-selection counter change as a like-for-like speedup.

Raw runs:

- `results/facmac_arch_baseline_20261017_a.csv`
- `results/facmac_arch_optimized_20261017_a.csv`
- `results/facmac_arch_baseline_20261017_b.csv`
- `results/facmac_arch_optimized_20261017_b.csv`

## Reproduce

Always use new output names.

```powershell
uv run python tests/test_gavdac_sequence.py
uv run python tools/benchmark_gavdac_learner.py --compare-source-root results/facmac_arch_reference_20261017 --iterations 12 --threads 6 --output results/<new-file>.json
uv run python tests/test_training_timing.py --config gavdac_ea --timesteps 14400 --output results/<new-file>.csv
```

The paired command requires the captured source directory. Ordinary current
implementation benchmarks can omit `--compare-source-root`. `--state-output`
can save network, gradient, optimizer, and loss tensors; `--profile` adds an
untimed CPU profiler sample. No new dependencies are required.
