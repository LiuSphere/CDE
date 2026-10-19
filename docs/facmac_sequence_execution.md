# FACMAC Sequence Execution

Date: 2026-10-17
Branch: `codex/facmac-performance-optimization`

## Scope

This second optimization round changes execution structure, not the model's
parameters, learning objective, or training intensity. All model operations
remain float32 on CPU. No mixed precision, quantization, reduced batches,
shortened recurrent histories, or skipped updates were introduced.

Existing environment, evaluation, data, and unrelated working-tree changes
were retained. The starting controller, policy, learner, and graph modules
were captured under `results/facmac_arch_reference_20261017/` before editing.
Those generated reference files are local artifacts, not new production paths.

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
