# X05: Bind Model And Shared State Pairs Once

Registered 2026-10-22 before implementation. Execution-only, not economic
optimization or authorization to modify frozen formal sources.

## Hypothesis And Intervention

The X04 worker still reconstructs NumPy dictionaries and invokes recursive
`load_state_dict` each phase. Models, parameter names and shapes are fixed
for the pool lifetime. Bind validated module/shared tensor pairs at pool
initialization, then copy only the phase's original modules directly.
Preserve local parameter storage, synchronization frequency, Actor graph,
all recurrent history and the original parent optimizer ordering.

## Reference And Gates

- Frozen X04 source under `results/x05_ref_a/`, captured after X04 correctness
  and timing gates. Raw hashes recorded by each comparison manifest.
- Same 2/2 workers, CPU float32 and one thread/process. No source changes or
  concurrent benchmarks during a timing measurement.
- Focused equivalence plus exact predecessor trace including additionally
  captured critic gradients. All final regression tests must pass.
- Initial screen: one reference/candidate pair at 28,800 interactions.
  Advance only if steady-update throughput improves at least 2%.
- Acceptance: two AB/BA pairs at 57,600 interactions and 945 updates;
  both total times improve and aggregate total throughput gain >= 2%.
- On failure restore this task's isolated intervention to X04 and retain
  all evidence. No commit/push, formal deployment, or remote speedup claims.

## State

Not accepted. The predefined screen gate failed; implementation restored to
X04 before opening another candidate.

- `results/x05_focused_a.log`: all 6 focused tests passed in 30.609 seconds.
- `results/x05_trace_ref_a.json` / `x05_trace_candidate_a.json`: exact
  101-episode / 48-update trace, including additional critic-gradient digests.
- `results/x05_screen_a/`: 28,800 interactions and 345 updates per case.
  Reference 62.742 seconds, candidate 63.189; total throughput -0.71%,
  steady updates -0.72%. The initial >=2% steady gate failed.
- One short pair is not evidence that caching is universally slower. It does
  not justify retaining this extra execution path on this host.
- Rejected candidate preserved under `results/x05_rejected_a/`; source and
  its candidate-only test restored manually to the frozen X04 files. The
  reusable trace tool's added critic-gradient capture is retained.
- Final byte-for-byte restoration and 170-test regression are recorded in
  [X04](X04_shared_transport.md#final-restoration-and-regression).
