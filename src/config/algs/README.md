# Retained Algorithm Configurations

All algorithm YAMLs merge with `src/config/default.yaml`. The environment is
configured separately by `src/config/envs/microgrid.yaml`.

| Configuration | Purpose |
| --- | --- |
| `gavdac_ea_main` | Selected main: v03 dimensions/budget, `crossover_prob: 0.0`; selection, mutation and RL injection retained. |
| `gavdac_ea_main_no_ea` | Fresh C1 control: entry-merged main parameters, only `EA: false` and the name differ. |
| `gavdac_mappo` | Independent feed-forward shared local Gaussian actor, centralized critic, SB3 PPO core; see `docs/mappo_baseline.md`. |
| `gavdac_ea_tune_v03_long` | Historical v03 with crossover: 6M steps, batch128, hidden48, mixer32, buffer50000. |
| `gavdac_ea` | Historical generic GAVDAC + EA using shared defaults, including crossover; not v03 dimensions. |
| `gavdac_no_ea` | Generic baseline with `EA: false`. |
| `gavdac_ea_no_mutation` | Generic ablation with `mutation_prob: 0.0`. |
| `gavdac_ea_no_crossover` | Generic ablation with `crossover_prob: 0.0`. |
| `gavdac_ea_no_variation` | Generic selection/cloning control without mutation or crossover. |
| `gavdac_v03_no_ea` | v03-matched paper ablation, only `EA: false`. |
| `gavdac_v03_no_mutation` | v03-matched paper ablation, only `mutation_prob: 0.0`. |
| `gavdac_v03_no_crossover` | v03-matched paper ablation, only `crossover_prob: 0.0`. |
| `gavdac_v03_crossover025` | Exploratory v03 candidate, only `crossover_prob: 0.25`; not part of the frozen paper matrix or the default. |

The generic ablations are retained mechanisms, not budget/dimension-matched
v03 paper experiments. Create separate aligned configurations before formal
comparisons; keep previous configurations immutable.

```powershell
uv run python src/direct_main.py --config gavdac_ea_main --env-config microgrid --override seed=0
```

The entry point without `--config` selects `gavdac_ea_main`. Its only behavioral
change from historical v03 is removing crossover. The shared default and old
YAMLs retain their original values for explicit historical comparisons.
The main still inherits elite/parent policies and mutates non-elites; removing
crossover does not remove ordinary cloning or evolutionary augmentation.

## Opt-In Multiprocess Execution

Both shared execution controls default to `0` (serial):

- `rollout_workers`: persistent CPU processes for independent deterministic EA
  population rollouts and fixed validation days.
- `learner_workers`: synchronous episode-batch gradient shards. The parent alone
  clips gradients, steps critic then actor, and updates target networks.

The direct entry accepts `--override rollout_workers=4 --override learner_workers=4`.
For newly qualified, authorized slot jobs, use the existing supervisor with
`--rollout-workers 4 --learner-workers 4 --threads 1`; it records these settings
without changing the interaction budget. Preserve new-token, `--detach`,
credential-file and exclusive-slot requirements. The supervisor rejects these
options for MAPPO and frozen evaluation manifests. Do not switch a running
experiment's source or bypass checkpoint hashes to enable the new path.

The Windows screen favored 4/4 for startup-inclusive short-run time; 4/8 had
faster steady learner updates but a larger startup/memory footprint. Counts are
not a universal optimum: qualify and benchmark the actual host before a new
formal wave. See `docs/facmac_sequence_execution.md` for exact/tolerance
equivalence boundaries, benchmark commands, evidence and throughput definitions.
The accepted X01/X02 execution iterations subsequently measured 22.73% higher
total throughput for 4/4 versus serial in two alternating-order 57,600-step local
repetitions. X03 did not accept 5/4: its extra 1.79% missed the original 2%
resource gate. These results do not guarantee remote/6M-step speedups or better
dispatch; the shared execution defaults still remain serial.
The 2026-10-22 shared-transport follow-up measured a further +3.41% total
throughput against its own predecessor at fixed 2/2 workers. Its 4/4 attempt
was stopped for low available RAM alongside a formal run; the new result is
not a 4/4 qualification and must not be added to the earlier speedup figure.
H02 (`gavdac_c2_h02`) is retired after its three-seed target failure and is
not dispatchable. The optional learner sharder currently handles one
critic/mixer pair; a future multi-critic design needs its own explicit support
and complete-update checks, not ignored worker options.
The exploratory campaign now requires explicit multiprocessing settings for
future rounds; its initial qualification setting is 2/2 with one compute
thread per process. Shared defaults stay serial for compatibility.

`gavdac_v03_no_crossover` has the same algorithm parameters as the new main;
it is no longer a meaningful ablation of the new main. The retained paper
controls and 0.25 experiment describe the old v03 protocol, not a new matched
ablation matrix. Do not relabel historical results or claim new multi-seed
validation from this default change.

Compare total environment interactions including EA rollouts, validation-selected
checkpoints, physical feasibility, inventory diagnostics and all evaluation days.
Do not select RL/EA actors independently for each test day.

The fresh comparison uses C1/C2/MAPPO seeds 0/1/2, without reusing historical
models or results. The full-connectivity mixer and continuous evaluation
protocol are frozen before formal runs; MAPPO is a separate implementation.
See `docs/paper_experiment_plan.md` for phase gates and interpretation limits.

Execution of the fixed paper matrix is recorded in `docs/paper_execution.md`.
Historical VMAS and optimizer/noise sweeps are preserved in the pre-cleanup backup
and Git history, not in this active configuration directory.
