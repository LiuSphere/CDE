# Retained Algorithm Configurations

All algorithm YAMLs merge with `src/config/default.yaml`. The environment is
configured separately by `src/config/envs/microgrid.yaml`.

| Configuration | Purpose |
| --- | --- |
| `gavdac_ea_main` | Selected main: v03 dimensions/budget, `crossover_prob: 0.0`; selection, mutation and RL injection retained. |
| `gavdac_ea_main_no_ea` | Fresh C1 control: entry-merged main parameters, only `EA: false` and the name differ. |
| `gavdac_c2_h02` | Exploratory opt-in clipped double mixed-Q targets; common main settings retained. See `docs/c2_hypotheses/H02.md`; not the default or an accepted economic improvement. |
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
