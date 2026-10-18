# Retained Algorithm Configurations

All algorithm YAMLs merge with `src/config/default.yaml`. The environment is
configured separately by `src/config/envs/microgrid.yaml`.

| Configuration | Purpose |
| --- | --- |
| `facmac_ea_tune_v03_long` | Selected default: 6M steps, batch128, hidden48, mixer32, buffer50000. |
| `facmac_ea` | Generic FACMAC + EA using shared defaults, not v03 dimensions. |
| `facmac_no_ea` | Generic baseline with `EA: false`. |
| `facmac_ea_no_mutation` | Generic ablation with `mutation_prob: 0.0`. |
| `facmac_ea_no_crossover` | Generic ablation with `crossover_prob: 0.0`. |
| `facmac_ea_no_variation` | Generic selection/cloning control without mutation or crossover. |

The generic ablations are retained mechanisms, not budget/dimension-matched
v03 paper experiments. Create separate aligned configurations before formal
comparisons; keep previous configurations immutable.

```powershell
uv run python src/direct_main.py --config facmac_ea_tune_v03_long --env-config microgrid --override seed=0
```

Compare total environment interactions including EA rollouts, validation-selected
checkpoints, physical feasibility, inventory diagnostics and all evaluation days.
Do not select RL/EA actors independently for each test day.

Historical VMAS and optimizer/noise sweeps are preserved in the pre-cleanup backup
and Git history, not in this active configuration directory.
