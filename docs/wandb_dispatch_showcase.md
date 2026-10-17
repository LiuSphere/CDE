# W&B dispatch reports

Training defaults to `wandb_mode: online`. During training, only allowlisted
aggregate analysis metrics are uploaded: validation returns, losses, train/test
return mean and spread, costs and savings, battery charge/discharge and SOC,
constraint diagnostics, generation, grid import, peer transfers, losses and spill.
`environment_step` is the common x axis. `wandb_metric_patterns` controls this
allowlist; excluded diagnostics remain available in local console logs.

Representative-day evaluation and plotting run only after normal training
completion, using immutable best RL and EA checkpoints. No representative-day
rollouts, plots or tables are produced during training. An interrupted or failed
run does not produce a final report. EA reports require an evolved policy to have
been validated and saved.

## Defaults

| Setting | Default |
| --- | --- |
| `showcase_enabled` | `true` |
| `showcase_days` | `2` per saved policy |
| `showcase_candidate_rows` | `[740, 766, 808, 862, 880, 904]` |
| `showcase_soc_tolerance` | `0.03` absolute SOC |

Reports run once at training completion. Reporting requires
`save_model: true` and `save_best_policy: true`. Change settings through
`src/direct_main.py --override key=value`; set `showcase_enabled=False` to disable
these reports.

## Selection Protocol

Candidate days must belong to the validation split; test days are rejected.
Every candidate starts with the configured `evaluation_initial_soc` (default
0.35). A qualified day must satisfy all of:

- Every battery's final SOC differs from its initial SOC by at most 0.03.
- Every battery charges and discharges for at least two hours each, with actual
  power exceeding 1% of its respective rated charge/discharge power.
- Operating cost plus an inventory adjustment is below the no-control baseline.
- Maximum hourly agent power-balance residual is at most `1e-5`.

The inventory adjustment is the sum, over batteries, of
`abs(final_soc - initial_soc) * capacity * 1.25 * (max_training_price + cycle_cost)
/ charge_efficiency`. It penalizes both depletion and excess terminal inventory
for comparable displays; it does not modify the environment reward.

Qualified days are ranked by inventory-adjusted saving ratio, then terminal SOC
error and row index. If none qualify, the best diagnostic days are shown with
`DIAGNOSTIC: NOT QUALIFIED` titles. If fewer than two qualify, only those qualified
days are shown. These are deliberately selected best cases, not typical-day or
held-out test performance. Aggregate validation metrics remain separate.

## W&B Contents

For each source (`rl`, `ea`):

- `dispatch/<source>/best_case_<rank>/inputs`: load, PV and tariff.
- `dispatch/<source>/best_case_<rank>/dispatch`: per-agent power balance.
- `dispatch/<source>/best_case_<rank>/battery`: all battery SOC and actual power.
- `dispatch/<source>/best_case_<rank>/exchange`: peer sends, receipts and losses.
- `dispatch/<source>/best_case_<rank>/costs`: hourly components and cumulative costs.
- `dispatch/<source>/candidate_days`: metrics for every candidate, including failures.
- `dispatch/<source>/qualified_count`: number of qualified candidates.

W&B receives only the final figures, candidate metrics table and qualification
count. Raw hourly trajectories, selection JSON and actor checkpoints are retained
locally, not uploaded as directory artifacts. Local reports are under
`results/dispatch_showcases/<run_token>/` by default, or the configured
`local_results_path`. W&B uses the current `environment_step` as the display axis;
the original checkpoint step is retained in captions and local metadata.

Evaluation uses an independent runner and saved actor without changing training
weights, training environment state, training step counter or random generators.
