# AGENTS.md

## Scope

- Python 3.12+ continuous-action FACMAC + optional EA for microgrid dispatch; float32 model execution on CPU.
- Default: `facmac_ea_tune_v03_long` + `microgrid`. Configuration and run commands belong in `README.md` and `src/config/algs/README.md`.
- Verify facts against code/configs; plans and run states do not authorize launching or resuming experiments.

## Repository Index

| Area | Start Here |
| --- | --- |
| Setup/configs | `README.md`, `pyproject.toml`, `src/config/default.yaml`, `src/config/envs/microgrid.yaml`, `src/config/algs/README.md` |
| Entry/training loop | `src/direct_main.py`, `src/run.py` |
| Environment/data | `src/envs/maenv.py`, `src/envs/data/` |
| Rollouts/replay | `src/runners/episode_runner.py`, `src/components/episode_buffer.py` |
| Actor | `src/controllers/cqmix_controller.py`, `src/modules/policy.py` |
| Critic/mixer | `src/learners/`, `src/modules/critic.py`, `src/modules/graph_mixer.py`, `src/modules/graph/` |
| Evolution | `src/ea/mod_neuro_evo.py`, `src/ea/ssne_base.py` |
| Evaluation/reports | `src/utils/policy_evaluation.py`, `src/utils/dispatch_showcase.py`, `src/utils/logging.py` |
| Analysis/performance | `tools/analyze_facmac_tuning.py`, `tools/recover_facmac_showcase.py`, `tools/benchmark_facmac_*.py`, `tests/test_training_timing.py` |
| Reference experiments | `tools/run_physical_microgrid_milp.py`, `tools/train_sb3_microgrid_ppo.py`, `tools/sb3_*env.py` |
| Paper protocol/tests | `tools/freeze_paper_protocol.py`, `tests/test_paper_experiments.py`, `tests/test_*.py` |

## Task Documentation

Consult only what the task needs; historical reports are evidence, not current implementation specifications.

| Task | Documentation |
| --- | --- |
| Environment/model | `docs/facmac_environment_v3.md`, `src/config/envs/microgrid_explained.md` |
| Sequence execution/performance | `docs/facmac_sequence_execution.md` |
| Evaluation/figures | `docs/wandb_dispatch_showcase.md`, `src/utils/policy_evaluation.py` |
| Experiment protocol | `docs/paper_execution.md`, `docs/paper_experiment_plan.md` |
| Exploratory crossover test | `docs/crossover_parameter_test.md` |

Keep this file for navigation and durable, cross-cutting constraints. Record detailed rationale or procedures in an existing topic document; add a new document only when needed for reuse or audit. Do not duplicate code/config values, temporary status or readily discoverable implementation details. History belongs in `docs/archive/` or run artifacts.

## Manuscript Instructions

- Read and follow `paper/manuscript/MANUSCRIPT_AGENTS.md` only for manuscript modifications (text, equations, figures, tables or LaTeX). Do not read/apply it for training, code, experiments or read-only review.

## Implementation Rules

- Use explicit imports/direct construction, not registries. Support only continuous Gymnasium `Box` actions; do not restore retired environment paths. Keep namespace packages without empty `__init__.py` files.
- Preserve runner -> batch/replay -> controller -> learner tensor contracts; check shape/agent changes across all boundaries. Keep rollouts, replay, sampled batches, networks and EA on CPU.
- Preserve recurrent history, gradient boundaries and optimization order when changing execution. Evolve the complete shared actor once per generation, not once per agent; retain shared EA bases and ablations.
- Let errors surface; no broad handlers, silent fallbacks or warning suppression.

## Configuration And Artifacts

- Use `uv run` exclusively. Dependency edits update `pyproject.toml` and `uv.lock` together, followed by `uv sync` and `uv lock --check`.
- Keep shared parameters in `src/config/default.yaml`, intentional differences in algorithm YAMLs, and training intensity configuration-controlled. Verify matched comparison configs rather than assuming generic controls match v03.
- Do not change rewards, constraints or source CSVs during unrelated work. Write generated artifacts under `results/` using new, short tokens/paths; never overwrite or delete existing artifacts without explicit instruction.
- Keep logs aggregated and interval-controlled. Do not continuously monitor jobs unless requested.

## Evaluation And Checkpoints

- No test-based tuning or per-day RL/EA winner selection. Disclose historical holdout reuse, information/inventory boundaries and actual training interactions, including EA rollouts; follow the experiment protocol above.
- Preserve checkpoint integrity checks and provenance. Do not claim exact resume, deterministic reproduction or comparator equivalence without evidence.

## Validation

- Suite: `uv run python -m unittest discover -s tests` (includes real training). Documentation-only edits need focused checks, not training.
- Load config changes through `src/direct_main.py`. Loop/buffer changes must cross warmup and complete a learner update; EA changes must exercise a cycle and affected generation settings.
- Sequence/vectorization changes require float32 output, gradient and full-update equivalence checks. See `tests/test_facmac_sequence.py`, `tests/test_facmac_performance.py` and the performance document for timing procedures.
- Before committing, inspect `git status` and confirm author name/email. No amend or force-push without explicit approval.
- After completing and validating a change you judge substantial, proactively ask the user whether to commit and push it to the remote repository. Do not commit or push without explicit user approval.
