# Repository Cleanup

Date: 2026-10-18. User authorized a backup followed by cleanup.

## Backup

Location: `C:/git/search-sdk-backups/before-cleanup-20261018-191125`.

- `repository.bundle`: all Git refs and complete reachable history.
- `workspace.tar.gz`: pre-cleanup workspace including retained results, hidden
  assistant configuration and source data; excludes `.git`, `.venv` and
  `__pycache__`. Virtual dependencies can be rebuilt with the backed-up uv lock.
- `experiment_tuning_history/`: old VMAS notes moved out of the active repository.
- `campaign_history/`: retired decisions, scope note and one-shot cleanup script.

Bundle verification and archive listing succeeded before cleanup.

SHA256:

```text
repository.bundle B77CE0A984541797B389B6F43B2D5B825AA626E20A14746705B21BC7D6094699
workspace.tar.gz  49B990F3CE4142A20D5886CE0E3C6B821CFABB57C495DC12AA172537BB1A5744
```

To restore, use a new empty directory, not the current checkout:

```powershell
git clone C:/git/search-sdk-backups/before-cleanup-20261018-191125/repository.bundle C:/git/search-sdk-restored
tar -xzf C:/git/search-sdk-backups/before-cleanup-20261018-191125/workspace.tar.gz -C C:/git/search-sdk-restored
```

Then run `uv sync` in that restored directory. The backup is local only; it is
not an off-machine disaster-recovery copy.

## Active Repository

Removed VMAS runtime support, its YAML/test/dependency, old remote-control
scripts, historical optimizer/noise/seed YAMLs, compact predecessors and v04-v06.
Retained v03, generic FACMAC and generic EA ablations. Existing generic ablations
are not silently changed to v03 dimensions; paper-aligned configs remain planned.

Updated README, config index, W&B project default and tests to the retained v03.
Archived the retired tuning protocol; the current status document is not a
continuation instruction. Local state now lists only retained v03 runs and has
no active run or heartbeat.

Environment/reward/source CSV and v03 YAML are not changed. Existing v03 models,
logs, trajectories and recovery records are retained. No new full training,
commit, push or automation is part of this cleanup.

Historical environment-validation documents and old-model MILP regression
tools remain intentionally: they explain model evolution and test compatibility.
They are not evidence for current formal experiments. Developer skills and the
Python virtual environment are also intentionally retained, not treated as junk.

## Verification

- `uv sync --offline` and `uv lock --check --offline` succeeded.
- Full unittest discovery: 93 tests passed in 82.878 seconds.
- Three subsequently added repository-scope checks also passed.
- Real v03 short training crossed warmup, with 8 learner updates and 11 EA calls.
- Compared 2,400 file hashes against the extracted pre-cleanup archive:
  environment code, source CSVs, v03 YAML and all retained v03 run artifacts
  were unchanged.
- `git diff --check` passed. No training/test subprocess was left running.

Validation logs are local under `results/cleanup_validation_20261018`.
Optional cache deletion was blocked by the execution policy; generated Python
caches remain and do not affect the retained source/configuration scope.
