# C2 Execution Hosts

User authorized the fourth slot on2026-10-21 and disabled715 on2026-10-25.
Current available capacity is three slots: local/0 and707/0,707/1.
715 remains in the historical inventory but is not eligible for checks,
credential synchronization, qualification or new launches until the user
explicitly reauthorizes it. See `results/c2opt21/decision_074.md` and the
[optimization protocol](c2_optimization.md) and
[hypothesis ledger](c2_hypotheses/README.md).
Qualification and live occupancy belong in `results/c2opt21/state.json`;
an entry in this table alone is not proof that a host is ready or idle.

| Slot | SSH Alias | Actual Host | Project Root | Capacity |
| --- | --- | --- | --- | --- |
| Local/0 | Local | Current Windows host | `C:/Users/xyx/Desktop/GDE-main` | 1 |
| 707/0 | `707` | `arch707` | `/home/arch707/data/CDE` | 2 total |
| 707/1 | `707` | `arch707` | `/home/arch707/data/CDE` | 2 total |
| arch715/0 (disabled) | `715` | `arch715` | `/home/bless/projects/CDE_c2opt21` | 0 available; 1 physical |

The configured alias `715` uses user `bless`, port2222. Plain `ssh arch715`
does not select that alias and would use the wrong default port/user.
The existing `/home/bless/projects/CDE` and other projects remain untouched.
The dedicated clone isolates its dependencies, results and slot lock.

## Qualification And Launch

Use the locked dependencies and the campaign Python version. Check source,
configuration and numeric input equivalence, record hardware/OS/thread details,
run host qualification with only the explicit old desktop SB3 exclusions,
and exercise the exclusive slot before declaring it ready.
CPU float32, OMP/MKL/OPENBLAS1thread per task; do not substitute GPU execution.

Check the new host's exclusive slot:

```powershell
ssh 715 'uv --directory /home/bless/projects/CDE_c2opt21 run --frozen python tools/experiment_slot.py --slots 1 --status'
```

Every launch there uses `--slots 1 --slot 0` and
`--credential-file /home/bless/.config/gavdac/wandb.json`, with a new token,
`--detach`, actual process identities and exit evidence.
Never infer vacancy from SSH failure or delete a stale lock automatically.

## Credential Isolation

Before a new remote wave, synchronize the effective local W&B key over SSH
stdin. Do not print or commit it. For the new host preserve global logins:

```powershell
uv run python tools/sync_remote_wandb.py --host 715 --project /home/bless/projects/CDE_c2opt21 --credential-file-only
```

This validates the remote API and writes only the repository-external0600
task file under a0700directory, without changing `.netrc` or other projects'
environment. Each supervised process explicitly overrides any old W&B
environment key from that task file. Apply the same isolation to future waves.

## Scheduling And Comparison Boundaries

The current available capacity is three slots, not four seeds.
The following fourth-slot procedures describe historical authorization;
they do not override the2026-10-25 user disablement.
H01retains its three already-running seeds on their original hosts.
Do not duplicate/migrate them or add a fourth seed to fill capacity.
The extra slot can perform qualified auxiliary work consistent with the
current single-hypothesis protocol. Any later training-host reassignment
must be registered and checked before launch, with hardware differences
disclosed; it is not an algorithmic speedup claim.

The same `gavdac` heartbeat checks every registered host and schedules at the
earliest active ETA plus a small buffer. A free fourth slot does not permit
an unregistered second hypothesis or changes to active frozen source.
Only synchronize repositories when their own active jobs have ended.

## Qualified Multiprocessing Execution

After H02closed, execution commit `9f03926` was merged into main and both
idle remotes synchronized ff-only. The future campaign's initial accepted
per-slot setting is `--rollout-workers 2 --learner-workers 2 --threads 1`,
not four new seeds or one process per slot. Two707slots have ten PyTorch
processes including their two training parents, plus supervisor/uv/service
helpers. Check actual free RAM and competing jobs again before each wave.

Local172tests passed;707and715each168tests passed with four explicitly
excluded old desktop fixtures, zero other failures/errors/skips. All118
source/config/data/dependency/tool/test files match after LFnormalization,
with72actual raw EOLdifferences per remote retained; parsed float64data
match. Never replace original-host checkpoint rawhash checks with this proof.

Four fixed12500joint-budget exclusive2/2engineering smokes passed:
local `mpls0a`,707 `mprs0a`/`mprs1a`,715 `mp715a`.
The707tasks actually overlapped25.750473seconds. W&Bwas disabled; no formal
seed, extra economic pilot or economic benefit was inferred. Baseline smoke
code does not emit final exact counters; normal budget-crossing completion,
finite runtime gradient/loss guards, finite available returns and original
launch/source/identity hashes were checked without inventing unlogged counts.
Formal training still requires its actual final W&Binteraction counters.

Evidence under `results/c2opt21/`: immutable `mp_adoption_protocol_a.json`,
`mp_adoption_acceptance_a.json`, `mp_exclusive_smokes_acceptance_a.json`,
host comparisons/qualifications and `decision_014.md`. These qualify execution,
not long-horizon identical policies, guaranteed remote speedup, or a new
economic hypothesis. Shared YAMLdefaults remain serial for compatibility;
campaign launch manifests must explicitly include the accepted worker counts.
