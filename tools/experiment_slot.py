"""Supervise one CPU experiment with an exclusive slot and durable exit evidence."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.GetProcessTimes.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        )
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            creation, exit_time, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                         ctypes.byref(kernel_time), ctypes.byref(user_time)):
                raise ctypes.WinError(ctypes.get_last_error())
            return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    path = Path("/proc") / str(pid) / "stat"
    try:
        contents = path.read_text()
    except FileNotFoundError:
        return None
    fields = contents.rsplit(")", 1)[1].split()
    return fields[19]


def job_environment(threads, credential_file=None):
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("CPU threads must be a positive integer")
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES="",
               PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    # Independent jobs must own a backend that outlives their short-lived launcher.
    env.pop("WANDB_SERVICE", None)
    if credential_file:
        path = Path(credential_file).expanduser()
        if os.name != "nt" and path.stat().st_mode & 0o077:
            raise ValueError("Credential file must not be accessible by group/other")
        credential = json.loads(path.read_text())
        if not credential.get("api_key") or not credential.get("base_url"):
            raise ValueError("Credential file is incomplete")
        env.update(WANDB_API_KEY=credential["api_key"], WANDB_BASE_URL=credential["base_url"])
    return env


def slot_status(slots):
    result = []
    for slot in range(slots):
        path = ROOT / "results/cmp20/slot_locks" / f"slot{slot}.json"
        try:
            lock = json.loads(path.read_text())
        except FileNotFoundError:
            result.append({"slot": slot, "state": "free"})
            continue
        identity = process_identity(lock["supervisor_pid"])
        result.append({"slot": slot, "state": "running" if identity == lock["supervisor_identity"]
                       else "stale_lock", **lock})
    return result


def validate_job(config, seed, token, slot, slots):
    if config not in {"gavdac_ea_main", "gavdac_ea_main_no_ea", "gavdac_mappo"}:
        raise ValueError("Only the fresh formal matrix is supported")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", token):
        raise ValueError("Use a new short lowercase alphanumeric token")
    if not 0 <= slot < slots <= 2 or seed not in (0, 1, 2):
        raise ValueError("Invalid slot or formal seed")


def supervise(config, seed, token, slot, slots, threads=1, credential_file=None, smoke=False,
              engineering=False, evaluation_manifest=None):
    validate_job(config, seed, token, slot, slots)
    if engineering and (smoke or config != "gavdac_mappo"):
        raise ValueError("Engineering budget is only for the independent MAPPO pilot")
    if evaluation_manifest and (engineering or smoke):
        raise ValueError("Evaluation cannot use training smoke/engineering flags")
    manifest_path = Path(evaluation_manifest).resolve() if evaluation_manifest else None
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path else None
    env = job_environment(threads, credential_file)
    if not (ROOT / "src/config/algs" / f"{config}.yaml").is_file():
        raise ValueError("Algorithm configuration is not implemented")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required")
    locks = ROOT / "results/cmp20/slot_locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock_path = locks / f"slot{slot}.json"
    lock = {
        "token": token, "supervisor_pid": os.getpid(),
        "supervisor_identity": process_identity(os.getpid()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with lock_path.open("x", encoding="utf-8") as file:
        json.dump(lock, file)
    process = None
    try:
        directory = ROOT / "results" / token
        directory.mkdir(exist_ok=False)
        command = [uv, "run", "--frozen", "python", "src/direct_main.py", "--config", config,
                   "--env-config", "microgrid"]
        overrides = {
            "seed": seed, "unique_token": token, "local_results_path": directory.as_posix(),
            "env_args.result_dir": directory.as_posix(),
            "wandb_dir": (directory / "wandb").as_posix(), "use_cuda": False,
            "wandb_mode": "disabled" if smoke else "online",
        }
        if smoke:
            overrides.update(t_max=12500, save_model=False, showcase_enabled=False)
        if engineering:
            overrides.update(t_max=500000)
        for key, value in overrides.items():
            command += ["--override", f"{key}={json.dumps(value)}"]
        if evaluation_manifest:
            command = [
                uv, "run", "--frozen", "python", "tools/run_comparison_queue.py",
                "--manifest", str(manifest_path),
                "--seed", str(seed), "--token", token,
            ]
        source_hashes = {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src").rglob("*"))
            if path.suffix in {".py", ".yaml", ".csv"}
        }
        started = time.monotonic()
        with (directory / "train.stdout.log").open("x", encoding="utf-8") as stdout, \
             (directory / "train.stderr.log").open("x", encoding="utf-8") as stderr:
            launch_options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                       **launch_options)
            launch = {
                **lock, "config": config, "seed": seed, "slot": slot, "slots": slots,
                "command": command, "child_pid": process.pid,
                "child_identity": process_identity(process.pid),
                "platform": platform.platform(), "cpu_threads": threads,
                "python_version": platform.python_version(),
                "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "credential_file_used": bool(credential_file), "smoke": smoke,
                "wandb_service_isolation": "independent_child_backend",
                "engineering": engineering,
                "evaluation_manifest": str(manifest_path) if manifest_path else None,
                "evaluation_manifest_sha256": manifest_hash,
                "source_sha256": source_hashes,
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                ).strip(),
            }
            with (directory / "launch.json").open("x", encoding="utf-8") as file:
                json.dump(launch, file, indent=2)
            print(json.dumps({"token": token, "slot": slot, "child_pid": process.pid}), flush=True)
            returncode = process.wait()
        result = {
            "token": token, "slot": slot, "returncode": returncode,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "status": "exited_pending_validation",
        }
        with (directory / "exit.json").open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        print(json.dumps(result), flush=True)
        return returncode
    finally:
        # This ephemeral lock belongs exclusively to this supervisor, not to a run artifact.
        if process is None or process.poll() is not None:
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--slots", type=int, choices=[1, 2], required=True)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--token")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(slot_status(args.slots)))
        return
    if any(value is None for value in (args.slot, args.config, args.seed, args.token)):
        parser.error("Launching requires --slot, --config, --seed and --token")
    validate_job(args.config, args.seed, args.token, args.slot, args.slots)
    if args.detach:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required")
        logs = ROOT / "results/cmp20/supervisors"
        logs.mkdir(parents=True, exist_ok=True)
        command = [uv, "run", "--frozen", "python", str(Path(__file__).resolve()),
                   *(arg for arg in sys.argv[1:] if arg != "--detach")]
        options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
                   if os.name == "nt" else {"start_new_session": True})
        with (logs / f"{args.token}.stdout.log").open("x", encoding="utf-8") as stdout, \
             (logs / f"{args.token}.stderr.log").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                close_fds=True, env=job_environment(args.threads), **options,
            )
        print(json.dumps({"token": args.token, "manager_pid": process.pid,
                          "manager_identity": process_identity(process.pid),
                          "status": "detached_pending_slot_acquisition"}))
        return
    raise SystemExit(supervise(args.config, args.seed, args.token, args.slot, args.slots,
                               args.threads, args.credential_file, args.smoke, args.engineering,
                               args.evaluation_manifest))


if __name__ == "__main__":
    main()
