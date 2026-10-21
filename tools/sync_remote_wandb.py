"""Copy only the effective W&B credential to an explicitly authorized SSH host."""

import argparse
import json
import os
import re
import shlex
import subprocess

import wandb
from wandb.sdk.lib import wbauth


REMOTE_SETUP = r"""
import json, os, pathlib, sys, tempfile
import requests
from wandb.sdk.lib import wbauth
payload = json.load(sys.stdin)
base, key = payload["base_url"], payload["api_key"]
response = requests.post(
    base.rstrip("/") + "/graphql", auth=("api", key),
    json={"query": "query { viewer { username } }"}, timeout=30,
)
if response.status_code != 200:
    raise RuntimeError("W&B credential verification failed, HTTP " + str(response.status_code))
result = response.json()
if result.get("errors") or not result.get("data", {}).get("viewer"):
    raise RuntimeError("W&B viewer verification failed")
persist_netrc = payload.get("persist_netrc", True)
if persist_netrc:
    wbauth.write_netrc_auth(host=base, api_key=key)
directory = pathlib.Path.home() / ".config" / "gavdac"
directory.mkdir(parents=True, exist_ok=True, mode=0o700)
directory.chmod(0o700)
path = directory / "wandb.json"
with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".wandb_", delete=False) as file:
    temporary = pathlib.Path(file.name)
    json.dump(payload, file)
    file.flush()
    os.fsync(file.fileno())
temporary.chmod(0o600)
temporary.replace(path)
stored = json.loads(path.read_text())
if stored["api_key"] != key or (persist_netrc and wbauth.read_netrc_auth(host=base) != key):
    raise RuntimeError("Remote credential persistence check failed")
report = {
    "verified": True, "matches_local": True,
    "username": result["data"]["viewer"]["username"],
    "credential_file": str(path), "permissions": oct(path.stat().st_mode & 0o777),
    "conflicting_environment": bool(os.environ.get("WANDB_API_KEY")
                                    and os.environ["WANDB_API_KEY"] != key),
    "netrc_modified": persist_netrc,
}
if payload.get("probe_name"):
    os.environ.update(WANDB_API_KEY=key, WANDB_BASE_URL=base)
    import wandb
    probe_dir = pathlib.Path.cwd() / "results" / payload["probe_name"]
    probe_dir.mkdir(parents=True, exist_ok=False)
    run = wandb.init(
        project=payload["wandb_project"], name=payload["probe_name"], mode="online",
        dir=str(probe_dir), config={"credential_probe": True, "host": payload["host"]},
        settings=wandb.Settings(init_timeout=30),
    )
    run.log({"preflight/credential_probe": 1})
    report["probe_run_id"] = run.id
    report["probe_url"] = run.url
    run.finish()
    report["probe_finish_returned"] = True
print(json.dumps(report))
"""


def sync(host, project, probe_name=None, wandb_project="gde-microgrid", persist_netrc=True):
    if probe_name is not None and not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", probe_name):
        raise ValueError("Use a new short alphanumeric probe token")
    settings = wandb.setup().settings
    base = settings.base_url
    key = os.environ.get("WANDB_API_KEY") or settings.api_key or wbauth.read_netrc_auth(host=base)
    if not key:
        raise RuntimeError("No local W&B API key is configured")
    remote = shlex.join(["uv", "--directory", project, "run", "--frozen",
                         "python", "-c", REMOTE_SETUP])
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote],
        input=json.dumps({"base_url": base, "api_key": key, "probe_name": probe_name,
                          "wandb_project": wandb_project, "persist_netrc": persist_netrc,
                          "host": host}),
        text=True, encoding="utf-8", capture_output=True, timeout=90,
    )
    if result.returncode:
        # Never include the SSH stdin payload or an unsanitized SDK error in logs.
        detail = result.stderr.replace(key, "[REDACTED]")
        raise RuntimeError(f"Remote credential setup failed ({result.returncode}): {detail}")
    output = json.loads(result.stdout)
    if not output.get("verified") or not output.get("matches_local"):
        raise RuntimeError("Remote credential verification did not confirm equality")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--probe-name")
    parser.add_argument("--wandb-project", default="gde-microgrid")
    parser.add_argument("--credential-file-only", action="store_true",
                        help="Preserve existing global W&B login; write only the task credential file.")
    args = parser.parse_args()
    print(json.dumps(sync(args.host, args.project, args.probe_name, args.wandb_project,
                          persist_netrc=not args.credential_file_only)))


if __name__ == "__main__":
    main()
