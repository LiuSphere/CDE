"""Minimal online W&B initialization/log/finish probe, without training."""

import argparse
import faulthandler
import json
import time
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    faulthandler.dump_traceback_later(45)
    print("PROBE_INIT_BEGIN", flush=True)
    run = wandb.init(
        project="gde-microgrid", mode="online", dir=str(args.output),
        name=args.output.name, job_type="initialization-check",
        config={"purpose": "isolated online init/log/finish; no training"},
        settings=wandb.Settings(console="off", init_timeout=45, x_service_wait=30),
    )
    print(f"PROBE_INIT_OK id={run.id}", flush=True)
    run.log({"diagnostic/init_ok": 1})
    run.finish()
    faulthandler.cancel_dump_traceback_later()
    result = {"status": "init_log_finish_ok", "wandb_id": run.id,
              "seconds": time.monotonic() - start}
    with (args.output / "success.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
