import logging
from collections import defaultdict
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import wandb


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger
        self.stats = defaultdict(lambda: [])
        self.wandb_run = None
        self.pending_metrics = {}
        self.pending_step = None
        self.metric_patterns = None

    def setup_wandb(self, config, name, project, entity, mode, directory):
        self.metric_patterns = config.get("wandb_metric_patterns")
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            mode=mode,
            dir=directory,
            settings=wandb.Settings(console="off"),
        )
        self.wandb_run.define_metric("environment_step")
        self.wandb_run.define_metric("*", step_metric="environment_step")

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))
        if self.metric_patterns is not None and not any(
            fnmatchcase(key, pattern) for pattern in self.metric_patterns
        ):
            return
        if self.pending_step is not None and t != self.pending_step:
            self.flush()
        self.pending_step = t
        self.pending_metrics[key] = value.item() if hasattr(value, "item") else value

    def flush(self):
        if self.pending_metrics:
            self.pending_metrics["environment_step"] = self.pending_step
            # W&B advances its own history counter; environment_step is the axis.
            self.wandb_run.log(self.pending_metrics)
            self.pending_metrics = {}
            self.pending_step = None

    def log_dispatch_report(self, directory, report, images, step):
        source = report["policy_source"]
        if self.pending_step is not None and step != self.pending_step:
            self.flush()
        self.pending_step = step
        prefix = f"dispatch/{source}"
        columns = list(report["candidates"][0])
        self.pending_metrics[f"{prefix}/candidate_days"] = wandb.Table(
            columns=columns, data=[[r[k] for k in columns] for r in report["candidates"]],
        )
        self.pending_metrics[f"{prefix}/qualified_count"] = report["qualified_count"]
        for rank, plots in images.items():
            chosen = report["selected"][rank - 1]
            caption = (
                f"Selected validation case, not typical | row {chosen['row_idx']} | "
                f"qualified={chosen['qualified']} | initial SOC={report['initial_soc']} | "
                f"terminal error={chosen['max_terminal_soc_error']:.4f} | "
                f"checkpoint={report['checkpoint_step']}"
            )
            for name, path in plots.items():
                self.pending_metrics[f"{prefix}/best_case_{rank}/{name}"] = wandb.Image(
                    str(path), caption=caption,
                )
        self.wandb_run.summary[f"{prefix}/latest_report"] = str(Path(directory).resolve())
        self.wandb_run.summary[f"{prefix}/selection_scope"] = report["scope"]
        # Keep both policy reports and subsequent scalars in one step payload.

    def finish(self):
        self.flush()
        self.wandb_run.finish()

    def print_recent_stats(self, t_env):
        log_str = f"Recent Stats | t_env: {t_env:>10}\n"
        i = 0
        for k, v in sorted(self.stats.items()):
            i += 1
            values = []
            for _, raw_value in self.stats[k][-5:]:
                if isinstance(raw_value, (int, float)):
                    values.append(raw_value)
                elif hasattr(raw_value, "item"):
                    values.append(raw_value.item())
                else:
                    values.append(raw_value)
            item = f"{np.mean(values):.4f}"
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")
    return logger
