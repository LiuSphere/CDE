"""H12: scaled unit-Huber TD regression, inherited Actor2 and every-step targets."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from learners.delayed_actor_learner import (
    DelayedActorGAVDACLearner, validate_update_intervals,
)
from learners.gavdac_learner import GAVDACLearner


def huber_enabled(args):
    value = getattr(args, "scaled_huber_critic", False)
    if type(value) is not bool:
        raise ValueError("scaled_huber_critic must be boolean")
    if value:
        if validate_update_intervals(args) != (2, 1):
            raise ValueError("H12 requires Actor interval2 and target interval1")
        if getattr(args, "use_cuda", False):
            raise ValueError("H12 requires CPU float32")
    return value


class HuberDelayedActorGAVDACLearner(DelayedActorGAVDACLearner):
    def __init__(self, mac, scheme, logger, args):
        self.scaled_huber_critic = huber_enabled(args)
        if not self.scaled_huber_critic:
            raise ValueError("H12 learner requires scaled_huber_critic:true")
        super().__init__(mac, scheme, logger, args)

    def _td_error_loss(self, masked_td_error, denominator):
        if (masked_td_error.device.type != "cpu" or masked_td_error.dtype != torch.float32
                or not torch.isfinite(masked_td_error).all()):
            raise ValueError("H12 requires finite CPUfloat32 TD errors")
        if not torch.isfinite(torch.as_tensor(denominator)).all() or denominator <= 0:
            raise ValueError("H12 requires finite positive valid-transition count")
        return 2 * F.huber_loss(
            masked_td_error, torch.zeros_like(masked_td_error), reduction="none", delta=1.,
        ).sum() / denominator

    def write_diagnostics(self, path, **context):
        value = {
            "lineage_role": "H12_child_of_reconstructed_H03",
            "actor_update_interval": self.actor_update_interval,
            "target_update_interval": self.target_update_interval,
            "target_update_tau": getattr(self.args, "target_update_tau", .001),
            "scaled_huber_critic": True, "huber_delta": 1., "huber_multiplier": 2.,
            "critic_optimizer_steps": self.critic_steps, "actor_optimizer_steps": self.actor_steps,
            "target_updates": self.target_updates, "finite_update_checks": self.finite_checks,
            "optimizer_post_step_finite_checks": self.optimizer_finite_checks,
            "train_updates": self.finite_checks, "exact_resume_supported": False, **context,
        }
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)

    def save_models(self, path):
        GAVDACLearner.save_models(self, path)
        self.write_diagnostics(Path(path) / "h12_diagnostics.json")

    def load_models(self, path):
        saved = json.loads((Path(path) / "h12_diagnostics.json").read_text(encoding="utf-8"))
        expected = {"lineage_role": "H12_child_of_reconstructed_H03",
                    "actor_update_interval": 2, "target_update_interval": 1,
                    "scaled_huber_critic": True, "huber_delta": 1., "huber_multiplier": 2.,
                    "exact_resume_supported": False}
        if any(json.dumps(saved.get(k)) != json.dumps(v) for k, v in expected.items()):
            raise ValueError("H12 saved diagnostic configuration mismatch")
        super().load_models(path)

