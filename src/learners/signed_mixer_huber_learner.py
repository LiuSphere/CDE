"""H14: bounded signed mixer weights; all H12 losses and update phases inherit."""

import json
from pathlib import Path

from learners.delayed_actor_huber_learner import HuberDelayedActorGAVDACLearner, huber_enabled
from learners.gavdac_learner import GAVDACLearner


def signed_mixer_enabled(args):
    value = getattr(args, "signed_mixer_hyperweights", False)
    if type(value) is not bool:
        raise ValueError("signed_mixer_hyperweights must be boolean")
    if value:
        if (not huber_enabled(args) or getattr(args, "target_update_tau", None) != .002
                or getattr(args, "mixer_weight_scale", .1) != .1
                or getattr(args, "rnn_hidden_dim", None) != 48
                or getattr(args, "recurrent_critic", False)):
            raise ValueError("H14 requires the exact retained H12 architecture and update configuration")
        retired = (
            "actor_feature_layer_norm", "actor_mixer_context_grad", "clipped_double_critic",
            "five_step_critic", "critic_layer_norm", "target_policy_smoothing",
            "critic_small_output_init", "actor_no_action_l2", "critic_center_generator_action",
        )
        if any(name in vars(args) for name in retired):
            raise ValueError("H14 cannot include retired mechanism flags")
    return value


class SignedMixerHuberGAVDACLearner(HuberDelayedActorGAVDACLearner):
    def __init__(self, mac, scheme, logger, args):
        self.signed_mixer_hyperweights = signed_mixer_enabled(args)
        if not self.signed_mixer_hyperweights or getattr(args, "n_agents", 0) < 2:
            raise ValueError("H14 learner requires signed_mixer_hyperweights:true and a mixer")
        super().__init__(mac, scheme, logger, args)

    def write_diagnostics(self, path, **context):
        value = {
            "lineage_role": "H14_signed_mixer_child_of_retained_H12",
            "signed_mixer_hyperweights": True, "weights_operation": "signed_sigmoid",
            "mixer_weight_scale": .1,
            "actor_update_interval": self.actor_update_interval,
            "target_update_interval": self.target_update_interval,
            "target_update_tau": self.args.target_update_tau,
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
        self.write_diagnostics(Path(path) / "h14_diagnostics.json")

    def load_models(self, path):
        saved = json.loads((Path(path) / "h14_diagnostics.json").read_text(encoding="utf-8"))
        expected = {
            "lineage_role": "H14_signed_mixer_child_of_retained_H12",
            "signed_mixer_hyperweights": True, "weights_operation": "signed_sigmoid",
            "mixer_weight_scale": .1, "actor_update_interval": 2,
            "target_update_interval": 1, "target_update_tau": .002,
            "scaled_huber_critic": True, "huber_delta": 1., "huber_multiplier": 2.,
            "exact_resume_supported": False,
        }
        if any(json.dumps(saved.get(k)) != json.dumps(v) for k, v in expected.items()):
            raise ValueError("H14 saved diagnostic configuration mismatch")
        GAVDACLearner.load_models(self, path)
