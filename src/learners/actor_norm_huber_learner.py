"""H13 identification; H12 losses, update ordering and finite hooks are inherited."""

import json
from pathlib import Path

from learners.delayed_actor_huber_learner import HuberDelayedActorGAVDACLearner
from learners.gavdac_learner import GAVDACLearner
from modules.actor_feature_norm import LayerNormRNNAgent, actor_feature_norm_enabled


class ActorNormHuberGAVDACLearner(HuberDelayedActorGAVDACLearner):
    def __init__(self, mac, scheme, logger, args):
        if not actor_feature_norm_enabled(args) or type(mac.agent) is not LayerNormRNNAgent:
            raise ValueError("H13 requires its explicit normalized Actor")
        super().__init__(mac, scheme, logger, args)

    def write_diagnostics(self, path, **context):
        super().write_diagnostics(
            path, **{**context, "lineage_role": "H13_child_of_retained_H12",
                     "actor_feature_layer_norm": True, "actor_feature_norm_width": 48,
                     "actor_feature_norm_eps": 1e-5, "actor_feature_norm_affine": True},
        )

    def save_models(self, path):
        GAVDACLearner.save_models(self, path)
        self.write_diagnostics(Path(path) / "h13_diagnostics.json")

    def load_models(self, path):
        saved = json.loads((Path(path) / "h13_diagnostics.json").read_text(encoding="utf-8"))
        expected = {
            "lineage_role": "H13_child_of_retained_H12", "actor_update_interval": 2,
            "target_update_interval": 1, "scaled_huber_critic": True,
            "huber_delta": 1., "huber_multiplier": 2.,
            "actor_feature_layer_norm": True, "actor_feature_norm_width": 48,
            "actor_feature_norm_eps": 1e-5, "actor_feature_norm_affine": True,
            "exact_resume_supported": False,
        }
        if any(json.dumps(saved.get(k)) != json.dumps(v) for k, v in expected.items()):
            raise ValueError("H13 saved diagnostic configuration mismatch")
        GAVDACLearner.load_models(self, path)
