"""H11's sole increment: synchronize targets only after even Actor updates."""

from learners.delayed_actor_learner import DelayedActorGAVDACLearner


class DelayedTargetsGAVDACLearner(DelayedActorGAVDACLearner):
    target_update_interval = 2
    diagnostics_filename = "h11_diagnostics.json"
    lineage_role = "H11_child_of_reconstructed_H03"

    def _target_phase(self):
        if self.critic_steps % self.target_update_interval == 0:
            return super()._target_phase()
        # A skipped synchronization must not hide an invalid configured mode.
        mode = getattr(self.args, "target_update_mode", "hard")
        if mode not in ("hard", "soft", "exponential_moving_average"):
            raise ValueError(f"unknown target update mode: {mode}")
