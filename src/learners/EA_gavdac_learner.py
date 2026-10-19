from .gavdac_learner import GAVDACLearner


class EA_GAVDACLearner(GAVDACLearner):
    # Continuous EA relies on rollout/evolution/sync in `run.py`, not population-aware learner updates.
    uses_population_training = False

    def train(self, batch, t_env, episode_num):
        return super().train(batch, t_env, episode_num)
