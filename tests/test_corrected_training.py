"""Exercise the real entry point through warmup, gradients and evolution."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import direct_main
from ea.ssne_base import SSNEBase
from learners.gavdac_learner import GAVDACLearner
from runners.episode_runner import EpisodeRunner


class TrainingTest(unittest.TestCase):
    def test_corrected_model_crosses_warmup_and_ea_cycles(self):
        for generations in (1, 3, 0):
            with self.subTest(generations=generations):
                with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
                    calls = {"learner": 0, "ea": 0}
                    train = GAVDACLearner.train
                    epoch = SSNEBase.epoch
                    rollout = EpisodeRunner.run
                    scenario_runs = []

                    def count_rollout(runner, *args, **kwargs):
                        options = kwargs.get("training_options")
                        if options is not None:
                            scenario_runs.append((options, kwargs.get("explore", True)))
                        return rollout(runner, *args, **kwargs)

                    def count_train(learner, batch, t_env, episode_num):
                        calls["learner"] += 1
                        self.assertGreaterEqual(t_env, 512 * 24)
                        self.assertEqual(batch.batch_size, 512)
                        self.assertEqual(tuple(batch["obs"].shape[-2:]), (3, 20))
                        self.assertEqual(batch["state"].shape[-1], 60)
                        self.assertEqual(tuple(batch["actions"].shape[-2:]), (3, 2))
                        self.assertEqual(batch["obs"].device.type, "cpu")
                        self.assertTrue(bool(batch["obs"].isfinite().all()))
                        value = train(learner, batch, t_env, episode_num)
                        for network in (learner.mac.agent, learner.critic, learner.mixer):
                            for parameter in network.parameters():
                                self.assertEqual(parameter.device.type, "cpu")
                                if parameter.grad is not None:
                                    self.assertTrue(bool(parameter.grad.isfinite().all()))
                        return value

                    def count_epoch(evolver, *args, **kwargs):
                        calls["ea"] += 1
                        if scenario_runs:
                            self.assertEqual(len(scenario_runs), 6)
                            self.assertTrue(all(s[0] == scenario_runs[0][0] for s in scenario_runs))
                            self.assertEqual([s[1] for s in scenario_runs], [False] * 5 + [True])
                            scenario_runs.clear()
                        return epoch(evolver, *args, **kwargs)

                    argv = [
                        "direct_main.py", "--config",
                        "gavdac_ea" if generations else "gavdac_no_ea",
                        "--env-config", "microgrid",
                        "--override", "wandb_mode=disabled",
                        "--override", "save_model=False",
                        "--override", "t_max=12500",
                        "--override", "test_interval=12000",
                        "--override", "fixed_eval_days=2",
                        "--override", f"ea_generations_per_cycle={generations or 1}",
                        "--override", f"env_args.result_dir={directory}",
                        "--override", f"wandb_dir={directory}/wandb",
                    ]
                    with patch.object(sys, "argv", argv), \
                         patch.object(GAVDACLearner, "train", count_train), \
                         patch.object(SSNEBase, "epoch", count_epoch), \
                         patch.object(EpisodeRunner, "run", count_rollout):
                        direct_main.main()
                    self.assertGreater(calls["learner"], 0)
                    if generations:
                        self.assertGreater(calls["ea"], 0)
                        self.assertEqual(calls["ea"] % generations, 0)
                    else:
                        self.assertEqual(calls["ea"], 0)
                    print(f"Verified generation={generations}: {calls}")


if __name__ == "__main__":
    unittest.main()
