"""The selected main method omits crossover while retaining historical configs."""

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import direct_main
from ea.mod_neuro_evo import SSNE
from learners.gavdac_learner import GAVDACLearner


MAIN = "gavdac_ea_main"


def load_config(arguments):
    configs = []
    with patch.object(sys, "argv", ["direct_main.py", *arguments]), \
         patch.object(direct_main, "run", lambda config, logger: configs.append(config)):
        direct_main.main()
    return configs[0]


class MainAlgorithmTest(unittest.TestCase):
    def test_default_omits_crossover_without_changing_other_v03_parameters(self):
        historical = load_config(["--config", "gavdac_ea_tune_v03_long"])
        expected = {**deepcopy(historical), "name": MAIN, "crossover_prob": 0.0}
        self.assertEqual(load_config([]), expected)
        self.assertEqual(load_config(["--config", MAIN]), expected)
        self.assertTrue(expected["EA"])
        self.assertEqual(historical["crossover_prob"], 1.0)
        endpoint = load_config(["--config", "gavdac_v03_no_crossover"])
        self.assertEqual(expected, {**endpoint, "name": MAIN})

    def test_historical_crossover_candidate_is_unchanged(self):
        self.assertEqual(
            load_config(["--config", "gavdac_v03_crossover025"])["crossover_prob"],
            0.25,
        )

    def test_real_default_training_has_no_extra_clone_recombination(self):
        train, epoch, mutation = (
            GAVDACLearner.train, SSNE.epoch, SSNE.mutate_inplace,
        )
        for generations in (1, 3):
            counts = {"learner": 0, "ea": 0, "mutation": 0}

            def learn(learner, batch, t_env, episode):
                self.assertEqual(learner.args.name, MAIN)
                self.assertEqual(learner.args.crossover_prob, 0)
                self.assertGreaterEqual(t_env, 512 * 24)
                self.assertEqual(batch.batch_size, 128)
                before = [p.detach().clone() for p in learner.mac.parameters()]
                result = train(learner, batch, t_env, episode)
                self.assertTrue(any(not th.equal(a, b) for a, b in zip(
                    before, learner.mac.parameters(), strict=True,
                )))
                for network in (learner.mac.agent, learner.critic, learner.mixer):
                    for parameter in network.parameters():
                        self.assertEqual(parameter.device.type, "cpu")
                        self.assertTrue(bool(parameter.isfinite().all()))
                counts["learner"] += 1
                return result

            def evolve(evolver, population, fitness, index, agent_level=False):
                self.assertTrue(agent_level)
                self.assertEqual(evolver.args.crossover_prob, 0)
                self.assertEqual(evolver.num_elitists, 1)
                selected = []
                tournament = evolver.selection_tournament

                def select(*args, **kwargs):
                    result = tournament(*args, **kwargs)
                    selected.extend(result)
                    return result

                with patch.object(evolver, "selection_tournament", side_effect=select), \
                     patch.object(evolver, "clone", wraps=evolver.clone) as clones, \
                     patch.object(evolver, "crossover_inplace",
                                  side_effect=AssertionError("Crossover was called")):
                    result = epoch(evolver, population, fitness, index, agent_level=True)
                elite = int(np.argsort(fitness)[-1])
                unselected = sum(
                    i not in selected and i != elite for i in range(len(population))
                )
                # One elite copy plus two parent copies per remaining pair;
                # a clone-recombination call would exceed this count.
                remaining = max(0, unselected - 1)
                self.assertEqual(clones.call_count, 1 + 2 * ((remaining + 1) // 2))
                counts["ea"] += 1
                return result

            def mutate(evolver, *args, **kwargs):
                counts["mutation"] += 1
                return mutation(evolver, *args, **kwargs)

            with self.subTest(generations=generations), \
                 tempfile.TemporaryDirectory(prefix="main_", dir=ROOT / "results") as directory:
                argv = ["direct_main.py"]
                overrides = {
                    "t_max": 12500, "wandb_mode": "disabled", "save_model": False,
                    "showcase_enabled": False, "unique_token": f"main_g{generations}",
                    "local_results_path": directory, "env_args.result_dir": directory,
                    "wandb_dir": f"{directory}/wandb",
                    "ea_generations_per_cycle": generations,
                }
                for key, value in overrides.items():
                    argv += ["--override", f"{key}={value}"]
                with patch.object(sys, "argv", argv), \
                     patch.object(GAVDACLearner, "train", learn), \
                     patch.object(SSNE, "epoch", evolve), \
                     patch.object(SSNE, "mutate_inplace", mutate):
                    direct_main.main()
            self.assertGreater(counts["learner"], 0)
            self.assertGreater(counts["ea"], 0)
            self.assertGreater(counts["mutation"], 0)
            self.assertEqual(counts["ea"] % generations, 0)
            print(f"Verified crossover-free main, generations={generations}: {counts}")


if __name__ == "__main__":
    unittest.main()
