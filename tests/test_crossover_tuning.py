"""Reduced clone recombination: entry config, actual operator and CPU smoke."""

import sys
import tempfile
import unittest
from copy import deepcopy
from itertools import chain, repeat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import direct_main
from ea.mod_neuro_evo import SSNE
from learners.gavdac_learner import GAVDACLearner


CONFIG = "gavdac_v03_crossover025"


def load_config(name):
    configs = []
    with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
         patch.object(direct_main, "run", lambda config, logger: configs.append(config)):
        direct_main.main()
    return configs[0]


class CrossoverTuningTest(unittest.TestCase):
    def test_entry_changes_only_probability_and_experiment_name(self):
        parent = load_config("gavdac_ea_tune_v03_long")
        expected = {**deepcopy(parent), "name": CONFIG, "crossover_prob": 0.25}
        self.assertEqual(load_config(CONFIG), expected)
        self.assertEqual(parent["crossover_prob"], 1.0)
        self.assertEqual(load_config("gavdac_v03_no_crossover")["crossover_prob"], 0)

    def test_actual_epoch_reduces_identical_child_pairs(self):
        collapsed = {}
        with th.random.fork_rng(devices=[]):
            th.manual_seed(0)
            for probability in (0.0, 0.25, 1.0):
                collapsed[probability] = 0
                for sample in range(100):
                    args = SimpleNamespace(
                        prob_reset_and_sup=0.05, frac=1, pop_size=5,
                        elite_fraction=0.2, mutation_prob=0,
                        crossover_prob=probability,
                    )
                    evolver = SSNE(args)
                    population = []
                    for value in range(5):
                        agent = th.nn.Linear(1, 1, bias=False)
                        with th.no_grad():
                            agent.weight.fill_(value)
                        population.append(SimpleNamespace(agent=agent))
                    # Leave two distinct child slots (2, 3); isolate the gate
                    # from tournament noise and subsequent mutation.
                    draws = chain([(sample + 0.5) / 100], repeat(0.5))
                    with patch.object(evolver, "selection_tournament", return_value=[4, 4]), \
                         patch("ea.ssne_base.random.shuffle"), \
                         patch("ea.ssne_base.random.choice", side_effect=lambda xs: xs[0]), \
                         patch("ea.ssne_base.random.random", side_effect=lambda: next(draws)):
                        elite = evolver.epoch(
                            population, [5, 4, 3, 2, 1], 0, agent_level=True,
                        )
                    self.assertEqual(elite, 1)
                    self.assertEqual(population[elite].agent.weight.item(), 0)
                    left, right = (population[i].agent.weight for i in (2, 3))
                    collapsed[probability] += int(th.equal(left, right))
                    self.assertTrue(all(bool(
                        candidate.agent.weight.isfinite().all()
                    ) for candidate in population))
        self.assertEqual(collapsed, {0.0: 0, 0.25: 25, 1.0: 100})
        print("Controlled identical-child pairs per 100 eligible pairs:", collapsed)

    def test_real_entry_crosses_warmup_and_uses_partial_crossover(self):
        counts = {"learner": 0, "ea": 0}
        original_train, original_epoch = GAVDACLearner.train, SSNE.epoch

        def learn(learner, batch, t_env, episode):
            self.assertEqual(learner.args.crossover_prob, 0.25)
            self.assertGreaterEqual(t_env, 512 * 24)
            self.assertEqual(batch.batch_size, 128)
            result = original_train(learner, batch, t_env, episode)
            counts["learner"] += 1
            for network in (learner.mac.agent, learner.critic, learner.mixer):
                for parameter in network.parameters():
                    self.assertEqual(parameter.device.type, "cpu")
                    self.assertTrue(bool(parameter.isfinite().all()))
            return result

        def evolve(evolver, *args, **kwargs):
            self.assertEqual(evolver.args.crossover_prob, 0.25)
            self.assertTrue(kwargs["agent_level"])
            result = original_epoch(evolver, *args, **kwargs)
            counts["ea"] += 1
            return result

        with tempfile.TemporaryDirectory(prefix="cx25_", dir=ROOT / "results") as directory:
            argv = ["direct_main.py", "--config", CONFIG, "--env-config", "microgrid"]
            overrides = {
                "t_max": 12500, "wandb_mode": "disabled", "save_model": False,
                "showcase_enabled": False, "unique_token": "cx25sm",
                "local_results_path": directory, "env_args.result_dir": directory,
                "wandb_dir": f"{directory}/wandb",
            }
            for key, value in overrides.items():
                argv += ["--override", f"{key}={value}"]
            with patch.object(sys, "argv", argv), \
                 patch.object(GAVDACLearner, "train", learn), \
                 patch.object(SSNE, "epoch", evolve):
                direct_main.main()
        self.assertGreater(counts["learner"], 0)
        self.assertGreater(counts["ea"], 0)
        print("Reduced-crossover CPU smoke counts:", counts)


if __name__ == "__main__":
    unittest.main()
