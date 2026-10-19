"""Budget-matched v03 ablation configurations and real training checks."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import direct_main
from ea.ssne_base import SSNEBase
from learners.gavdac_learner import GAVDACLearner


VARIANTS = {
    "gavdac_v03_no_ea": {"EA": False},
    "gavdac_v03_no_mutation": {"mutation_prob": 0.0},
    "gavdac_v03_no_crossover": {"crossover_prob": 0.0},
}


def load_config(name):
    configs = []
    with patch.object(sys, "argv", ["direct_main.py", "--config", name]), \
         patch.object(direct_main, "run", lambda config, logger: configs.append(config)):
        direct_main.main()
    return configs[0]


class PaperConfigTest(unittest.TestCase):
    def test_no_crossover_disables_agent_level_clone_recombination(self):
        from ea.mod_neuro_evo import SSNE
        for probability, clones, draws in ((0, 3, 4), (1, 4, 5)):
            args = SimpleNamespace(
                prob_reset_and_sup=.05, frac=1, pop_size=5, elite_fraction=.2,
                mutation_prob=0, crossover_prob=probability,
            )
            evolver = SSNE(args)
            with patch.object(evolver, "selection_tournament", return_value=[3, 4]), \
                 patch.object(evolver, "clone") as clone, \
                 patch("ea.ssne_base.random.shuffle"), \
                 patch("ea.ssne_base.random.random", return_value=.5) as random:
                evolver.epoch(list(range(5)), [5, 4, 3, 2, 1], 0, agent_level=True)
            self.assertEqual(clone.call_count, clones)
            self.assertEqual(random.call_count, draws)

    def test_only_declared_parameter_and_name_differ(self):
        parent = load_config("gavdac_ea_tune_v03_long")
        for name, changes in VARIANTS.items():
            with self.subTest(name=name):
                expected = {**parent, "name": name, **changes}
                self.assertEqual(load_config(name), expected)

    def test_real_ablations_cross_warmup_and_use_declared_operators(self):
        train, epoch = GAVDACLearner.train, SSNEBase.epoch
        from ea.mod_neuro_evo import SSNE
        actual_mutate = SSNE.mutate_inplace
        crossover = SSNE.crossover_inplace
        for name in VARIANTS:
            counts = {"learner": 0, "ea": 0, "mutation": 0, "crossover": 0}

            def learn(self, *args, **kwargs):
                counts["learner"] += 1
                return train(self, *args, **kwargs)

            def evolve(self, *args, **kwargs):
                counts["ea"] += 1
                return epoch(self, *args, **kwargs)

            def mutation(self, *args, **kwargs):
                counts["mutation"] += 1
                return actual_mutate(self, *args, **kwargs)

            def cross(self, *args, **kwargs):
                counts["crossover"] += 1
                return crossover(self, *args, **kwargs)

            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=ROOT / "results") as d:
                argv = ["direct_main.py", "--config", name]
                overrides = {
                    "t_max": 12500, "wandb_mode": "disabled", "save_model": False,
                    "showcase_enabled": False, "local_results_path": d,
                    "env_args.result_dir": d, "wandb_dir": f"{d}/wandb",
                    "unique_token": name,
                }
                for key, value in overrides.items():
                    argv += ["--override", f"{key}={value}"]
                with patch.object(sys, "argv", argv), \
                     patch.object(GAVDACLearner, "train", learn), \
                     patch.object(SSNEBase, "epoch", evolve), \
                     patch.object(SSNE, "mutate_inplace", mutation), \
                     patch.object(SSNE, "crossover_inplace", cross):
                    direct_main.main()
                self.assertGreater(counts["learner"], 0)
                if name == "gavdac_v03_no_ea":
                    self.assertEqual(counts["ea"], 0)
                else:
                    self.assertGreater(counts["ea"], 0)
                if name.endswith("no_mutation"):
                    self.assertEqual(counts["mutation"], 0)
                if name.endswith("no_crossover"):
                    self.assertEqual(counts["crossover"], 0)
                print(name, counts)


if __name__ == "__main__":
    unittest.main()
