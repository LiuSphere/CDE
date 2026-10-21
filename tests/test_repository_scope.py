"""Guard the microgrid-only runtime and the retained active configurations."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from runners.episode_runner import EpisodeRunner
from read_wandb_runs import DEFAULT_PROJECT
import direct_main


class RepositoryScopeTest(unittest.TestCase):
    def test_all_algorithm_configs_load_through_entrypoint_with_matching_names(self):
        for path in sorted((ROOT / "src/config/algs").glob("*.yaml")):
            with self.subTest(config=path.stem):
                with patch.object(sys, "argv", ["direct_main.py", "--config", path.stem]), \
                     patch.object(direct_main, "run") as run:
                    direct_main.main()
                config, _ = run.call_args.args
                self.assertTrue(path.stem.startswith("gavdac_"))
                self.assertEqual(config["name"], path.stem)
                self.assertEqual(config["env_args"]["model_version"], "gavdac_dispatch_v3")

    def test_active_configs_are_v03_and_generic_controls(self):
        self.assertEqual(
            {path.stem for path in (ROOT / "src/config/algs").glob("*.yaml")},
            {
                "gavdac_ea_main", "gavdac_ea_main_no_ea", "gavdac_mappo",
                "gavdac_ea", "gavdac_ea_tune_v03_long", "gavdac_no_ea",
                "gavdac_ea_no_mutation", "gavdac_ea_no_crossover",
                "gavdac_ea_no_variation",
                "gavdac_v03_no_ea", "gavdac_v03_no_mutation",
                "gavdac_v03_no_crossover",
                "gavdac_v03_crossover025",
            },
        )

    def test_non_microgrid_environment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported environment"):
            EpisodeRunner(SimpleNamespace(batch_size_run=1, env="retired"), None)

    def test_wandb_reader_targets_microgrid(self):
        self.assertEqual(DEFAULT_PROJECT, "blessingbrandro/gde-microgrid")


if __name__ == "__main__":
    unittest.main()
