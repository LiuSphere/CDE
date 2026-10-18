"""Guard the microgrid-only runtime and the retained active configurations."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from runners.episode_runner import EpisodeRunner
from read_wandb_runs import DEFAULT_PROJECT


class RepositoryScopeTest(unittest.TestCase):
    def test_active_configs_are_v03_and_generic_controls(self):
        self.assertEqual(
            {path.stem for path in (ROOT / "src/config/algs").glob("*.yaml")},
            {
                "facmac_ea", "facmac_ea_tune_v03_long", "facmac_no_ea",
                "facmac_ea_no_mutation", "facmac_ea_no_crossover",
                "facmac_ea_no_variation",
            },
        )

    def test_non_microgrid_environment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported environment"):
            EpisodeRunner(SimpleNamespace(batch_size_run=1, env="retired"), None)

    def test_wandb_reader_targets_microgrid(self):
        self.assertEqual(DEFAULT_PROJECT, "blessingbrandro/gde-microgrid")


if __name__ == "__main__":
    unittest.main()
