"""Check compact dimensions and the real training entry point after warmup."""

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import direct_main
from controllers.cqmix_controller import CQMixMAC
from ea.ssne_base import SSNEBase
from envs.maenv import MGEnv
from learners.facmac_learner import FACMACLearner


def load_runtime_config(name):
    configs = []
    argv = ["direct_main.py", "--config", name, "--env-config", "microgrid"]
    with patch.object(sys, "argv", argv), \
         patch.object(direct_main, "run", lambda config, logger: configs.append(config)):
        direct_main.main()
    return configs[0]


class CompactConfigTest(unittest.TestCase):
    def test_dimensions_and_inherited_training_settings(self):
        baseline = load_runtime_config("facmac_ea")
        config = load_runtime_config("facmac_ea_compact")
        for key in baseline:
            if key not in {
                "name", "batch_size", "rnn_hidden_dim", "mixing_embed_dim", "test_interval",
            }:
                self.assertEqual(config[key], baseline[key], key)
        self.assertEqual(baseline["batch_size"], 512)
        self.assertEqual(baseline["rnn_hidden_dim"], 64)
        self.assertEqual(baseline["mixing_embed_dim"], 64)
        self.assertEqual(config["batch_size"], 128)
        self.assertEqual(config["t_max"], 3000000)
        self.assertEqual(config["t_max"], baseline["t_max"])
        self.assertFalse(config["use_cuda"])
        self.assertEqual(config["buffer_warmup"], 512)
        self.assertEqual(config["hypernet_embed"], 64)
        self.assertEqual(config["test_interval"], 10000)

        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            env_args = deepcopy(config["env_args"])
            env_args["result_dir"] = directory
            env = MGEnv(env_args, config["name"])
            try:
                info = env.get_env_info()
                self.assertEqual(
                    (info["n_agents"], info["obs_shape"], info["state_shape"], info["n_actions"]),
                    (3, 20, 60, 2),
                )
                config.update(
                    n_agents=info["n_agents"], n_actions=info["n_actions"],
                    state_shape=info["state_shape"], action_spaces=info["action_spaces"],
                )
                args = SimpleNamespace(**config)
                scheme = {
                    "obs": {"vshape": info["obs_shape"], "group": "agents"},
                    "actions": {"vshape": (info["n_actions"],), "group": "agents"},
                }
                mac = CQMixMAC(scheme, {"agents": info["n_agents"]}, args)
                learner = FACMACLearner(
                    mac, scheme, SimpleNamespace(log_stat=lambda *unused: None), args
                )
                self.assertEqual(mac.agent.fc1.in_features, 25)
                self.assertEqual(mac.agent.rnn.hidden_size, 48)
                self.assertEqual(mac.agent.fc2.out_features, 2)
                self.assertEqual(learner.critic.fc1.in_features, 22)
                self.assertEqual(learner.critic.fc1.out_features, 48)
                self.assertEqual(learner.mixer.embed_dim, 32)
                self.assertEqual(learner.mixer.mixing_GNN.layers[0].w2[-1].out_features, 512)
                self.assertEqual(learner.mixer.mixing_GNN.layers[0].w2[0].out_features, 64)
            finally:
                env.close()

    def test_compact_training_crosses_original_warmup_and_evolution(self):
        self._check_training("facmac_ea_compact", 100000)

    def test_v2_only_changes_storage_and_checkpoint_settings(self):
        original = load_runtime_config("facmac_ea_compact")
        updated = load_runtime_config("facmac_ea_compact_v2")
        expected = deepcopy(original)
        expected.update(
            name="facmac_ea_compact_v2", buffer_size=50000, save_model_interval=50000,
        )
        self.assertEqual(updated, expected)
        self.assertEqual(original["buffer_size"], 100000)
        self.assertEqual(original["save_model_interval"], 10000)
        self.assertEqual(updated["t_max"], 3000000)
        self.assertEqual(updated["batch_size"], 128)
        self.assertEqual(updated["buffer_warmup"], 512)
        self.assertFalse(updated["use_cuda"])

    def test_v2_training_crosses_original_warmup_and_evolution(self):
        self._check_training("facmac_ea_compact_v2", 50000)

    def _check_training(self, config_name, expected_buffer_size):
        calls = {"learner": 0, "ea": 0}
        original_train = FACMACLearner.train
        original_epoch = SSNEBase.epoch

        def check_train(learner, batch, t_env, episode_num):
            self.assertEqual(learner.args.buffer_size, expected_buffer_size)
            self.assertGreaterEqual(t_env, 512 * 24)
            self.assertEqual(batch.batch_size, 128)
            self.assertEqual(tuple(batch["obs"].shape[-2:]), (3, 20))
            self.assertEqual(batch["state"].shape[-1], 60)
            self.assertEqual(tuple(batch["actions"].shape[-2:]), (3, 2))
            self.assertEqual(learner.mac.agent.rnn.hidden_size, 48)
            before = [parameter.detach().clone() for parameter in learner.mac.parameters()]
            result = original_train(learner, batch, t_env, episode_num)
            self.assertTrue(any(
                not th.equal(left, right)
                for left, right in zip(before, learner.mac.parameters(), strict=True)
            ))
            for network in (learner.mac.agent, learner.critic, learner.mixer):
                for parameter in network.parameters():
                    self.assertEqual(parameter.device.type, "cpu")
                    self.assertTrue(bool(parameter.isfinite().all()))
                    if parameter.grad is not None:
                        self.assertTrue(bool(parameter.grad.isfinite().all()))
            calls["learner"] += 1
            return result

        def check_epoch(evolver, population, *args, **kwargs):
            self.assertTrue(all(mac.agent.rnn.hidden_size == 48 for mac in population))
            calls["ea"] += 1
            return original_epoch(evolver, population, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            argv = [
                "direct_main.py", "--config", config_name,
                "--env-config", "microgrid",
                "--override", "wandb_mode=disabled",
                "--override", "save_model=False",
                "--override", "t_max=12500",
                "--override", f"env_args.result_dir={directory}",
                "--override", f"wandb_dir={directory}/wandb",
            ]
            with patch.object(sys, "argv", argv), \
                 patch.object(FACMACLearner, "train", check_train), \
                 patch.object(SSNEBase, "epoch", check_epoch):
                direct_main.main()
        self.assertGreater(calls["learner"], 0)
        self.assertGreater(calls["ea"], 0)
        print(f"Verified {config_name} training: {calls}")


if __name__ == "__main__":
    unittest.main()
