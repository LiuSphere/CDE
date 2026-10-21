"""Spawned rollout equivalence, ordering, failures, and complete learner updates."""

import copy
import hashlib
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fastrand
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from components.episode_buffer import ReplayBuffer
from controllers.cqmix_controller import CQMixMAC
from envs.maenv import MGEnv
from learners.gavdac_learner import GAVDACLearner
from run import run
from runners.episode_runner import EpisodeRunner
from runners.rollout_pool import _snapshot
from test_paper_experiments import load_config
from utils.logging import get_logger
from utils.policy_evaluation import evaluate_fixed
import direct_main


class StatsLog:
    def __init__(self):
        self.values = []

    def log_stat(self, *args):
        self.values.append(args)


def runner_fixture(directory, workers):
    args = SimpleNamespace(**load_config("gavdac_ea_main"))
    args.device = "cpu"
    args.rollout_workers = workers
    args.env_args["result_dir"] = directory
    runner = EpisodeRunner(args, StatsLog())
    for name, value in runner.get_env_info().items():
        setattr(args, name, value)
    scheme = {
        "state": {"vshape": args.state_shape},
        "obs": {"vshape": args.obs_shape, "group": "agents"},
        "actions": {"vshape": (args.n_actions,), "group": "agents"},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    mac = CQMixMAC(scheme, groups, args)
    runner.setup(scheme, groups, {}, mac)
    return runner, mac


def tensor_digest(value):
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            digest.update(item.detach().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                digest.update(str(key).encode())
                add(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                add(child)
        else:
            digest.update(repr(item).encode())
    add(value)
    return digest.hexdigest()


def training_trace(directory, workers, timesteps=2400):
    random.seed(39)
    np.random.seed(39)
    torch.manual_seed(39)
    fastrand.pcg32_seed(39)
    config = load_config("gavdac_ea_main")
    config.update(
        rollout_workers=workers, t_max=timesteps - 1, buffer_size=128,
        buffer_warmup=8, batch_size=8, start_timesteps=48,
        test_interval=480, fixed_eval_days=2, test_nepisode=2,
        wandb_mode="disabled", save_model=False, showcase_enabled=False,
        log_interval=100000, runner_log_interval=100000,
        learner_log_interval=100000, local_results_path=directory,
        unique_token=f"mp{workers}",
    )
    config["env_args"]["result_dir"] = directory
    trace = {"replay": [], "updates": [], "rng": None}
    original_insert = ReplayBuffer.insert_episode_batch
    original_train = GAVDACLearner.train
    original_init = MGEnv.__init__

    def seed_spaces(env, *args, **kwargs):
        original_init(env, *args, **kwargs)
        for index, space in enumerate(env.get_env_info()["action_spaces"]):
            space.seed(39 + index)

    def insert(buffer, batch):
        trace["replay"].append(tensor_digest(batch.data.transition_data))
        return original_insert(buffer, batch)

    def train(learner, *args, **kwargs):
        result = original_train(learner, *args, **kwargs)
        state = {
            name: getattr(learner, name).state_dict()
            for name in ("critic", "mixer", "target_critic", "target_mixer",
                         "agent_optimiser", "critic_optimiser")
        }
        state["actor"] = learner.mac.agent.state_dict()
        state["target_actor"] = learner.target_mac.agent.state_dict()
        state["gradients"] = [p.grad for p in learner.mac.parameters()]
        trace["updates"].append((args[1:], tensor_digest(state)))
        return result

    logger = get_logger()
    logger.setLevel("ERROR")
    with patch.object(MGEnv, "__init__", seed_spaces), \
         patch.object(ReplayBuffer, "insert_episode_batch", insert), \
         patch.object(GAVDACLearner, "train", train):
        run(config, logger)
    trace["rng"] = tensor_digest((
        torch.get_rng_state(), np.random.get_state(), random.getstate(),
    ))
    return trace


class ParallelRolloutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        (ROOT / "results").mkdir(exist_ok=True)

    def test_ea_batch_inventory_hidden_and_stats_are_exact(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            serial, mac = runner_fixture(directory, 0)
            parallel, _ = runner_fixture(directory, 2)
            policies = [copy.deepcopy(mac) for _ in range(5)]
            candidates = copy.deepcopy(policies)
            try:
                scenario = serial.env.training_scenario()
                expected = serial.run_group(policies, training_options=scenario)
                actual = parallel.run_group(candidates, training_options=scenario)
                self.assertEqual(serial.t_env, 120)
                self.assertEqual(parallel.t_env, serial.t_env)
                for (a, av), (b, bv) in zip(expected, actual, strict=True):
                    self.assertEqual(av, bv)
                    torch.testing.assert_close(a.data.transition_data,
                                               b.data.transition_data, rtol=0, atol=0)
                for a, b in zip(policies, candidates, strict=True):
                    torch.testing.assert_close(a.hidden_states, b.hidden_states,
                                               rtol=0, atol=0)
                self.assertEqual(tensor_digest(_snapshot(serial.env)),
                                 tensor_digest(_snapshot(parallel.env)))
                self.assertEqual(serial.logger.values, parallel.logger.values)
                self.assertEqual(len(parallel.rollout_pool.worker_pids), 2)
            finally:
                serial.close_env()
                parallel.close_env()
            self.assertIsNone(parallel.rollout_pool)

    def test_fixed_days_and_recorded_csv_preserve_training_restore(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            serial, mac = runner_fixture(f"{directory}/serial", 0)
            parallel, _ = runner_fixture(f"{directory}/parallel", 2)
            candidate = copy.deepcopy(mac)
            try:
                serial.run(mac, explore=False)
                parallel.run(candidate, explore=False)
                for rows in ([731, 740, 732], [750, 731]):
                    expected = evaluate_fixed(serial, mac, rows, .35, "validation",
                                              "rl", 24, record=True)
                    actual = evaluate_fixed(parallel, candidate, rows, .35, "validation",
                                            "rl", 24, record=True)
                    self.assertEqual(expected, actual)
                    self.assertEqual(tensor_digest(_snapshot(serial.env)),
                                     tensor_digest(_snapshot(parallel.env)))
                a, av = serial.run(mac, explore=False)
                b, bv = parallel.run(candidate, explore=False)
                self.assertEqual(av, bv)
                torch.testing.assert_close(a.data.transition_data,
                                           b.data.transition_data, rtol=0, atol=0)
                self.assertEqual(serial.logger.values, parallel.logger.values)
                left = next(serial.env.render_dir.glob("*.csv")).read_bytes()
                right = next(parallel.env.render_dir.glob("*.csv")).read_bytes()
                self.assertEqual(left, right)
            finally:
                serial.close_env()
                parallel.close_env()

    def test_worker_errors_propagate_without_parent_commit(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            runner, mac = runner_fixture(directory, 2)
            try:
                before = tensor_digest(_snapshot(runner.env))
                with self.assertRaisesRegex(ValueError, "not in the validation split"):
                    runner.run_group(
                        [mac, mac], test_mode=True, test_options=[
                            {"row_idx": 731, "source": "validation", "initial_soc": .35},
                            {"row_idx": 0, "source": "validation", "initial_soc": .35},
                        ],
                    )
                self.assertEqual(before, tensor_digest(_snapshot(runner.env)))
                self.assertEqual(runner.t_env, 0)
            finally:
                runner.close_env()

    def test_complete_training_replay_updates_optimizers_and_rng_match(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            serial = training_trace(f"{directory}/s", 0)
            parallel = training_trace(f"{directory}/p", 2)
        self.assertGreater(len(serial["updates"]), 10)
        self.assertEqual(serial, parallel)

    def test_cli_combined_pools_cross_warmup_and_close(self):
        learners = []
        original = GAVDACLearner.train

        def capture(learner, *args, **kwargs):
            result = original(learner, *args, **kwargs)
            self.assertTrue(all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in learner.agent_params + learner.critic_params
            ))
            if not learners:
                learners.append(learner)
            return result

        with tempfile.TemporaryDirectory(dir=ROOT / "results") as directory:
            argv = ["direct_main.py", "--config", "gavdac_c2_h01"]
            for key, value in {
                "t_max": 2400, "buffer_size": 128, "buffer_warmup": 8,
                "batch_size": 16, "start_timesteps": 48,
                "rollout_workers": 2, "learner_workers": 2,
                "test_interval": 1000, "fixed_eval_days": 2,
                "wandb_mode": "disabled", "save_model": False,
                "showcase_enabled": False, "local_results_path": directory,
                "env_args.result_dir": directory, "wandb_dir": directory,
                "unique_token": "combined_pools",
            }.items():
                argv += ["--override", f"{key}={value}"]
            with patch.object(sys, "argv", argv), \
                 patch.object(GAVDACLearner, "train", capture):
                direct_main.main()
        self.assertEqual(len(learners), 1)
        self.assertIsNone(learners[0].gradient_pool)


if __name__ == "__main__":
    unittest.main()
