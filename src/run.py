import datetime
import json
import os
import pprint
import threading
import time
from os.path import abspath, dirname
from types import SimpleNamespace as SN
from copy import deepcopy
from pathlib import Path
from contextlib import ExitStack

import numpy as np
import torch as th

from components.episode_buffer import ReplayBuffer
from controllers.cqmix_controller import CQMixMAC
from ea import mod_neuro_evo as utils_ne
from learners.EA_gavdac_learner import EA_GAVDACLearner
from learners.gavdac_learner import GAVDACLearner
from learners.target_smoothing_learner import TargetSmoothedGAVDACLearner, target_policy_smoothing_enabled
from runners.episode_runner import EpisodeRunner
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from utils.policy_evaluation import (
    BestPolicyStore, evaluate_fixed, evaluate_saved_policy, fixed_rows,
)

def rl_to_evo(rl_agent, evo_net):
    for target_param, param in zip(
        evo_net.agent.parameters(), rl_agent.agent.parameters(), strict=False
    ):
        target_param.data.copy_(param.data)


def run(_config, _log):
    target_policy_smoothing_enabled(SN(**_config))
    if _config["name"] == "gavdac_mappo":
        from run_mappo import run_mappo
        return run_mappo(_config, _log)
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    if args.use_cuda:
        th.set_float32_matmul_precision(
            getattr(args, "float32_matmul_precision", "highest")
        )

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    unique_token = getattr(args, "unique_token", None) or "{}__{}".format(
        args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    args.unique_token = unique_token
    if args.env == "microgrid":
        output_id = getattr(args, "evaluation_output_id", "") if args.evaluate else ""
        args.env_args["run_id"] = output_id or unique_token
        trajectory_path = (Path(args.env_args["result_dir"]) / "microgrid_trajs"
                           / args.name / args.env_args["run_id"])
        if trajectory_path.exists() and any(trajectory_path.iterdir()):
            raise FileExistsError(f"Use a new run/output id; artifacts exist at {trajectory_path}")
    logger.setup_wandb(
        config=_config,
        name=unique_token,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        directory=args.wandb_dir,
    )

    # Run and train
    run_sequential(args=args, logger=logger)
    logger.finish()

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print(f"Thread {t.name} is alive! Is daemon: {t.daemon}")
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

def run_sequential(args, logger):
    workers = getattr(args, "rollout_workers", 0)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("rollout_workers must be a nonnegative integer")
    runner = EpisodeRunner(args=args, logger=logger)
    with ExitStack() as resources:
        resources.callback(runner.close_env)
        return _run_with_runner(args, logger, runner, resources)


def _run_with_runner(args, logger, runner, resources):

    saved_config = deepcopy(vars(args))
    if getattr(args, "policy_checkpoint", "") and not args.evaluate:
        raise ValueError("policy_checkpoint is inference-only; set evaluate=True")
    if args.env == "microgrid" and args.use_cuda:
        raise ValueError("Microgrid training and evaluation must run on CPU")
    # Init runner so we can get env info

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.action_spaces = env_info["action_spaces"]
    args.actions_dtype = env_info["actions_dtype"]
    args.normalise_actions = env_info.get("normalise_actions", False)

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {
            "vshape": (max(space.shape[0] for space in args.action_spaces),),
            "group": "agents",
            "dtype": th.float,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        # Keep environment rollouts and replay storage on CPU. Training samples
        # are moved to the model device immediately before learner updates.
        device="cpu",
    )
    evolver = utils_ne.SSNE(args)
    # Setup multiagent controller here
    if args.EA:
        pop = []
        mac = CQMixMAC(buffer.scheme, groups, args)
        for i in range(args.pop_size):
            pop.append(CQMixMAC(buffer.scheme, groups, args))
    else:
        mac = CQMixMAC(buffer.scheme, groups, args)
        pop = []
    fitness = []
    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    if getattr(args, "policy_checkpoint", ""):
        evaluate_saved_policy(runner, mac, args)
        return

    policy_store = None
    showcase = None
    validation_rows = None
    if args.env == "microgrid":
        validation_rows = fixed_rows(
            runner.env, "validation", args.validation_rows, args.fixed_eval_days,
        )
        if args.save_model and args.save_best_policy and not args.evaluate:
            policy_store = BestPolicyStore(
                Path(args.local_results_path) / "models" / args.unique_token / "best_policies",
                saved_config,
            )
            if args.showcase_enabled:
                from utils.dispatch_showcase import DispatchShowcase
                showcase = DispatchShowcase(
                    Path(args.local_results_path) / "dispatch_showcases" / args.unique_token,
                    logger, args,
                )
    # Learner
    if target_policy_smoothing_enabled(args):
        learner = TargetSmoothedGAVDACLearner(mac, buffer.scheme, logger, args)
    elif args.EA:
        learner = EA_GAVDACLearner(mac, buffer.scheme, logger, args)
    else:
        learner = GAVDACLearner(mac, buffer.scheme, logger, args)
    resources.callback(learner.close)

    if args.use_cuda:
        learner.cuda(device=args.device)
        for population_mac in pop:
            population_mac.cuda(device=args.device)

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                f"Checkpoint directiory {args.checkpoint_path} doesn't exist"
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info(f"Loading model from {model_path}")
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate:
            for _ in range(args.test_nepisode):
                runner.run(mac, test_mode=True)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    ea_cycle = 0
    ea_elite_index = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(f"Beginning training for {args.t_max} timesteps")

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        is_ea_cycle = (
            args.EA
            and runner.t_env > args.start_timesteps
            and episode % args.EA_freq == 0
        )
        if is_ea_cycle:
            ea_cycle += 1
            fitness = []
            # Compare policies on identical exogenous data and initial energy.
            # These remain training rollouts counted in t_env and replay.
            scenario = runner.env.training_scenario() if args.env == "microgrid" else None
            ea_options = ({"training_options": scenario, "explore": False}
                          if scenario is not None else {})
            tail_result = None
            if getattr(args, "rollout_workers", 0):
                results = runner.run_group(pop, training_options=scenario, tail_policy=mac)
                population_results, tail_result = results[:-1], results[-1]
            else:
                population_results = (
                    runner.run(ea_mac, test_mode=False, **ea_options) for ea_mac in pop
                )
            for episode_batch, episode_return in population_results:
                fitness.append(episode_return)
                buffer.insert_episode_batch(episode_batch)
            elite_index = np.argmax(fitness)
            episode_batch, _ = (tail_result if tail_result is not None else
                                runner.run(mac, test_mode=False, training_options=scenario))
            buffer.insert_episode_batch(episode_batch)

            # Each generation updates the complete shared policy once.
            for _ in range(args.ea_generations_per_cycle):
                elite_index = evolver.epoch(pop, fitness, 0, agent_level=True)
            ea_elite_index = elite_index
        else:
            fitness = [0.0 for _ in range(args.pop_size)]
            elite_index = 0
            episode_batch, _ = runner.run(mac, test_mode=False)
            buffer.insert_episode_batch(episode_batch)
        if is_ea_cycle:
            learner_update_count = args.ea_learner_updates
        elif episode % args.learner_update_interval == 0:
            learner_update_count = args.learner_updates_per_episode
        else:
            learner_update_count = 0

        for _ in range(learner_update_count):
            if buffer.can_sample(args.batch_size) and (
                buffer.episodes_in_buffer >= getattr(args, "buffer_warmup", 0)
            ):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                if args.EA:
                    if getattr(learner, "uses_population_training", False):
                        all_teams = []
                        for index_n in range(args.pop_size):
                            all_teams.append(pop[index_n])
                        all_teams.append(mac)
                        learner.train(episode_sample, all_teams, runner.t_env, episode)
                    else:
                        learner.train(episode_sample, runner.t_env, episode)
                else:
                    learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(f"t_env: {runner.t_env} / {args.t_max}")
            logger.console_logger.info(
                f"Estimated time left: {time_left(last_time, last_test_T, runner.t_env, args.t_max)}. Time passed: {time_str(time.time() - start_time)}"
            )
            last_time = time.time()

            last_test_T = runner.t_env

            rl_eval_reward = 0
            if validation_rows is not None:
                rl_evaluation = evaluate_fixed(
                    runner, mac, validation_rows, args.evaluation_initial_soc,
                    "validation", "rl", runner.t_env,
                )
                rl_eval_reward = rl_evaluation["mean_return"]
                if policy_store is not None:
                    policy_store.consider(mac, "rl", runner.t_env, rl_evaluation, runner.env)
            else:
                for _ in range(n_test_runs):
                    _, rl_tp_reward = runner.run(mac, test_mode=True)
                    rl_eval_reward += rl_tp_reward
                rl_eval_reward = rl_eval_reward / n_test_runs
            print("RL eval ", rl_eval_reward)
            logger.log_stat(
                "eval/rl_reward", rl_eval_reward, runner.t_env
            )
            if args.EA:
                ea_eval_reward = 0
                last_test_T = runner.t_env
                if validation_rows is not None:
                    ea_evaluation = evaluate_fixed(
                        runner, pop[ea_elite_index], validation_rows,
                        args.evaluation_initial_soc, "validation", "ea", runner.t_env,
                    )
                    ea_eval_reward = ea_evaluation["mean_return"]
                    if policy_store is not None and ea_cycle > 0:
                        policy_store.consider(pop[ea_elite_index], "ea", runner.t_env,
                                              ea_evaluation, runner.env)
                else:
                    for _ in range(n_test_runs):
                        _, ea_tp_reward = runner.run(pop[ea_elite_index], test_mode=True)
                        ea_eval_reward += ea_tp_reward
                    ea_eval_reward = ea_eval_reward / n_test_runs
            else:
                ea_eval_reward = 0
            print("EA eval ", ea_eval_reward)
            logger.log_stat(
                "eval/ea_reward", ea_eval_reward, runner.t_env
            )
            logger.log_stat(
                "eval/best_reward",
                max(rl_eval_reward, ea_eval_reward) if args.EA else rl_eval_reward,
                runner.t_env,
            )
        if is_ea_cycle:
            # Replace any index different from the new elite
            replace_index = np.argmin(fitness)
            if replace_index == elite_index:
                replace_index = (replace_index + 1) % args.pop_size
            rl_to_evo(mac, pop[replace_index])
            evolver.rl_policy = replace_index
            if ea_cycle % args.ea_log_interval == 0:
                logger.console_logger.info(
                    "EA cycle %s at t_env=%s: elite=%s, synced_rl_to_population=%s",
                    ea_cycle,
                    runner.t_env,
                    elite_index,
                    replace_index,
                )

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info(f"Saving models to {save_path}")

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            # learner.save_models(save_path, args.unique_token, model_save_time)

            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("train/t_env", runner.t_env, runner.t_env)
            logger.flush()
            logger.print_recent_stats(runner.t_env)
            last_log_T = runner.t_env

    if showcase is not None:
        showcase.publish_final(runner, mac, policy_store, runner.t_env)
    if target_policy_smoothing_enabled(args):
        completion = {
            **learner.diagnostics(), "joint_interactions": runner.t_env,
            "agent_transitions": runner.t_env * args.n_agents, "ea_cycles": ea_cycle,
        }
        (Path(args.local_results_path) / "h07_completion.json").write_text(
            json.dumps(completion, indent=2, allow_nan=False), encoding="utf-8")
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
