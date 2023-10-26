import argparse
import json
import os
import random

import numpy as np
import ray
from ray import tune, air
from ray.dashboard.utils import Dict
from ray.rllib import RolloutWorker, BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.evaluation import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec, Policy
from ray.tune import register_env
from ray.tune.experiment.trial import ExportFormat
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.tune.schedulers import PopulationBasedTraining

from behaviors.random_walking_policy import make_randomBehavior
from behaviors.reactive_walking_policy import make_reactiveBehavior
from environment.mpe import pop_v0
from training.model.cnn_global_global import CNN_GLOBAL_GLOBAL
from training.model.cnn_global_global_light import CNN_GLOBAL_GLOBAL_LIGHT
from training.model.cnn_local_global import CNN_LOCAL_GLOBAL
from training.model.cnn_local_global_light import CNN_LOCAL_GLOBAL_LIGHT
from training.model.cnn_local_local import CNN_LOCAL_LOCAL
from training.model.cnn_local_local_light import CNN_LOCAL_LOCAL_LIGHT


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies,
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        '''
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        '''
        # Create lists to store angles in
        episode.user_data["obs_target"] = []
        episode.hist_data["obs_target"] = []

    def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies,
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

        info = episode._last_infos
        if info:
            if 'agent_0' in info:
                episode.user_data["obs_target"].append(info['agent_0']['target_obs'])

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies,
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.config.batch_mode == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        obs_target = np.mean(episode.user_data["obs_target"])
        episode.custom_metrics["obs_target"] = obs_target
        episode.hist_data["obs_target"] = episode.user_data["obs_target"]


def main(args):
    ray.init()

    # Inspiration from : https://github.com/ray-project/ray/blob/ray-2.3.0/rllib/examples/custom_env.py

    params = []
    with open(args.params) as json_param:
        params = json.load(json_param)

    dir_name = args.name
    dir_resume = os.path.expanduser('~') + "/ray_results/" + dir_name
    to_resume = os.path.isdir(dir_resume)
    num_workers = args.workers
    samples = args.samples
    '''
    num_rollout_workers = 3
    num_cpu_per_workers = num_workers // (samples * num_rollout_workers)
    '''
    num_rollout_workers = num_workers // (samples) - 1
    num_cpu_per_workers = 1
    reward = args.reward
    model = args.model

    # Issue with train batch size : https://github.com/ray-project/ray/issues/19222
    # Maybe solution : https://discuss.ray.io/t/rllib-batch-size-for-complete-episodes-issue/2022

    train_batch_size_min = 1000
    train_batch_size_max = train_batch_size_min * 2

    print("NUMBER OF WORKERS TOTAL :")
    print(num_workers)
    print("NUMBER OF ROLLOUT WORKERS :")
    print(num_rollout_workers)
    print("CPU PER WORKERS : ")
    print(num_cpu_per_workers)

    if model == 'local_local':
        ModelCatalog.register_custom_model("CNN_MODEL", CNN_LOCAL_LOCAL)
        actor_centered = True
        critic_centered = True
    elif model == 'local_global':
        ModelCatalog.register_custom_model("CNN_MODEL", CNN_LOCAL_GLOBAL)
        actor_centered = True
        critic_centered = False
    elif model == 'global_global':
        ModelCatalog.register_custom_model("CNN_MODEL", CNN_GLOBAL_GLOBAL)
        actor_centered = False
        critic_centered = False
    else:
        print(model)
        print(" is not a defined model. Stop here.")
        exit()

    env_creator = lambda params: pop_v0.env(nbr_agent=params['nbr_agents'], nbr_target=params['nbr_targets'],
                                            obs_range=params['obs_range'], com_range=params['com_range'],
                                            safety_range=params['safety_range'],
                                            dangerous_range=params['dangerous_range'],
                                            map_size=params['map_size'], obs_to_normalize=False,
                                            max_cycles=params["max_cycles"], obs2images=True, has_protection_force=True,
                                            max_target_speed=params['max_target_speed'], share_target=True,
                                            max_agent_speed=params['max_target_speed'], reward_type=reward,
                                            actor_centered=actor_centered, critic_centered=critic_centered)

    register_env('pop', lambda config: PettingZooEnv(env_creator(params)))

    # https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_custom_policy.py
    # https://docs.ray.io/en/latest/rllib/rllib-env.html

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id.startswith("agent_"):  # We have a single policy for agent
            return 'default_policy'
        else:  # But each target has one
            id = agent_id.split("adversary_", 1)[1]
            return 'policy_target_' + id

    policies_target = {"policy_target_{}".format(i): PolicySpec(policy_class=make_randomBehavior(params["map_size"]))
                       for i in
                       range(params['nbr_targets'])}

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=num_rollout_workers,
                  batch_mode='complete_episodes',  # complete_episodes, truncate_episode
                  rollout_fragment_length='auto',
                  num_envs_per_worker=1)
        .environment(env='pop')
        .framework("tf")
        .multi_agent(
            # The multiagent Policy map.
            policies={
                # The Policy we are actually learning.
                "default_policy": PolicySpec(
                    policy_class=None,  # infer automatically from Algorithm
                    observation_space=None,  # infer automatically from env
                    action_space=None,  # infer automatically from env
                ),
                # Random policy for targets
                **policies_target
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["default_policy"],
            # count_steps_by="env_steps"
        )
        .training(gamma=0.9,
                  lambda_=0.95,  # These params are tuned from a fixed starting value
                  clip_param=0.2,
                  lr=1e-4,
                  num_sgd_iter=tune.choice([10, 20, 30]),
                  # sgd_minibatch_size=tune.choice([128, 512, 2048]),
                  sgd_minibatch_size=tune.choice([32, 64, 128]),
                  #train_batch_size=tune.choice([train_batch_size_min, train_batch_size_min*2, train_batch_size_max]),
                  train_batch_size=train_batch_size_min,
                  # train_batch_size=tune.choice([1e4, 2e4, 4e4]),
                  model={"custom_model": "CNN_MODEL"}
                  )
        .debugging(log_level="DEBUG")
        .resources(num_cpus_per_worker=num_cpu_per_workers)
        .callbacks(MyCallbacks)
        # .build()
    )

    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            # "lr": lambda: random.uniform(1., 1e-5),
            "lr": [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            # "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "entropy_coeff": lambda: random.uniform(0, 0.1),
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 512),  # 128, 16384
            #"train_batch_size": lambda: random.randint(train_batch_size_min, train_batch_size_max),  # 160000
        },
        custom_explore_fn=explore)

    stopping_criteria = {"training_iteration": 100000}

    # https://docs.ray.io/en/latest/tune/examples/pbt_guide.html#function-api-with-population-based-training

    tuner = tune.Tuner(
        # ExportingPPOTrainer,
        "PPO",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=samples
        ),
        param_space=algo.to_dict(),
        run_config=air.RunConfig(stop=stopping_criteria,
                                 verbose=3,
                                 name=dir_name,
                                 checkpoint_config=air.CheckpointConfig(
                                     checkpoint_score_attribute="episode_reward_mean",
                                     checkpoint_score_order="max",
                                     num_to_keep=5,
                                     checkpoint_frequency=5,
                                     checkpoint_at_end=True
                                 )),
    )

    if to_resume:
        tuner = tuner.restore(path=dir_resume, trainable="PPO", resume_errored=False,
                              restart_errored=True)

    results = tuner.fit()

    print("best hyperparameters: ", results.get_best_result(metric="episode_reward_mean", mode="max"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params_default = "params.list"
    name_default = "pbt_local_global"
    samples = 1
    num_workers = 3
    reward = 'glo'  # => 'ind', 'glo'
    model = 'local_global'  # => 'local_local', 'local_global', 'global_global'
    parser.add_argument("-p", "--params", type=str, default=params_default)
    parser.add_argument("-n", "--name", type=str, default=name_default)
    parser.add_argument("-s", "--samples", type=int, default=samples)
    parser.add_argument("-w", "--workers", type=int, default=num_workers)
    parser.add_argument("-r", "--reward", type=str, default=reward)
    parser.add_argument("-m", "--model", type=str, default=model)

    args = parser.parse_args()
    main(args)
