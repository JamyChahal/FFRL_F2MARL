import argparse
import ast
import json
import os
import random

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.tune.experiment.trial import ExportFormat
from ray.tune.logger import pretty_print
from ray.tune.schedulers import PopulationBasedTraining

from ray.rllib.agents.ppo import PPOTrainer

from behaviors.random_walking_policy import make_randomBehavior
from behaviors.reactive_walking_policy import make_reactiveBehavior
from environment.mpe import pop_v0

class ExportingPPOTrainer(PPOTrainer):
    # https://discuss.ray.io/t/save-model-parameters-on-each-checkpoint/2892/15
    def save_checkpoint(self, checkpoint_dir: str) -> str:
        path = super().save_checkpoint(checkpoint_dir)
        if not os.path.exists(os.path.join(checkpoint_dir, "default_policy")):
            print("[DEBUG]", flush=True)
            print(self.get_policy("default_policy"), flush=True)
            # if self.get_policy('policy_agent') is not None:
            self.export_policy_model(os.path.join(checkpoint_dir, "default_policy"))

        else:
            print("Error : Cannot save model, already exists in this folder. Let's continue the training.")
        return path


def main(args):
    ray.init()

    params = []
    with open(args.params) as json_param:
        params = json.load(json_param)

    dir_name = args.name
    to_resume = os.path.isdir(os.path.expanduser('~') + "/ray_results/" + dir_name)
    num_workers = args.workers
    model = ast.literal_eval(args.model)
    reward = args.reward
    samples = args.samples
    rollout_workers = 1
    cpu_per_workers = (num_workers-1) // samples

    env_creator = lambda params: pop_v0.env(nbr_agent=params['nbr_agents'], nbr_target=params['nbr_targets'],
                                            obs_range=params['obs_range'], com_range=params['com_range'],
                                            safety_range=params['safety_range'],
                                            dangerous_range=params['dangerous_range'],
                                            map_size=params['map_size'], obs_to_normalize=False,
                                            max_cycles=params["max_cycles"], has_protection_force=True,
                                            max_target_speed=params['max_target_speed'], share_target=True,
                                            max_agent_speed=params['max_target_speed'], reward_type=reward)

    register_env('pop', lambda config: PettingZooEnv(env_creator(params)))

    # https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_custom_policy.py
    # https://docs.ray.io/en/latest/rllib/rllib-env.html

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id.startswith("agent_"):  # We have a single policy for agent
            return 'default_policy'
        else:  # But each target has one
            id = agent_id.split("adversary_", 1)[1]
            return 'policy_target_' + id

    '''
    policies_target = {"policy_target_{}".format(i): PolicySpec(policy_class=make_randomBehavior(params["map_size"]))
                       for i in
                       range(params['nbr_targets'])}
    '''

    policies_target = {}
    for i in range(params['nbr_targets']):
        if i % 2 == 0:
            policies_target.update({'policy_target_{}'.format(i) :
                                        PolicySpec(policy_class=make_randomBehavior(params['map_size']))})
        else:
            policies_target.update({'policy_target_{}'.format(i) :
                                        PolicySpec(policy_class=make_reactiveBehavior(params['map_size'], params['det_range']))})

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=rollout_workers)
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
                  train_batch_size=tune.choice([1e3, 2e3, 4e3]),
                  # train_batch_size=tune.choice([1e4, 2e4, 4e4]),
                  model={"fcnet_hiddens": model,
                         #"use_lstm": params['model']['use_lstm'],
                         #"lstm_cell_size": params['model']['lstm_cell_size']
                         }
                  )
        .debugging(log_level="WARN")
        .resources(num_cpus_per_worker=cpu_per_workers)
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
            "lr": [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            #"lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "entropy_coeff": lambda: random.uniform(0, 0.1),
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 512),  # 128, 16384
            "train_batch_size": lambda: random.randint(32, 5120),  # 160000
        },
        custom_explore_fn=explore)

    stopping_criteria = {"training_iteration": 100000}

    # "episode_reward_mean": 12.0 * params["max_cycles"],

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
                                     num_to_keep=3,
                                     checkpoint_frequency=5,
                                     checkpoint_at_end=True
                                 )),
    )

    results = tuner.fit()

    print("best hyperparameters: ", results.get_best_result(metric="episode_reward_mean", mode="max"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params_default = "params_pbt_FFRL.list"
    name_default = "pbt_tuner2"
    num_workers = 8  # 63
    model = "[64, 64]"
    reward = 'ind' # or 'col'
    samples = 4
    parser.add_argument("-p", "--params", type=str, default=params_default)
    parser.add_argument("-n", "--name", type=str, default=name_default)
    parser.add_argument("-w", "--workers", type=int, default=num_workers)
    parser.add_argument("-m", "--model", type=str, default=model)
    parser.add_argument("-r", "--reward", type=str, default=reward)
    parser.add_argument("-s", "--samples", type=int, default=samples)

    args = parser.parse_args()
    main(args)
