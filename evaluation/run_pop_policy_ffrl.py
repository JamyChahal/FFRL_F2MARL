import argparse
from enum import Enum, auto

import numpy as np
from ray.rllib.policy.policy import Policy

from behaviors.random_walking_policy import RandomBehavior
from behaviors.target_reactive import TargetReactive
from environment.mpe import pop_v0


class METHOD(Enum):
    FFRL = auto()
    F2MARL = auto()
    OUTSIDE = auto()
    NONE = auto()
    RANDOM = auto()
    A_CMOMMT = auto()
    A_CMOMMT_OBS = auto()
    I_CMOMMT = auto()

    @classmethod
    def is_trained(cls, method):
        return (method == METHOD.FFRL or method == METHOD.F2MARL)

def flatten_obs(observation):
    obs_flat = []
    for key in observation:
        arr = observation[key]
        for a in arr:
            for b in a:
                obs_flat.append(b)
    #print(obs_flat)
    obs_flat = np.array(obs_flat)
    return obs_flat


def target_policy(observation, agent, targetEntities):
    action = (0, 0)
    if "adversary" in agent:
        id = int(agent.split("adversary_", 1)[1])
        targetEntities[id].set_observation(observation)
        action = targetEntities[id].get_action()
    return action


def agent_policy(reward, observation, agent, agentEntities, method):
    action = (0, 0)
    if "agent" in agent:
        id = int(agent.split("agent_", 1)[1])
        if not METHOD.is_trained(method):
            agentEntities[id].set_observation(observation)
            action = agentEntities[id].get_action()
        else:
            action = agentEntities[id].do_action(reward, observation)
    return action


def main(args):
    # Parameters
    map_size = args.map_size
    obs_range = args.obs_range
    com_range = args.com_range
    nbr_agent = args.nbr_agent
    nbr_target = args.nbr_target
    max_cycles = args.sim_time
    gui = args.gui
    method_name = args.method
    max_agent_speed = args.max_agent_speed
    max_target_speed = args.max_target_speed
    target_behavior = args.target_behavior
    method = METHOD[method_name]

    checkpoint_dir = ""  # TODO : Specify your checkpoint
    if checkpoint_dir == "":
        print("Please specify the checkpoint directory at line 74. It looks like ~/ray_results/.../checkpoint_xxx")
        exit()

    # Create the target's behaviors
    targetEntities = []
    for i in range(0, nbr_target):
        if target_behavior == "evasive":
            targetEntities.append(TargetReactive(map_size=map_size, det_range=obs_range + 2, is_in_training=False))
        elif target_behavior == "random":
            targetEntities.append(RandomBehavior(map_size=map_size))
        else:
            print("ERROR : target behavor is not evasive nor random. Exit simulation here.")
            exit()

    agentEntities = []
    policy = Policy.from_checkpoint(checkpoint_dir)
    # https://docs.ray.io/en/latest/rllib/rllib-saving-and-loading-algos-and-policies.html#how-do-i-restore-from-a-policy-checkpoint
    for i in range(0, nbr_agent):
        agentEntities.append(policy['default_policy'])

    obs_to_normalize = METHOD.is_trained(method)
    if gui:
        render_mode = 'human'
    else:
        render_mode = None

    env = pop_v0.env(nbr_agent=nbr_agent, nbr_target=nbr_target, obs_range=obs_range, com_range=com_range,
                     safety_range=safety_range,
                     map_size=map_size, obs_to_normalize=obs_to_normalize,
                     max_cycles=max_cycles,
                     max_target_speed=max_target_speed, share_target=True,
                     max_agent_speed=max_agent_speed, reward_type='ind',
                     render_mode=render_mode)
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            if 'agent' in agent:
                id = int(agent.split("agent_", 1)[1])
                action = agentEntities[id].compute_single_action(flatten_obs(observation))[0]
            else:
                action = target_policy(observation, agent, targetEntities)

        env.step(action)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    map_size = 50  # 100
    obs_range = 10  # 2
    com_range = 20
    nbr_agent = 8
    nbr_target = 8
    simulation_time = 60 * 60  # 1000
    gui = True  # Default False
    method = "RANDOM"  #
    debug = False  # Default False
    max_target_speed = 1
    max_agent_speed = 2
    safety_range = 1

    target_behavior = "random" # random or evasive

    parser.add_argument("-t", "--sim_time", type=int, default=simulation_time)
    parser.add_argument("-m", "--map_size", type=int, default=map_size)
    parser.add_argument("-o", "--obs_range", type=int, default=obs_range)
    parser.add_argument("-c", "--com_range", type=int, default=com_range)
    parser.add_argument("-a", "--nbr_agent", type=int, default=nbr_agent)
    parser.add_argument("-w", "--nbr_target", type=int, default=nbr_target)
    parser.add_argument("-s", "--safety_range", type=float, default=safety_range)

    parser.add_argument("--method", type=str, default=method)

    parser.add_argument("--max_target_speed", type=float, default=max_target_speed)
    parser.add_argument("--max_agent_speed", type=float, default=max_agent_speed)

    parser.add_argument("--target_behavior", type=str, default=target_behavior)

    parser.add_argument('--game_displayed', dest='is_game_displayed', action='store_true')
    parser.add_argument('--not_game_displayed', dest='is_game_displayed', action='store_false')
    parser.set_defaults(gui=gui)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no_debug', dest='debug', action='store_false')
    parser.set_defaults(debug=debug)
    args = parser.parse_args()

    main(args)
