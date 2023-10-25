import argparse
from enum import Enum, auto

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

from behaviors.random_walking_policy import RandomBehavior
from behaviors.target_reactive import TargetReactive
from environment.mpe import pop_v0

from training.model.cnn_global_global import CNN_GLOBAL_GLOBAL
from training.model.cnn_local_local import CNN_LOCAL_LOCAL
from training.model.cnn_local_global import CNN_LOCAL_GLOBAL

matplotlib.use('TkAgg')
fig, axes = plt.subplots(nrows=2, ncols=2)


def display_all_maps(m):
    i = 0
    for ax in axes.flat:
        # im = ax.imshow((np.flip(m[:, :, i])), vmin=0, vmax=1, origin="lower")
        im = ax.imshow(np.flip(np.rot90(m[:, :, i], -1), axis=1), vmin=0, vmax=1, origin="lower")
        # im = ax.imshow(np.flip(np.rot90(m[:, :, i], -1)), vmin=0, vmax=1, origin="lower")
        if i == 0:
            ax.set_title("Environment's map", pad=5)
        if i == 1:
            ax.set_title("Target's map", pad=5)
        if i == 2:
            ax.set_title("Agent's map", pad=5)
        if i == 3:
            ax.set_title("Idleness's map", pad=5)
        i = i + 1

    fig.subplots_adjust(right=0.8)
    # fig.tight_layout()
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.subplots_adjust(bottom=0.05, top=0.95)

    plt.show(block=False)
    plt.pause(0.001)


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


def flatten_obs(observation, display=False):
    # TODO : Change the dimension according to the observation shape of the F2MARL policy
    nbr_poses = 2 + 8 * 2 + 8 * 2
    arr_poses = np.zeros((nbr_poses))

    arr_image = observation['d_image']
    if display:
        display_all_maps(arr_image)

    arr_image = np.array(arr_image).flatten()
    # If global : 30*30
    # If local : 18*18
    arr_image_critic = np.zeros((30 * 30 * 4))

    obs_flat_final = np.concatenate((arr_poses, arr_image, arr_image_critic))

    return obs_flat_final


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
    is_backup = args.is_backup
    max_agent_speed = args.max_agent_speed
    max_target_speed = args.max_target_speed
    target_behavior = args.target_behavior
    method = METHOD[method_name]

    # TODO : Change the model according to yours, like LOCAL_LOCAL, LOCAL_GLOBAL or GLOBAL_GLOBAL
    ModelCatalog.register_custom_model("CNN_MODEL", CNN_LOCAL_GLOBAL)

    checkpoint_dir = ""  # TODO : Specify your checkpoint
    if checkpoint_dir == "":
        print("Please specify the checkpoint directory at line 126. It looks like ~/ray_results/.../checkpoint_xxx")
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
                     max_cycles=max_cycles, obs2images=True,
                     max_target_speed=max_target_speed, share_target=True,
                     max_agent_speed=max_agent_speed, reward_type='glo',
                     render_mode=render_mode)
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            if 'agent' in agent:
                id = int(agent.split("agent_", 1)[1])
                if 'agent_0' in agent:
                    action = agentEntities[id].compute_single_action(flatten_obs(observation, True))[0]
                else:
                    action = agentEntities[id].compute_single_action(flatten_obs(observation, False))[0]
            else:
                action = target_policy(observation, agent, targetEntities)

        env.step(action)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    map_size = 50  # 100
    obs_range = 5  # 2
    com_range = 20
    nbr_agent = 8
    nbr_target = 8
    simulation_time = 60 * 60  # 1000
    gui = True  # Default False
    method = "RANDOM"  # Method of the agents
    debug = False  # Default False
    max_target_speed = 1
    max_agent_speed = 2
    safety_range = 1

    target_behavior = "random"  # random or evasive

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
