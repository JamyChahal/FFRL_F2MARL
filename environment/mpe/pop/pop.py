
# noqa
"""
# Simple Adversary

```{figure} mpe_simple_adversary.gif
:width: 140px
:name: simple_adversary
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_adversary_v3` |
|--------------------|--------------------------------------------------|
| Actions            | Discrete/Continuous                              |
| Parallel API       | Yes                                              |
| Manual Control     | No                                               |
| Agents             | `agents= [adversary_0, agent_0,agent_1]`         |
| Agents             | 3                                                |
| Action Shape       | (5)                                              |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5))                   |
| Observation Shape  | (8),(10)                                         |
| Observation Values | (-inf,inf)                                       |
| State Shape        | (28,)                                            |
| State Values       | (-inf,inf)                                       |


In this environment, there is 1 adversary (red), N good agents (green), N landmarks (default N=2). All agents observe the position of landmarks and other agents. One landmark is the 'target landmark' (colored green). Good agents are rewarded based on how close the closest one of them is to the
target landmark, but negatively rewarded based on how close the adversary is to the target landmark. The adversary is rewarded based on distance to the target, but it doesn't know which landmark is the target landmark. All rewards are unscaled Euclidean distance (see main MPE documentation for
average distance). This means good agents have to learn to 'split up' and cover all landmarks to deceive the adversary.

Agent observation space: `[self_pos, self_vel, goal_rel_position, landmark_rel_position, other_agent_rel_positions]`

Adversary observation space: `[landmark_rel_position, other_agents_rel_positions]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

Adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False)
```



`N`:  number of good agents and landmarks

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""
import copy
import math
from collections import OrderedDict

import numpy as np
from gymnasium.utils import EzPickle

from behaviors.Tools.Map_Obs import Map_Env
from environment.mpe._mpe_utils.core import Agent, Landmark, World
from environment.mpe._mpe_utils.scenario import BaseScenario
from environment.mpe._mpe_utils.pop_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(self, nbr_agent=2, nbr_target=2, max_cycles=25, obs_range=1, com_range=2, map_size=10,
                 safety_range=2, dangerous_range=1, max_target_speed=1, max_agent_speed=2,
                 reward_type='ind', share_target=False, obs_to_normalize=False, obs2images=False, actor_centered=False,
                 critic_centered=True, render_mode=None, has_protection_force=False):
        EzPickle.__init__(
            self,
            nbr_agent=nbr_agent,
            nbr_target=nbr_target,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(
            nbr_agent=nbr_agent,
            nbr_target=nbr_target,
            max_cycles=max_cycles,
            obs_range=obs_range,
            com_range=com_range,
            map_size=map_size,
            safety_range=safety_range,
            dangerous_range=dangerous_range,
            reward_type=reward_type,
            share_target=share_target,
            obs_to_normalize=obs_to_normalize,
            obs2images=obs2images,
            actor_centered=actor_centered,
            critic_centered=critic_centered
        )
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            max_target_speed=max_target_speed,
            max_agent_speed=max_agent_speed,
            map_size=map_size,
            obs_range=obs_range,
            com_range=com_range,
            obs2images=obs2images,
            actor_centered=actor_centered,
            critic_centered=critic_centered,
            has_protection_force=has_protection_force
        )
        self.metadata["name"] = "pop_v0"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

MAX_SEEN_TARGET = 8
MAX_SEEN_AGENT = 8


class Scenario(BaseScenario):

    def make_world(self, nbr_agent=2, nbr_target=2, obs_range=1, com_range=2, map_size=10, safety_range=2,
                   dangerous_range=1, reward_type='ind', share_target=False, obs_to_normalize=False, obs2images=False,
                   actor_centered=True, critic_centered=False, max_cycles=25):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = nbr_agent + nbr_target
        world.num_agents = num_agents
        self.num_adversaries = nbr_target
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < self.num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < self.num_adversaries else i - self.num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # add other parameters
        self.max_cycles = max_cycles
        self.obs_range = obs_range
        self.com_range = com_range
        self.map_size = map_size
        self.obs_to_normalize = obs_to_normalize
        self.share_target = share_target
        self.reward_type = reward_type
        self.obs2images = obs2images
        self.actor_centered = actor_centered
        self.critic_centered = critic_centered

        # Prepare maps for images
        if self.obs2images:
            self.my_map = dict()
            self.critic_map = dict()
            for i in range(0, nbr_agent):
                self.my_map.update({"agent_" + str(i): Map_Env(self.map_size, self.obs_range, self.com_range,
                                                               discretization=0.3)})
                self.critic_map.update({"agent_" + str(i): Map_Env(self.map_size, self.obs_range, self.com_range,
                                                                   discretization=0.3)})

        # For protection force variables
        self.sr = safety_range
        self.dr = dangerous_range
        if abs(self.sr - self.dr) < 0.0001:
            self.sr = self.dr + 1  # Avoid zero division

        self.fr1a = 1 / float(self.sr - self.dr)
        self.fr1b = self.sr  # -1 - self.fr1a * self.dr1
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i in range(0, world.num_agents):
            if i < self.num_adversaries:
                world.agents[i].color = np.array([0.85, 0.35, 0.35])
            else:
                world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-self.map_size, +self.map_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return 0  # TODO
        else:
            dists = []
            # TODO
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )

    def get_distance(self, agent1, agent2):
        return math.sqrt(math.pow(agent1.state.p_pos[0] - agent2.state.p_pos[0], 2) +
                         math.pow(agent1.state.p_pos[1] - agent2.state.p_pos[1], 2))

    def agent_reward_ind(self, agent, adversary_agents):
        reward = 0
        for target in adversary_agents:
            if self.get_distance(target, agent) <= self.obs_range:
                reward += 1
        return reward

    def agent_reward_shared(self, good_agents, adversary_agents):
        reward = 0
        for target in adversary_agents:
            for ga in good_agents:
                if self.get_distance(target, ga) <= self.obs_range:
                    reward += 1  # Definition of the normalized A metric (sum(obs)/sum(times))
                    break  # At least one agent see the target, now move on the next target
        #reward = reward / (self.max_cycles * len(adversary_agents))
        return reward

    def agent_reward_collaboration(self, agent, good_agents, adversary_agents):
        # Get reward only if a target is seen by you, or by an agent with who you are communicating
        reward = 0
        for target in adversary_agents:
            for ga in good_agents:
                if self.get_distance(target, ga) <= self.obs_range:
                    # The target is seen by ga, but do you communicate with him ?
                    if self.get_distance(agent, ga) <= self.com_range:
                        reward += 1
                        break  # At least one agent under com' see the target, now move on the next target
        return reward

    def agent_reward_collide(self, agent, good_agents):
        reward = 0
        for ga in good_agents:
            if agent is ga:
                continue
            dist_collide = self.get_distance(ga, agent)
            if self.sr <= dist_collide < self.dr and self.gradual_reward:
                reward -= 1 / (self.sr - self.dr) * dist_collide - self.sr / (self.sr - self.dr)  # ax+b
            if dist_collide <= self.dr:
                reward -= 1
        return reward

    def agent_reward_outside(self, agent):
        reward = 0
        if self.is_agent_outside_map(agent, 2):
            reward -= 10
        return reward

    def agent_reward_exploration(self, agent):
        # Get between 0 and 1 the idleness normalized
        reward = 0
        x, y = agent.state.p_pos
        f = self.my_map[agent.name].observe(x, y, update_idleness=False)
        mmax_idleness = self.my_map[agent.name].get_max_idleness()
        if mmax_idleness > 0:
            reward = f / (math.pow(self.obs_range, 2) * mmax_idleness)
        return reward

    def agent_reward(self, agent, world):
        '''
        Return the reward for the agent
        +1 for each target seen by at least one agent
        -1 if too close
        '''

        reward = 0
        adversary_agents = self.adversaries(world)
        good_agents = self.good_agents(world)

        if self.reward_type == 'ind':
            reward += self.agent_reward_ind(agent, adversary_agents)
        elif self.reward_type == 'glo':
            reward += self.agent_reward_shared(good_agents, adversary_agents)
        elif self.reward_type == 'col':
            reward += self.agent_reward_collaboration(agent, good_agents, adversary_agents)

        # Collide negative reward
        reward += self.agent_reward_collide(agent, good_agents)

        # Outside negative reward
        reward += self.agent_reward_outside(agent)

        if self.obs2images:
            reward += self.agent_reward_exploration(agent)

        # reward /= self.max_cycles

        return reward

    def is_agent_outside_map(self, agent, limit):
        if agent.state.p_pos[0] > self.map_size + limit or \
                agent.state.p_pos[0] < -self.map_size - limit or \
                agent.state.p_pos[1] > self.map_size + limit or \
                agent.state.p_pos[1] < -self.map_size - limit:
            #print("Agent outside !")
            return True
        else:
            return False

    def how_far_agent_outside_map(self, agent):
        x_max = agent.state.p_pos[0] - self.map_size
        x_min = -self.map_size - agent.state.p_pos[0]
        y_max = agent.state.p_pos[1] - self.map_size
        y_min = -self.map_size - agent.state.p_pos[1]
        out = max(x_max, 0) + max(x_min, 0) + max(y_max, 0) + max(y_min, 0)
        return out

    def apply_protection_force(self, agent, world, x_speed, y_speed):
        def get_robot_magnitude(distance):
            m = 0
            if distance <= self.dr:
                m = -1
            if self.dr < distance <= self.sr:
                m = self.fr1a * distance + self.fr1b
            if distance > self.sr:
                m = 0
            return m

        x_coord, y_coord = x_speed, y_speed

        # Repulsion from the outside world
        x_force, y_force = 0, 0
        limit = 1
        if agent.state.p_pos[0] - self.map_size > limit:
            x_force = -1
        if (-self.map_size - agent.state.p_pos[0]) > limit:
            x_force = 1
        if agent.state.p_pos[1] - self.map_size > limit:
            y_force = -1
        if -self.map_size - agent.state.p_pos[1] > limit:
            y_force = 1

        # Repulsion from the other agents
        '''
        good_agents = self.good_agents(world)
        for ga in good_agents:
            if agent is ga:
                continue
            dist_collide = self.get_distance(ga, agent)
            if dist_collide <= self.sr:
                magnitude = get_robot_magnitude(dist_collide)
                vector = ga.state.p_pos - agent.state.p_pos
                x_force -= vector[0] * magnitude
                y_force -= vector[1] * magnitude
        '''
        # Final calculation
        x_final_force = x_coord + x_force
        y_final_force = y_coord + y_force

        return x_final_force, y_final_force

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        return 0  # For now, targets has no rewards

    def infos(self, agent, world):
        adversary_agents = self.adversaries(world)
        good_agents = self.good_agents(world)

        reward = 0

        for target in adversary_agents:
            for ga in good_agents:
                if self.get_distance(target, ga) <= self.obs_range:
                    reward += 1  # Definition of the normalized A metric (sum(obs)/sum(times))
                    break  # At least one agent see the target, now move on the next target

        info = {'target_obs': reward}
        return info

    def observation(self, agent, world):
        def sorting(pose_list):
            pose_list.sort(key=lambda x: (x[0] ** 2 + x[1] ** 2))
            return pose_list[0:8]

        def distance(relative_pose):
            return math.sqrt(relative_pose[0] ** 2 + relative_pose[1] ** 2)

        def remove_redundant_pose(seq):
            return list(OrderedDict((tuple(x), x) for x in seq).values())

        self_pos = [copy.copy(agent.state.p_pos)]
        agent_pose = []
        target_pose = []
        for other in world.agents:
            if other is agent:
                continue
            relative_pose = other.state.p_pos - agent.state.p_pos
            dist = math.sqrt(relative_pose[0] ** 2 + relative_pose[1] ** 2)
            if other.adversary:
                if dist <= self.obs_range:
                    target_pose.append(relative_pose)
            elif dist <= self.com_range:
                agent_pose.append(relative_pose)

        # Communicate with the surrounding agent about targets (can be incorporated in the previous for loop)
        """
        if np.shape(target_pose)[0] < MAX_SEEN_TARGET and self.share_target:
            for other in world.agents:
                if not other.adversary:
                    relative_pose = other.state.p_pos - agent.state.p_pos
                    dist = math.sqrt(relative_pose[0] ** 2 + relative_pose[1] ** 2)
                    if dist <= self.com_range:  # Within our communication range
                        agent_com_pose = other.state.p_pos
                        # Check its surrounding targets
                        for other_com in world.agents:
                            if other_com.adversary:
                                relative_com_pose = other_com.state.p_pos - agent_com_pose
                                if distance(relative_com_pose) <= self.obs_range:
                                    target_pose.append(other_com.state.p_pos - agent.state.p_pos)
        """
        if np.shape(target_pose)[0] < MAX_SEEN_TARGET and self.share_target:
            list_check_com = []  # Agent which are within the ad-hoc range
            list_tocheck_com = []  # Agent within ad-hoc range but need to check the neighboorhood com'
            list_outside_com = []  # Agent outside com' range for now
            # Fill the tocheck
            for other in world.agents:
                if not other.adversary and agent is not other:
                    list_outside_com.append(other)

            # Fill the check com'
            list_tocheck_com.append(agent)

            # Check for all the com' to check
            for a in list_tocheck_com:
                for other in world.agents:
                    if not other.adversary:
                        relative_pose = other.state.p_pos - agent.state.p_pos
                        dist = math.sqrt(relative_pose[0] ** 2 + relative_pose[1] ** 2)
                        if dist <= self.com_range:  # Within our communication range
                            if other not in list_check_com:
                                list_check_com.append(other)
                                if other not in list_tocheck_com:
                                    list_tocheck_com.append(other)

                # Delete the agent from the tocheck list
                aa = list_tocheck_com.pop(0)
                list_check_com.append(aa)

            # Check all the targets of the agent
            for other in world.agents:
                if other.adversary:
                    for a in list_check_com:
                        relative_com_pose = other.state.p_pos - a.state.p_pos
                        if distance(relative_com_pose) <= self.obs_range:
                            target_pose.append(other.state.p_pos - agent.state.p_pos)
                            break

        # Limit it here for 4 agents and 4 targets, only the closest
        if np.shape(agent_pose)[0] > MAX_SEEN_AGENT:
            agent_pose = sorting(agent_pose)
        if np.shape(target_pose)[0] > MAX_SEEN_TARGET:
            target_pose = sorting(target_pose)

        # Remove redundant
        if self.share_target:
            target_pose = remove_redundant_pose(target_pose)


        # OBS AS IMAGE
        if self.obs2images:
            # Prepare observation image for actor
            if "agent" in agent.name:
                self.my_map[agent.name].next_time()
                self.my_map[agent.name].observe(agent.state.p_pos[0], agent.state.p_pos[1])
                for a in agent_pose:
                    self.my_map[agent.name].add_agent_friend(a[0] + agent.state.p_pos[0], a[1] + agent.state.p_pos[1])
                for t in target_pose:
                    self.my_map[agent.name].add_target(t[0] + agent.state.p_pos[0], t[1] + agent.state.p_pos[1])
                self.my_map[agent.name].add_agent_moi(self_pos[0][0], self_pos[0][1])
                # Update idleness from surrounding agents
                for other in world.agents:
                    if other is agent:
                        continue
                    relative_pose = other.state.p_pos - agent.state.p_pos
                    dist = math.sqrt(relative_pose[0] ** 2 + relative_pose[1] ** 2)
                    if not other.adversary and dist <= self.com_range:
                        self.my_map[agent.name].update_from_map(self.my_map[other.name].get_idleness_map())

                obs_map = self.my_map[agent.name].get_map(self_pos[0][0], self_pos[0][1], centered=self.actor_centered)
            else:
                obs_map = Map_Env(self.map_size, self.obs_range, self.com_range, discretization=0.3).get_map(self_pos[0][0], self_pos[0][1], centered=self.actor_centered)

            # Prepare observation image for critic
            if "agent" in agent.name:
                self.critic_map[agent.name].next_time()
                for other in world.agents:
                    if agent.name in other.name:
                        self.critic_map[agent.name].add_agent_moi(other.state.p_pos[0], other.state.p_pos[1])
                        self.critic_map[agent.name].observe(other.state.p_pos[0], other.state.p_pos[1])
                    elif "agent" in other.name:
                        self.critic_map[agent.name].add_agent_friend(other.state.p_pos[0], other.state.p_pos[1])
                        self.critic_map[agent.name].observe(other.state.p_pos[0], other.state.p_pos[1])
                    elif "adversary" in other.name:
                        self.critic_map[agent.name].add_target(other.state.p_pos[0], other.state.p_pos[1])

                obs_map_critic = self.critic_map[agent.name].get_map(self_pos[0][0], self_pos[0][1], centered=self.critic_centered)
            else:
                obs_map_critic = Map_Env(self.map_size, self.obs_range, self.com_range, discretization=0.3).get_map(self_pos[0][0], self_pos[0][1], centered=self.critic_centered)



        # Completing to have specifically 4 agents and 4 targets info (with full 0)
        while np.shape(agent_pose)[0] < MAX_SEEN_AGENT:
            agent_pose.append(np.array((0, 0)))
        while np.shape(target_pose)[0] < MAX_SEEN_TARGET:
            target_pose.append(np.array((0, 0)))

        # Normalization of the self_pose (with map_size), agent_pose (with com_range) and target_pose (with obs_range)
        if not agent.adversary and self.obs_to_normalize:
            self_pos[0][0] = self_pos[0][0] / self.map_size
            self_pos[0][1] = self_pos[0][1] / self.map_size
            for i in range(0, len(agent_pose)):
                agent_pose[i][0] = agent_pose[i][0] / self.com_range
                agent_pose[i][1] = agent_pose[i][1] / self.com_range
            for i in range(0, len(target_pose)):
                target_pose[i][0] = target_pose[i][0] / self.obs_range
                target_pose[i][1] = target_pose[i][1] / self.obs_range

        all_pose = []  # world.agents is already sort (adversary then agent)
        # Previously : All the true pose
        '''
        for other in world.agents:
            all_pose.append(other.state.p_pos)
        '''
        # Now : Relative pose of all the agents
        for other in world.agents:
            all_pose.append(other.state.p_pos - agent.state.p_pos)


        obs = {
            "a_self_pose": self_pos,
            "b_agent_pose": agent_pose,
            "c_target_pose": target_pose,
            #"d_all_pose": all_pose  # Only for centralized critic
        }  # Sort by alphabetic, because rllib open the Dict in this way

        if self.obs2images:
            obs.update({
                "d_image": obs_map,
                "e_image_critic": obs_map_critic
            })

        return obs
