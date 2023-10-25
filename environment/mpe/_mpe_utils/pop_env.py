import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.mpe._mpe_utils.core import Agent
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

MAX_SEEN_TARGET = 8
MAX_SEEN_AGENT = 8

def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        #else:
        #    env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
            self,
            scenario,
            world,
            max_cycles,
            render_mode=None,
            local_ratio=None,
            max_target_speed=1,
            max_agent_speed=1,
            map_size=10,
            com_range=10,
            obs_range=5,
            obs2images=False,
            actor_centered=True,
            critic_centered=False,
            has_protection_force=False
    ):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )

        # Set up the drawing window

        self.renderOn = False
        self._seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = False
        self.local_ratio = local_ratio
        self.has_protection_force = has_protection_force
        self.max_target_speed = max_target_speed
        self.max_agent_speed = max_agent_speed
        self.map_size = map_size
        self.com_range = com_range
        self.obs_range = obs_range
        self.obs2images = obs2images
        self.actor_centered = actor_centered
        self.critic_centered = critic_centered

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Tuple((
                    spaces.Box(low=-np.float32(1), high=+np.float32(1), shape=(1,)),  # X, Y speed
                    spaces.Box(low=-np.float32(1), high=+np.float32(1), shape=(1,)))  # Y speed
                )
            else:

                self.action_spaces[agent.name] = spaces.Tuple([
                    spaces.Discrete(5),  # X
                    spaces.Discrete(5)  # Y
                ])

                # self.action_spaces[agent.name] = spaces.MultiDiscrete([5, 5])

            # REMOVE OBSERVATION SPACE TO THE OLD ONE
            '''
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )
            '''

            pose_dim = 2
            if not self.obs2images:
                DICT_SPACE = spaces.Dict({
                    "a_self_pose": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                              shape=(1, pose_dim)),
                    "b_agent_pose": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                               shape=(MAX_SEEN_AGENT, pose_dim)),
                    "c_target_pose": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                                shape=(MAX_SEEN_TARGET, pose_dim))
                })  # a,b,c because it is an alphabetic order in rllib
            else:
                discretization = 0.3
                if self.critic_centered:
                    env_size_critic = int(com_range * 2 * 2 * discretization)
                else:
                    env_size_critic = int(map_size * 2 * discretization)
                if self.actor_centered:
                    env_size = int(com_range * 2 * 2 * discretization)
                else:
                    env_size = int(map_size * 2 * discretization)


                DICT_SPACE = spaces.Dict({
                    "a_self_pose": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                              shape=(1, pose_dim), dtype=np.float32),
                    "b_agent_pose": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                               shape=(MAX_SEEN_AGENT, pose_dim)),
                    "c_target_pose": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                                shape=(MAX_SEEN_TARGET, pose_dim)),
                    "d_image": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                          shape=(env_size, env_size, 4), dtype=np.float32),
                    "e_image_critic": spaces.Box(-np.float32(np.inf), high=+np.float32(np.inf),
                                                 shape=(env_size_critic, env_size_critic, 4), dtype=np.float32),
                })  # a,b,c because it is an alphabetic order in rllib

            self.observation_spaces[agent.name] = DICT_SPACE


        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        # return self.scenario.observation(self.world.agents[self._index_map[agent]], self.world).astype(np.float32)
        return self.scenario.observation(self.world.agents[self._index_map[agent]], self.world)

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ) #.astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {'target_obs': 0.} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                scenario_action.append(action)
                '''
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
                '''
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        for agent in self.world.agents:
            agent_info = self.scenario.infos(agent, self.world)
            self.infos[agent.name] = agent_info

        self.world.step()

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            self.rewards[agent.name] = agent_reward


    # Set the discrete action space into true 2D motions
    def get_continuous_action(self, x_speed_choice, y_speed_choice):
        if x_speed_choice == 0:
            x_speed = 1
        elif x_speed_choice == 1:
            x_speed = 0.5
        elif x_speed_choice == 2:
            x_speed = 0
        elif x_speed_choice == 3:
            x_speed = -0.5
        elif x_speed_choice == 4:
            x_speed = -1
        else:
            x_speed = 0

        if y_speed_choice == 0:
            y_speed = 1
        elif y_speed_choice == 1:
            y_speed = 0.5
        elif y_speed_choice == 2:
            y_speed = 0
        elif y_speed_choice == 3:
            y_speed = -0.5
        elif y_speed_choice == 4:
            y_speed = -1
        else:
            y_speed = 0

        return x_speed, y_speed

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                x_speed = action[0][0]
                y_speed = action[0][1]
                # TODO : To be verified
            else:
                # process discrete action
                x_speed_choice = action[0][0]
                y_speed_choice = action[0][1]

                x_speed, y_speed = self.get_continuous_action(x_speed_choice, y_speed_choice)

                if not agent.adversary and self.has_protection_force:
                    x_speed, y_speed = self.scenario.apply_protection_force(agent, self.world, x_speed, y_speed)
                    # Override the command
                    agent.action.u[0] = x_speed
                    agent.action.u[1] = y_speed

            if agent.adversary:
                x_speed = x_speed * self.max_target_speed
                y_speed = y_speed * self.max_target_speed
            else:
                x_speed = x_speed * self.max_agent_speed
                y_speed = y_speed * self.max_agent_speed

            agent.action.u[0] = x_speed
            agent.action.u[1] = y_speed


            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            return

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # Set a minimum
        cam_range = max(cam_range, self.map_size)

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                    (x / cam_range) * self.width // 2
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2
            x += self.width // 2
            y += self.height // 2
            if "agent_0" in entity.name:
                pygame.draw.circle(
                    self.screen, (0, 255, 0), (x, y), entity.size * 350 / (0.1 * cam_range)
                )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            else:
                pygame.draw.circle(
                    self.screen, entity.color * 200, (x, y), entity.size * 350 / (0.1 * cam_range)
                )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350 / (0.1 * cam_range), 1
            )  # borders

            if 'agent' in entity.name:
                # Add obs range
                pygame.draw.circle(
                    self.screen, (255, 0, 0), (x, y), self.obs_range / cam_range * self.width // 2, 1
                )
                pygame.draw.circle(
                    self.screen, (0, 0, 255), (x, y), self.com_range / cam_range * self.width // 2, 1
                )

            assert (
                    0 <= x <= self.width and 0 <= y <= self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                            "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False
