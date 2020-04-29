from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box, Discrete
import gym
import math
import os

class Gridworld(gym.Env):
    def __init__(self):
        self.grid_size = 5
        self.grid = np.zeros((self.grid_size,self.grid_size)).astype(np.int32)
        self.num_agents = 3
        self.bad_agent_pos = np.zeros((2)).astype(np.int32)
        self.horizon = 50
        self.num_steps = 0
        self.action_space = Discrete(5)

    def apply_action(self, act):
        if act == 0:
            return np.array([0,0])
        elif act == 1:
            return np.array([0,-1])
        elif act == 2:
            return np.array([0,1])
        elif act == 3:
            return np.array([-1,0])
        elif act == 4:
            return np.array([1,0])

    def step(self, action):
        assert(action.shape == (self.num_agents,))
        #print(action)
        #### 0 = None, 1 = left, 2 = right, 3 = up, 4 = down
        ## Good Agents Step
        for i, act in enumerate(action):
            current_agent = i + 1
            next_pos = self.agents[current_agent] + self.apply_action(act)
            next_pos = next_pos.clip(0, self.grid_size - 1)
            if self.grid[next_pos[0], next_pos[1]] == 0:
                self.agents[current_agent] = next_pos
            self.update_grid()

        ## Bad Agent Step
        next_pos = self.bad_agent_pos + self.apply_action(np.random.randint(5))
        next_pos = next_pos.clip(0, self.grid_size - 1)
        if self.grid[next_pos[0], next_pos[1]] == 0:
            self.bad_agent_pos = next_pos

        rew = -np.linalg.norm(self.bad_agent_pos)
        self.update_grid()
        self.num_steps += 1
        done = (self.num_steps == self.horizon)
        return self.grid, rew, done, {}

    def update_grid(self):
        self.grid = np.zeros((self.grid_size,self.grid_size)).astype(np.int32)
        self.grid[self.bad_agent_pos[0], self.bad_agent_pos[1]] = -1
        for a in self.agents.keys():
            self.grid[self.agents[a][0], self.agents[a][1]] = a

    def reset(self):
        self.bad_agent_pos = np.zeros((2)).astype(np.int32)
        self.agents = {}
        for a in range(1, self.num_agents+1):
            pos = np.random.randint(1, self.grid_size, (2)).astype(np.int32)
            while self.grid[pos[0], pos[1]] != 0:
                pos = np.random.randint(1, self.grid_size, (2)).astype(np.int32)
            self.agents[a] = pos
            self.update_grid()

        self.update_grid()
        self.num_steps = 0
        return self.grid





