from collections import OrderedDict
import numpy as np
from gym.spaces import Dict, Box, Discrete
import gym
import math
import os
import ipdb
from .diagonal_movement import DiagonalMovement
from .astar import AStarFinder
from .grid import Grid
import matplotlib.pyplot as plt


class Gridworld(gym.Env):
    def __init__(self):
        self.grid_size = 10
        self.grid = np.zeros((self.grid_size, self.grid_size)).astype(np.int32)
        self.true_grid_obs = [[1] * self.grid_size for _ in range(self.grid_size)]
        self.num_agents = 4
        self.num_obstacles = 10
        self.horizon = 50
        self.num_steps = 0
        self.action_space = Discrete(4)

    def step(self, consensus_obstacle_list=None):
        if consensus_obstacle_list is not None:
            # build an grid based on this list
            self.grid_obs = [[1] * self.grid_size for _ in range(self.grid_size)]
            for obs in consensus_obstacle_list:
                if obs[0] > self.grid_size - 1: obs[0] = self.grid_size - 1
                if obs[0] < 0: obs[0] = 0
                if obs[1] > self.grid_size - 1: obs[1] = self.grid_size - 1
                if obs[1] < 0: obs[1] = 0
                self.grid_obs[obs[0]][obs[1]] = 0
        else:
            self.grid_obs = self.true_grid_obs

        # Agents Step
        # print("SAME GRID, BEFORE ACTION")
        # print(self.grid)
        for a in range(1, self.num_agents + 1):
            grid = Grid(matrix=self.grid_obs)
            start = grid.node(self.agents[a][0], self.agents[a][1])
            end = grid.node(self.goal[0], self.goal[1])
            finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
            path, runs = finder.find_path(start, end, grid)
            if len(path) > 1:
                next_pos = path[1]
            elif len(path) == 1:
                next_pos = path[0]
            elif len(path) == 0:
                # print("STAYING PUT")
                next_pos = self.agents[a]  # stay put
            if self.grid[next_pos[0], next_pos[1]] == -1:  ### NOTE THAT WE ARE USING self.grid NOT self.grid_obs
                # print("STAYING PUT")
                next_pos = self.agents[a]  # stay put

            #print("curr: ", self.agents[a], "next: ", next_pos, a)

            assert self.grid[next_pos[0], next_pos[1]] != -1
            assert self.true_grid_obs[next_pos[0]][next_pos[1]] != 0
            self.agents[a] = next_pos
            self.update_grid()
        # print("SAME GRID, AFTER ACTION")
        # print(self.grid)

        # If all agents have reached goal, DONE
        done = False
        n_done = 0
        for a in range(1, self.num_agents + 1):
            if list(self.agents[a]) == self.goal:
                n_done += 1

        if n_done == self.num_agents:
            done = True

        # Place obstacles
        # print("UPDATING GRIDS")
        self.obstacles = []
        for o in range(self.num_obstacles):
            pos = self.place()
            pos = [int(x) for x in pos]
            self.obstacles.append(list(pos))
            self.update_grid()

        self.update_grid()
        # print(self.grid)
        # print("-"*50)
        self.num_steps += 1

        return self.true_grid_obs, self.obstacles, done, {}

    def update_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size)).astype(np.int32)
        self.true_grid_obs = [[1] * self.grid_size for _ in range(self.grid_size)]

        self.grid[self.goal[0], self.goal[1]] = 5
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = -1
            self.true_grid_obs[obs[0]][obs[1]] = 0
        for a in self.agents.keys():
            self.grid[self.agents[a][0], self.agents[a][1]] = a

    def place(self):
        pos = np.random.randint(1, self.grid_size, (2)).astype(np.int32)
        while self.grid[pos[0], pos[1]] != 0:
            pos = np.random.randint(1, self.grid_size, (2)).astype(np.int32)
        return pos

    def reset(self):
        self.agents = {}
        self.obstacles = []

        # Place goal
        self.goal = [self.grid_size - 1, self.grid_size - 1]
        self.update_grid()

        # Place agents
        for a in range(1, self.num_agents + 1):
            pos = self.place()
            self.agents[a] = pos
            self.update_grid()

        # Place obstacles
        for o in range(self.num_obstacles):
            pos = self.place()
            pos = [int(x) for x in pos]
            self.obstacles.append(list(pos))
            self.update_grid()

        self.update_grid()
        self.num_steps = 0
        return self.true_grid_obs, self.obstacles

    def render(self):
        print(self.grid)
        #plt.imshow(self.grid)
        #plt.savefig(f"gifs/test/t{self.num_steps}")
        #plt.show()
