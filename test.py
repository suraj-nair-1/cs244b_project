import numpy as np
import multiagent_gridworld
import gym_minigrid
import gym


env = gym.make("MultiGrid-v0")
#env = gym.make("MiniGrid-Dynamic-Obstacles-16x16-v0")

for _ in range(1):
    env.reset()
    done = False
    while not done:
        obs, done, _ = env.step()
