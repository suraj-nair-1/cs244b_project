import numpy as np
import multiagent_gridworld
import gym


env = gym.make("MultiGrid-v0")

for _ in range(1):
    env.reset()
    done = False
    while not done:
        actions = np.random.randint(0, 5, (3))
        obs, r, done, _ = env.step(actions)
        print(obs)
        print(r)
        print("-"*50)
        