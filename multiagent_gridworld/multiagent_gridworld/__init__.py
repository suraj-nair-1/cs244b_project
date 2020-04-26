from gym.envs.registration import register

register(
    id='MultiGrid-v0',
    entry_point='multiagent_gridworld.gridworld:Gridworld',
)
