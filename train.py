# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gym
import numpy as np
import random, math
from DQN import DQN, ReplayMemory, Transition
from logger import Logger
import multiagent_gridworld
import torch
import torch.nn.functional as F
import torch.optim as optim
import ipdb

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 5
N_STEPS = 3000
N_EPOCHS = 50
HIDDEN_SIZE = 128

env = gym.make("MultiGrid-v0")
eval_env = gym.make("MultiGrid-v0")
n_actions = env.action_space.n
n_agents = env.num_agents
grid_size = env.grid_size
input_size = grid_size * grid_size
output_size = n_actions * n_agents
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(input_size, HIDDEN_SIZE, output_size, n_actions, n_agents).to(device)
target_net = DQN(input_size, HIDDEN_SIZE, output_size, n_actions, n_agents).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

logger = Logger('./logs/ours2/')


def select_actions(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1]
    else:
        return torch.tensor([[random.randrange(n_actions) for _ in range(n_agents)]], device=device, dtype=torch.long)


def optimize_model():
    global policy_net, loss, optimizer
    if len(memory) < BATCH_SIZE:  # buffer needs to be large enough
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))  # converts batch-array of Transitions to Transition of batch-arrays
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)  # b x n_agents
    reward_batch = torch.cat(batch.reward)  # b x n_agents

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    out = policy_net(state_batch)
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)) # b x 1 x n_agents

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, n_agents, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # b x n_agents

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.double(), expected_state_action_values.unsqueeze(1).double())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def evaluate(epoch):
    global policy_net
    print("\n evaluating")
    state = eval_env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float32).flatten().unsqueeze(0)
    rews = []
    done = False
    while not done:
        with torch.no_grad():
            policy_net.eval()
            actions = policy_net(state).max(1)[1]
            #actions = torch.tensor([[random.randrange(n_actions) for _ in range(n_agents)]], device=device, dtype=torch.long)  # random actions
        next_state, r, done, _ = eval_env.step(actions[0].numpy())
        state = torch.tensor(next_state, device=device, dtype=torch.float32).flatten().unsqueeze(0)
        rews.append(r)
    print("episode rew val: ", np.sum(rews))
    logger.scalar_summary("episode rew val", np.sum(rews), epoch)



def train():
    global policy_net, target_net
    for epoch in range(N_EPOCHS):
        state = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).flatten().unsqueeze(0)
        done = False
        policy_net.train()
        for step in range(N_STEPS):
            # select and perform an action
            actions = select_actions(state)
            next_state, reward, done, _ = env.step(actions[0].numpy())
            next_state = torch.tensor(next_state, device=device, dtype=torch.float32).flatten().unsqueeze(0)
            reward = torch.tensor([reward]*n_agents, device=device).unsqueeze(0)

            # store transition
            memory.push(state, actions, next_state, reward)

            # determine next state
            if done:
                state = env.reset()
                state = torch.tensor(state, device=device, dtype=torch.float32).flatten().unsqueeze(0)
            else:
                state = next_state

            # perform optimization
            optimize_model()

        evaluate(epoch)
        # update target network with policy network's params
        if epoch % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())


if __name__=='__main__':
    train()
