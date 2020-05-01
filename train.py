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
import cv2 
from PIL import Image


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 200000
TARGET_UPDATE = 1
N_EP = 100000
HIDDEN_SIZE = 128

env = gym.make("MultiGrid-v0")
eval_env = gym.make("MultiGrid-v0")
n_actions = env.action_space.n
n_agents = env.num_agents
grid_size = env.grid_size
input_size = grid_size * grid_size
output_size = 1 #n_actions * n_agents
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(input_size, HIDDEN_SIZE, output_size, n_actions, n_agents).to(device)
target_net = DQN(input_size, HIDDEN_SIZE, output_size, n_actions, n_agents).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(1000000)

steps_done = 0

logger = Logger('./logs/ours9/')

all_acts = []
for agent1_act in range(n_actions):
    for agent2_act in range(n_actions):
        for agent3_act in range(n_actions):
            a = torch.zeros((n_agents, n_actions)).to(device)
            a[0, agent1_act] = 1
            a[1, agent2_act] = 1
            a[2, agent3_act] = 1
            all_acts.append(a)
all_acts = torch.stack(all_acts, 0)


def max_act(net, state):
    ## TODO (surajn): find a better way to do this.
    state_input = state.unsqueeze(1).repeat(1, all_acts.size(0), 1).reshape(
                (all_acts.size(0) * state.size(0) , -1))
    act_input = all_acts.unsqueeze(0).repeat(state.size(0), 1, 1, 1).reshape(
              (all_acts.size(0) * state.size(0) , -1))
    qvals = net(state_input, act_input)
    qvals = qvals.reshape(state.size(0), all_acts.size(0), 1)
    best_act = all_acts[qvals.max(1)[1].long()]
    best_q = qvals.max(1)[0].view(state.size(0), 1)
    best_a = best_act.max(-1)[0].view(state.size(0), n_agents)
    return best_q, best_a.long()
  
  
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
            _, ma = max_act(policy_net, state)
            return ma
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
    
    full_action_batch = torch.zeros((BATCH_SIZE, n_actions * n_agents)).float().to(device)
    full_action_batch[action_batch[0]] = 1
    full_action_batch[n_actions + action_batch[1]] = 1
    full_action_batch[2 * n_actions + action_batch[2]] = 1

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch, full_action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, 1, device=device)
    tq, ta = max_act(target_net, non_final_next_states)
    next_state_values[non_final_mask] = tq.float()
    

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # b x n_agents

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.float())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss


def evaluate(epoch, num_trials = 10):
    global policy_net
    print("\n evaluating")
    all_rews = []
    for k in range(num_trials):
        state = eval_env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).flatten().unsqueeze(0)
        rews = []
        images = []
        ts = 0
        done = False
        while not done:
            with torch.no_grad():
                policy_net.eval()
                _, actions = max_act(policy_net, state) #policy_net(state).max(1)[1]
            im = np.zeros((grid_size*grid_size, 3))
            for s in range(state.size(1)):
                if state[0,s] == -1:
                    im[s, :] = 255
                elif state[0,s] == 1:
                    im[s, 0] = 255
                elif state[0,s] == 2:
                    im[s, 1] = 255
                elif state[0,s] == 3:
                    im[s, 2] = 255
            im = cv2.resize(im.reshape((grid_size,grid_size, 3)), (128, 128))
            
            images.append(Image.fromarray(im.astype(np.uint8)))
            next_state, r, done, _ = eval_env.step(actions[0].cpu().detach().numpy())
            state = torch.tensor(next_state, device=device, dtype=torch.float32).flatten().unsqueeze(0)
            rews.append(r)
            ts += 1
            
        all_rews.append(np.sum(rews))
        images[0].save(f'ims/eval_epoch_{epoch}_{k}.gif',
                   save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)
    print("episodes rew vals: ", np.mean(all_rews))
    logger.scalar_summary("episodes rew vals", np.mean(all_rews), epoch)



def train():
    global policy_net, target_net
    for episode in range(N_EP):
        state = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).flatten().unsqueeze(0)
        done = False
        policy_net.train()
        while not done:
            # select and perform an action
            actions = select_actions(state)

            next_state, reward, done, _ = env.step(actions[0].cpu().detach().numpy())
            next_state = torch.tensor(next_state, device=device, dtype=torch.float32).flatten().unsqueeze(0)
            reward = torch.tensor([reward], device=device).unsqueeze(0)

            # store transition
            memory.push(state, actions, next_state, reward)

            # determine next state
            state = next_state

            # perform optimization
        for _ in range(100):
            l = optimize_model()
        # update target network with policy network's params
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print(l)
        if l is not None:
            logger.scalar_summary("loss", l.cpu().detach().numpy(), episode)
        if episode % 100 == 0:
            evaluate(episode)
        


if __name__=='__main__':
    train()
