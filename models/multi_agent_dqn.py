#itializing environment

import gymnasium
import pprint
import highway_env
from matplotlib import pyplot as plt
import numpy as np
import pickle
import warnings
from datetime import date
from torch.utils.tensorboard import SummaryWriter
#%matplotlib inline

env_version = 'v1' #'v0' or 'v1'
render = False
env = gymnasium.make('intersection-'+env_version, render_mode=None)

#config
n_agents = 4
discrete = True

writer = SummaryWriter()

# Set action type based on discrete flag
action_type = "DiscreteMetaAction" if discrete else "ContinuousAction"

# Multi-agent environment configuration
env.unwrapped.config.update({
  "controlled_vehicles": n_agents,
  "observation": {
    "type": "MultiAgentObservation",
    "observation_config": {
      "type": "Kinematics",
    }
  },
  "action": {
    "type": "MultiAgentAction",
    "action_config": {
      "type": action_type,
    }
  }
})
env.reset()

#print config of env
print("configuration dict of environment:")
pprint.pprint(env.unwrapped.config)

#learning
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# if GPU is to be used
device = torch.device(
    # "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print("using ", device)

#replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

#save transitions to memory
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#Q-network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, n_agents=4):
        super(DQN, self).__init__()
        self.n_agents = n_agents
        
        # Shared feature extraction layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Separate output heads for each agent
        self.output_heads = nn.ModuleList([nn.Linear(128, output_dim) for _ in range(n_agents)])
        
    def forward(self, x):
        # Shared feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Get actions for each agent through separate heads
        agent_actions = []
        for head in self.output_heads:
            agent_action = F.softmax(head(x), dim=-1)
            agent_actions.append(agent_action)
            
        # Stack all agent actions
        return torch.stack(agent_actions, dim=1)  # Shape: [batch_size, n_agents, n_actions]
    

#hyperparameters and utilities
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3

#Get number of actions from gym action space
n_actions = 3
# Get the number of state observations
n_observations = 100
render_env = False

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state = state.to(device)
            with torch.no_grad():
                expected_reward = policy_net(state)
            
            #torch.tensor([tuple(torch.argmax(policy_net(state)) for _ in range(n_agents))], device=device, dtype=torch.long)
            print("policy net action")
            return torch.tensor([[[torch.argmax(head).item() for head in expected_reward[0]]]], device=device, dtype=torch.long), tuple(torch.argmax(head).item() for head in expected_reward[0])
    else:
        #return torch.tensor(tuple(env.action_space.sample()), device=device, dtype=torch.long)
        print("random action")
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), tuple(env.action_space.sample())


def get_observation(obs):
    obs_list = []
    for agent in obs:
        for row in agent:
            for element in row:
                obs_list.append(element)
    torch_obs = torch.tensor([[obs_list]], device=device, dtype=torch.float32, requires_grad=True)
    return torch_obs

episode_durations = []
episode_rewards = []


def plot_rewards(show_result=False):
    plt.figure(2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def get_action_value(network_output):
        return torch.tensor([[torch.max(head).item() for head in network_output[0]]], device=device, dtype=torch.float32)

# #optimizing network
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    print("Batch Completed. Optimizing... ")
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # for state in state_batch:
    #     network_output = policy_net(state)
    #     action = get_action_value(network_output=network_output)
    #     print(action)
    state_action_values = torch.tensor((), device=device, dtype = torch.float32)
    for state in state_batch:
        network_output = target_net(state)
        state_action_values = torch.cat((state_action_values, get_action_value(network_output)), 0)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.


    # Compute the expected Q values
    next_state_values = torch.tensor((), device=device, dtype = torch.float32)
    non_final_idx = 0
    with torch.no_grad():
        for mask in non_final_mask:
            if mask == True:
                network_output = target_net(non_final_next_states[non_final_idx])
                next_state_values = torch.cat((next_state_values, get_action_value(network_output)), 0)
            else:
                next_state_values = torch.cat((next_state_values, torch.tensor([[-1, -1, -1, -1]], device=device, dtype=torch.float32)), 0)

    # for mask in non_final_mask:
    #     if mask == True:
    #         state = non_final_next_states[non_final_idx]
    #         non_final_idx+=1
    #         with torch.no_grad():
    #             print(state)
    #             network_output = target_net(state)
    #         action = get_action_value(network_output=network_output)
    #         action_list = torch.cat((action_list, action), 1)
    #     else:
    #         action_list = torch.cat((action_list, torch.tensor([[-1, -1, -1, -1]], device = device, dtype = torch.float32)), 1)
    # next_state_values = torch.stack(action_list, dim=0)

    expected_state_action_values = torch.tensor((), device=device, dtype = torch.float32)
    reward_batch = reward_batch[:, None]
    reward_batch = torch.repeat_interleave(reward_batch, 4, 1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    state_action_values = state_action_values.requires_grad_()
    expected_state_action_values = expected_state_action_values.requires_grad_()

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    print("loss ", loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    #torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss



# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
    
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                           batch.next_state)), device=device, dtype=torch.bool)
#     non_final_mask = non_final_mask.to(device)

#     non_final_next_states = torch.cat([s for s in batch.next_state
#                                                 if s is not None])
#     non_final_next_states = non_final_next_states.to(device)

#     state_batch = torch.cat(batch.state)
#     state_batch = state_batch.to(device)

#     action_batch = torch.cat(batch.action)
#     action_batch = action_batch.to(device)

#     reward_batch = torch.cat(batch.reward)
#     reward_batch = reward_batch.to(device)

#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
    
#     print(state_batch.shape)
#     print(action_batch.shape)
#     state_action_values = policy_net(state_batch).gather(1, action_batch)
#     print(state_action_values.shape)

#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1).values
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     print(non_final_mask)
#     print(non_final_next_states.shape)
#     with torch.no_grad():
#         next_state_values[non_final_mask] = target_net(non_final_next_states[:, 0]).max(1).values
#     # Compute the expected Q values
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch

#     # Compute Huber loss
#     criterion = nn.SmoothL1Loss()
#     print(state_action_values.shape)
#     print(expected_state_action_values.shape)
#     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

#     # Optimize the model
#     optimizer.zero_grad()
#     loss.backward()
#     # In-place gradient clipping
#     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
#     optimizer.step()



num_episodes = 100000

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = get_observation(state)
    print(f"started episode {i_episode}: ")

    episode_reward = 0

    for t in count():
        print(f"running step {t}")
        action, action_tuple = select_action(state)
        print("selected action ", action_tuple)
        observation, reward, terminated, truncated, info = env.step(action_tuple)
        print("got reward ", reward)
        episode_reward+=reward
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        done = terminated or truncated
        if render_env is True:
            env.render()

        if terminated:
            next_state = None
        else:
            next_state = get_observation(observation) 

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(episode_reward)
            plot_durations()
            plot_rewards()
            break
        
        # Log metrics to tensorboard
        writer.add_scalar('Training/Episode Duration', t + 1, i_episode)
        writer.add_scalar('Training/Episode Reward', episode_reward, i_episode)
        if loss is not None:
            writer.add_scalar('Training/Loss', loss.item(), i_episode)

        # Save models and data every 1000 episodes
        if (i_episode + 1) % 1000 == 0:
            # Save model parameters using PyTorch
            model_save_path = f"saved_models/model"+date+"_episode_{i_episode+1}.pt"
            torch.save({
                'episode': i_episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'episode_durations': episode_durations
            }, model_save_path)
            
            print(f"Saved checkpoint at episode {i_episode+1}")

        #code to load stored model
        #checkpoint = torch.load(model_save_path)
        # policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        # target_net.load_state_dict(checkpoint['target_net_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

# Plot final metrics
plt.figure(figsize=(12, 4))

# Plot episode durations
plt.subplot(1, 2, 1)
plt.plot(episode_durations)
plt.title('Episode Durations')
plt.xlabel('Episode')
plt.ylabel('Duration')

# Plot episode rewards
plt.subplot(1, 2, 2)
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.tight_layout()
plt.show()

# Save final metrics
final_metrics_path = f"final_metrics.pkl"
with open(final_metrics_path, 'wb') as f:
    pickle.dump({
        'episode_durations': episode_durations,
        'episode_rewards': episode_rewards
    }, f)