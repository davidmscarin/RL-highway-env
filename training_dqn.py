#itializing environment

import gymnasium
import pprint
import highway_env
from matplotlib import pyplot as plt
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

env_version = 'v0' #'v0' or 'v1'
env = gymnasium.make('intersection-'+env_version, render_mode='rgb_array')

#print config of env
print("configuration dict of environment:")
pprint.pprint(env.unwrapped.config)

#example to change the number of lanes
env.unwrapped.config['controlled_vehicles'] = 1
env.reset()

done = False
discrete = True if env_version == 'v0' else False #False if env is v1 true if env is v0


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
    # "mps" if torch.backends.mps.is_available() else
    "cpu"
)

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
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

#hyperparameters and utilities
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

#Get number of actions from gym action space
n_actions = 3
# Get the number of state observations
n_observations = 105

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
            print("reward expected: ", policy_net(state))
            print("index with highest expected reward: ", torch.argmax(policy_net(state)))
            return torch.tensor([[torch.argmax(policy_net(state))]], device=device, dtype=torch.long)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def get_observation(obs):
    obs_list = []
    for row in obs:
        for element in row:
            obs_list.append(element)
    torch_obs = torch.tensor([obs_list], dtype=torch.float32)
    return torch_obs

episode_durations = []
episode_rewards = []


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

#optimizing network
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
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
    
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



num_episodes = 10000

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = get_observation(state)
    print(f"state episode {i_episode}: ", state)

    episode_reward = 0

    for t in count():
        print(f"running step {t}")
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        print("reward: ", reward)
        episode_reward+=reward
        reward = torch.tensor([[reward]], device=device)
        done = terminated or truncated
        #env.render()

        if terminated:
            next_state = None
        else:
            next_state = get_observation(observation) 

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

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
            break
        
        # Save models and data every 1000 episodes
        if (i_episode + 1) % 1000 == 0:
            # Save model parameters using PyTorch
            model_save_path = f"saved_models/model_episode_{i_episode+1}.pt"
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
    



        