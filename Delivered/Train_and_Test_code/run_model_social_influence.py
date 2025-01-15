import gymnasium
import highway_env
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from itertools import count

env_version = 'v1' #'v0' or 'v1'
render = True
if not render:
    render_mode = None
else:
    render_mode = 'human'
env = gymnasium.make('intersection-'+env_version, render_mode=render_mode)

device = torch.device(
    "cpu"
)

#config
n_agents = 4
discrete = True

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

import torch
import os
from glob import glob


#DQN policy net
class PolicyNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    #return action tensor
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
n_actions = 3
n_observations = 25

#initialize representations of agents
#each agent has a respective policy network, target network (used for soft updates to parameters) and optimizer
agents = {i: PolicyNetwork(n_observations, n_actions).to(device) for i in range(n_agents)}

def load_dqn_model(agents):
    
    for i_agent in range(n_agents):

        # Load the checkpoint
        checkpoint_path = f"../models/saved_models/model_default_agents_agent_{i_agent}_2024-12-15_episode_19999_LR_1e-05.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        # Load the state dictionaries
        agents[i_agent].load_state_dict(checkpoint['agent_policy_net_state_dict'])

        print(f"agent {i_agent} using model "+checkpoint_path)
        
        # Set starting episode if needed
        if i_agent == 0:  # only need to do this once
            starting_episode = checkpoint['episode'] + 1
            episode_rewards = checkpoint['episode_rewards']
            episode_durations = checkpoint['episode_durations']


def get_network_action(state, policy_net):
    return torch.argmax(policy_net(state)).item()

max_eps = 100

collisions = []
arrivals = []
rewards = []

load_dqn_model(agents)

for episode in range(max_eps):
    states, info = env.reset()

    num_crashed = 0
    num_arrived = 0
    episode_reward = 0

    for t in count():

        crashed = 0

        print(f"starting step {t}")
        actions_tuple = tuple()

        for i_state in range(len(states)):
            action = get_network_action(torch.tensor(states[i_state].flatten()), agents[i_state])
            actions_tuple = actions_tuple + (action, )
        
        print("action chosen: ", actions_tuple)


        observation, reward, terminated, truncated, info = env.step(actions_tuple)
        print("terminated", terminated)
        print("truncated", truncated)
        print(info)
        episode_reward+=reward
    
        print("mean reward: ", reward)
        #episode_reward+=reward
        done = terminated or truncated
        next_states = observation
        states = next_states

        if done:
            num_crashed = info['rewards']['collision_reward']/0.25
            num_arrived = info['rewards']['arrived_reward']/0.25
            collisions.append(num_crashed)
            arrivals.append(num_arrived)
            rewards.append(episode_reward)
            break

import matplotlib.pyplot as plt

# Create a figure with two subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(10, 8))

# Plot rewards
ax1.plot(rewards, 'b-', label='Rewards')
ax1.set_title('Episode Rewards')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.grid(True)
ax1.legend()

# Plot crash metric
ax2.plot(collisions, 'r-', label='Collisions')
ax2.set_title('Collisions')
ax2.set_xlabel('Step')
ax2.set_ylabel('No. Agents Crashed')
ax2.grid(True)
ax2.legend()

# Plot crashes per episode
ax3.plot(arrivals, 'g-', label=f'Arrivals')
ax3.set_title('Arrivals')
ax3.set_xlabel('Episode')
ax3.set_ylabel('No. Agents Arrived')
ax3.grid(True)
ax3.legend()

ax4.bar_plot()

# Adjust layout and display
plt.tight_layout()
plt.show()
        
