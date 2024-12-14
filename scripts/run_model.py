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

def load_dqn_model(model_path=None, policy_net=None, target_net=None, optimizer=None):
    """
    Load a saved DQN model and its training state.
    
    Args:
        model_path (str): Path to specific model file. If None, loads most recent model.
        policy_net: The policy network to load weights into
        target_net: The target network to load weights into
        optimizer: The optimizer to load state into
    
    Returns:
        dict: Contains loaded model information including:
            - policy_net_state_dict
            - target_net_state_dict
            - optimizer_state_dict
            - episode_rewards
            - episode_durations
            - episode
    """
    
    if model_path is None:
        # Find most recent model file
        print('no path given')
        exit()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Load model states if networks are provided
    if policy_net is not None:
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    if target_net is not None:
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded model from {model_path}")
    print(f"Model was saved at episode {checkpoint['episode']}")
    
    return checkpoint

def get_network_action(state):
    expected_reward = policy_net(state)
    return tuple(torch.max(head) for head in expected_reward[0])

def get_state(obs):
    obs_list = []
    for agent in obs:
        for row in agent:
            for element in row:
                obs_list.append(element)
    torch_obs = torch.tensor([[obs_list]], device=device, dtype=torch.float32, requires_grad=True)
    return torch_obs


#Get number of actions from gym action space
n_actions = 3
# Get the number of state observations
n_observations = 100
render_env = False

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3, amsgrad=True)

model_path = '/Users/davidscarin/Documents/Uni/MIA/MS/RL-highway-env/models/saved_models/model_default_agents2024-12-10_episode_999.pt'

# Load most recent model
checkpoint = load_dqn_model(model_path=model_path, policy_net=policy_net, target_net=target_net, optimizer=optimizer)

# Access model information
episode = checkpoint['episode']
rewards = checkpoint['episode_rewards']
durations = checkpoint['episode_durations']

print(f"Total episodes trained: {episode}")
print(f"Final average reward: {sum(rewards[-100:]) / 100}")  # Last 100 episodes

obs, info = env.reset()

while True:
    time.sleep(1)
    state = get_state(obs)
    actions = get_network_action(state)    
    obs, reward, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    if done:
        obs, info = env.reset()
        
    
        
