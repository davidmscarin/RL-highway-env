import gymnasium
import highway_env
import pprint
import matplotlib.pyplot as plt
import numpy as np
import random

print(highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_ALL)
print(highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_LONGI)

action_space = {1: 'IDLE', 2: 'FASTER', 3: 'SLOWER'}

env_version = 'v1' #'v0' or 'v1'
env = gymnasium.make('intersection-'+env_version, render_mode='human')

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
      "lateral": False,
      "longitudinal": True
    }
  }
})
env.reset()

#print config of env
#print("configuration dict of environment:")
#pprint.pprint(env.unwrapped.config)

done = False
n_episodes = 300

stored_rewards = []
crashes = [0, 0, 0, 0]

for i_episode in range(n_episodes):
    #action = env.action_space.sample()
    action = (0, 0, 0, 0)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    agents_terminated = info['agents_terminated']
    rewards = info['agents_rewards']

    for i in range(len(crashes)):
        if agents_terminated[i]: crashes[i]+=1
    
    average_reward = sum(rewards)/len(rewards)
    stored_rewards.append(average_reward)

    if done:
        env.reset()

print(sum(stored_rewards)/len(stored_rewards))
print(crashes)