import gymnasium
import highway_env
import pprint
import matplotlib.pyplot as plt
import numpy as np
import random

print(highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_ALL)
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
    }
  }
})
env.reset()

#print config of env
print("configuration dict of environment:")
pprint.pprint(env.unwrapped.config)

done = False

while True:
    if discrete:
        #for discrete action space
        # Create tuple of random actions (integers) from action space keys
        actions = tuple(random.randint(1, 3) for _ in range(n_agents))
    else:
        #for continuous action space
        #acceleration = np.random.uniform(env.action_space.low, env.action_space.high)
        acceleration = np.random.choice([0, 5, -5])
        steering = np.random.uniform(env.action_space.low/4, env.action_space.high/4)
        action = [acceleration, steering]
        
    obs, reward, done, truncated, info = env.step(actions)
    obs = np.array(obs)
    print(obs.shape)
    if done:
        env.reset()
    env.render()

    

plt.imshow(env.render())
plt.show()