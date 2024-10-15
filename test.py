import gymnasium
import pprint
import highway_env
from matplotlib import pyplot as plt
import numpy as np
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

#print the action space
if discrete:
    print("action space: ", env.unwrapped.action_type.actions_indexes)

else:
    env.action_space.low = env.unwrapped.config['action']['steering_range'][0]
    env.action_space.high = env.unwrapped.config['action']['steering_range'][1]
    print("steering action space range: ", env.action_space.low, env.action_space.high)

while not done:
    if discrete:
        #for discrete action space
        action = env.unwrapped.action_type.actions_indexes["IDLE"] #pick action
    else:
        #for continuous action space
        #acceleration = np.random.uniform(env.action_space.low, env.action_space.high)
        acceleration = np.random.choice([0, 5, -5])
        steering = np.random.uniform(env.action_space.low/4, env.action_space.high/4)
        action = [acceleration, steering]
        
    print(action)
    obs, reward, done, truncated, info = env.step(action)
    print(obs.shape)
    env.render()

print("observations: ", obs)
print("observations shape: ", obs.shape)
print("info: ", info)

plt.imshow(env.render())
plt.show()