import gymnasium
import pprint
import highway_env
from matplotlib import pyplot as plt
#%matplotlib inline

env = gymnasium.make('intersection-v0', render_mode='rgb_array')

#print config of env
print("configuration dict of environment:")
pprint.pprint(env.unwrapped.config)

#example to change the number of lanes
env.unwrapped.config["lanes_count"] = 2
env.reset()

done = False

while not done:
    action = env.unwrapped.action_type.actions_indexes["IDLE"] #pick action
    print(action)
    obs, reward, done, truncated, info = env.step(action)
    env.render()

print("observations: ", obs)
print("reward: ", reward)
print("done: ", done)
print("truncated: ", truncated)
print("info: ", info)

plt.imshow(env.render())
plt.show()