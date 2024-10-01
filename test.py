import gymnasium
import highway_env
from matplotlib import pyplot as plt
#%matplotlib inline

env = gymnasium.make('intersection-v0', render_mode='rgb_array')
env.reset()
while True:
    action = env.unwrapped.action_type.actions_indexes["IDLE"] #pick action
    obs, reward, done, truncated, info = env.step(action)
    if done:
        print("actions: ", env.unwrapped.action_space)
        print("observations: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("truncated: ", truncated)
        print("info: ", info)
    env.render()

plt.imshow(env.render())
plt.show()