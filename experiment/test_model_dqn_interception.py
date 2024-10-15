import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

import highway_env  # noqa: F401

env = gym.make("intersection-v0", render_mode="rgb_array")

model = DQN.load("highway_dqn/model", env=env)


while(True):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        # Predict
        action, _states = model.predict(obs, deterministic=True)
        # Get reward
        obs, reward, done, truncated, info = env.step(action)
        # Render
        env.render()