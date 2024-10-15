import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env

env = gym.make("racetrack-v0", render_mode="rgb_array")

model = PPO.load("racetrack_ppo/model", env=env)

while(True):
    obs, info = env.reset()
    done=truncated=False
    while not (done or truncated):
        # Predict
        action, _states = model.predict(obs, deterministic=True)
        # Get reward
        obs, reward, done, truncated, info = env.step(action)
        # Render
        env.render()