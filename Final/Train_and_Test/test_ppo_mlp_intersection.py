import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

import highway_env  
import pprint

def test_env():
    
    return gym.make(
        "intersection-v0",
        render_mode="rgb_array",
        config={
            'vehicles_count': 10,
            'controlled_vehicles': 4,
            'destination': 'o1',
        }
    )

env = test_env()

pprint.pprint(env.unwrapped.config)

model = PPO.load("intersection_ppo_mlp_multi/model", env=env)

NUM_EP=100

for i in range(1,NUM_EP + 1):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        # Predict
        action, _states = model.predict(obs, deterministic=True)
        
        print("Action:",action)
        # Get reward
        obs, reward, done, truncated, info = env.step(action)
        print("Obervation:",obs)
        print("Info:",info)
        print(f"Done:{done} Truncated:{truncated}")
        print("Reward:",reward)
        # Render
        env.render()

    NUM_EP+=1