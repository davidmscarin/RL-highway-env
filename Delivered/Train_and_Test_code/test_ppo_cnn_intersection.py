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
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,
            },
        },
    )


#env = gym.make("intersection-v0", render_mode="rgb_array")

env = test_env()

pprint.pprint(env.unwrapped.config)

model = PPO.load("intersection_ppo_cnn_multi/model", env=env)


while(True):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        # Predict
        action, _states = model.predict(obs, deterministic=True)
       # print("Obervation:",obs)
        print("Action:",action)
        # Get reward
        obs, reward, done, truncated, info = env.step(action)
        print("Reward:",reward)
        # Render
        env.render()