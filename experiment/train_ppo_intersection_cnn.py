import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env  # noqa: F401


TRAIN = True

if __name__ == "__main__":
    n_cpu = 12
    batch_size = 64
    env = gym.make(
        "intersection-v1",
        config={
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,
            },
        },
    )
    model = PPO(
        "CnnPolicy",
        env,
        #policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log="intersection_ppo_cnn/",
    )
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(5e5))
        model.save("intersection_ppo_cnn/model")
        del model

    '''
    # Run the algorithm
    model = PPO.load("racetrack_ppo/model", env=env)

    env = gym.make("racetrack-v0")
    env = RecordVideo(
        env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)

    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
'''