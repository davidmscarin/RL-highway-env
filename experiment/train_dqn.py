import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

import highway_env  # noqa: F401


TRAIN_FIRST_TIME = True

if __name__ == "__main__":
    n_cpu = 24
    batch_size = 64
    env = make_vec_env("intersection-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
   
    if (TRAIN_FIRST_TIME == True):
        model = DQN(
            "MlpPolicy",
            env,
            #policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="intersection_dqn/model",
            device="cuda"
        )
    
    else:
            model=DQN.load("intersection_dqn/model", env=env)
    

    # Train the model
    model.learn(total_timesteps=int(1e5),reset_num_timesteps=False)
    model.save("intersection_dqn/model")
    del model
        
    '''
    
    
    # Run the trained model and record video
    model = DQN.load("highway_dqn/model", env=env)
    env = RecordVideo(
        env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)
    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering

    for videos in range(10):
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