import os
import csv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import highway_env  

TRAIN_FIRST_TIME = True
n_cpu = 24 

if __name__ == "__main__":
   
    model_dir = "intersection_ppo_mlp_multi/"
    log_file = f"{model_dir}/training_log.csv"
    final_model_path = f"{model_dir}/model"

    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(log_file):
        print(f"Creating new log file at {log_file}")

    env = make_vec_env(
        "intersection-v0", 
        n_envs=n_cpu, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs={
            'config':{
                'vehicles_count': 10,
                'controlled_vehicles': 4,
                'destination': 'o1',
            },
        }
    )

    if TRAIN_FIRST_TIME:
        batch_size = 64
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.9,
            verbose=2,
            tensorboard_log=model_dir,
        )
    else:
        model = PPO.load(final_model_path, env=env)
    
    total_timesteps = int(1e6)
    eval_interval = int(1e5)  # Evaluate every 100,000 timesteps
    best_mean_reward = float('-inf')

    with open(log_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Timestep', 'Mean Reward', 'Std Reward', 'Best Model Saved'])

    for step in range(0, total_timesteps, eval_interval):
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

        best_model_saved = False
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            model.save(f"{model_dir}/best_model")
            best_model_saved = True

        with open(log_file, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([step + eval_interval, mean_reward, std_reward, best_model_saved])

    model.save(final_model_path)
    print(f"Final model saved at {final_model_path}")

