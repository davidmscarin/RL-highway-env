import os
import csv
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import highway_env  

TRAIN_FIRST_TIME = True
n_cpu = 24 

if __name__ == "__main__":

    model_dir = "intersection_dqn_cnn_multi/"
    best_model_path = f"{model_dir}/best_model"
    final_model_path = f"{model_dir}/model"
    log_file = f"{model_dir}/training_log.csv"

    os.makedirs(model_dir, exist_ok=True)

    env = make_vec_env(
        "intersection-v0", 
        n_envs=n_cpu, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs={
            'config':{
                'initial_vehicle_count': 10,
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
        }
    )

    best_mean_reward = float('-inf')
    best_timestep = 0

    # Check if we need to load the best model or train from scratch
    if TRAIN_FIRST_TIME:
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            exploration_fraction=0.7,
            verbose=2,
            tensorboard_log=model_dir,
        )
        
        current_timestep = 0

    else:
       
        model = DQN.load(final_model_path, env=env)

        
        current_timestep = 0
        if os.path.exists(log_file):
            with open(log_file, 'r') as csv_file:
                reader = csv.reader(csv_file)
                rows = list(reader)
                if rows:
                    last_row = rows[-3]  # Get the last row in the file
                    current_timestep = int(last_row[1])  # The first column stores the timestep

    total_timesteps = int(2e6)
    eval_interval = int(1e5)  # Evaluate every 100,000 timesteps

    file_exists = os.path.isfile(log_file)

    with open(log_file, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
       
        if not file_exists:
            writer.writerow(['Timestep', 'Mean Reward', 'Std Reward', 'Best Model Saved'])

        for step in range(current_timestep, total_timesteps + current_timestep, eval_interval):
            model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
            
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
            
            best_model_saved = False
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_timestep = step + eval_interval
                model.save(best_model_path)
                best_model_saved = True
            
            # Write to the log file
            writer.writerow([step + eval_interval, mean_reward, std_reward, best_model_saved])

            csv_file.flush()

    model.save(final_model_path)

    with open(log_file, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([])
        writer.writerow(['Final Model Info'])
        writer.writerow(['Final Timestep', total_timesteps])
        writer.writerow(['Best Model Timestep', best_timestep])
        writer.writerow(['Best Mean Reward', best_mean_reward])

        csv_file.flush()


    del model


