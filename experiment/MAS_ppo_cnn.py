import  gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import pprint
import highway_env

env = gym.make(
  "highway-v0",
  render_mode="rgb_array",
  config={
    "observation": {
      "type": "MultiAgentObservation",
      "observation_config": {
        "type": "Kinematics",
      }
    }
  }
)



#print.pprint(obs)

# Multi-agent environment configuration
env.unwrapped.config.update({
  "controlled_vehicles": 2,
  "observation": {
    "type": "MultiAgentObservation",
    "observation_config": {
      "type": "Kinematics",
    }
  },
  "action": {
    "type": "MultiAgentAction",
    "action_config": {
      "type": "DiscreteMetaAction",
    }
  }
})
n_cpu = 12
batch_size = 64
model = PPO(
        "CnnPolicy",
        env,
        #policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log="intersection_ppo_cnn/",
    )


obs, info = env.reset()
model.learn(total_timesteps=int(1e5))
model.save("intersection_ppo_cnn/model")


'''
# A training episode
obs, info = env.reset()
done = truncated = False
while not (done or truncated):
  # Dispatch the observations to the model to get the tuple of actions
  action = tuple(model.predict(obs_i) for obs_i in obs)
  # Execute the actions
  next_obs, reward, done, truncated, info = env.step(action)
  # Update the model with the transitions observed by each agent
  for obs_i, action_i, next_obs_i in zip(obs, action, next_obs):
    model.update(obs_i, action_i, next_obs_i, reward, info, done, truncated)
  obs = next_obs
'''