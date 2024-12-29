import gymnasium as gym
import sys
sys.path.insert(0, '/home/fsilva/Desktop/MIA/1ano/MS/project/rl-agents/scripts/')
#from HighwayEnv.scripts.utils import show_videos
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

env_config = 'configs/IntersectionEnv/env.json'
agent_config = 'configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json'

NUM_EPISODES = 100  #@param {type: "integer"}

env = load_environment(env_config)
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=NUM_EPISODES, display_env=True, display_agent=True)
print(f"Ready to train {agent} on {env}")


#%tensorboard --logdir "{evaluation.directory}"

evaluation.train()



     