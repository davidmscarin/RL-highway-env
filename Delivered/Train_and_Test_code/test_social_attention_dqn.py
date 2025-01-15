import gymnasium as gym
import sys
sys.path.insert(0, '/home/isiauser/MIA/1ano/MS/MIA_MOD_SIM/rl-agents/scripts/')
sys.path.insert(0, '/home/isiauser/MIA/1ano/MS/MIA_MOD_SIM/rl-agents/')
from HighwayEnv.scripts.utils import show_videos

from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment


# Get the environment and agent configurations from the rl-agents repository
#%cd /content/rl-agents/scripts/
env_config = 'configs/IntersectionEnv/env.json'
agent_config = 'configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json'


#Run the learned policy for a few episodes
env = load_environment(env_config)
env.config["offscreen_rendering"] = True
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=20, training = False, recover = True)
print(evaluation)
evaluation.test()
show_videos(evaluation.run_directory)