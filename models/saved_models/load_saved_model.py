#load a model and initialize an environment picking actions from that model
import gymnasium
import highway_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import pprint
import matplotlib.pyplot as plt

# Define DQN class again since we need it to load the model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Initialize environment
env_version = 'v0'
env = gymnasium.make('intersection-'+env_version, render_mode='rgb_array')
env.unwrapped.config['controlled_vehicles'] = 1
env.reset()

# Initialize network with same parameters as training
device = torch.device("cpu")
n_observations = 105
n_actions = 3

# Create policy network and load saved state
policy_net = DQN(n_observations, n_actions).to(device)

# Load the saved model - update path to your specific saved model
model_path = "model_episode_9000.pt"  # Adjust episode number as needed
checkpoint = torch.load(model_path)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
policy_net.eval()  # Set to evaluation mode

print(f"Loaded model from episode {checkpoint['episode']}")

#print config of env
print("configuration dict of environment:")
pprint.pprint(env.unwrapped.config)

def get_observation(obs):
    obs_list = []
    for row in obs:
        for element in row:
            obs_list.append(element)
    return torch.tensor([obs_list], dtype=torch.float32)

# Run episodes with loaded model
num_test_episodes = 100
completed = []

for episode in range(num_test_episodes):
    state, info = env.reset()
    state = get_observation(state)
    done = False
    
    while not done:
        # Get action from policy network
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
        
        # Take step in environment
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        if reward == -5 and done:
            completed.append(0)
            print('collision')
        elif done:
            completed.append(1)
            print('complete')
        print(completed)

        if not done:
            state = get_observation(observation)
            
        env.render()
    
    print(f"Episode {episode + 1} finished")

# Plot completion statistics
plt.figure(figsize=(8, 6))
successes = completed.count(1)
failures = completed.count(0)

plt.bar(['Successes', 'Failures'], [successes, failures], 
        color=['green', 'red'])
plt.title('Episode Completion Statistics')
plt.ylabel('Number of Episodes')
plt.show()
print(len(completed))
env.close()
