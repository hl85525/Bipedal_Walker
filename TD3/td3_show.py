import gym
import torch
from TD3.agent import Agent

# Set the path to your trained model
model_path = "td3_actor_1500.pth"

# Create the environment
env = gym.make("BipedalWalker-v3", render_mode="human")

# Get the observation and action dimensions
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Set the maximum action value
max_action = float(env.action_space.high[0])

# Create an instance of the Agent
kwargs = {
    "env_with_Dead": True,
    "state_dim": obs_dim,
    "action_dim": act_dim,
    "max_action": max_action,
    # Add other necessary arguments
}
agent = Agent(**kwargs)

# Load the trained model
agent.actor.load_state_dict(torch.load(model_path))

# Evaluate the agent
num_episodes = 5  # Number of episodes to render

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.sample_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        episode_reward += reward
        env.render()

    print(f"Episode {episode + 1}: Reward = {episode_reward}")

env.close()