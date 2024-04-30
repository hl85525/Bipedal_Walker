import gym
from agent import DDPG_Agent
import torch

# Make sure you specify the appropriate render mode if necessary
env = gym.make("BipedalWalker-v3", render_mode="human")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

kwargs = {
    "state_dim": obs_dim,
    "action_dim": act_dim,
    "max_action": max_action,
    "gamma": 0.99,
    "net_width": 128,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "tau": 0.005,
    "batch_size": 256,
    "replay_buffer_size": int(1e6)
}

agent = DDPG_Agent(**kwargs)
agent.actor.load_state_dict(torch.load("ddpg_actorfinal_model.pth"))
agent.critic.load_state_dict(torch.load("ddpg_criticfinal_model.pth"))

num_episodes = 5

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.sample_action(state)
        # Correctly unpack values considering potential additional outputs
        next_state, reward, done, _ = env.step(action)[:4]
        episode_reward += reward
        state = next_state

    print(f"Episode {episode+1}: Reward = {episode_reward}")

env.close()
