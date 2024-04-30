import gym
from agent import SAC_Agent
import torch

env = gym.make("BipedalWalker-v3", render_mode="human")  # or BipedalWalkerHardcore-v3
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

kwargs = {
    "state_dim": obs_dim,
    "action_dim": act_dim,
    "max_action": max_action,
    "gamma": 0.99,
    "net_width": 128,
    "alpha": 0.2,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "batch_size": 256,
    "replay_buffer_size": int(1e6)
}

agent = SAC_Agent(**kwargs)
agent.actor.load_state_dict(torch.load("sac_actorfinal_model.pth"))
agent.q_critic1.load_state_dict(torch.load("sac_q_critic_1final_model.pth"))
agent.q_critic2.load_state_dict(torch.load("sac_q_critic_2final_model.pth"))

num_episodes = 5

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.sample_action(state)
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        state = next_state

    print(f"Episode {episode+1}: Reward = {episode_reward}")

env.close()