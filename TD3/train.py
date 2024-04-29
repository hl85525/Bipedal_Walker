import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from td3.agent import Agent
from td3.replay_buffer import ReplayBuffer

env = gym.make("BipedalWalker-v3")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
seed = 42
torch.manual_seed(seed)
env.action_space.seed(seed)
np.random.seed(seed)
env_with_Dead = True
max_action = float(env.action_space.high[0])
kwargs = {
    "env_with_Dead": env_with_Dead,
    "state_dim": obs_dim,
    "action_dim": act_dim,
    "max_action": max_action,
    "gamma": 0.99,
    "net_width": 200,
    "a_lr": 1e-4,
    "c_lr": 1e-4,
    "Q_batchsize": 256
}
agent = Agent(**kwargs)
replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(1e6))
max_episodes = 2000
max_steps = 1000  # Maximum number of steps per episode
render = False
ep_reward = 0
reward_list = []
plot_data = []

for episode in range(1, max_episodes + 1):
    state, _ = env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        action = agent.sample_action(state)
        next_state, reward, done, _, _ = env.step(action)
        if reward <= -100:
            reward = -1
            replay_buffer.add(state, action, reward, next_state, True)
        else:
            replay_buffer.add(state, action, reward, next_state, False)
        if replay_buffer.size > 2000:
            agent.train(replay_buffer)
        state = next_state
        ep_reward += reward
        step += 1
    reward_list.append(ep_reward)
    print(f"Episode: {episode}, Steps: {step}, Reward: {ep_reward}")
    ep_reward = 0
    if episode % 10 == 0:
        plot_data.append([episode, np.array(reward_list).mean(), np.array(reward_list).std()])
        reward_list = []

# Save the final model
agent.save("final_model")

# Plot the reward graph
plt.figure(figsize=(8, 5))
plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', color='tab:blue')
plt.fill_between([x[0] for x in plot_data], [x[1] - x[2] for x in plot_data], [x[1] + x[2] for x in plot_data], alpha=0.2, color='tab:blue')
plt.xlabel('Episode number')
plt.ylabel('Episode reward')
plt.title('Training Reward Plot')
plt.savefig("reward_plot.png")
plt.show()