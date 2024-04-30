import numpy as np
from torch import nn
import gym
import random
import torch
import matplotlib.pyplot as plt
from agent import SAC_Agent, ReplayBuffer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env = gym.make("BipedalWalker-v3")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

max_action = float(env.action_space.high[0])

agent = SAC_Agent(
    state_dim=obs_dim,
    action_dim=act_dim,
    max_action=max_action
)

replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(1e6))
max_episodes = 10
max_steps_per_episode = 1000
save_interval = 400
render = False
load = False
ModelIdex = 4800
if load: agent.load(ModelIdex)

ep_reward = 0
reward_list = []
plot_data = []

for episode in range(1, max_episodes + 1):
    state, done = env.reset(), False
    step_count = 0  # Reset the step count at the start of each episode
    while not done:
        action = agent.sample_action(state)
        next_state, reward, done, info, _ = env.step(action)
        step_count += 1  # Increment the step count

        if reward <= -100:
            reward = -1
            replay_buffer.add(state, action, reward, next_state, True)
        else:
            replay_buffer.add(state, action, reward, next_state, False)

        if replay_buffer.size > 2000: agent.train()

        if done or step_count >= max_steps_per_episode:
            break

        state = next_state
        ep_reward += reward

    if episode % save_interval == 0:
        agent.save(episode)

    reward_list.append(ep_reward)

    print('episode: {}, reward: {}'.format(episode, ep_reward))
    ep_reward = 0

    if episode % 10 == 0:
        plot_data.append([episode, np.array(reward_list).mean(), np.array(reward_list).std()])
        reward_list = []

# Save the final model
agent.save("final_model")

plt.figure(figsize=(8, 6))
plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', color='tab:blue')
plt.fill_between([x[0] for x in plot_data], [x[1] - x[2] for x in plot_data], [x[1] + x[2] for x in plot_data],
                 alpha=0.2, color='tab:blue')
plt.xlabel('Episode number')
plt.ylabel('Episode reward')
plt.title('SAC Agent - BipedalWalker-v3')
plt.savefig("reward_plot.png")
plt.show()