import gym
from agent import Agent

env = gym.make("BipedalWalker-v3", render_mode="human") # or BipedalWalkerHardcore-v3
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

kwargs = {
    "env_with_Dead": True,
    "state_dim": obs_dim,
    "action_dim": act_dim,
    "max_action": max_action,
    "gamma": 0.99,
    "net_width": 200, # or 200 for normal version
    "a_lr": 1e-4,
    "c_lr": 1e-4,
    "Q_batchsize": 256
}

agent = Agent(**kwargs)
agent.load("final_model")

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