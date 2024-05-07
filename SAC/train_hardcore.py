import gym
import numpy as np
import torch
from archs.ff_models import Actor, Critic
from sac_agent import SACAgent
import matplotlib.pyplot as plt
from collections import deque


def train(env, agent, n_episodes=8000, model_type='unk', env_type='unk', score_limit=300.0, explore_episode=50,
          max_t_step=750):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        done = False
        agent.train_mode()
        t = int(0)

        while not done and t < max_t_step:
            t += int(1)
            action = agent.get_action(state, explore=True)
            action = action.clip(min=env.action_space.low, max=env.action_space.high)
            next_state, reward, done, _ = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            state = next_state
            score += reward
            agent.step_end()

        if i_episode > explore_episode:
            agent.episode_end()
            for i in range(t):
                agent.learn_one_step()

        scores_deque.append(score)
        avg_score_100 = np.mean(scores_deque)
        scores.append((i_episode, score, avg_score_100))
        print(f'Episode: {i_episode}, Steps: {t}, Reward: {score:.2f}')

        if avg_score_100 > score_limit:
            break

    return np.array(scores).transpose()


# Set up the environment
env = gym.make('BipedalWalkerHardcore-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
clip_low, clip_high = env.action_space.low, env.action_space.high

# Set up the agent
agent = SACAgent(Actor, Critic, clip_low, clip_high, state_size=state_size, action_size=action_size)

# Train the agent
scores = train(env, agent, n_episodes=3, model_type='BipedalWalkerHardcore-v3', env_type='BipedalWalkerHardcore-v3',
               score_limit=300.0, explore_episode=50, max_t_step=750)

# Save the trained models
agent.save_ckpt("BipedalWalkerHardcore-v3", "BipedalWalkerHardcore-v3")
