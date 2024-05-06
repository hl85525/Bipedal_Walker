import gym
import numpy as np
import torch
from sac_agent import SACAgent
from archs.ff_models import Actor, Critic


def test_model_with_rendering(env, agent, n_episodes=10, max_t_step=750):
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        t = 0
        while not done and t < max_t_step:
            env.render()  # Render the environment at each step
            t += 1
            action = agent.get_action(state, explore=False)
            action = action.clip(min=env.action_space.low, max=env.action_space.high)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        print(f'Episode {i_episode} Total Reward: {total_reward}')


# Set up the environment
env = gym.make('BipedalWalkerHardcore-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
clip_low, clip_high = env.action_space.low, env.action_space.high

# Initialize the SAC agent
agent = SACAgent(Actor, Critic, clip_low, clip_high, state_size=state_size, action_size=action_size)

# Load the trained models
agent.load_ckpt("BipedalWalkerHardcore-v3", "BipedalWalkerHardcore-v3")

# Run the testing with rendering
test_model_with_rendering(env, agent, n_episodes=5)

# Close the environment to clean up resources
env.close()
