import numpy as np
import gym
import random
import torch
import matplotlib.pyplot as plt
from ff_models import Actor, Critic
from agent import ReplayBuffer
from improved_ddpg_agent import DDPG_Agent
import config
from test_DDPG import test

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
seed = config.SEED
max_episodes = config.MAX_EPISODES
max_steps_per_episode = config.MAX_STEPS_PER_EPISODE
save_interval = config.SAVE_INTERVAL
render = False
load = False
ModelIdex = 4800
plot_data = []


def train(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(obs_dim, act_dim)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)

    actor = Actor(obs_dim, act_dim)
    critic = Critic(obs_dim, act_dim)
    agent = DDPG_Agent(
        Actor=actor, Critic=critic, state_dim=obs_dim, action_dim=act_dim
    )

    replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(1e6))

    if load:
        print("Loading model")
        agent.load(ModelIdex)

    ep_reward = 0
    reward_list = []
    reward = 0
    has_agent_succeeded = False
    done = False

    for episode in range(max_episodes):
        state, *_ = env.reset()
        step_count = 0  # Reset the step count at the start of each episode

        # Test it every 50 episodes
        if episode % 50 == 0 or has_agent_succeeded:
            has_agent_succeeded = test(env, agent)
            if has_agent_succeeded:
                print("Agent has solved the environment!")
                break
                
        while not done:
            print("STATE: ", state)
            action = agent.get_action(state)
            # Use a wildcard to capture all extra values
            next_state, reward, done, *_ = env.step(action)
            step_count += 1  # Increment the step count

            if reward <= -100:
                reward = -1
                replay_buffer.add(state, action, reward, next_state, True)
            else:
                replay_buffer.add(state, action, reward, next_state, False)

            if replay_buffer.size > config.MIN_BUFFER_LIMIT:
                agent.train(state, action, reward, next_state, done)

            if done or step_count >= max_steps_per_episode:
                break

            state = next_state
            ep_reward += reward

        if episode % save_interval == 0:
            agent.save_ckpt(str(episode))

        reward_list.append(ep_reward)

        if ep_reward >= 300:
            has_agent_succeeded = True
            # Will be tested in the next episode

        print("[Episode {}] total reward = {}".format(episode, ep_reward))
        reward = ep_reward
        ep_reward = 0

        if episode % 10 == 0:
            plot_data.append(
                [episode, np.array(reward_list).mean(), np.array(reward_list).std()]
            )
            reward_list = []

        episode += 1

    # Save the final model
    agent.save_ckpt("final_model")

if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    train(env)

    plt.figure(figsize=(8, 6))
    plt.plot(
        [x[0] for x in plot_data], [x[1] for x in plot_data], "-", color="tab:blue"
    )
    # plt.fill_between(
    #     [x[0] for x in plot_data],
    #     [x[1] - x[2] for x in plot_data],
    #     [x[1] + x[2] for x in plot_data],
    #     alpha=0.2,
    #     color="tab:blue",
    # )
    plt.xlabel("Episode number")
    plt.ylabel("Episode reward")
    plt.title("Vanilla DDPG Agent - BipedalWalker-v3")
    plt.savefig("reward_plot.png")
    plt.show()
