import gym
import numpy as np
from improved_ddpg_agent import DDPG_Agent
import torch
import config


def test(env, agent: DDPG_Agent):
    all_ep_rewards = []
    for episode in range(5):
        state, *_ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, *_ = env.step(action)
            episode_reward += reward
            state = next_state

            if episode_reward <= -200:
                print("[Evaluation] Episode ended due to low reward")
                break

            print(f"[Evaluation] Step taken and got Reward: {episode_reward}")

        print(f"[Episode {episode+1}] Total Reward = {episode_reward}")
        all_ep_rewards.append(episode_reward)

    print(
        f"Finished testing. Max Reward: {max(all_ep_rewards)} Mean Reward: {np.mean(all_ep_rewards)}"
    )
    return np.mean(all_ep_rewards) >= 300


if __name__ == "__main__":
    # Make sure you specify the appropriate render mode if necessary
    env = gym.make("BipedalWalker-v3", render_mode="human")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": obs_dim,
        "action_dim": act_dim,
        "gamma": config.GAMMA,
        "net_width": config.NET_WIDTH,
        "actor_lr": config.ACTOR_LR,
        "critic_lr": config.CRITIC_LR,
        "tau": config.TAU,
        "batch_size": config.BATCH_SIZE,
        "replay_buffer_size": config.BUFFER_SIZE,
    }

    agent = DDPG_Agent(**kwargs)
    agent.actor.load_state_dict(torch.load("ddpg_actorfinal_model.pth"))
    agent.critic.load_state_dict(torch.load("ddpg_criticfinal_model.pth"))

    for episode in range(5):
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
