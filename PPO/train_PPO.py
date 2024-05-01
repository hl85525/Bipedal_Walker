import os
import gym
import torch
import numpy as np
from envrunner import EnvRunner
from model import PolicyNet, ValueNet
from agent import PPO
import matplotlib.pyplot as plt
from datetime import datetime


# Run an episode using the policy net
def play(policy_net):
    render_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

    with torch.no_grad():
        state, _ = render_env.reset()
        total_reward = 0
        length = 0

        while True:
            render_env.render()
            state_tensor = torch.tensor(
                np.expand_dims(state, axis=0), dtype=torch.float32, device="cpu"
            )
            action = (
                policy_net.choose_action(state_tensor, deterministic=True).cpu().numpy()
            )
            state, reward, done, info, _ = render_env.step(action[0])
            total_reward += reward
            length += 1

            if done:
                print(
                    "[Evaluation] Total reward = {:.6f}, length = {:d}".format(
                        total_reward, length
                    ),
                    flush=True,
                )
                break

    render_env.close()


plot_data = []


# Train the policy net & value net using the agent
def train(env, runner, policy_net, value_net, agent, max_episode=5000):
    mean_total_reward = 0
    mean_length = 0
    save_dir = "./save"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(max_episode):
        # Run and episode to collect data
        with torch.no_grad():
            mb_states, mb_actions, mb_old_a_logps, mb_values, mb_returns, mb_rewards = (
                runner.run(env, policy_net, value_net)
            )
            mb_advs = mb_returns - mb_values
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)

        # Train the model using the collected data
        pg_loss, v_loss, ent = agent.train(
            mb_states, mb_actions, mb_values, mb_advs, mb_returns, mb_old_a_logps
        )
        mean_total_reward += mb_rewards.sum()
        mean_length += len(mb_states)
        print(
            "[Episode {:4d}] total reward = {:.6f}, length = {:d}".format(
                i, mb_rewards.sum(), len(mb_states)
            )
        )

        if i % 10 == 0:
            plot_data.append([i, mean_total_reward / 10, mean_length / 10])
            mean_total_reward = 0
            mean_length = 0

        # Show the current result & save the model
        if i % 200 == 0:
            print("\n[{:5d} / {:5d}]".format(i, max_episode))
            print("----------------------------------")
            print("actor loss = {:.6f}".format(pg_loss))
            print("critic loss = {:.6f}".format(v_loss))
            print("entropy = {:.6f}".format(ent))
            print("\nSaving the model ... ", end="")
            torch.save(
                {
                    "it": i,
                    "PolicyNet": policy_net.state_dict(),
                    "ValueNet": value_net.state_dict(),
                },
                os.path.join(save_dir, "model.pt"),
            )
            print("Done.")
            play(policy_net)


if __name__ == "__main__":
    # Create the environment
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.shape[0]
    print("State Space Dimensions: ", state_space_dim)
    print("Action Space Dimensions: ", action_space_dim)

    # Create the policy net & value net
    policy_net = PolicyNet(state_space_dim, action_space_dim)
    value_net = ValueNet(state_space_dim)
    print(policy_net)
    print(value_net)

    # Create the runner
    runner = EnvRunner(state_space_dim, action_space_dim)

    # Create a PPO agent for training
    agent = PPO(policy_net, value_net)

    # Train the network
    train(env, runner, policy_net, value_net, agent)

    # Plot the reward graph
    plt.figure(figsize=(8, 5))
    plt.plot(
        [x[0] for x in plot_data], [x[1] for x in plot_data], "-", color="tab:blue"
    )
    plt.fill_between(
        [x[0] for x in plot_data],
        [x[1] - x[2] for x in plot_data],
        [x[1] + x[2] for x in plot_data],
        alpha=0.2,
        color="tab:blue",
    )
    plt.xlabel("Episode number")
    plt.ylabel("Episode reward")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plt.title("Training Reward Plot")
    plt.savefig(f"reward_plot_{timestamp}.png")
    plt.show()
    env.close()
