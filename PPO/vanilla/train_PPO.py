import os
import gym
import torch
import numpy as np
from envrunner import EnvRunner
from model import PolicyNN, ValueNN
from agent import Agent
import matplotlib.pyplot as plt
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPISODES = 10000

# Run an episode using actor
def play(policy_net):
    render_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    flag = False

    with torch.no_grad():
        state, _ = render_env.reset()
        total_reward = 0
        length = 0
        num_steps_taken = 0
        prev_reward = -50000 # Arbitrary large negative number
        has_reward_changed = False

        while True:
            render_env.render()
            state_tensor = torch.tensor(
                np.expand_dims(state, axis=0), dtype=torch.float32, device=device
            )
            action = (
                policy_net.choose_action(state_tensor, deterministic=True)
                .to(device)
                .cpu()
                .numpy()
            )
            state, reward, done, info, _ = render_env.step(action[0])
            total_reward += reward
            # If reward (up to 4 decimal places) has changed, reset flag
            has_reward_changed = (has_reward_changed) or (round(reward, 4) != round(prev_reward, 4))
            prev_reward = reward
            length += 1

            # Check if reward has not changed in the last 100 steps
            num_steps_taken += 1
            if num_steps_taken > 100 and not has_reward_changed: 
                    # Handle the case where the agent makes no progress
                    print(
                        "[Evaluation] Reward has remained same for 100 steps. Stopping evaluation."
                    )
                    break
            elif num_steps_taken > 100:
                num_steps_taken = 0
                has_reward_changed = False

            print(f"[Evaluation] Step taken and got Reward: {reward}", flush=True)

            if done:
                print(
                    "[Evaluation] Total reward = {:.6f}, length = {:d}".format(
                        total_reward, length
                    ),
                    flush=True,
                )
                if total_reward >= 300:
                    flag = True
                break

    render_env.close()
    return flag


plot_data = []


# Train the policy net & value net using the agent
def train(
    env,
    runner: EnvRunner,
    policy_net: PolicyNN,
    value_net: ValueNN,
    agent: Agent,
    max_episode=MAX_EPISODES,
):
    mean_total_reward = 0
    mean_length = 0
    save_dir = "./save"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for ep in range(max_episode):
        # Run and episode to collect data
        with torch.no_grad(): # No need to store gradients
            mb_states, mb_actions, mb_old_a_logps, mb_values, mb_returns, mb_rewards = (
                runner.run(env, policy_net, value_net)
            )
            mb_advs = mb_returns - mb_values
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)

        # Train the model using the collected data
        pg_loss, v_loss, ent = agent.train(
            mb_states, mb_actions, mb_values, mb_advs, mb_returns, mb_old_a_logps
        )
        reward = mb_rewards.sum()
        mean_total_reward += reward
        mean_length += len(mb_states)
        print(
            "[Episode {:4d}] total reward = {:.6f}, length = {:d}".format(
                ep, reward, len(mb_states)
            )
        )

        if ep % 10 == 0:
            plot_data.append([ep, mean_total_reward / 10, mean_length / 10])
            mean_total_reward = 0
            mean_length = 0

        # Show the current result & save the model
        if ep % 200 == 0:
            print("\n[{:5d}]".format(ep))
            print("----------------------------------")
            print("actor loss = {:.6f}".format(pg_loss))
            print("critic loss = {:.6f}".format(v_loss))
            print("entropy = {:.6f}".format(ent))
            print("\nSaving the model ... ", end="")
            torch.save(
                {
                    "it": ep,
                    "PolicyNN": policy_net.state_dict(),
                    "ValueNN": value_net.state_dict(),
                },
                os.path.join(save_dir, f"model_{ep}.pt"),
            )
            print("Done.")
            has_done_well = play(policy_net)
            if has_done_well:
                print("Agent has done well. Stopping training.")
                break

        ep += 1
    # Evaluate the trained model
    has_done_well = play(policy_net)
    if has_done_well:
        print("Agent has done well. Stopping training.")


if __name__ == "__main__":
    # Create the environment
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        "recordings",
        name_prefix="rl-video" + datetime.now().strftime("%Y%m%d%H%M%S"),
    )
    env = gym.wrappers.ClipAction(env)
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.shape[0]
    print("State Space Dimensions: ", state_space_dim)
    print("Action Space Dimensions: ", action_space_dim)

    # Create the policy net & value net
    policy_net = PolicyNN(state_space_dim, action_space_dim)
    value_net = ValueNN(state_space_dim)
    print(policy_net)
    print(value_net)

    # Create the runner
    runner = EnvRunner(state_space_dim, action_space_dim)

    # Create a PPO agent for training
    agent = Agent(policy_net, value_net)

    # Train the network
    train(env, runner, policy_net, value_net, agent)

    # Plot the reward graph
    plt.figure(figsize=(8, 5))
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
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plt.title("Vanilla PPO - BipedalWalker-v3")
    plt.savefig(f"reward_plot_{timestamp}.png")
    plt.show()
    env.close()
