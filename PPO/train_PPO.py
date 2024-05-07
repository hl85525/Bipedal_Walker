import gym
from matplotlib import pyplot as plt
import numpy as np
import config
from agent import Agent
from tester import Test
import itertools
from datetime import datetime

def train(env, plot_data: list):
    state, _ = env.reset()

    agent = Agent(
        state_size=state.shape[0],
        action_size=env.action_space.shape[0],
        batch_size=config.BATCH_SIZE,
    )

    tester = Test(
        state_size=state.shape[0], action_size=env.action_space.shape[0]
    )

    for step in range(config.NUMBER_OF_STEPS):
        should_continue_training = True

        # Learning Rate and Epsilon decay.
        # Another PPO implementation improvement.
        agent.set_optimizer_lr_eps(step)
    
        # Test the model after 50 steps or if the average reward is >= 300
        if (step + 1) % 150 == 0 or (
            len(env.return_queue) >= 100
            and np.mean(list(itertools.islice(env.return_queue, 90, 100))) >= 300
        ):
            print("[Evaluation] Testing the model")
            end_train: bool = tester.test(
                agent.agent_control.policy_nn, env
            )
            if end_train:
                # We have reached the target reward of >= 300
                break

        # Collect batch_size number of samples
        for ep in range(config.BATCH_SIZE):

            # if (step + 1) % 50 == 0:
            #     env.render()
    
            # Feed current state to the policy NN and get action and its probability
            actions, actions_logprob = agent.get_action(state)

            # Use given action and retrieve new state, reward agent recieved 
            # and whether episode is finished flag
            new_state, reward, done, _, _ = env.step(actions)

            # Store step information to memory for future use
            agent.add_to_memory(
                state, actions, actions_logprob, new_state, reward, done, ep
            )
            state = new_state


            if done:
                state, _ = env.reset()

        # For value (critic) function clipping, we need NN output before update
        # which we will use as baseline to see how much new output is different 
        # and to clip it if it's too different
        agent.calculate_old_value_state()
    
        # Calculate advantage for policy NN loss
        agent.calculate_advantage()

        # Instead of shuffling whole memory, we will create indices and shuffle them after each update
        # This is another PPO implementation improvement.
        batch_indices = np.arange(config.BATCH_SIZE)

        # We will use every collected step to update NNs config.UPDATE_STEPS times
        agent.reset_noise()
        for _ in range(config.UPDATE_STEPS):
            np.random.shuffle(batch_indices)

            # Split the memory to mini-batches and use them to update NNs
            for i in range(0, config.BATCH_SIZE, config.MINIBATCH_SIZE):
                should_continue_training = agent.update(batch_indices[i : i + config.MINIBATCH_SIZE])
                if not should_continue_training:
                    # "Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    print(f"Early stopping at step {step} due to reaching max kl")
                    break
            if not should_continue_training:
                break
        
        if not should_continue_training:
            continue

        # Record losses and rewards and print them to console
        agent.record_results(step, env, plot_data)

    tester.env.close()
    env.close()

if __name__ == "__main__":
    env = gym.make(config.ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(
        env,
        "recordings",
        name_prefix="rl-video" + datetime.now().strftime("%Y%m%d%H%M%S"),
    )
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda rew: np.clip(rew, -10, 10))
    np.random.seed(config.SEED)
    plot_data = []
    train(env, plot_data)

    # Save the plot data
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
    plt.title("PPO Agent - BipedalWalker-v3")
    plt.savefig(f"reward_plot_{timestamp}.png")
    plt.show()
