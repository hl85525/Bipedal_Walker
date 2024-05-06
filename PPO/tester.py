import gym
import numpy as np
import torch
import config
from model import PolicyNN
import json
from datetime import datetime


class Test:

    def __init__(self, state_size, action_size):
        self.env = gym.make(config.ENV_NAME, render_mode="rgb_array")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = gym.wrappers.ClipAction(self.env)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.policy_nn = PolicyNN(
            input_shape=state_size, output_shape=action_size
        ).to(self.device)

    def test(self, trained_policy_nn, env) -> bool:
        self.policy_nn.load_state_dict(trained_policy_nn.state_dict())
        self.env = gym.make(config.ENV_NAME, render_mode="rgb_array")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = gym.wrappers.ClipAction(self.env)
        state, _ = self.env.reset()
        state = (state - env.obs_rms.mean) / np.sqrt(env.obs_rms.var + env.epsilon)
        state = np.clip(state, -10, 10)
        print("Testing...")
        print("Episodes done [", end="")
        num_steps_taken = 0
        prev_reward = -50000  # Arbitrary large negative number
        has_reward_changed = False
        total_reward = 0
        for n_episode in range(config.NUMBER_OF_EPISODES):
            print(".", end="")
            while True:
                if n_episode % 25 == 0:
                    self.env.render()
                actions, _, _ = self.policy_nn(
                    torch.tensor(state, dtype=torch.float, device=self.device)
                )
                new_state, reward, done, _, _ = self.env.step(
                    actions.cpu().detach().numpy()
                )
                has_reward_changed = (has_reward_changed) or (
                    round(reward, 1) != round(prev_reward, 1)
                )
                prev_reward = reward
                total_reward += reward
                print(f"[Evaluation] Step taken and got Reward: {reward}")
                if num_steps_taken > 100 and not has_reward_changed:
                    print(
                        "[Evaluation] Reward has remained same for 100 steps. Stopping evaluation."
                    )
                    break
                if total_reward < -400:
                    print(
                        "[Evaluation] Total reward = {:.6f}".format(total_reward),
                        flush=True,
                    )
                    break
                state = new_state
                state = (state - env.obs_rms.mean) / np.sqrt(
                    env.obs_rms.var + env.epsilon
                )
                state = np.clip(state, -10, 10)
                if done:
                    state, _ = self.env.reset()
                    break
        print("]")
        print(
            "  Mean 100 test reward: "
            + str(np.round(np.mean(self.env.return_queue), 2))
        )
        print(self.env.return_queue)
        print("Done testing!")

        if np.mean(self.env.return_queue) >= 300:
            print(
                "Goal reached! Mean reward over 100 episodes is "
                + str(np.mean(self.env.return_queue))
            )
            datetime_save = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(
                self.policy_nn.state_dict(), "models/model" + datetime_save + ".p"
            )
            data = {
                "obs_rms_mean": env.obs_rms.mean.tolist(),
                "obs_rms_var": env.obs_rms.var.tolist(),
                "eps": env.epsilon,
            }
            with open(
                "models/data" + datetime_save + ".json", "w", encoding="utf-8"
            ) as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            state, _ = self.env.reset()
            state = (state - env.obs_rms.mean) / np.sqrt(env.obs_rms.var + env.epsilon)
            while True:
                actions, _, _ = self.policy_nn(
                    torch.tensor(state, dtype=torch.float, device=self.device)
                )
                new_state, reward, done, _, _ = self.env.step(
                    actions.cpu().detach().numpy()
                )
                state = new_state
                state = (state - env.obs_rms.mean) / np.sqrt(
                    env.obs_rms.var + env.epsilon
                )
                state = np.clip(state, -10, 10)
                if done:
                    break
            return True
        return False
