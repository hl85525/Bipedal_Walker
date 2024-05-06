import gym
import numpy as np
import torch
import config
from model import PolicyNN
import json

if __name__ == "__main__":
    PATH_DATA = "models/best.json"
    PATH_MODEL = "models/best.p"

    with open(PATH_DATA, "r") as f:
        json_load = json.loads(f.read())
    obs_rms_mean = np.asarray(json_load["obs_rms_mean"])
    obs_rms_var = np.asarray(json_load["obs_rms_var"])
    epsilon = json_load["eps"]

    env = gym.make(config.ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.RecordVideo(
        env,
        "bestRecordings",
        name_prefix="rl-video" + PATH_MODEL[12:22],
    )
    state, _ = env.reset()

    state = (state - obs_rms_mean) / np.sqrt(obs_rms_var + epsilon)
    state = np.clip(state, -10, 10)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    policy_nn = PolicyNN(
        input_shape=state.shape[0], output_shape=env.action_space.shape[0]
    ).to(device)
    policy_nn.load_state_dict(torch.load(PATH_MODEL, map_location=device))
    for n_episode in range(config.NUMBER_OF_EPISODES):
        print(".", end=",")
        env.start_video_recorder()
        while True:
            actions, _, _ = policy_nn(torch.tensor(state, dtype=torch.float, device=device))
            new_state, reward, done, _, _ = env.step(actions.cpu().detach().numpy())
            state = new_state
            state = (state - obs_rms_mean) / np.sqrt(obs_rms_var + epsilon)
            state = np.clip(state, -10, 10)
            if done:
                state, _ = env.reset()
                print("Done: " + str(n_episode))
                break
        env.close_video_recorder()
    print("  Mean 100 test reward: " + str(np.round(np.mean(env.return_queue), 2)))
    env.close()
