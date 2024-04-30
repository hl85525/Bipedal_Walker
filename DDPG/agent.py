import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dead = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, dead):
        if isinstance(state, tuple):
            state = state[0]  # assuming the first element is always the array you need
        if isinstance(next_state, tuple):
            next_state = next_state[0]  # assuming the first element is always the array you need

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead  # 0,0,0，...，1

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.dead[ind]).to(self.device)
        )


class DDPG_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(DDPG_Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, action_dim),
            nn.Tanh()  # Ensure actions are bounded within [-1, 1]
        )

    def forward(self, state):
        return self.net(state)


class DDPG_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(DDPG_Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)


class DDPG_Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            gamma=0.99,
            net_width=128,
            actor_lr=1e-3,
            critic_lr=1e-3,
            tau=0.005,
            batch_size=256,
            replay_buffer_size=int(1e6)
    ):
        self.actor = DDPG_Actor(state_dim, action_dim, net_width).to(device)
        self.actor_target = DDPG_Actor(state_dim, action_dim, net_width).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = DDPG_Critic(state_dim, action_dim, net_width).to(device)
        self.critic_target = DDPG_Critic(state_dim, action_dim, net_width).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_buffer_size)

    def sample_action(self, state, exploration_noise=0.1):
        if isinstance(state, tuple):
            state = state[0]

        with torch.no_grad():
            state = np.array(state).reshape(1, -1)
            state = torch.FloatTensor(state).to(device)
            action = self.actor(state)
            noise = torch.randn_like(action) * exploration_noise
            action = (action + noise).clamp(-1, 1)  # Clip the action to be within the valid range
        return action.cpu().numpy().flatten()

    def save(self, episode):
        torch.save(self.actor.state_dict(), f"ddpg_actor{episode}.pth")
        torch.save(self.critic.state_dict(), f"ddpg_critic{episode}.pth")

    def load(self, episode):
        self.actor.load_state_dict(torch.load(f"ddpg_actor{episode}.pth"))
        self.critic.load_state_dict(torch.load(f"ddpg_critic{episode}.pth"))

    def train(self):
        if self.replay_buffer.size < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Compute target Q-value
        target_Q = reward + (1 - done) * self.gamma * self.critic_target(next_state, self.actor_target(next_state))

        # Update critic
        critic_loss = F.mse_loss(self.critic(state, action), target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)