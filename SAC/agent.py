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
        state_list = []
        for s in state:
            if isinstance(s, dict):
                state_list.extend(s.values())
            else:
                state_list.append(s)
        state_flat = np.array(state_list, dtype=np.float32)

        next_state_list = []
        for s in next_state:
            if isinstance(s, dict):
                next_state_list.extend(s.values())
            else:
                next_state_list.append(s)
        next_state_flat = np.array(next_state_list, dtype=np.float32)

        self.state[self.ptr] = state_flat
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state_flat
        self.dead[self.ptr] = dead

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


class SAC_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(SAC_Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, action_dim)
        )

    def forward(self, state):
        return self.net(state)


class SAC_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(SAC_Q_Critic, self).__init__()
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


class SAC_Agent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            gamma=0.99,
            net_width=128,
            alpha=0.2,
            actor_lr=1e-3,
            critic_lr=1e-3,
            batch_size=256,
            replay_buffer_size=int(1e6)
    ):
        self.actor = SAC_Actor(state_dim, action_dim, net_width).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.q_critic1 = SAC_Q_Critic(state_dim, action_dim, net_width).to(device)
        self.q_critic1_optimizer = torch.optim.Adam(self.q_critic1.parameters(), lr=critic_lr)

        self.q_critic2 = SAC_Q_Critic(state_dim, action_dim, net_width).to(device)
        self.q_critic2_optimizer = torch.optim.Adam(self.q_critic2.parameters(), lr=critic_lr)

        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha)).to(device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=actor_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_buffer_size)
        self.alpha = alpha

    def sample_action(self, state):
        with torch.no_grad():
            state_list = []
            for s in state:
                if isinstance(s, dict):
                    state_list.extend(s.values())
                else:
                    state_list.append(s)
            state = np.array(state_list, dtype=np.float32)
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            a = self.actor(state)
        return a.cpu().numpy().flatten()

    def save(self, episode):
        torch.save(self.actor.state_dict(), "sac_actor{}.pth".format(episode))
        torch.save(self.q_critic1.state_dict(), "sac_q_critic_1{}.pth".format(episode))
        torch.save(self.q_critic2.state_dict(), "sac_q_critic_2{}.pth".format(episode))

    def load(self, episode):
        self.actor.load_state_dict(torch.load(f"sac_actor{episode}.pth"))
        self.q_critic1.load_state_dict(torch.load(f"sac_q_critic_1{episode}.pth"))
        self.q_critic2.load_state_dict(torch.load(f"sac_q_critic_2{episode}.pth"))

    def train(self):
        if self.replay_buffer.size < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        target_Q = reward + (1 - done) * self.gamma * torch.min(
            self.q_critic1(next_state, self.actor(next_state)),
            self.q_critic2(next_state, self.actor(next_state))
        )
        q_loss1 = F.mse_loss(self.q_critic1(state, action), target_Q.detach())
        q_loss2 = F.mse_loss(self.q_critic2(state, action), target_Q.detach())

        self.q_critic1_optimizer.zero_grad()
        q_loss1.backward()
        self.q_critic1_optimizer.step()

        self.q_critic2_optimizer.zero_grad()
        q_loss2.backward()
        self.q_critic2_optimizer.step()

        actor_loss = (self.alpha * self.actor(state) - self.q_critic1(state, self.actor(state))).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (self.actor(state) + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
