from typing import Tuple
import config
from memory import Memory
import numpy as np
from model import PolicyNN, ValueNN
import torch


class AgentControl:
    """
    Encapsulates lower level operations of the agent such as
    - getting action from policy network
    - getting value from critic network
    - calculating log probabilities
    - calculating ratio
    - updating policy
    - updating critic
    """

    def __init__(self, state_size, action_size):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.policy_nn = PolicyNN(input_shape=state_size, output_shape=action_size).to(
            self.device
        )
        self.critic_nn = ValueNN(input_shape=state_size).to(self.device)
        self.optimizer_policy = torch.optim.Adam(
            params=self.policy_nn.parameters(),
            lr=config.LEARNING_RATE_POLICY,
            eps=config.EPSILON_START,
        )
        self.optimizer_critic = torch.optim.Adam(
            params=self.critic_nn.parameters(),
            lr=config.LEARNING_RATE_CRITIC,
            eps=config.EPSILON_START,
        )
        self.loss_critic = torch.nn.MSELoss()
        self.mu = np.zeros(action_size)
        self.prev_noise = np.zeros_like(self.mu)

    def set_optimizer_lr_eps(self, n_step):
        """
        Anneals the learning rate and epsilon of the optimizer.

        Annealing the learning rate is another PPO implementation improvement.
        """
        step_ratio = n_step / config.NUMBER_OF_STEPS
        frac = 1.0 - step_ratio
        lr_policy = frac * config.LEARNING_RATE_POLICY
        lr_critic = frac * config.LEARNING_RATE_CRITIC

        self.optimizer_policy.param_groups[0]["lr"] = lr_policy
        self.optimizer_critic.param_groups[0]["lr"] = lr_critic
        self.optimizer_critic.param_groups[0]["eps"] = (
            config.EPSILON_START
            - step_ratio * (config.EPSILON_START - config.EPSILON_END)
        )

    def get_action(self, state):
        tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        actions, actions_logprob, _ = self.policy_nn(tensor)
        return actions, actions_logprob

    def get_critic_value(self, state):
        return self.critic_nn(state)

    def reset_noise(self):
        self.prev_noise = np.zeros_like(self.mu)

    def noise(self, actions):
        if config.SHOULD_ADD_NOISE:
            noise = (
                self.prev_noise
                + config.THETA * (self.mu - self.prev_noise) * config.DT
                + config.SIGMA
                * np.sqrt(config.DT)
                * np.random.normal(size=self.mu.shape)
            )
            actions += noise
        return actions

    def calculate_logprob(self, states, actions):
        _, new_actions_logprob, entropy = self.policy_nn(states, actions)
        return new_actions_logprob, entropy

    def calculate_ratio(self, new_action_logprob, action_logprobs):
        """
        To calculate the ratio we can subtract since both
        numerator and denominator and logarithmic.

        Because we have 4 actions and not 1, we calculate joint probability which will
        represent single action in R^4. Since actions are independent, the joint probability
        will be a product of the probabilites. Product of probabilities is equal to sum of log probabilities.

        Finally we take the exponent of the log probability to get the probability.
        """
        return torch.exp(
            torch.sum(new_action_logprob, dim=1)
            - torch.sum(action_logprobs, dim=1).detach()
        )

    def update_policy(
        self, advantages, ratios, entropy: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        Given a batch of states, advantages and ratios of action probabilities (r(theta) in PPO),
        we update the policy network by finding the policy loss
        then calculating the gradients and doing gradient descent.

        Returns policy loss and whether to early stop training.
        """
        should_continue = True
        advantages_norm = (advantages - advantages.mean()) / (
            advantages.std() + config.POLICY_EPSILON
        )
        # Core PPO Improvement: Clipped Surrogate Objective
        policy_loss = torch.minimum(
            ratios * advantages_norm,
            torch.clamp(
                ratios, 1 - config.CLIPPING_EPSILON, 1 + config.CLIPPING_EPSILON
            )
            * advantages_norm,
        ).mean()
        policy_loss = (entropy.mean() * config.ENTROPY_COEF) - policy_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html

        # Another PPO implementation improvement.
        with torch.no_grad():
            approx_kl_div = torch.mean((torch.exp(ratios) - 1) - ratios).cpu().numpy()

        if config.TARGET_KL is not None and approx_kl_div > 1.5 * config.TARGET_KL:
            should_continue = False

        self.optimizer_policy.zero_grad()
        policy_loss.backward()

        # To prevent gradient explosion we clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy_nn.parameters(), config.MAX_GRAD_NORM
        )
        self.optimizer_policy.step()
        return policy_loss, should_continue

    def update_critic(self, gt, states, old_value_state):
        # We need to calculate regular loss where we just find squared difference between estimated value and return
        estimated_value = self.critic_nn(states).squeeze(-1)
        critic_loss1 = torch.square(estimated_value - gt)
        # and we need to calculate clipped loss where estimated value is replaced with old estimated value + clipped difference
        estimated_value_clipped = old_value_state + torch.clamp(
            self.critic_nn(states).squeeze(-1) - old_value_state,
            -config.CLIPPING_EPSILON,
            config.CLIPPING_EPSILON,
        )
        critic_loss2 = torch.square(estimated_value_clipped - gt)
        # Compare two losses and take bigger and calculate mean to get final critic loss
        critic_loss = 0.5 * (torch.maximum(critic_loss1, critic_loss2)).mean()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        # The gradients of the value (critic) network are clipped so that the “global l2 norm” (i.e. the norm of the
        # concatenated gradients of all parameters) does not exceed 0.5
        torch.nn.utils.clip_grad_norm_(
            self.critic_nn.parameters(), config.MAX_GRAD_NORM
        )
        self.optimizer_critic.step()
        return critic_loss


class Agent:
    # Role of Agent class is to coordinate between AgentControl where we do all calculations
    # and memory where we store all of the data
    def __init__(self, state_size, action_size, batch_size):
        self.agent_control = AgentControl(
            state_size=state_size, action_size=action_size
        )
        self.memory = Memory(state_size, action_size, batch_size)
        self.policy_loss_m = []
        self.critic_loss_m = []
        self.policy_loss_mm = [0] * 100
        self.critic_loss_mm = [0] * 100
        self.max_reward = -300
        self.ep_count = 0

    def set_optimizer_lr_eps(self, n_step):
        """
        Anneals the learning rate and epsilon of the optimizer.
        """
        self.agent_control.set_optimizer_lr_eps(n_step)

    def get_action(self, state):
        """
        Get action from policy network.
        """
        actions, actions_logprob = self.agent_control.get_action(state)
        action_return_value = self.agent_control.noise(
            actions=actions.cpu().detach().numpy()
        )
        return action_return_value, actions_logprob

    def add_to_memory(
        self, state, action, actions_logprob, new_state, reward, done, n_batch_step
    ):
        """
        Add step information to memory.
        """
        self.memory.add(
            state, action, actions_logprob, new_state, reward, done, n_batch_step
        )

    def reset_noise(self):
        self.agent_control.reset_noise()

    def calculate_old_value_state(self):
        """
        Get NN output from collected states and pass it to the memory
        """
        self.memory.set_old_value_state(
            self.agent_control.get_critic_value(self.memory.states).squeeze(-1).detach()
        )

    def calculate_advantage(self):
        """
        Generalized Advantage Estimator (GAE) lets us decide if we want each state advantage to be calculated with
        reward + estimate(next state) - estimate(state) which has low variance but high bias or
        with reward + gamma * next_reward + ... + gamma^n * estimate(last next state) - estimate(state)
        which has high variance but low bias.

        We can decide to calculate advantage with something between those two where lambda
        is a hyperparameter.
        """
        values = (
            self.agent_control.get_critic_value(self.memory.states).squeeze(-1).detach()
        )

        # We use reward + gamma * next_reward + ... + gamma^n * estimate(last next state) - estimate(state)
        # GAE is one of the PPO implementation improvements
        next_values = (
            self.agent_control.get_critic_value(self.memory.new_states)
            .squeeze(-1)
            .detach()
        )

        self.memory.calculate_gae_advantage(values, next_values)

    def update(self, indices):
        """
        We update policy neural network by calculating derivative of loss function and doing gradient descent.
        We follow the following steps:
        1. Calculate ratio of new and old action probabilities.
        2. Find minimum between ratio * advantage and clipped_ratio * advantage.
        3. Find mean of minibatch losses.

        Returns whether to continue training or not.
        """
        # Step 1: Calculate ratio of new and old action probabilities
        new_action_logprob, entropy = self.agent_control.calculate_logprob(
            self.memory.states[indices], self.memory.actions[indices]
        )
        ratios = self.agent_control.calculate_ratio(
            new_action_logprob, self.memory.action_logprobs[indices]
        )
        policy_loss, should_continue_training = self.agent_control.update_policy(
            self.memory.advantages[indices], ratios, entropy
        )
        if not should_continue_training:
            return False
        # Similar to ratio in policy loss, we also clipped values from critic. For that we need old_value_state which
        # represent old estimate of states before updates.
        critic_loss = self.agent_control.update_critic(
            self.memory.gt[indices],
            self.memory.states[indices],
            self.memory.old_value_state[indices],
        )

        # Calculating mean losses for statistics
        self.policy_loss_m.append(policy_loss.detach().item())
        self.critic_loss_m.append(critic_loss.detach().item())
        return True

    def record_results(self, n_step, env, plot_data):
        self.max_reward = np.maximum(self.max_reward, np.max(env.return_queue))
        self.policy_loss_mm[n_step % 100] = np.mean(self.policy_loss_m)
        self.critic_loss_mm[n_step % 100] = np.mean(self.critic_loss_m)

        print(
            "[Episode "
            + str(n_step)
            + "/"
            + str(config.NUMBER_OF_STEPS)
            + "] "
            + " Mean 100 policy loss: "
            + str(np.round(np.mean(self.policy_loss_mm[: min(n_step + 1, 100)]), 4))
            + " Mean 100 critic loss: "
            + str(np.round(np.mean(self.critic_loss_mm[: min(n_step + 1, 100)]), 4))
            + " Max reward: "
            + str(np.round(self.max_reward, 2))
            + " Mean 100 reward: "
            + str(np.round(np.mean(env.return_queue), 2))
            + " Steps "
            + str(env.episode_count)
        )
        self.critic_loss_m = []
        self.policy_loss_m = []
        self.ep_count = env.episode_count
        # Store reward plot data
        plot_data.append([n_step, np.round(np.mean(env.return_queue), 2) + 15])
