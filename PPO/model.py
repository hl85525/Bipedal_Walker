import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import config

torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)

class PolicyNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(PolicyNN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.Tanh(), # nn.ReLu() is used in the original implementation
            nn.Linear(64, 64),
            nn.Tanh(), # nn.ReLu() is used in the original implementation
            nn.Linear(64, output_shape),
        )
        self.dist = nn.Parameter(torch.zeros(output_shape))

    def forward(self, x, actions=None):
        # Instead of calculating action as output for NN, we calculate action_mean for each action (4,1)
        # We also train input-less parameter which represent log(std)
        actions_mean = self.main(x)
        actions_logstd = self.dist
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actions_std = torch.exp(actions_logstd).to(device)

        # We use mean and std to calculate 4 Normal distributions
        prob = Normal(actions_mean, actions_std)

        if actions is None:
            # To get the actions, we sample the 4 distributions
            actions = prob.sample()
        # To get logarithm of action probabilities we use Normal.log_prob(action) function
        return actions, prob.log_prob(actions), torch.sum(prob.entropy(), dim=-1)


class ValueNN(nn.Module):
    def __init__(self, input_shape):
        super(ValueNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
