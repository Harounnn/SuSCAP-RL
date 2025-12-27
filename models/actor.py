import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLM

class Actor(nn.Module):
    def __init__(self, obs_dim, cond_dim, hidden_sizes=(128,128), action_dim=1):
        super().__init__()
        # shared encoder
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        self.film = FiLM(hidden_sizes[-1], cond_dim)

        self.mu = nn.Linear(hidden_sizes[-1], action_dim)
        self.logstd = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs, cond):
        h = self.net(obs)
        h = self.film(h, cond)
        mu = self.mu(h)
        logstd = self.logstd(h).clamp(-20, 2)
        std = torch.exp(logstd)
        return mu, std

    def sample(self, obs, cond):
        mu, std = self.forward(obs, cond)
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)  
        logp = dist.log_prob(x).sum(-1)
        return a, logp, mu
