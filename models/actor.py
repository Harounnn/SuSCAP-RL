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

        # Initialize mu to near-zero to prevent early tanh saturation.
        # When mu is large, tanh(mu) = ±1 and the tanh derivative is ~0,
        # permanently starving the actor gradient ("gradient death").
        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.constant_(self.mu.bias, 0.0)
        # Initialize logstd for conservative initial exploration (std ≈ 0.37).
        nn.init.uniform_(self.logstd.weight, -3e-3, 3e-3)
        nn.init.constant_(self.logstd.bias, -1.0)

    def forward(self, obs, cond):
        device = next(self.parameters()).device
        obs = obs.to(device) if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32, device=device)
        cond = cond.to(device) if isinstance(cond, torch.Tensor) else torch.tensor(cond, dtype=torch.float32, device=device)
        h = self.net(obs)
        h = self.film(h, cond)
        mu = self.mu(h)
        logstd = self.logstd(h).clamp(-20, 2)
        std = torch.exp(logstd)
        return mu, std

    def sample(self, obs, cond):
        device = next(self.parameters()).device
        obs = obs.to(device) if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32, device=device)
        cond = cond.to(device) if isinstance(cond, torch.Tensor) else torch.tensor(cond, dtype=torch.float32, device=device)
        mu, std = self.forward(obs, cond)
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)
        # Tanh squashing correction (SAC appendix C):
        # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u_i))
        # This ensures the log-prob correctly accounts for the tanh bijection.
        # Without this correction, the entropy bonus pushes the policy toward
        # tanh saturation boundaries, causing gradient death at a = ±1.
        logp = dist.log_prob(x).sum(-1)
        logp -= torch.log(1 - torch.tanh(x)**2 + 1e-6).sum(dim=-1)
        return a, logp, mu
