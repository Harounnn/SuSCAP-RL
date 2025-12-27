import torch
import torch.nn as nn
import torch.nn.functional as F
from .film import FiLM

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, cond_dim, hidden_sizes=(128,128)):
        super().__init__()
        self.s_net = nn.Linear(obs_dim, hidden_sizes[0])
        self.ac_net = nn.Linear(action_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.film = FiLM(hidden_sizes[1], cond_dim)
        self.out = nn.Linear(hidden_sizes[1], 1)

    def forward(self, obs, action, cond):
        h = F.relu(self.s_net(obs) + self.ac_net(action))
        h = F.relu(self.fc2(h))
        h = self.film(h, cond)
        return self.out(h).squeeze(-1)
