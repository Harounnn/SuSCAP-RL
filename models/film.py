import torch
import torch.nn as nn

class FiLM(nn.Module):
    """
    Simple FiLM layer: given conditioning vector z -> compute gamma, beta.
    Applies elementwise: output = gamma * x + beta
    """
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, in_dim)
        self.beta = nn.Linear(cond_dim, in_dim)

    def forward(self, x, cond):
        g = self.gamma(cond)
        b = self.beta(cond)
        return g * x + b
