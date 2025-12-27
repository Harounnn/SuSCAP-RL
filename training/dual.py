import numpy as np

class DualController:
    def __init__(self, n_constraints, lr=1e-4, clip_max=1e3):
        self.lambdas = np.zeros(n_constraints, dtype=float)
        self.lr = lr
        self.clip_max = clip_max

    def step(self, est_costs, targets):
        self.lambdas += self.lr * (est_costs - targets)
        self.lambdas = np.clip(self.lambdas, 0.0, self.clip_max)
        return self.lambdas.copy()
