import numpy as np

class DualController:
    def __init__(self, n_constraints, lr=1e-4, clip_max=1e3, lambda_init=None):
        # Initialize with provided values (warm-start) or zeros
        if lambda_init is not None:
            self.lambdas = np.array(lambda_init, dtype=float)
        else:
            self.lambdas = np.zeros(n_constraints, dtype=float)
        self.lr = lr
        self.clip_max = clip_max

    def step(self, est_costs, targets):
        # Gradient ascent on lagrangian: max lambda s.t. cost(lambda) <= target
        # So: lambda_new = lambda + lr * (cost - target)
        self.lambdas += self.lr * (est_costs - targets)
        # Clip: keep minimum at 0.001 to maintain signal even when constraints satisfied
        # This prevents lambdas from dying when constraint is satisfied
        self.lambdas = np.clip(self.lambdas, 0.001, self.clip_max)
        return self.lambdas.copy()
