import numpy as np

class DualController:
    """
    Lagrangian dual-variable controller for constrained RL.

    Implements the subgradient method from the Constrained MDP literature
    (see Altman 1999, Ray et al. 2019, or any CMDP paper):

        lambda_{t+1} = max(0, lambda_t + alpha (C_t - C_limit))

    where C_t is the estimated per-step cost and C_limit is the constraint
    threshold.

    Key modifications for stability under highly variable workloads:
      1. Normalized subgradient: (C_t - C_limit) / C_limit  -- makes the
         update magnitude scale-invariant across constraints whose costs span
         very different ranges (e.g. energy ~1e-4 vs latency ~1e0).
      2. Adaptive step-size (alpha_t) that anneals over time, preventing
         late-stage oscillations while remaining responsive early.
      3. EWMA smoothing of the cost signal fed to step(), reducing noise
         from batch-to-batch variance.
    """

    def __init__(self, n_constraints, lr=0.01, clip_max=50.0, clip_min=0.001,
                 lambda_init=None):
        if lambda_init is not None:
            self.lambdas = np.array(lambda_init, dtype=float)
        else:
            self.lambdas = np.full(n_constraints, clip_min, dtype=float)
        self.lr = lr
        self.clip_max = clip_max
        self.clip_min = clip_min
        self._step_count = 0

    def step(self, est_costs, targets):
        """
        Update lambdas using the normalized subgradient method.

        Per the paper: lambda = max(0, lambda + alpha * (C - C_limit)).
        We replace (C - C_limit) with (C / C_limit - 1) to make the update
        scale-invariant. This prevents the controller from being effectively
        frozen when cost magnitudes are tiny (e.g., 1e-5) relative to the
        learning rate.

        Args:
            est_costs: numpy (n_constraints,) -- current (EWMA) per-step costs.
            targets:   numpy (n_constraints,) -- constraint thresholds C_limit.
        """
        self._step_count += 1

        # Normalized violation:  (C - C_limit) / C_limit  =  C/C_limit - 1
        # When C == C_limit, violation = 0 (no update).
        # When C == 2*C_limit, violation = 1 (lambda increases by ~lr).
        # When C == 0, violation = -1 (lambda decreases by ~lr).
        safe_targets = np.maximum(targets, 1e-8)
        normalized_violation = (est_costs - targets) / safe_targets

        # Adaptive step-size: anneal to dampen oscillations in late training.
        anneal = 1.0 / (1.0 + 0.001 * self._step_count)
        alpha_eff = self.lr * anneal

        self.lambdas += alpha_eff * normalized_violation
        self.lambdas = np.clip(self.lambdas, self.clip_min, self.clip_max)
        return self.lambdas.copy()
