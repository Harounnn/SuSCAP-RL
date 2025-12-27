"""
Evaluator: runs deterministic evaluation across W x C grid and returns arrays.

Functions:
- evaluate_grid(agent, env, W_grid, C_bank, episodes_per_cell)
  -> returns: return_grid (n_w, n_c, d), cost_grid (n_w, n_c, m), var_grid (n_w, n_c) [optional]
"""

import numpy as np
import torch
from typing import List, Tuple, Optional

class Evaluator:
    def __init__(self, env, agent, device="cpu"):
        """
        env: a Gym-like environment (SchedulerEnv)
        agent: object exposing `actor` with `.sample()` method returning (action, logp, mu)
        device: torch device string
        """
        self.env = env
        self.agent = agent
        self.device = device

    def deterministic_action(self, obs: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """
        Get deterministic action from actor: use mu (pre-squash) and apply tanh if necessary.
        """
        import torch
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        cond_t = torch.tensor(cond[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.agent.actor.forward(obs_t, cond_t)  # returns mu, std
            mu, _ = out
            a = torch.tanh(mu).cpu().numpy()[0]
        a = (a + 1.0) / 2.0
        return a

    def evaluate_cell(self, w: np.ndarray, c: np.ndarray, scenario: str, episodes: int = 3) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """
        Evaluate the agent on one (w,c) cell.
        Returns:
          mean_return_vec: (d,)
          mean_cost_vec: (m,)
          critic_var: optional scalar average of critic variance across trajectory (if available)
        """
        returns = []
        costs = []
        critic_vars = []

        for ep in range(episodes):
            self.env.scenario = scenario
            obs, _ = self.env.reset()
            done = False
            ep_rewards = []
            ep_costs = []
            ep_vars = []
            while True:
                cond = np.concatenate([w, c]).astype(np.float32)
                action = self.deterministic_action(obs, cond)
                obs, reward_vec, terminated, truncated, info = self.env.step(action)
                ep_rewards.append(reward_vec)
                ep_costs.append(info["cost"])

                try:
                    obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
                    cond_t = torch.tensor(cond[None], dtype=torch.float32, device=self.device)
                    a_t = torch.tensor(action[None], dtype=torch.float32, device=self.device)
                    # SACWithConstraints provides critic_mean that returns (mean, var)
                    q_mean, q_var = self.agent.critic_mean(obs_t, a_t, cond_t)
                    ep_vars.append(float(q_var.mean().cpu().numpy()))
                except Exception:
                    pass

                if terminated or truncated:
                    break

            returns.append(np.sum(ep_rewards, axis=0))
            costs.append(np.sum(ep_costs, axis=0))
            if ep_vars:
                critic_vars.append(np.mean(ep_vars))

        mean_return = np.mean(returns, axis=0)
        mean_cost = np.mean(costs, axis=0)
        mean_var = np.mean(critic_vars) if critic_vars else None
        return mean_return, mean_cost, mean_var

    def evaluate_grid(self, W_grid: np.ndarray, C_bank: List[Tuple[str, np.ndarray]], episodes: int = 3, parallel: bool = False):
        """
        Evaluate over grid of preferences W_grid (n_w, 3) and scenarios (list of (scenario_name, c_vector)).
        Returns:
          return_grid: (n_w, n_c, d)
          cost_grid:   (n_w, n_c, m)
          var_grid:    (n_w, n_c) or None
        """
        n_w = W_grid.shape[0]
        n_c = len(C_bank)
        d = self.env.observation_space.shape[0]  
        test_w = W_grid[0]
        scenario_name, cvec = C_bank[0]
        r0, cost0, var0 = self.evaluate_cell(test_w, cvec, scenario_name, episodes=1)
        d = len(r0)
        m = len(cost0)

        return_grid = np.zeros((n_w, n_c, d), dtype=float)
        cost_grid = np.zeros((n_w, n_c, m), dtype=float)
        var_grid = np.zeros((n_w, n_c), dtype=float)

        for i in range(n_w):
            for j in range(n_c):
                scenario_name, cvec = C_bank[j]
                mean_r, mean_cost, mean_var = self.evaluate_cell(W_grid[i], cvec, scenario_name, episodes=episodes)
                return_grid[i, j, :] = mean_r
                cost_grid[i, j, :] = mean_cost
                var_grid[i, j] = mean_var if mean_var is not None else 0.0

        return return_grid, cost_grid, var_grid
