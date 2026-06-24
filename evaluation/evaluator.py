import numpy as np
import torch
from typing import List, Tuple, Optional

class Evaluator:
    def __init__(self, env, agent, device="cpu"):
        """
        env: a Gym-like environment (SchedulerEnv)
        agent: object exposing `actor` with `.sample()` and `.forward()` methods and `critic_mean`
        device: torch device string
        """
        self.env = env
        self.agent = agent
        self.device = device

    def deterministic_action(self, obs: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """
        Deterministic action from actor: use mu (pre-squash) and apply tanh->[0,1]
        """
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        cond_t = torch.tensor(cond[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, _ = self.agent.actor.forward(obs_t, cond_t)
            a = torch.tanh(mu).cpu().numpy()[0]
        a = (a + 1.0) / 2.0
        return a

    def stochastic_action(self, obs: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """
        Stochastic action sampled from policy distribution (actor.sample)
        """
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        cond_t = torch.tensor(cond[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a, _, _ = self.agent.actor.sample(obs_t, cond_t)
        a = a.cpu().numpy()[0]
        return a

    def evaluate_cell(self, w: np.ndarray, c: np.ndarray, scenario: str,
                      episodes: int = 3, stochastic: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[float], List[np.ndarray], List[np.ndarray]]:
        """
        Evaluate the agent on one (w,c) cell.

        Returns:
          mean_return_vec: (d,)
          mean_cost_vec: (m,)
          mean_critic_var: scalar
          per_episode_returns: list of (d,) arrays, env-sign (negative costs)
          per_episode_costs: list of (m,) arrays
        """
        returns = []
        costs = []
        critic_vars = []
        per_ep_returns = []
        per_ep_costs = []

        for ep in range(episodes):
            self.env.scenario = scenario
            obs, _ = self.env.reset()
            ep_rewards = []
            ep_costs = []
            ep_vars = []
            while True:
                cond = np.concatenate([w, c]).astype(np.float32)
                if stochastic:
                    action = self.stochastic_action(obs, cond)
                else:
                    action = self.deterministic_action(obs, cond)

                next_obs, reward_vec, terminated, truncated, info = self.env.step(action)

                ep_rewards.append(np.array(reward_vec, dtype=float))
                ep_costs.append(np.array(info.get("cost", [0.0, 0.0]), dtype=float))

                # try to get policy entropy (measure of policy uncertainty/stochasticity)
                try:
                    obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
                    cond_t = torch.tensor(cond[None], dtype=torch.float32, device=self.device)
                    # sample from policy and get log_prob to compute entropy
                    _, log_prob, _ = self.agent.actor.sample(obs_t, cond_t)
                    entropy = -log_prob.mean().cpu().item()
                    ep_vars.append(float(entropy))
                except Exception:
                    pass

                obs = next_obs
                if terminated or truncated:
                    break

            ep_return = np.sum(np.stack(ep_rewards, axis=0), axis=0)
            # costs are recorded per step; evaluate average per-step cost to match
            # thresholds defined as per-step limits (not per-episode sums)
            ep_cost = np.mean(np.stack(ep_costs, axis=0), axis=0)

            per_ep_returns.append(ep_return)
            per_ep_costs.append(ep_cost)

            returns.append(ep_return)
            costs.append(ep_cost)
            if ep_vars:
                critic_vars.append(np.mean(ep_vars))

        mean_return = np.mean(np.stack(returns, axis=0), axis=0)
        mean_cost = np.mean(np.stack(costs, axis=0), axis=0)
        mean_var = float(np.mean(critic_vars)) if len(critic_vars) > 0 else None

        return mean_return, mean_cost, mean_var, per_ep_returns, per_ep_costs

    def evaluate_grid(self, W_grid: np.ndarray, C_bank: List[Tuple[str, np.ndarray]], episodes: int = 3,
                      stochastic: bool = False, show_progress: bool = True):
        """
        Evaluate over grid of preferences W_grid (n_w, 3) and scenarios (list of (scenario_name, c_vector)).
        Returns:
          return_grid: (n_w, n_c, d) mean returns per cell
          cost_grid:   (n_w, n_c, m) mean costs per cell
          var_grid:    (n_w, n_c) mean critic var per cell (0 if not available)
          per_episode_points: (N_points, d) flattened per-episode returns (env-sign)
          per_episode_meta: (N_points, 3) ints (w_idx, c_idx, ep_idx)
        """
        n_w = W_grid.shape[0]
        n_c = len(C_bank)

        # probe to get dims
        scenario_name, cvec = C_bank[0]
        r0, cost0, var0, _, _ = self.evaluate_cell(W_grid[0], cvec, scenario_name, episodes=1, stochastic=stochastic)
        d = len(r0)
        m = len(cost0)

        return_grid = np.zeros((n_w, n_c, d), dtype=float)
        cost_grid = np.zeros((n_w, n_c, m), dtype=float)
        var_grid = np.zeros((n_w, n_c), dtype=float)

        per_episode_points = []
        per_episode_meta = []

        total = n_w * n_c
        rng = range(n_w)
        if show_progress:
            try:
                from tqdm import trange
                outer = trange(n_w, desc="Eval W")
            except Exception:
                outer = rng
        else:
            outer = rng

        for i in outer:
            for j in range(n_c):
                scenario_name, cvec = C_bank[j]
                mean_r, mean_cost, mean_var, per_ep_returns, per_ep_costs = self.evaluate_cell(
                    W_grid[i], cvec, scenario_name, episodes=episodes, stochastic=stochastic
                )
                return_grid[i, j, :] = mean_r
                cost_grid[i, j, :] = mean_cost
                var_grid[i, j] = mean_var if mean_var is not None else 0.0

                # append per-episode points and meta
                for ep_idx, ep_ret in enumerate(per_ep_returns):
                    per_episode_points.append(np.array(ep_ret, dtype=float))
                    per_episode_meta.append((i, j, ep_idx))

        per_episode_points = np.vstack(per_episode_points) if len(per_episode_points) > 0 else np.zeros((0, d))
        per_episode_meta = np.array(per_episode_meta, dtype=int) if len(per_episode_meta) > 0 else np.zeros((0,3), dtype=int)

        return return_grid, cost_grid, var_grid, per_episode_points, per_episode_meta
