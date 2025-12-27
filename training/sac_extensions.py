import torch
import torch.nn.functional as F
import copy
import numpy as np

from torch.optim import Adam

class SACWithConstraints:
    def __init__(self, cfg, env, device="cpu"):
        self.device = device
        obs_dim = cfg["obs_dim"]
        cond_dim = cfg["cond_dim"]
        action_dim = cfg["action_dim"]
        hidden = tuple(cfg.get("hidden_sizes",[128,128]))
        M = cfg.get("ensemble_size", 2)

        # actor
        from models.actor import Actor
        self.actor = Actor(obs_dim, cond_dim, hidden, action_dim).to(device)
        self.actor_opt = Adam(self.actor.parameters(), lr=cfg.get("actor_lr",1e-4))

        # critics ensemble
        from models.critic import QNetwork
        self.critics = []
        self.critic_opts = []
        self.critic_targets = []
        for k in range(M):
            qc = QNetwork(obs_dim, action_dim, cond_dim, hidden).to(device)
            self.critics.append(qc)
            self.critic_opts.append(Adam(qc.parameters(), lr=cfg.get("critic_lr",3e-4)))
            self.critic_targets.append(copy.deepcopy(qc))

        # constraint critics (one per constraint, ensemble per constraint)
        self.n_constraints = cfg.get("n_constraints", 2)
        self.constraint_critics = []
        self.constraint_opts = []
        self.constraint_targets = []
        for i in range(self.n_constraints):
            lst = []
            outs = []
            tgts = []
            for k in range(M):
                qc = QNetwork(obs_dim, action_dim, cond_dim, hidden).to(device)
                lst.append(qc)
                outs.append(Adam(qc.parameters(), lr=cfg.get("critic_lr",3e-4)))
                tgts.append(copy.deepcopy(qc))
            self.constraint_critics.append(lst)
            self.constraint_opts.append(outs)
            self.constraint_targets.append(tgts)

        self.M = M
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.alpha = cfg.get("entropy_coef", 1e-3)

    def critic_mean(self, obs, action, cond):
        qs = []
        for qc in self.critics:
            qs.append(qc(obs, action, cond))
        return torch.stack(qs, dim=0).mean(0), torch.stack(qs, dim=0).var(0)

    def constraint_mean(self, obs, action, cond, constraint_i):
        qs = []
        for qc in self.constraint_critics[constraint_i]:
            qs.append(qc(obs, action, cond))
        return torch.stack(qs, dim=0).mean(0), torch.stack(qs, dim=0).var(0)

    def soft_update_targets(self):
        for k, qc in enumerate(self.critics):
            for p, tp in zip(qc.parameters(), self.critic_targets[k].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for i in range(self.n_constraints):
            for k, qc in enumerate(self.constraint_critics[i]):
                for p, tp in zip(qc.parameters(), self.constraint_targets[i][k].parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def update(self, batch, w_batch, c_batch, lambda_vec, relabel=False, mode="linear"):
        obs = torch.tensor(batch["s"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["a"], dtype=torch.float32, device=self.device)
        rvecs = torch.tensor(batch["r_vec"], dtype=torch.float32, device=self.device)  # shape B x 3
        costs = torch.tensor(batch["cost"], dtype=torch.float32, device=self.device)    # B x m
        next_obs = torch.tensor(batch["s_next"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["done"].astype(float), dtype=torch.float32, device=self.device)

        cond = torch.tensor(np.concatenate([w_batch, c_batch], axis=1), dtype=torch.float32, device=self.device)

        # scalarize
        if mode == "linear":
            w_t = torch.tensor(w_batch, dtype=torch.float32, device=self.device)
            r_scalar = (rvecs * w_t).sum(dim=1)
        else:
            r_scalar = torch.tensor(batch["r_scalar"], dtype=torch.float32, device=self.device)

        # critic targets
        with torch.no_grad():
            # next actions from actor
            a_next, _, _ = self.actor.sample(next_obs, cond)
            q_next_mean, _ = self.critic_mean(next_obs, a_next, cond)
            y = r_scalar + self.gamma * (1.0 - dones) * q_next_mean

        # update each reward critic
        for k, (qc, opt) in enumerate(zip(self.critics, self.critic_opts)):
            opt.zero_grad()
            q_val = qc(obs, actions, cond)
            loss = F.mse_loss(q_val, y)
            loss.backward()
            opt.step()

        # update constraint critics
        for i in range(self.n_constraints):
            # target for constraint i
            with torch.no_grad():
                a_next, _, _ = self.actor.sample(next_obs, cond)
                q_next_list = []
                for tgt in self.constraint_targets[i]:
                    q_next_list.append(tgt(next_obs, a_next, cond))
                q_next_mean = torch.stack(q_next_list, dim=0).mean(0)
                y_cost = costs[:, i] + self.gamma * (1.0 - dones) * q_next_mean

            # train ensemble for constraint i
            for k, (qc, opt) in enumerate(zip(self.constraint_critics[i], self.constraint_opts[i])):
                opt.zero_grad()
                q_val = qc(obs, actions, cond)
                loss = F.mse_loss(q_val, y_cost)
                loss.backward()
                opt.step()

        # actor update
        # sample actions for current obs
        a_pi, logp, _ = self.actor.sample(obs, cond)
        q_r_mean, _ = self.critic_mean(obs, a_pi, cond)

        # constraints mean
        q_c_means = []
        for i in range(self.n_constraints):
            mean_i, _ = self.constraint_mean(obs, a_pi, cond, i)
            q_c_means.append(mean_i)
        q_c_stack = torch.stack(q_c_means, dim=1)  # shape B x n_constraints

        lambda_t = torch.tensor(lambda_vec, dtype=torch.float32, device=self.device)
        penalty = (q_c_stack * lambda_t).sum(dim=1)

        actor_loss = - (q_r_mean - penalty).mean() + self.alpha * (-logp).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # soft update
        self.soft_update_targets()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": loss.item()
        }
