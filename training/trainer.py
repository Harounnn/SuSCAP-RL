import time
import numpy as np
import torch
import os
from training.replay_buffer import ReplayBuffer
from training.sac_extensions import SACWithConstraints
from training.dual import DualController
from curriculum.scheduler import SchedulerCurriculum

class Trainer:
    def __init__(self, cfg, env, device="cpu"):
        self.cfg = cfg
        self.env = env
        # Use device from YAML if not explicitly passed
        self.device = cfg.get("device", device)

        # Sub-configs for cleaner access
        env_cfg = cfg["env"]
        cond_cfg = cfg["conditioning"]
        train_cfg = cfg["training"]
        eval_cfg = cfg["evaluation"]
        model_cfg = cfg["model"]  
        cons_cfg = cfg["constraints"] 

        obs_dim = env_cfg["obs_dim"]
        action_dim = env_cfg["action_dim"]
        n_constraints = env_cfg["n_constraints"]
        cond_dim = cond_cfg["cond_dim"]

        sac_cfg = {
            "obs_dim": obs_dim,
            "cond_dim": cond_dim,
            "action_dim": action_dim,
            "hidden_sizes": model_cfg.get("hidden_sizes", [128, 128]),
            "ensemble_size": model_cfg.get("ensemble_size", 2),
            "gamma": train_cfg.get("gamma", 0.99), 
            "tau": train_cfg.get("tau", 0.005),
            "critic_lr": model_cfg.get("critic_lr", 3e-4),
            "actor_lr": model_cfg.get("actor_lr", 1e-4),
            "entropy_coef": model_cfg.get("entropy_coef", 1e-3),
            "n_constraints": n_constraints
        }

        self.agent = SACWithConstraints(sac_cfg, env, device=self.device)
        self.replay = ReplayBuffer(capacity=cfg.get("replay_size", 200_000))
        
        self.dual = DualController(
            n_constraints, 
            lr=cons_cfg["dual_lr"], 
            clip_max=cons_cfg["lambda_max"]
        )

        self.total_steps = train_cfg.get("total_steps", 200_000)
        self.batch_size = train_cfg.get("batch_size", 256)
        self.updates_per_step = train_cfg.get("updates_per_step", 1)
        
        self.dual_update_freq = cons_cfg.get("dual_update_freq", 500)
        self.eval_freq = eval_cfg.get("eval_freq", 5000)

        self.curriculum = SchedulerCurriculum()

        self.cost_ewma = np.zeros(n_constraints, dtype=float)
        self.ewma_alpha = 0.01

    def collect_episode(self, w, c, deterministic=False):
        obs, _ = self.env.reset()
        done = False
        steps = 0
        while True:
            # build cond vector from w and c (here we assume c is numeric vector)
            cond = np.concatenate([w, c], axis=0).astype(np.float32)
            # action: sample from actor
            obs_t = torch.tensor(obs[None,:], dtype=torch.float32)
            cond_t = torch.tensor(cond[None,:], dtype=torch.float32)
            with torch.no_grad():
                a, _, _ = self.agent.actor.sample(obs_t, cond_t)
            action = a.cpu().numpy()[0]
            next_obs, reward_vec, terminated, truncated, info = self.env.step(action)
            cost = info["cost"]

            transition = {
                "s": obs.astype(np.float32),
                "a": action.astype(np.float32),
                "r_vec": reward_vec.astype(np.float32),
                "cost": cost.astype(np.float32),
                "s_next": next_obs.astype(np.float32),
                "w": w.astype(np.float32),
                "c": c.astype(np.float32),
                "done": float(terminated or truncated)
            }
            self.replay.push(transition)
            obs = next_obs
            steps += 1
            if terminated or truncated:
                break
        return steps

    def sample_curriculum(self, step):
        w, c, scenario = self.curriculum.sample(step)
        self.env.scenario = scenario
        return w, c

    def train(self):
        step = 0
        print("total training steps:", self.total_steps)
        while step < self.total_steps:
            # sample (w,c) using curriculum
            w, c = self.sample_curriculum(step)

            # collect 1 episode
            self.collect_episode(w, c)

            # perform updates
            for _ in range(self.updates_per_step):
                if len(self.replay) < self.batch_size:
                    continue
                batch = self.replay.sample(self.batch_size)
                # prepare w_batch and c_batch arrays for batch
                w_batch = np.stack(batch["w"])
                c_batch = np.stack(batch["c"])
                # convert batch values to the format expected by update()
                # here we pass batch arrays and vectors
                lambda_vec = self.dual.lambdas.copy()
                info = self.agent.update(batch, w_batch, c_batch, lambda_vec, relabel=True, mode="linear")

                # update EWMA costs for dual updates using sampled batch average
                batch_costs = np.stack(batch["cost"]).mean(axis=0)
                self.cost_ewma = (1 - self.ewma_alpha) * self.cost_ewma + self.ewma_alpha * batch_costs

            # dual update occasionally
            if step % self.dual_update_freq == 0 and step > 0:
                self.dual.step(self.cost_ewma, targets=np.array([self.cfg.get("cost_threshold",1.0)]*self.dual.lambdas.size))

            # periodic evaluation hook placeholder
            if step % self.eval_freq == 0:
                print(f"[TRAIN] step={step}, replay={len(self.replay)}")

            step += 1

        print("Training finished")
