import time
import numpy as np
import torch
import os
import logging
from training.replay_buffer import ReplayBuffer
from training.sac_extensions import SACWithConstraints
from training.dual import DualController
from curriculum.scheduler import SchedulerCurriculum
from curriculum.preference import PreferenceCurriculum

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
            clip_max=cons_cfg["lambda_max"],
            lambda_init=cons_cfg.get("lambda_init", None)
        )

        self.total_steps = train_cfg.get("total_steps", 200_000)
        self.batch_size = train_cfg.get("batch_size", 256)
        self.updates_per_step = train_cfg.get("updates_per_step", 1)
        
        self.dual_update_freq = cons_cfg.get("dual_update_freq", 500)
        self.eval_freq = eval_cfg.get("eval_freq", 5000)

        # scalarization control: default linear; can switch to 'chebyshev' at step chebyshev_start
        self.scalar_mode = train_cfg.get("scalar_mode", "linear")
        self.chebyshev_start = train_cfg.get("chebyshev_start", 6000)

        self.curriculum = SchedulerCurriculum()
        self.preference = PreferenceCurriculum()

        self.cost_ewma = np.zeros(n_constraints, dtype=float)
        # EWMA factor for smoothing batch cost estimates (configurable)
        self.ewma_alpha = cons_cfg.get("ewma_alpha", 0.1)
        # keep most recent batch costs to use when EWMA is still near-zero
        self.last_batch_costs = None

        # Phase tracking
        self.current_phase = "corners"
        self.global_step = 0

        self.ckpt_dir = "checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, step: int, name: str | None = None):
        ckpt = {
            "step": step,
            "actor": self.agent.actor.state_dict(),
            "reward_critics": [c.state_dict() for c in self.agent.critics],
            "constraint_critics": [
                [c.state_dict() for c in group]
                for group in self.agent.constraint_critics
            ],
            "dual_vars": self.dual.lambdas.copy(),
        }

        fname = name or f"step_{step:06d}.pt"
        path = os.path.join(self.ckpt_dir, fname)
        torch.save(ckpt, path)
        print(f"[Checkpoint] Saved to {path}")

    def set_phase(self, phase: str):
        """Switch to a new curriculum phase."""
        self.current_phase = phase
        self.curriculum.set_phase(phase)
        self.preference.set_phase(phase)
        print(f"\n[PHASE] Switched to '{phase}' curriculum phase")

    def collect_episode(self, w, c, deterministic=False):
        obs, _ = self.env.reset()
        done = False
        steps = 0
        while True:
            cond = np.concatenate([w, c], axis=0).astype(np.float32)
            obs_t = torch.tensor(obs[None,:], dtype=torch.float32, device=self.device)
            cond_t = torch.tensor(cond[None,:], dtype=torch.float32, device=self.device)
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
        # sample scenario + encoding first (needed for both standard and polishing paths)
        c, scenario = self.curriculum.sample(step, phase=self.current_phase)

        if self.current_phase == "polishing":
            # --- Active Uncertainty Polishing ---
            # Set scenario before temporarily resetting env for candidate evaluation
            self.env.scenario = scenario
            obs, _ = self.env.reset()
            obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=self.device)

            # Generate 10 candidate preference vectors from Dirichlet distribution
            num_candidates = 10
            candidates = np.random.dirichlet(np.ones(3), size=num_candidates).astype(np.float32)

            variances = []
            with torch.no_grad():
                for i in range(num_candidates):
                    w_candidate = candidates[i]
                    cond = np.concatenate([w_candidate, c], axis=0).astype(np.float32)
                    cond_t = torch.tensor(cond[None, :], dtype=torch.float32, device=self.device)

                    # Sample candidate action from actor
                    action, _, _ = self.agent.actor.sample(obs_t, cond_t)

                    # Query each critic in the reward ensemble
                    q_values = []
                    for critic in self.agent.critics:
                        q = critic(obs_t, action, cond_t)
                        q_values.append(q.item())

                    # Compute variance of ensemble predictions
                    variances.append(float(np.var(q_values)))

            # Select candidate with maximum ensemble disagreement (highest uncertainty)
            best_idx = int(np.argmax(variances))
            w = candidates[best_idx]

            if step % max(1, int(self.total_steps / 20)) == 0:
                print(f"[CURRICULUM] step={step} phase={self.current_phase} "
                      f"w={w.tolist()} scenario={scenario} "
                      f"max_ensemble_var={variances[best_idx]:.6f}")
        else:
            # Standard curriculum phase sampling
            w = self.preference.sample(step, phase=self.current_phase)

            if step % max(1, int(self.total_steps / 20)) == 0:
                print(f"[CURRICULUM] step={step} phase={self.current_phase} w={w.tolist()} scenario={scenario}")

        # set env scenario (already set above for polishing; idempotent for other phases)
        self.env.scenario = scenario

        return w, c

    def train(self):
        step = 0
        print("total training steps:", self.total_steps)

        # Pre-fill replay buffer above batch_size threshold to prevent sampling deadlocks
        while len(self.replay) < self.batch_size:
            w, c = self.sample_curriculum(0)
            self.collect_episode(w, c)
        print(f"[TRAIN] Replay buffer pre-filled: {len(self.replay)} samples")

        while step < self.total_steps:
            # sample (w,c) using curriculum
            w, c = self.sample_curriculum(step)

            # collect 1 episode
            self.collect_episode(w, c)

            # choose scalarization mode for this step
            current_mode = self.scalar_mode
            if self.chebyshev_start is not None and step >= self.chebyshev_start:
                current_mode = "chebyshev"

            # perform updates
            # initialize dual cost fallback
            cost_for_dual = self.cost_ewma.copy()
            for _ in range(self.updates_per_step):
                if len(self.replay) < self.batch_size:
                    continue
                batch = self.replay.sample(self.batch_size)
                # prepare w_batch and c_batch arrays for batch
                w_batch = np.stack(batch["w"])
                c_batch = np.stack(batch["c"])
                # convert batch values to the format expected by update()
                lambda_vec = self.dual.lambdas.copy()
                info = self.agent.update(batch, w_batch, c_batch, lambda_vec, relabel=True, mode=current_mode)

                # update EWMA costs for dual updates using sampled batch average
                batch_costs = np.stack(batch["cost"]).mean(axis=0)
                self.cost_ewma = (1 - self.ewma_alpha) * self.cost_ewma + self.ewma_alpha * batch_costs
                # store last batch costs for fallback if EWMA is still near-zero
                self.last_batch_costs = batch_costs
                # use last batch costs when ewma hasn't warmed up
                cost_for_dual = np.where(self.cost_ewma < 1e-6, self.last_batch_costs, self.cost_ewma)

            # dual update occasionally
            if step % self.dual_update_freq == 0 and step > 0:
                targets = np.array(self.cfg["constraints"]["cost_thresholds"])
                try:
                    self.logger.info(
                        f"Dual update step={step} targets={targets} ewma={self.cost_ewma} last_batch={self.last_batch_costs} lambdas={self.dual.lambdas}"
                    )
                except Exception:
                    pass
                self.dual.step(cost_for_dual, targets)
                try:
                    self.logger.info(f"Dual updated lambdas={self.dual.lambdas}")
                except Exception:
                    pass

            # periodic evaluation hook / checkpoint
            if step % self.eval_freq == 0:
                print(f"[TRAIN] step={step}, replay={len(self.replay)} scalar_mode={current_mode}")
                self.save_checkpoint(step, name="latest.pt")

            step += 1

        self.save_checkpoint(step, name="final.pt")
        print("Training finished")

    def train_phase(self, phase_steps: int):
        """Train for a single curriculum phase."""
        phase_start = self.global_step
        print(f"\n{'='*60}")
        print(f"Training Phase: {self.current_phase}")
        print(f"Steps: {phase_steps}")
        print(f"{'='*60}\n")

        for local_step in range(phase_steps):
            step = local_step  # For curriculum sampling
            
            # sample (w,c) using curriculum
            w, c = self.sample_curriculum(step)

            # collect 1 episode
            self.collect_episode(w, c)

            # choose scalarization mode for this step
            current_mode = self.scalar_mode
            if self.chebyshev_start is not None and self.global_step >= self.chebyshev_start:
                current_mode = "chebyshev"

            # perform updates
            # initialize dual cost fallback
            cost_for_dual = self.cost_ewma.copy()
            for _ in range(self.updates_per_step):
                if len(self.replay) < self.batch_size:
                    continue
                batch = self.replay.sample(self.batch_size)
                # prepare w_batch and c_batch arrays for batch
                w_batch = np.stack(batch["w"])
                c_batch = np.stack(batch["c"])
                # convert batch values to the format expected by update()
                lambda_vec = self.dual.lambdas.copy()
                info = self.agent.update(batch, w_batch, c_batch, lambda_vec, relabel=True, mode=current_mode)

                # update EWMA costs for dual updates using sampled batch average
                batch_costs = np.stack(batch["cost"]).mean(axis=0)
                self.cost_ewma = (1 - self.ewma_alpha) * self.cost_ewma + self.ewma_alpha * batch_costs
                # store last batch costs for fallback if EWMA is still near-zero
                self.last_batch_costs = batch_costs
                # use last batch costs when ewma hasn't warmed up
                cost_for_dual = np.where(self.cost_ewma < 1e-6, self.last_batch_costs, self.cost_ewma)

            # dual update occasionally
            if self.global_step % self.dual_update_freq == 0 and self.global_step > 0:
                targets = np.array(self.cfg["constraints"]["cost_thresholds"])
                try:
                    self.logger.info(
                        f"Dual update step={self.global_step} targets={targets} ewma={self.cost_ewma} last_batch={self.last_batch_costs} lambdas={self.dual.lambdas}"
                    )
                except Exception:
                    pass
                self.dual.step(cost_for_dual, targets)
                try:
                    self.logger.info(f"Dual updated lambdas={self.dual.lambdas}")
                except Exception:
                    pass

            # periodic evaluation hook / checkpoint
            if self.global_step % self.eval_freq == 0:
                print(f"[TRAIN] global_step={self.global_step}, phase_step={local_step}, replay={len(self.replay)}, "
                      f"scalar_mode={current_mode}, lambdas={self.dual.lambdas}")
                self.save_checkpoint(self.global_step, name="latest.pt")

            self.global_step += 1
        
        print(f"\n[PHASE_COMPLETE] {self.current_phase} finished at step {self.global_step}")
        self.save_checkpoint(self.global_step, name=f"{self.current_phase}_final.pt")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.agent.actor.load_state_dict(ckpt["actor"])

        for c, sd in zip(self.agent.critics, ckpt["reward_critics"]):
            c.load_state_dict(sd)

        for group, sd_group in zip(self.agent.constraint_critics, ckpt["constraint_critics"]):
            for c, sd in zip(group, sd_group):
                c.load_state_dict(sd)

        self.dual.lambdas = ckpt["dual_vars"].copy()

        print(f"[Checkpoint] Loaded from {path}")
