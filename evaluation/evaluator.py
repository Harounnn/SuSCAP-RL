import os
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sns.set(style="whitegrid")

from env.scheduler_env import SchedulerEnv
from training.trainer import Trainer
from curriculum.scenario import ScenarioCurriculum
import evaluation.metrics as metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_CHECKPOINT = "checkpoints/polishing_final.pt"
DEFAULT_SCENARIO = "normal"
DEFAULT_EPISODE_LENGTH = 1440

PREFERENCE_PROFILES = {
    "Extreme Green": np.array([0.7, 0.3, 0.0], dtype=np.float32),
    "Extreme Performance": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "Balanced Compromise": np.array([0.34, 0.33, 0.33], dtype=np.float32),
}

PROFILE_COLORS = {
    "Extreme Green": "#2ecc71",
    "Extreme Performance": "#e74c3c",
    "Balanced Compromise": "#3498db",
}

PROFILE_LINESTYLES = {
    "Extreme Green": "-",
    "Extreme Performance": "--",
    "Balanced Compromise": ":",
}

# ---------------------------------------------------------------------------
# Evaluator class  (grid-based evaluation used by scripts/evaluate.py)
# ---------------------------------------------------------------------------
class Evaluator:
    """Evaluates a trained policy over (preference, scenario) grid cells."""

    def __init__(self, env, agent, device="cpu"):
        self.env = env
        self.agent = agent
        self.device = device

    def deterministic_action(self, obs: np.ndarray, cond: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        cond_t = torch.tensor(cond[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, _ = self.agent.actor.forward(obs_t, cond_t)
            a = torch.tanh(mu).cpu().numpy()[0]
        a = (a + 1.0) / 2.0
        return a

    def stochastic_action(self, obs: np.ndarray, cond: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        cond_t = torch.tensor(cond[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a, _, _ = self.agent.actor.sample(obs_t, cond_t)
        a = a.cpu().numpy()[0]
        return a

    def evaluate_cell(
        self, w: np.ndarray, c: np.ndarray, scenario: str,
        episodes: int = 3, stochastic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[float], List[np.ndarray], List[np.ndarray]]:
        returns, costs, critic_vars = [], [], []
        per_ep_returns, per_ep_costs = [], []

        for ep in range(episodes):
            self.env.scenario = scenario
            obs, _ = self.env.reset()
            ep_rewards, ep_costs, ep_vars = [], [], []
            while True:
                cond = np.concatenate([w, c]).astype(np.float32)
                action = self.stochastic_action(obs, cond) if stochastic else self.deterministic_action(obs, cond)
                next_obs, reward_vec, terminated, truncated, info = self.env.step(action)

                ep_rewards.append(np.array(reward_vec, dtype=float))
                ep_costs.append(np.array(info.get("cost", [0.0, 0.0]), dtype=float))

                try:
                    obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
                    cond_t = torch.tensor(cond[None], dtype=torch.float32, device=self.device)
                    _, log_prob, _ = self.agent.actor.sample(obs_t, cond_t)
                    ep_vars.append(float(-log_prob.mean().cpu().item()))
                except Exception:
                    pass

                obs = next_obs
                if terminated or truncated:
                    break

            ep_return = np.sum(np.stack(ep_rewards, axis=0), axis=0)
            ep_cost = np.mean(np.stack(ep_costs, axis=0), axis=0)
            per_ep_returns.append(ep_return)
            per_ep_costs.append(ep_cost)
            returns.append(ep_return)
            costs.append(ep_cost)
            if ep_vars:
                critic_vars.append(np.mean(ep_vars))

        mean_return = np.mean(np.stack(returns, axis=0), axis=0)
        mean_cost = np.mean(np.stack(costs, axis=0), axis=0)
        mean_var = float(np.mean(critic_vars)) if critic_vars else None
        return mean_return, mean_cost, mean_var, per_ep_returns, per_ep_costs

    def evaluate_grid(
        self, W_grid: np.ndarray, C_bank: List[Tuple[str, np.ndarray]],
        episodes: int = 3, stochastic: bool = False, show_progress: bool = True
    ):
        n_w, n_c = W_grid.shape[0], len(C_bank)
        scenario_name, cvec = C_bank[0]
        r0, cost0, var0, _, _ = self.evaluate_cell(W_grid[0], cvec, scenario_name, episodes=1, stochastic=stochastic)
        d, m = len(r0), len(cost0)

        return_grid = np.zeros((n_w, n_c, d), dtype=float)
        cost_grid = np.zeros((n_w, n_c, m), dtype=float)
        var_grid = np.zeros((n_w, n_c), dtype=float)
        per_episode_points, per_episode_meta = [], []

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
                for ep_idx, ep_ret in enumerate(per_ep_returns):
                    per_episode_points.append(np.array(ep_ret, dtype=float))
                    per_episode_meta.append((i, j, ep_idx))

        per_episode_points = np.vstack(per_episode_points) if per_episode_points else np.zeros((0, d))
        per_episode_meta = np.array(per_episode_meta, dtype=int) if per_episode_meta else np.zeros((0, 3), dtype=int)
        return return_grid, cost_grid, var_grid, per_episode_points, per_episode_meta


# ---------------------------------------------------------------------------
# Single-profile deterministic evaluation
# ---------------------------------------------------------------------------
def evaluate_profile(
    env, trainer, w: np.ndarray, scenario: str,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    scen_curriculum: Optional[ScenarioCurriculum] = None,
    debug_actions: bool = True,
) -> dict:
    """Run one deterministic 1440-step evaluation episode."""
    env.scenario = scenario
    obs, _ = env.reset()
    c_vec = scen_curriculum.encode(scenario)

    total_energy = 0.0
    total_co2 = 0.0
    latencies_ms = []
    violations = 0
    actions_rescaled = []
    raw_actions_log = []

    for step_idx in range(episode_length):
        cond = np.concatenate([w, c_vec], axis=0).astype(np.float32)
        obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=trainer.device)
        cond_t = torch.tensor(cond[None, :], dtype=torch.float32, device=trainer.device)

        with torch.no_grad():
            mu, _ = trainer.agent.actor(obs_t, cond_t)
            raw_action = torch.tanh(mu).cpu().numpy()[0, 0]

        # Rescale from [-1, 1] to [0, 1] before passing to env
        action_env = float(np.clip((raw_action + 1.0) / 2.0, 0.0, 1.0))
        actions_rescaled.append(action_env)

        if debug_actions and step_idx < 5:
            raw_actions_log.append((step_idx, raw_action, action_env))

        next_obs, reward_vec, terminated, truncated, info = env.step(action_env)

        energy = -reward_vec[0]
        co2 = -reward_vec[1]
        latency_s = -reward_vec[2]

        total_energy += energy
        total_co2 += co2
        latencies_ms.append(latency_s * 1000.0)

        cost = info["cost"]
        if cost[0] > 0.0 or cost[1] > 0.0:
            violations += 1

        obs = next_obs
        if terminated or truncated:
            break

    if debug_actions and raw_actions_log:
        print("[ACTION_DEBUG] First 5 raw vs rescaled actions:")
        for sidx, raw, resc in raw_actions_log:
            print(f"  step={sidx}:  raw={raw:+.6f}  rescaled={resc:.6f}")

    return {
        "profile": None,
        "w0": float(w[0]), "w1": float(w[1]), "w2": float(w[2]),
        "total_energy_kwh": total_energy,
        "total_co2_kg": total_co2,
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "max_latency_ms": float(np.max(latencies_ms)) if latencies_ms else 0.0,
        "violation_count": violations,
        "steps_completed": len(latencies_ms),
        "actions": actions_rescaled,
    }


# ---------------------------------------------------------------------------
# Hypervolume indicator  (w.r.t. a nadir reference point)
# ---------------------------------------------------------------------------
def compute_hypervolume(results: list) -> float:
    """Empirical hypervolume over the three objectives (negate for maximisation)."""
    objs = []
    for r in results:
        objs.append([
            r["total_energy_kwh"],
            r["total_co2_kg"],
            r["mean_latency_ms"] / 1000.0,   # back to seconds for consistency
        ])
    objs = -np.array(objs, dtype=float)      # negate → maximisation space
    ref = objs.min(axis=0) - 1e-6
    hv = metrics.hypervolume_mc(objs, ref=ref, n_samples=50000)
    return float(hv)


# ---------------------------------------------------------------------------
# Plot: multi-panel pairwise Pareto trade-offs
# ---------------------------------------------------------------------------
def plot_pareto_tradeoffs(results: list, save_path: str):
    pairs = [
        ("total_energy_kwh", "mean_latency_ms", "Total Energy (kWh)", "Mean Latency (ms)"),
        ("total_energy_kwh", "total_co2_kg", "Total Energy (kWh)", "Total CO\u2082 (kg)"),
        ("total_co2_kg", "mean_latency_ms", "Total CO\u2082 (kg)", "Mean Latency (ms)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (xk, yk, xl, yl) in zip(axes, pairs):
        for r in results:
            ax.scatter(r[xk], r[yk], c=PROFILE_COLORS.get(r["profile"], "#333"),
                       s=120, label=r["profile"], edgecolors="black", linewidth=0.5, zorder=5)
            ax.annotate(r["profile"], (r[xk], r[yk]),
                        textcoords="offset points", xytext=(6, 5), fontsize=7)
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, frameon=True)
    fig.suptitle("Pareto Trade-offs Across Preference Profiles", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Pareto trade-offs saved to {save_path}")


# ---------------------------------------------------------------------------
# Plot: parallel coordinates
# ---------------------------------------------------------------------------
def plot_parallel_coordinates(results: list, save_path: str):
    df = pd.DataFrame([
        {
            "Profile": r["profile"],
            "Energy (kWh)": round(r["total_energy_kwh"], 4),
            "CO\u2082 (kg)": round(r["total_co2_kg"], 4),
            "Latency (ms)": round(r["mean_latency_ms"], 1),
            "Violations": r["violation_count"],
        }
        for r in results
    ])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pd.plotting.parallel_coordinates(
        df, "Profile", color=[PROFILE_COLORS[p] for p in df["Profile"]],
        ax=ax, linewidth=2, alpha=0.85,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, fontsize=9)
    ax.set_ylabel("Metric value", fontsize=10)
    ax.set_title("Parallel Coordinates: Multi-Objective Profile Comparison", fontsize=11)
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Parallel coordinates saved to {save_path}")


# ---------------------------------------------------------------------------
# Plot: 24-hour diurnal response (action + carbon intensity)
# ---------------------------------------------------------------------------
def plot_diurnal_response(
    env, trainer, scen_curriculum, profile_name: str = "Balanced Compromise",
    save_path: str = None,
):
    w = PREFERENCE_PROFILES[profile_name]
    c_vec = scen_curriculum.encode(DEFAULT_SCENARIO)

    env.scenario = DEFAULT_SCENARIO
    obs, _ = env.reset()
    start_idx = env._start_idx
    carbon_profile = env.carbon_intensities[start_idx:start_idx + DEFAULT_EPISODE_LENGTH]

    actions = []
    for step_idx in range(DEFAULT_EPISODE_LENGTH):
        cond = np.concatenate([w, c_vec], axis=0).astype(np.float32)
        obs_t = torch.tensor(obs[None, :], dtype=torch.float32, device=trainer.device)
        cond_t = torch.tensor(cond[None, :], dtype=torch.float32, device=trainer.device)
        with torch.no_grad():
            mu, _ = trainer.agent.actor(obs_t, cond_t)
            raw = torch.tanh(mu).cpu().numpy()[0, 0]
        action = float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))
        actions.append(action)
        next_obs, _, term, trunc, _ = env.step(action)
        obs = next_obs
        if term or trunc:
            carbon_profile = carbon_profile[:len(actions)]
            break

    minutes = np.arange(len(actions))
    hours = minutes / 60.0

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    color_action = PROFILE_COLORS.get(profile_name, "#3498db")
    ax1.plot(hours, actions, color=color_action, linewidth=1.5, label="CPU allocation action")
    ax1.set_xlabel("Hour of day", fontsize=11)
    ax1.set_ylabel("CPU allocation action", fontsize=11, color=color_action)
    ax1.tick_params(axis="y", labelcolor=color_action)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(hours, carbon_profile, color="#8e44ad", linewidth=1.2, alpha=0.7, label="Carbon intensity")
    ax2.set_ylabel("Carbon intensity (gCO\u2082/kWh)", fontsize=11, color="#8e44ad")
    ax2.tick_params(axis="y", labelcolor="#8e44ad")

    lines = [
        Line2D([0], [0], color=color_action, linewidth=1.5),
        Line2D([0], [0], color="#8e44ad", linewidth=1.2, alpha=0.7),
    ]
    labels = [f"Action ({profile_name})", "Carbon intensity"]
    ax1.legend(lines, labels, loc="upper left", frameon=True, fontsize=9)

    fig.suptitle("Diurnal Response: Scheduling Actions vs Carbon Intensity", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Diurnal response saved to {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="SuSCAP-RL Preference-sweep evaluation")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    import yaml
    cfg = yaml.safe_load(open(args.config))
    cfg["device"] = args.device
    env = SchedulerEnv(data_path="data/processed/merged_timeseries.csv",
                       episode_length=args.episode_length)
    trainer = Trainer(cfg, env, device=args.device)
    trainer.load_checkpoint(args.checkpoint)
    trainer.agent.actor.eval()
    print(f"[EVAL] Loaded checkpoint: {args.checkpoint}")

    scen_curriculum = ScenarioCurriculum()
    results = []

    for pname, w in PREFERENCE_PROFILES.items():
        print(f"\n[EVAL] Evaluating '{pname}'  w=[{w[0]:.2f} {w[1]:.2f} {w[2]:.2f}]")
        r = evaluate_profile(
            env, trainer, w, args.scenario,
            episode_length=args.episode_length,
            scen_curriculum=scen_curriculum,
            debug_actions=(pname == "Extreme Green"),
        )
        r["profile"] = pname
        results.append(r)
        print(f"       Energy={r['total_energy_kwh']:.3f} kWh  "
              f"CO2={r['total_co2_kg']:.3f} kg  "
              f"Mean Latency={r['mean_latency_ms']:.1f}ms  "
              f"Violations={r['violation_count']}/{r['steps_completed']}")

    # -- Hypervolume --
    hv = compute_hypervolume(results)
    print(f"\n[HYPERVOLUME] Empirical HV (3-objective): {hv:.6f}")

    # -- Save CSV --
    os.makedirs("evaluation", exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "profile": r["profile"],
            "w0": r["w0"], "w1": r["w1"], "w2": r["w2"],
            "total_energy_kwh": round(r["total_energy_kwh"], 4),
            "total_co2_kg": round(r["total_co2_kg"], 4),
            "mean_latency_ms": round(r["mean_latency_ms"], 2),
            "max_latency_ms": round(r["max_latency_ms"], 2),
            "violation_count": r["violation_count"],
            "steps_completed": r["steps_completed"],
        })
    df = pd.DataFrame(rows)
    csv_path = "evaluation/eval_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV] Summary saved to {csv_path}")
    print(df.to_string(index=False))

    # -- Generate plots --
    os.makedirs("figures", exist_ok=True)
    plot_pareto_tradeoffs(results, "figures/pareto_tradeoffs.png")
    plot_parallel_coordinates(results, "figures/parallel_coordinates.png")
    plot_diurnal_response(env, trainer, scen_curriculum,
                          profile_name="Balanced Compromise",
                          save_path="figures/diurnal_response.png")

    print("\n[EVAL] All evaluation tasks complete.")


if __name__ == "__main__":
    main()
