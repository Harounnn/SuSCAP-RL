import os
import yaml
import argparse
import numpy as np
import torch

from env.scheduler_env import SchedulerEnv
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
import evaluation.metrics as metrics
import evaluation.plots as plots

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
parser.add_argument("--mode", choices=["fast", "full"], default="fast")
parser.add_argument("--device", default="cpu")
parser.add_argument("--stochastic", action="store_true", help="use stochastic policy sampling during eval")
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))

# mode settings
e = cfg.get("evaluation", {})
if args.mode == "fast":
    episode_length = e.get("eval_episode_length", 20)
    episodes_per_cell = e.get("eval_episodes", 1)
    n_pref = e.get("eval_n_preferences", 7)
    use_all_scenarios = False
    hv_samples = e.get("hv_samples", 1000)
else:
    episode_length = e.get("eval_episode_length", 720)
    episodes_per_cell = e.get("eval_episodes", 28)
    n_pref = e.get("eval_n_preferences", 15)
    use_all_scenarios = True
    hv_samples = e.get("hv_samples", 20000)

device = args.device

# env + trainer + agent
env = SchedulerEnv(data_path="data/processed/merged_timeseries.csv", episode_length=episode_length)
trainer = Trainer(cfg, env, device=device)
if not os.path.exists(args.checkpoint):
    raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
trainer.load_checkpoint(args.checkpoint)
print("[EVAL] Loaded checkpoint:", args.checkpoint)
agent = trainer.agent
agent.actor.eval()

# build preference simplex grid (barycentric)
def build_simplex_grid(K):
    grid = []
    for i in range(K+1):
        for j in range(K+1 - i):
            k = K - i - j
            w = np.array([i, j, k], dtype=float) / float(K)
            if w.sum() > 0:
                grid.append(w)
    return np.array(grid, dtype=np.float32)

def choose_K_for_n(n_pref):
    K = 1
    while (K + 1) * (K + 2) // 2 < n_pref:
        K += 1
    return K

K = choose_K_for_n(n_pref)
W_grid = build_simplex_grid(K)
if len(W_grid) > n_pref:
    idx = np.linspace(0, len(W_grid)-1, n_pref).astype(int)
    W_grid = W_grid[idx]

# scenarios
from curriculum.scenario import ScenarioCurriculum
scen = ScenarioCurriculum()
all_scenarios = scen.scenarios
if use_all_scenarios:
    chosen_scenarios = all_scenarios
else:
    chosen_scenarios = all_scenarios[:min(2, len(all_scenarios))]
C_bank = [(s, scen.encode(s)) for s in chosen_scenarios]

print(f"[EVAL] preferences={len(W_grid)}, scenarios={len(C_bank)}, episodes={episodes_per_cell}, episode_length={episode_length}, stochastic={args.stochastic}")

# evaluator
evaluator = Evaluator(env, agent, device=device)
return_grid, cost_grid, var_grid, per_episode_points, per_episode_meta = evaluator.evaluate_grid(
    W_grid, C_bank, episodes=episodes_per_cell, stochastic=args.stochastic, show_progress=True
)

# convert per-episode returns to maximization space
points = metrics.to_maximization_space(per_episode_points)  # shape (N, d)

# normalization: scale each objective by 95th percentile to avoid domination by magnitude
if points.shape[0] > 0:
    scales = np.maximum(np.percentile(points, 95, axis=0), 1e-9)
    points_norm = points / scales
else:
    points_norm = points

# Pareto on per-episode points (normalized)
pf = metrics.pareto_front(points_norm) if points_norm.shape[0] > 0 else np.zeros((0, points.shape[1]))

# hypervolume: choose safe reference = min(points_norm) - eps
if points_norm.shape[0] > 0:
    ref = points_norm.min(axis=0) - 1e-6
    hv = metrics.hypervolume_mc(points_norm, ref=ref, n_samples=hv_samples)
else:
    hv = 0.0

print("[EVAL] Hypervolume (MC approx, normalized):", hv)

# coverage fraction: whether each (w,c) cell had any non-dominated episode
# map nondominated per-episode back to grid cells
N = points_norm.shape[0]
nd_mask_flat = np.zeros(N, dtype=bool)
for i in range(N):
    nd_mask_flat[i] = not metrics.is_dominated(points_norm[i], np.delete(points_norm, i, axis=0)) if N > 1 else True

n_w = W_grid.shape[0]
n_c = len(C_bank)
grid_mask = np.zeros((n_w, n_c), dtype=bool)
for idx, (iw, ic, ep) in enumerate(per_episode_meta):
    if nd_mask_flat[idx]:
        grid_mask[iw, ic] = True

coverage_frac = grid_mask.mean()
print("[EVAL] coverage fraction:", coverage_frac)

# aggregated return_grid is already in env sign (negative costs)
# compute regret heatmap using return_grid and W_grid
regret = metrics.grid_regret_matrix(return_grid, W_grid)

# ========== CONSTRAINT SATISFACTION ANALYSIS ==========
print("\n" + "="*70)
print("CONSTRAINT ANALYSIS")
print("="*70)

cost_thresholds = np.array(cfg["constraints"]["cost_thresholds"])
constraint_names = ["Energy (kWh)", "Latency (sec)"]

# Per-constraint violation rates
for c_idx, (c_name, threshold) in enumerate(zip(constraint_names, cost_thresholds)):
    violations = (cost_grid[:, :, c_idx] > threshold).astype(float)
    violation_rate = violations.mean()
    mean_cost = cost_grid[:, :, c_idx].mean()
    max_cost = cost_grid[:, :, c_idx].max()
    min_cost = cost_grid[:, :, c_idx].min()
    
    print(f"\n{c_name}:")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Violation Rate: {100*violation_rate:.1f}%")
    print(f"  Mean Cost: {mean_cost:.3f} (ratio to threshold: {mean_cost/threshold:.3f})")
    print(f"  Min Cost: {min_cost:.3f}")
    print(f"  Max Cost: {max_cost:.3f}")

# Overall constraint satisfaction (both constraints satisfied)
overall_satisfaction = ((cost_grid[:, :, 0] <= cost_thresholds[0]) & 
                        (cost_grid[:, :, 1] <= cost_thresholds[1])).astype(float).mean()
print(f"\nOverall Constraint Satisfaction (both): {100*overall_satisfaction:.1f}%")

# Per-scenario constraint stats
print("\nPer-Scenario Analysis:")
scenario_names = [c[0] for c in C_bank]
for ic, scenario_name in enumerate(scenario_names):
    scenario_costs = cost_grid[:, ic, :]
    scenario_satisfaction = ((scenario_costs[:, 0] <= cost_thresholds[0]) & 
                             (scenario_costs[:, 1] <= cost_thresholds[1])).astype(float).mean()
    print(f"  {scenario_name}: {100*scenario_satisfaction:.1f}% satisfaction")

# Constraint violation stats (original metrics.constraint_stats if available)
try:
    violation_rate, mean_violation = metrics.constraint_stats(cost_grid)
    print(f"\n[Legacy Metric] constraint violation rate: {violation_rate:.3f}, mean violation: {mean_violation:.3f}")
except Exception:
    pass

# Summary report
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)
print(f"Hypervolume: {hv:.4f}")
print(f"Coverage Fraction: {coverage_frac:.3f}")
print(f"Constraint Satisfaction: {100*overall_satisfaction:.1f}%")
print(f"Pareto Front Size: {len(pf)}")
print(f"Total Episodes Evaluated: {N}")

# Save raw artifacts
os.makedirs("figures", exist_ok=True)
np.save("figures/per_episode_points.npy", points_norm)
np.save("figures/return_grid.npy", return_grid)
np.save("figures/cost_grid.npy", cost_grid)
np.save("figures/var_grid.npy", var_grid)
np.save("figures/w_grid.npy", W_grid)

# Plots
plots.plot_regret_heatmap(regret, W_grid, scenario_names, save_path="figures/regret_heatmap.png")
plots.plot_ensemble_variance(var_grid, scenario_names, save_path="figures/var_heatmap.png")
if pf.size > 0:
    # pf is in normalized space; project back approx to original scale for plotting (optional)
    plots.plot_pareto_3d(pf, save_path="figures/pareto_3d.png")
plots.plot_coverage_map(grid_mask.flatten(), list(range(n_w)), scenario_names, save_path="figures/coverage_map.png")

print("\n[EVAL] Done. Figures saved in ./figures")

