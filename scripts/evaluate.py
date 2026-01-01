# scripts/evaluate.py
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

# ---- CLI ----
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
parser.add_argument("--mode", choices=["fast", "full"], default="fast",
                    help="fast: quick smoke-test; full: denser eval for figures")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))

# ---- build env (small episode for fast mode) ----
if args.mode == "fast":
    episode_length = 10
    episodes_per_cell = 1
    n_pref = 5
    n_scen = 2
    hv_samples = 1000
else:
    # full (paper) defaults — override in config if needed
    episode_length = cfg.get("eval_episode_length", 720)
    episodes_per_cell = cfg.get("eval_episodes", 7)
    n_pref = cfg.get("eval_n_preferences", 9)
    n_scen = None  # all scenarios
    hv_samples = cfg.get("hv_samples", 20000)

env = SchedulerEnv(data_path="data/processed/merged_timeseries.csv",
                   episode_length=episode_length)

# ---- load agent ----
trainer = Trainer(cfg, env, device=args.device)
if os.path.exists(args.checkpoint):
    # Use weights_only=False to allow numpy arrays in checkpoint (dual vars)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    trainer.load_checkpoint(args.checkpoint)
    print("[EVAL] Loaded checkpoint:", args.checkpoint)
else:
    print("[EVAL] Checkpoint not found, using untrained agent.")

agent = trainer.agent

# ---- build grids ----
from curriculum.preference import PreferenceCurriculum
from curriculum.scenario import ScenarioCurriculum

pref = PreferenceCurriculum()
scen = ScenarioCurriculum()

# sample preferences
W_grid = np.stack([pref.sample(i) for i in range(n_pref)])

# scenario bank (limit if fast)
all_scenarios = scen.scenarios
if args.mode == "fast":
    chosen_scen = all_scenarios[:n_scen]
else:
    chosen_scen = all_scenarios
C_bank = [(s, scen.encode(s)) for s in chosen_scen]

print(f"[EVAL] grid sizes: preferences={len(W_grid)}, scenarios={len(C_bank)}, episodes={episodes_per_cell}")

# ---- evaluate ----
evaluator = Evaluator(env, agent, device=args.device)
return_grid, cost_grid, var_grid = evaluator.evaluate_grid(W_grid, C_bank, episodes=episodes_per_cell)

# ---- metrics ----
# convert to maximization-space (higher better)
flat_points = metrics.to_maximization_space(return_grid.reshape(-1, return_grid.shape[-1]))

# compute pareto front (maximization)
pf = metrics.pareto_front(flat_points)

# choose reference point for HV: must be strictly WORSE than all points (smaller in each dimension)
# safe default: ref = (min(points) - margin)
margin = 1e-6
ref = flat_points.min(axis=0) - margin

hv = metrics.hypervolume_mc(flat_points, ref=ref, n_samples=hv_samples)
print("[EVAL] Hypervolume (MC approx):", hv)

# compute regret matrix
regret = metrics.grid_regret_matrix(return_grid, W_grid)

# constraint stats
violation_rate, mean_violation = metrics.constraint_stats(cost_grid)
print("[EVAL] constraint violation rate:", violation_rate, "mean violation:", mean_violation)

# coverage
coverage = metrics.coverage_fraction(flat_points, grid_shape=(len(W_grid), len(C_bank)))
print("[EVAL] coverage fraction:", coverage)

# save metrics
os.makedirs("figures", exist_ok=True)
np.save("figures/return_grid.npy", return_grid)
np.save("figures/cost_grid.npy", cost_grid)
np.save("figures/var_grid.npy", var_grid)
np.save("figures/w_grid.npy", W_grid)

# ---- plots (use all functions from plots.py) ----
scenario_names = [c[0] for c in C_bank]

# Regret heatmap
plots.plot_regret_heatmap(regret, W_grid, scenario_names, save_path="figures/regret_heatmap.png")

# Ensemble variance
plots.plot_ensemble_variance(var_grid, scenario_names, save_path="figures/var_heatmap.png")

# Pareto 3D (plot pareto front)
if pf.size > 0:
    plots.plot_pareto_3d(pf, save_path="figures/pareto_3d.png")
else:
    print("[EVAL] no pareto points to plot")

# Coverage map: boolean non-dominated mask per (w,c)
# compute nondominated mask across flattened points
nd_mask = np.zeros(flat_points.shape[0], dtype=bool)
for i, p in enumerate(flat_points):
    nd_mask[i] = not metrics.is_dominated(p, np.delete(flat_points, i, axis=0))
plots.plot_coverage_map(nd_mask, w_labels=list(range(len(W_grid))), scenario_names=scenario_names, save_path="figures/coverage_map.png")

# (Optional) if you have training logs: plot learning curves & constraints evolution
# try to load logs if available
logs_path = "logs/training_metrics.npz"
if os.path.exists(logs_path):
    logs = dict(np.load(logs_path))
    if "hv_list" in logs:
        plots.plot_learning_curves({"hypervolume": logs["hv_list"]}, save_path="figures/hv_curve.png")
    if "violation_rates" in logs:
        plots.plot_constraint_evolution(logs["violation_rates"], logs["avg_costs"], labels=["energy", "latency"], save_path="figures/constraints_evolution.png")

print("[EVAL] Done. Figures saved in ./figures")
