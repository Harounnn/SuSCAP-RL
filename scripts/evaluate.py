import numpy as np
import yaml

from env.scheduler_env import SchedulerEnv
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
import evaluation.metrics as metrics
import evaluation.plots as plots

# -----------------------------
# Load config
# -----------------------------
cfg = yaml.safe_load(open("configs/default.yaml"))

# -----------------------------
# Minimal environment
# -----------------------------
env = SchedulerEnv(
    data_path="data/processed/merged_timeseries.csv",
    episode_length=50
)

# -----------------------------
# Load trained agent
# -----------------------------
trainer = Trainer(cfg, env, device="cpu")
trainer.load_checkpoint("checkpoints/final.pt")
agent = trainer.agent

evaluator = Evaluator(env, agent, device="cpu")

# -----------------------------
# Minimal curriculum samples
# -----------------------------
from curriculum.preference import PreferenceCurriculum
from curriculum.scenario import ScenarioCurriculum

pref = PreferenceCurriculum()
scen = ScenarioCurriculum()

# 5 preferences only
W_grid = np.stack([pref.sample(i) for i in range(5)])

# Only first 2 scenarios
C_bank = [scen.sample(i) for i in range(2)]  # (name, encoding)

# ---- Evaluation ----
returns, costs, var_grid = evaluator.evaluate_grid(
    W_grid, C_bank, episodes=1
)

# ---- Metrics ----
# Regret
regret = metrics.grid_regret_matrix(returns, W_grid)

# Hypervolume
points = metrics.to_maximization_space(returns.reshape(-1, 3))
hv = metrics.hypervolume_mc(points, ref=np.zeros(3), n_samples=2000)

# Pareto front
pf = metrics.pareto_front(points)

# Constraint stats
violation_rate, mean_violation = metrics.constraint_stats(costs)

# Coverage
coverage = metrics.coverage_fraction(
    points, grid_shape=(len(W_grid), len(C_bank))
)

print("Hypervolume:", hv)
print("Coverage:", coverage)

# ---- Plots ----
scenario_names = [c[0] for c in C_bank]

plots.plot_regret_heatmap(
    regret, W_grid, scenario_names,
    save_path="figures/regret_heatmap.png"
)

plots.plot_ensemble_variance(
    var_grid, scenario_names,
    save_path="figures/variance_heatmap.png"
)

plots.plot_pareto_3d(
    pf,
    save_path="figures/pareto_3d.png"
)

# Coverage map (boolean non-dominated mask)
nd_mask = np.zeros(points.shape[0], dtype=bool)
for i, p in enumerate(points):
    nd_mask[i] = not metrics.is_dominated(p, np.delete(points, i, axis=0))

plots.plot_coverage_map(
    nd_mask,
    w_labels=list(range(len(W_grid))),
    scenario_names=scenario_names,
    save_path="figures/coverage_map.png"
)
