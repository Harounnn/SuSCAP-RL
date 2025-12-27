import numpy as np
import yaml
from env.scheduler_env import SchedulerEnv
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
import evaluation.metrics as metrics
import evaluation.plots as plots

cfg = yaml.safe_load(open("configs/default.yaml"))
env = SchedulerEnv(data_path="data/processed/merged_timeseries.csv", episode_length=50)
trainer = Trainer(cfg, env)
trainer.train()
agent = trainer.agent  

from curriculum.preference import PreferenceCurriculum
from curriculum.scenario import ScenarioCurriculum
pref = PreferenceCurriculum()
scen = ScenarioCurriculum()
W_grid = np.stack([pref.sample(i) for i in range(20)])  # sample 20 preferences
C_bank = [scen.sample(i) for i in range(len(scen.scenarios))]  # returns (name, encoding)

evaluator = Evaluator(env, agent, device="cpu")
returns, costs, var_grid = evaluator.evaluate_grid(W_grid, C_bank, episodes=3)

# metrics
regret = metrics.grid_regret_matrix(returns, W_grid)
hv_approx = metrics.hypervolume_mc(metrics.to_maximization_space(returns.reshape(-1,3)), ref=np.array([0.0,0.0,0.0]), n_samples=20000)

# plots
plots.plot_regret_heatmap(regret, W_grid, [c[0] for c in C_bank], save_path="figures/regret_heatmap.png")
plots.plot_ensemble_variance(var_grid, [c[0] for c in C_bank], save_path="figures/var_heatmap.png")
