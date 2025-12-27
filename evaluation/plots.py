"""
Plotting utilities aligned with the figures described in the paper draft.
Produces:
 - learning / ablation curves
 - constraint violation rate and average cost plots
 - per-(w,c) regret heatmaps
 - Pareto 3D scatter (energy, CO2, latency)
 - ensemble variance heatmaps
 - coverage heatmaps
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os

sns.set(style="whitegrid")

def plot_learning_curves(logs: dict, save_path: str = None):
    """
    logs: dict of label -> array-like (metric over training steps)
    e.g., {'full': hv_list, 'no_curriculum': hv2, ...}
    """
    plt.figure(figsize=(8,5))
    for k, v in logs.items():
        plt.plot(v, label=k)
    plt.xlabel("Training step (x eval interval)")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return plt.gcf()

def plot_constraint_evolution(violation_rates: np.ndarray, avg_costs: np.ndarray, labels: list, save_path: str = None):
    """
    violation_rates: (T, m) array
    avg_costs: (T, m) array
    labels: list of constraint names (m,)
    """
    T, m = violation_rates.shape
    fig, ax = plt.subplots(2, 1, figsize=(8,8), sharex=True)
    for i in range(m):
        ax[0].plot(violation_rates[:, i], label=labels[i])
        ax[1].plot(avg_costs[:, i], label=labels[i])
    ax[0].set_title("Constraint violation rate over time")
    ax[1].set_title("Average constraint cost over time")
    ax[1].set_xlabel("Evaluation step")
    ax[0].legend(); ax[1].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def plot_regret_heatmap(regret_matrix: np.ndarray, w_grid: np.ndarray, scenario_names: list, save_path: str = None):
    """
    regret_matrix: (n_w, n_c)
    w_grid: (n_w, 3)
    scenario_names: list length n_c
    We'll plot one heatmap per scenario (columns), x-axis = preference index
    """
    n_w, n_c = regret_matrix.shape
    fig, axes = plt.subplots(1, n_c, figsize=(3*n_c, 4), squeeze=False)
    for j in range(n_c):
        sns.heatmap(regret_matrix[:, j].reshape(-1,1), ax=axes[0,j], cmap="viridis", cbar=True)
        axes[0,j].set_title(f"Regret: {scenario_names[j]}")
        axes[0,j].set_ylabel("Preference index")
        axes[0,j].set_xlabel("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def plot_pareto_3d(points: np.ndarray, save_path: str = None):
    """
    points: (N, 3) array of objectives in maximization space (higher better)
    We'll scatter plot the Pareto front points.
    """
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], s=20)
    ax.set_xlabel("Energy (maximization)")
    ax.set_ylabel("CO2 (maximization)")
    ax.set_zlabel("Latency (maximization)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def plot_ensemble_variance(var_grid: np.ndarray, scenario_names: list, save_path: str = None):
    """
    var_grid: (n_w, n_c)
    """
    n_w, n_c = var_grid.shape
    fig, axes = plt.subplots(1, n_c, figsize=(3*n_c, 4), squeeze=False)
    for j in range(n_c):
        sns.heatmap(var_grid[:, j].reshape(-1,1), ax=axes[0,j], cmap="magma", cbar=True)
        axes[0,j].set_title(scenario_names[j])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def plot_coverage_map(points: np.ndarray, w_labels: list, scenario_names: list, save_path: str = None):
    """
    Given flattened evaluated points with indicators whether they are non-dominated,
    produce a coverage visualization.
    points: (n_w * n_c, ) boolean mask 1 if non-dominated
    w_labels: labels for preferences (length n_w)
    scenario_names: list of scenario names (length n_c)
    """
    mask = np.array(points).reshape((len(w_labels), len(scenario_names)))
    plt.figure(figsize=(3*len(scenario_names), 4))
    sns.heatmap(mask.astype(float), cmap="Greens", cbar=True)
    plt.xlabel("Scenario")
    plt.ylabel("Preference index")
    plt.xticks(np.arange(len(scenario_names))+0.5, scenario_names, rotation=45)
    plt.yticks(np.arange(len(w_labels))+0.5, np.arange(len(w_labels)))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return plt.gcf()
