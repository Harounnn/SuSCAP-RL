from typing import Sequence, Tuple, Optional
import numpy as np

def to_maximization_space(reward_vectors: np.ndarray) -> np.ndarray:
    """
    Convert environment reward vectors to maximization objectives.
    env reward vectors are expected to be negative costs (lower is better).
    We convert by negation: obj = -reward.
    Input:
      reward_vectors: (N, d) array
    Returns:
      objs: (N, d) array where higher is better
    """
    return -np.array(reward_vectors)

def is_dominated(p: np.ndarray, qs: np.ndarray) -> bool:
    """
    Check if point p is dominated by any point in qs (qs shape (M,d)).
    Maximization assumed (higher is better).
    """
    if qs.size == 0:
        return False
    ge = (qs >= p).all(axis=1)
    gt = (qs > p).any(axis=1)
    dominated = np.any(ge & gt)
    return bool(dominated)

def pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Return non-dominated subset of points (maximization).
    Preserves order approximately (not sorted).
    """
    pts = np.array(points)
    if pts.size == 0:
        return pts.reshape((0,0))
    nondom = []
    for i, p in enumerate(pts):
        others = np.delete(pts, i, axis=0)
        if not is_dominated(p, others):
            nondom.append(p)
    return np.array(nondom)

def hypervolume_mc(points: np.ndarray, ref: np.ndarray, n_samples: int = 100000, seed: Optional[int]=0) -> float:
    """
    Monte-Carlo approximate hypervolume for maximization objectives.
    points: (N, d) array of objective vectors (higher better).
    ref: reference point (d,) with values strictly less than best objectives.
    Returns approximated hypervolume (scalar).
    NOTE: MC is approximate and scales with n_samples. Use larger n_samples for precision.
    """
    rng = np.random.default_rng(seed)
    points = np.array(points)
    if points.size == 0:
        return 0.0
    d = points.shape[1]
    mins = np.minimum(points.min(axis=0), ref)
    low = mins
    high = ref
    samples = rng.random((n_samples, d)) * (high - low) + low
    dominated = np.zeros(n_samples, dtype=bool)
    for p in points:
        dominated |= np.all(samples <= p, axis=1)  
    fraction = dominated.mean()
    vol_box = np.prod(ref - low)
    return fraction * vol_box

def grid_regret_matrix(return_grid: np.ndarray, w_grid: np.ndarray, ref_map: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute scalarized regret per (w, c) cell.
    Inputs:
      - return_grid: shape (n_w, n_c, d) vector returns (in env sign: negative costs)
      - w_grid: shape (n_w, 3) preference vectors (sum=1)
      - ref_map: optional precomputed reference scalarized best per (w,c) (shape (n_w, n_c)) or None
    Returns:
      regret: (n_w, n_c) where regret >=0 (higher = worse)
    Behavior:
      if ref_map is None, compute reference as max scalarized across evaluation runs (i.e., empirical best).
    """
    n_w, n_c, d = return_grid.shape
    objs = -return_grid  
    scalar_grid = np.zeros((n_w, n_c))
    for i in range(n_w):
        for j in range(n_c):
            scalar_grid[i, j] = float(np.dot(w_grid[i], objs[i, j]))

    if ref_map is None:
        # empirical best across grid
        ref_map = scalar_grid.max(axis=1).reshape((n_w,1)) 
        # broadcast to n_c
        ref_map = np.repeat(ref_map, n_c, axis=1)

    # regret = best - achieved
    regret = ref_map - scalar_grid
    regret = np.maximum(regret, 0.0)
    return regret

def constraint_stats(cost_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    cost_grid: shape (n_w, n_c, m) - aggregated cost per eval (sum or mean)
    Returns:
      violation_rate: (m,) fraction of (w,c) where cost>0
      mean_violation: (m,) mean of positive cost across evaluated cells
    """
    pos = cost_grid > 0.0
    violation_rate = pos.any(axis=2).mean(axis=(0,1)) if cost_grid.ndim == 3 else (pos.mean(axis=0))
    mean_violation = np.mean(np.where(cost_grid>0, cost_grid, 0.0), axis=(0,1))
    return violation_rate, mean_violation

def coverage_fraction(points: np.ndarray, grid_shape: Tuple[int,int]) -> float:
    """
    Compute coverage as fraction of grid points that are non-dominated across all evaluated (w,c).
    points: (n_points, d) flattened evaluated objectives (maximization)
    grid_shape: (n_w,n_c) used to normalize
    """
    if len(points) == 0:
        return 0.0
    pf = pareto_front(points)
    return float(len(pf) / (grid_shape[0] * grid_shape[1]))

def ensemble_variance_map(critic_var_grid: np.ndarray) -> np.ndarray:
    """
    Return the provided critic variance grid (n_w, n_c) (already computed during evaluation),
    placeholder for any aggregation you want.
    """
    return critic_var_grid
