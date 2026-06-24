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
    rng = np.random.default_rng(seed)
    points = np.array(points)
    if points.size == 0:
        return 0.0
    d = points.shape[1]

    comp_min = points.min(axis=0)
    safe_ref = np.minimum(ref, comp_min) - 1e-6

    maxima = points.max(axis=0)
    low = safe_ref
    high = maxima

    if np.any(high <= low):
        return 0.0

    samples = rng.random((n_samples, d)) * (high - low) + low
    dominated = np.zeros(n_samples, dtype=bool)
    for p in points:
        dominated |= np.all(samples <= p, axis=1)
    fraction = dominated.mean()
    vol_box = np.prod(high - low)
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

def constraint_stats(cost_grid: np.ndarray, thresholds: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Compute constraint violation statistics.
    
    Args:
        cost_grid: shape (n_w, n_c, m) - costs per (preference, scenario, constraint)
        thresholds: optional (m,) threshold vector for constraint evaluation
    
    Returns:
        violation_rate: fraction of (w,c,m) cells exceeding threshold or >0
        mean_violation: mean positive cost across violated cells
    """
    if thresholds is not None:
        # Threshold-based violations
        violations = (cost_grid > thresholds[np.newaxis, np.newaxis, :]).astype(float)
        violation_rate = violations.mean()
        mean_violation = np.mean(np.where(violations > 0, cost_grid, 0.0))
    else:
        # Simple positivity check (legacy)
        pos = cost_grid > 0.0
        violation_rate = pos.mean()
        mean_violation = np.mean(np.where(pos, cost_grid, 0.0))
    
    return float(violation_rate), float(mean_violation)

def per_constraint_violation_rates(cost_grid: np.ndarray, thresholds: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute per-constraint violation rates.
    
    Args:
        cost_grid: shape (n_w, n_c, m)
        thresholds: optional (m,) thresholds
    
    Returns:
        violation_rates: (m,) array of violation rates per constraint
    """
    n_constraints = cost_grid.shape[2]
    rates = []
    
    for c_idx in range(n_constraints):
        constraint_costs = cost_grid[:, :, c_idx]
        if thresholds is not None:
            threshold = thresholds[c_idx]
            violations = (constraint_costs > threshold).astype(float)
        else:
            violations = (constraint_costs > 0.0).astype(float)
        
        rate = violations.mean()
        rates.append(rate)
    
    return np.array(rates)

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
