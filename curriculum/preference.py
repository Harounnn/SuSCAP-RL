import numpy as np

class PreferenceCurriculum:
    """
    Multi-phase preference curriculum with difficulty scaling.
    - corners: single pure objectives (1,0,0), (0,1,0), (0,0,1)
    - edges: 2D subsets + corners
    - grid: coarse grid
    - adversarial: random sampling (harder exploration)
    - refinement: fine grid (convergence)
    - polishing: random Dirichlet sampling (uncertainty-driven exploration)
    """

    def __init__(self):
        self.current_phase = "corners"
        self._build_phase_grids()
        self.phase_ptrs = {phase: 0 for phase in self.phase_grids.keys()}

    def _build_phase_grids(self):
        """Build preference grids for each curriculum phase."""
        # Pure objectives (corners of simplex)
        corners = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        
        # 2D edges (two objectives active)
        edges = []
        for w1 in np.arange(0.1, 1.0, 0.1):
            w2 = 1.0 - w1
            edges.append(np.array([w1, w2, 0.0]))
            edges.append(np.array([w1, 0.0, w2]))
            edges.append(np.array([0.0, w1, w2]))
        edges = list(set([tuple(e) for e in edges]))
        edges = [np.array(e) for e in edges]
        
        # Coarse grid (step=0.5)
        coarse_grid = self._build_grid(step=0.5)
        
        # Fine grid (step=0.25)
        fine_grid = self._build_grid(step=0.25)
        
        # Random sampling from simplex (adversarial phase uses this)
        # We'll generate on-the-fly via random_simplex_sample()
        
        self.phase_grids = {
            "corners": np.array(corners),
            "edges": np.array(edges),
            "grid": coarse_grid,
            "adversarial": None,  # generated dynamically
            "refinement": fine_grid,
            "polishing": None,  # generated dynamically via random simplex sampling
        }

    def _build_grid(self, step=0.25):
        """Build uniform grid on 3-simplex."""
        grid = []
        for w1 in np.arange(step, 1.0, step):
            for w2 in np.arange(step, 1.0 - w1, step):
                w3 = 1.0 - w1 - w2
                if w3 >= 0:
                    grid.append(np.array([w1, w2, w3]))
        # Add pure objectives
        grid.extend([
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ])
        return np.array(grid, dtype=np.float32)

    def _random_simplex_sample(self, rng=None):
        """Sample uniformly from 3-simplex."""
        if rng is None:
            rng = np.random.default_rng()
        # Exponential distribution method for uniform simplex sampling
        e = rng.exponential(scale=1.0, size=3)
        return (e / e.sum()).astype(np.float32)

    def set_phase(self, phase: str):
        """Set the current curriculum phase."""
        if phase not in self.phase_grids:
            raise ValueError(f"Unknown phase: {phase}. Choose from {list(self.phase_grids.keys())}")
        self.current_phase = phase
        if phase not in self.phase_ptrs:
            self.phase_ptrs[phase] = 0

    def sample(self, step: int, phase: str = None):
        """Sample preference weight vector with latency bias in intermediate phases.
        
        Args:
            step: training step counter (for adversarial: random seed)
            phase: curriculum phase (if None, uses current_phase)
        
        Returns:
            weight vector (3,) summing to 1 (energy, CO2, latency)
        """
        active_phase = phase if phase is not None else self.current_phase
        
        if active_phase == "adversarial":
            # Adversarial phase: 70% full simplex, 30% latency-biased
            if np.random.random() < 0.3:
                # Latency-biased: ensure w_latency >= 0.3
                return self._sample_latency_biased(min_latency=0.3)
            else:
                rng = np.random.default_rng(seed=step % (2**31))
                return self._random_simplex_sample(rng)
        elif active_phase == "polishing":
            # Polishing phase: pure random simplex sampling (unbiased Dirichlet)
            rng = np.random.default_rng(seed=step % (2**31))
            return self._random_simplex_sample(rng)
        else:
            # Deterministic cycling through phase grid
            grid = self.phase_grids[active_phase]
            ptr = self.phase_ptrs[active_phase]
            sample = grid[ptr % len(grid)].copy()
            self.phase_ptrs[active_phase] += 1
            
            # Bias intermediate phases (edges, grid) toward latency awareness
            # Corners should remain pure objectives (no bias)
            if active_phase in ["edges", "grid"]:
                # Ensure w_latency >= 0.3; if not, mix with latency direction
                for _ in range(10):
                    if sample[2] >= 0.3:  # sample[2] is latency weight (3rd objective)
                        return sample.astype(np.float32)
                    # Mix: 70% sample, 30% latency direction
                    latency_dir = np.array([0.0, 0.0, 1.0])
                    sample = 0.7 * sample + 0.3 * latency_dir
                    sample = sample / sample.sum()  # Re-normalize to simplex
                return sample.astype(np.float32)
            else:
                # Corners, adversarial, refinement: return as-is
                return sample.astype(np.float32)
    
    def _sample_latency_biased(self, min_latency=0.3, rng=None):
        """Sample from simplex with minimum latency weight constraint."""
        if rng is None:
            rng = np.random.default_rng()
        # Sample until constraint satisfied
        for _ in range(20):
            e = rng.exponential(scale=1.0, size=3)
            sample = (e / e.sum()).astype(np.float32)
            if sample[2] >= min_latency:  # sample[2] is latency (3rd objective)
                return sample
        # Fallback: return max-latency corner
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
