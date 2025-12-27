import numpy as np

class PreferenceCurriculum:
    """
    Scenario-driven preference curriculum.
    """

    def __init__(self):
        self.grid = self._build_grid(step=0.25)
        self.ptr = 0

    def _build_grid(self, step=0.25):
        grid = []
        for w1 in np.arange(step, 1.0, step):
            for w2 in np.arange(step, 1.0 - w1, step):
                w3 = 1.0 - w1 - w2
                if w3 > 0:
                    grid.append(np.array([w1, w2, w3]))
        grid += [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        return grid

    def sample(self, step: int):
        w = self.grid[self.ptr % len(self.grid)]
        self.ptr += 1
        return w.astype(np.float32)
