import numpy as np
import pandas as pd


class ScenarioBank:
    """
    Scenario definitions for curriculum-based training.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        # Quantile thresholds
        self.cpu_hi = df["cpu_mean"].quantile(0.9)
        self.cpu_lo = df["cpu_mean"].quantile(0.1)

        self.carbon_hi = df["carbon_intensity"].quantile(0.9)
        self.carbon_lo = df["carbon_intensity"].quantile(0.1)

    def label(self, idx: int) -> str:
        row = self.df.iloc[idx]

        if row.cpu_mean >= self.cpu_hi:
            return "peak_load"

        if row.cpu_mean <= self.cpu_lo:
            return "off_peak"

        if row.carbon_intensity >= self.carbon_hi:
            return "carbon_intensive"

        if row.carbon_intensity <= self.carbon_lo:
            return "renewable_rich"

        return "normal"

    def sample_start_index(self, scenario: str, episode_length: int):
        candidates = []

        for i in range(len(self.df) - episode_length):
            if self.label(i) == scenario:
                candidates.append(i)

        if not candidates:
            raise RuntimeError(f"No samples for scenario '{scenario}'")

        return np.random.choice(candidates)
