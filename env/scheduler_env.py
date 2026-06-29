import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .power_model import LinearPowerModel
from .scenarios import ScenarioBank


class SchedulerEnv(gym.Env):
    """
    Preference-conditioned multi-objective scheduling environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path: str,
        episode_length: int = 1440,
        timestep_sec: int = 60,
        energy_cap: float = 0.003,
        latency_cap: float = 1.0,
        scenario: str | None = None,
        power_model: LinearPowerModel | None = None
    ):
        super().__init__()
        self.df = pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        self.episode_length = episode_length
        self.timestep_sec = timestep_sec
        self.power_model = power_model or LinearPowerModel()
        self.scenario_bank = ScenarioBank(self.df)
        self.scenario = scenario
        self.energy_cap = energy_cap
        self.latency_cap = latency_cap

        # Pre-extract numpy arrays for microsecond lookups
        self.cpu_means = self.df["cpu_mean"].to_numpy(dtype=np.float32)
        self.mem_means = self.df["mem_mean"].to_numpy(dtype=np.float32)
        self.carbon_intensities = self.df["carbon_intensity"].to_numpy(dtype=np.float32)

        # Precompute cyclical time features once
        hours = self.df["timestamp"].dt.hour + self.df["timestamp"].dt.minute / 60.0
        self.sin_time = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
        self.cos_time = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 2000, 1, 1], dtype=np.float32)
        )
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        self._start_idx = None
        self._step_idx = None

    # Gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.scenario is None:
            self._start_idx = self.np_random.integers(
                0, len(self.df) - self.episode_length
            )
        else:
            self._start_idx = self.scenario_bank.sample_start_index(
                self.scenario, self.episode_length
            )

        self._step_idx = 0
        return self._get_obs(), {}

    def step(self, action):
        if isinstance(action, (np.ndarray, list, tuple)):
            act_val = action[0] if len(action) > 0 else float(action)
        else:
            act_val = action
        action = float(np.clip(act_val, 0.0, 1.0))

        idx = self._start_idx + self._step_idx
        cpu_mean = self.cpu_means[idx]
        carbon = self.carbon_intensities[idx]

        cpu_eff = cpu_mean * action
        energy = self.power_model.energy(cpu_eff, self.timestep_sec)
        co2 = energy * carbon / 1000.0

        latency = cpu_mean * (1.0 / (action + 1e-3))
        latency = min(latency, 10.0)

        reward_vec = np.array([-energy, -co2, -latency], dtype=np.float32)
        cost = np.array([max(0.0, energy - self.energy_cap), max(0.0, latency - self.latency_cap)], dtype=np.float32)

        self._step_idx += 1
        truncated = self._step_idx >= self.episode_length
        return self._get_obs(), reward_vec, False, truncated, {"reward_vec": reward_vec, "cost": cost, "scenario": self.scenario}

    # Helpers

    def _get_obs(self):
        idx = self._start_idx + self._step_idx
        return np.array([self.cpu_means[idx], self.mem_means[idx], self.carbon_intensities[idx], self.sin_time[idx], self.cos_time[idx]], dtype=np.float32)
