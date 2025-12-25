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
        episode_length: int = 1440,   # 1 day (minutes)
        timestep_sec: int = 60,
        energy_cap: float = 2.0,      # kWh per step
        latency_cap: float = 1.0,
        scenario: str | None = None,
        power_model: LinearPowerModel | None = None
    ):
        super().__init__()

        # Load dataset
        self.df = pd.read_csv(data_path, parse_dates=["timestamp"])
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        self.episode_length = episode_length
        self.timestep_sec = timestep_sec

        # Models
        self.power_model = power_model or LinearPowerModel()
        self.scenario_bank = ScenarioBank(self.df)
        self.scenario = scenario

        # Constraints
        self.energy_cap = energy_cap
        self.latency_cap = latency_cap

        # Spaces
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
        action = float(np.clip(action[0], 0.0, 1.0))

        row = self.df.iloc[self._start_idx + self._step_idx]

        cpu_eff = row.cpu_mean * action
        carbon = row.carbon_intensity

        energy = self.power_model.energy(cpu_eff, self.timestep_sec)
        co2 = energy * carbon / 1000.0 

        latency = row.cpu_mean * (1.0 / (action + 1e-3))
        latency = min(latency, 10.0)  

        reward_vec = np.array(
            [-energy, -co2, -latency],
            dtype=np.float32
        )

        cost = np.array(
            [
                max(0.0, energy - self.energy_cap),
                max(0.0, latency - self.latency_cap)
            ],
            dtype=np.float32
        )

        self._step_idx += 1
        truncated = self._step_idx >= self.episode_length

        info = {
            "reward_vec": reward_vec,
            "cost": cost,
            "scenario": self.scenario
        }

        return self._get_obs(), reward_vec, False, truncated, info


    # Helpers

    def _get_obs(self):
        row = self.df.iloc[self._start_idx + self._step_idx]

        hour = row.timestamp.hour + row.timestamp.minute / 60
        sin_t = np.sin(2 * np.pi * hour / 24)
        cos_t = np.cos(2 * np.pi * hour / 24)

        return np.array(
            [
                row.cpu_mean,
                row.mem_mean,
                row.carbon_intensity,
                sin_t,
                cos_t
            ],
            dtype=np.float32
        )
