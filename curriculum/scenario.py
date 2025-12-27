import numpy as np

SCENARIOS = [
    "normal",
    "peak_load",
    "off_peak",
    "carbon_intensive",
    "renewable_rich"
]

class ScenarioCurriculum:
    def __init__(self):
        self.scenarios = SCENARIOS

    def encode(self, scenario: str):
        one_hot = np.zeros(len(self.scenarios), dtype=np.float32)
        idx = self.scenarios.index(scenario)
        one_hot[idx] = 1.0
        return one_hot

    def sample(self, step: int):
        scenario = self.scenarios[step % len(self.scenarios)]
        return scenario, self.encode(scenario)
