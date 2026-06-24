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
        # Phase-based curriculum: difficulty progression
        self.phase_schedule = {
            "corners": ["normal"],  # Simplest: single baseline scenario
            "edges": ["normal", "peak_load", "off_peak"],  # Mix easy scenarios
            "grid": SCENARIOS.copy(),  # All scenarios equally
            "adversarial": ["carbon_intensive", "renewable_rich"],  # Challenging extremes
            "refinement": SCENARIOS.copy(),  # Full diversity for fine-tuning
            "polishing": SCENARIOS.copy()  # Full scenario coverage for active polishing
        }
        self.current_phase = "corners"

    def set_phase(self, phase: str):
        """Set the current curriculum phase."""
        if phase not in self.phase_schedule:
            raise ValueError(f"Unknown phase: {phase}. Choose from {list(self.phase_schedule.keys())}")
        self.current_phase = phase

    def encode(self, scenario: str):
        one_hot = np.zeros(len(self.scenarios), dtype=np.float32)
        idx = self.scenarios.index(scenario)
        one_hot[idx] = 1.0
        return one_hot

    def sample(self, step: int, phase: str = None):
        """Sample scenario based on phase curriculum.
        
        Args:
            step: training step counter
            phase: curriculum phase (if None, uses current_phase)
        
        Returns:
            (scenario_name, encoded_vector)
        """
        active_phase = phase if phase is not None else self.current_phase
        phase_scenarios = self.phase_schedule[active_phase]
        scenario = phase_scenarios[step % len(phase_scenarios)]
        return scenario, self.encode(scenario)
