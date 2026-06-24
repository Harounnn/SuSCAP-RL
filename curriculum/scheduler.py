from curriculum.scenario import ScenarioCurriculum

class SchedulerCurriculum:
    def __init__(self):
        self.scen = ScenarioCurriculum()

    def set_phase(self, phase: str):
        """Set the current curriculum phase."""
        self.scen.set_phase(phase)

    def sample(self, step: int, phase: str = None):
        """Sample scenario based on curriculum (supports optional phase override)."""
        scenario, c = self.scen.sample(step, phase=phase)
        return c, scenario

