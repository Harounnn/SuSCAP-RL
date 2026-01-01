from curriculum.scenario import ScenarioCurriculum

class SchedulerCurriculum:
    def __init__(self):
        self.scen = ScenarioCurriculum()

    def sample(self, step: int):
        scenario, c = self.scen.sample(step)
        return c, scenario
