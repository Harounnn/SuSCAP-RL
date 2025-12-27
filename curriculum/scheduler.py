from curriculum.preference import PreferenceCurriculum
from curriculum.scenario import ScenarioCurriculum

class SchedulerCurriculum:
    def __init__(self):
        self.pref = PreferenceCurriculum()
        self.scen = ScenarioCurriculum()

    def sample(self, step: int):
        w = self.pref.sample(step)
        scenario, c = self.scen.sample(step)
        return w, c, scenario
