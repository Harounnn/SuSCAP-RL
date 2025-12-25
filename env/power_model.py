class LinearPowerModel:
    """
    Standard linear server power model.

    P(u) = P_idle + (P_max - P_idle) * u
    """

    def __init__(
        self,
        p_idle: float = 100.0,   # Watts
        p_max: float = 300.0     # Watts
    ):
        self.p_idle = p_idle
        self.p_max = p_max

    def power(self, cpu_util: float) -> float:
        """
        Instantaneous power in Watts.
        cpu_util in [0,1]
        """
        return self.p_idle + (self.p_max - self.p_idle) * cpu_util

    def energy(self, cpu_util: float, delta_t_sec: float) -> float:
        """
        Energy consumed during timestep (kWh).
        """
        power_w = self.power(cpu_util)
        return (power_w * delta_t_sec) / 3_600_000.0
