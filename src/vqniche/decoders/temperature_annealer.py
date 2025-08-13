import torch
import torch.nn.functional as F


class TemperatureAnnealer:
    def __init__(
            self,
            start_temp: float,
            end_temp: float,
            total_steps: int,
            mode: str = "linear",
        ):
        """
        Anneals temperature from start_temp to end_temp over total_steps.

        Parameters
        ----------
        - start_temp: float
            Initial temperature.
        - end_temp: float
            Final temperature.
        - total_steps: int
            Number of steps over which to anneal.
        - mode: str
            "linear" or "exp".
        """
        assert mode in ("linear", "exp"), "mode must be 'linear' or 'exp'"
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps
        self.mode = mode
        self.step_count = 0


    def step(self) -> float:
        """Advance one step and return current temperature."""
        self.step_count += 1
        return self.get_temp()


    def get_temp(self) -> float:
        """Get current temperature without stepping."""
        progress = min(self.step_count / max(1, self.total_steps), 1.0)
        if self.mode == "linear":
            temp = self.start_temp + (self.end_temp - self.start_temp) * progress
        else:  # exponential
            decay_rate = (self.end_temp / self.start_temp) ** progress
            temp = self.start_temp * decay_rate
        return temp
