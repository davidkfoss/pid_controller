import numpy as np


class Plant:
    def __init__(self, params, disturbance_params):
        """Initialize the plant with parameters and optional disturbance."""
        self.params = params
        self.state = {}  # Store plant-specific state variables
        self.disturbance_range = disturbance_params.get(
            "disturbance_range", (0, 0)) if disturbance_params else (0, 0)

    def apply_disturbance(self):
        """Apply random noise from the disturbance range."""
        d_min, d_max = self.disturbance_range
        return np.random.uniform(d_min, d_max) if d_min != d_max else 0

    def update(self, control_signal):
        """Update the plant state based on the control signal (U)."""
        raise NotImplementedError("Subclasses must implement update()")

    def get_output(self):
        """Return the plant output (Y)."""
        raise NotImplementedError("Subclasses must implement get_output()")

    def get_error(self):
        """Return the difference between the current error and the target value"""
        raise NotImplementedError("Subclass must implement get_error()")
