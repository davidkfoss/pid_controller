import numpy as np


class Plant:
    def __init__(self, params):
        """Initialize the plant with parameters and optional disturbance."""
        self.initial_state = params

    def update(self, control_signal, plant_state, disturbance):
        """Update the plant state based on the control signal (U)."""
        raise NotImplementedError("Subclasses must implement update()")

    def get_error(self, state):
        """Return the difference between the current error and the target value"""
        raise NotImplementedError("Subclass must implement get_error()")
