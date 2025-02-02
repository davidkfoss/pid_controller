import numpy as np


class Controller:
    def __init__(self, params):
        self.errors = []
        self.learning_rate = self.params["learning_rate"]

    def update_params(self, params, grads):
        """Updates the parameters based on the new error (E)."""
        raise NotImplementedError("Subclass must implement update_params()")

    def control_signal(self, error):
        """Calculates and returns the control signal (U)."""
        raise NotImplementedError(
            "Subclass must implement get_control_signal()")
