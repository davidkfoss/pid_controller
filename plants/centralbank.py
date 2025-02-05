import numpy as np
from .plant import Plant


class MonetaryPolicyPlant(Plant):
    def __init__(self, params):
        """
        Initialize the monetary policy model (plant).

        Parameters:
        - params: Dictionary containing system parameters.
        - disturbance_params: Dictionary for disturbance settings.
        """
        super().__init__(params)

        # System parameters
        self.a = params["a"]
        self.b = params["b"]
        self.c = params["c"]
        self.d = params["d"]

        # Targets
        self.r_target = params["r_target"]
        self.y_target = params["y_target"]
        self.pi_target = params["pi_target"]

    def update(self, control_signal, plant_state, disturbance):
        """
        Updates the system based on the current interest rate (control signal).

        Parameters:
        - control_signal (float): The interest rate set by the controller.
        """
        # Retrieve state variables
        pi = plant_state["pi"]
        y = plant_state["y"]

        # Compute next state using discrete-time updates
        dy_dt = -self.a * (y - self.y_target) - \
            self.b * (control_signal - self.r_target)
        dpi_dt = self.c * (y - self.y_target) - \
            self.d * (pi - self.pi_target) + disturbance

        # Update the state
        new_y = y + dy_dt
        new_pi = pi + dpi_dt
        return {"pi": new_pi, "y": new_y}

    def get_error(self, state):
        """Returns the diffrence between current inflation rate and the target inflation rate."""
        return state["pi"] - self.pi_target
