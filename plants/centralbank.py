import numpy as np
from .plant import Plant


class MonetaryPolicyPlant(Plant):
    def __init__(self, params, disturbance_params):
        """
        Initialize the monetary policy model (plant).

        Parameters:
        - params: Dictionary containing system parameters.
        - disturbance_params: Dictionary for disturbance settings.
        """
        super().__init__(params, disturbance_params)

        # State variables (inflation rate and output gap)
        self.state["inflation"] = self.params["pi0"]
        self.state["output_gap"] = self.params["y0"]

        # System parameters
        self.a = self.params["a"]
        self.b = self.params["b"]
        self.c = self.params["c"]
        self.d = self.params["d"]

        # Targets
        self.r_target = self.params["r_target"]
        self.y_target = self.params["y_target"]
        self.pi_target = self.params["pi_target"]

    def update(self, control_signal):
        """
        Updates the system based on the current interest rate (control signal).

        Parameters:
        - control_signal (float): The interest rate set by the controller.
        """
        # Retrieve state variables
        pi = self.state["inflation"]
        y = self.state["output_gap"]

        # Get disturbance (D) to simulate shocks
        D = self.apply_disturbance()

        # Compute next state using discrete-time updates
        dy_dt = -self.a * (y - self.y_target) - \
            self.b * (control_signal - self.r_target)
        dpi_dt = self.c * (y - self.y_target) - \
            self.d * (pi - self.pi_target) + D

        # Update the state
        self.state["output_gap"] += dy_dt
        self.state["inflation"] += dpi_dt

    def get_output(self):
        """
        Returns the current inflation rate (pi), which the controller will try to stabilize.
        """
        return self.state["inflation"]

    def get_error(self):
        """Returns the diffrence between current inflation rate and the target inflation rate."""
        return self.get_output() - self.pi_target
