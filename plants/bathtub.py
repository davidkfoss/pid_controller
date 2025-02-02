from .plant import Plant
import numpy as np


class BathtubPlant(Plant):
    def __init__(self, params, disturbance_params):
        """Initialize the bathtub model with parameters."""
        super().__init__(params, disturbance_params)
        self.state["water_height"] = self.params["initial_height_H0"]
        self.gravity = 9.81
        self.target_height = self.params["target_height"]

    def update(self, control_signal):
        """
        Update the bathtub water height based on:
        - Control input (U)
        - Disturbance (D)
        - Drain flow rate (Q)
        """
        h = self.state["water_height"]  # Current height
        A = self.params["area_A"]  # Bathtub cross-sectional area
        C = self.params["drain_area_C"]  # Drain cross-sectional area

        # Compute velocity of water exiting through the drain
        V = np.sqrt(2 * self.gravity * max(h, 0))  # Ensure non-negative H

        # Compute outflow rate Q
        Q = V * C

        # Get random disturbance (D)
        D = self.apply_disturbance()

        # Change in volume
        db_dt = control_signal + D - Q

        # Change in height
        dh_dt = db_dt / A

        # Update water height
        self.state["water_height"] = max(
            0, self.state["water_height"] + dh_dt)  # Prevent negative height

    def get_output(self):
        """Return the current water height (Y)."""
        return self.state["water_height"]

    def get_error(self):
        return self.get_output() - self.target_height
