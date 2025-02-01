from .plant import Plant
import numpy as np


class BathtubModel(Plant):
    def __init__(self, params, disturbance_params):
        """Initialize the bathtub model with parameters."""
        super().__init__(params, disturbance_params)
        self.state["water_height"] = self.params["initial_height_H0"]
        self.gravity = 9.8  # Gravitational constant (m/s²)

    def update(self, control_signal):
        """
        Update the bathtub water height based on:
        - Control input (U)
        - Disturbance (D)
        - Drain flow rate (Q)
        """
        H = self.state["water_height"]  # Current height
        A = self.params["area_A"]  # Bathtub cross-sectional area
        C = self.params["drain_area_C"]  # Drain cross-sectional area

        # Compute velocity of water exiting through the drain
        V = np.sqrt(2 * self.gravity * max(H, 0))  # Ensure non-negative H

        # Compute outflow rate Q
        Q = V * C

        # Get random disturbance (D)
        D = self.apply_disturbance()

        # Change in volume
        dB_dt = control_signal + D - Q

        # Change in height
        dH_dt = dB_dt / A

        # Update water height
        self.state["water_height"] = max(
            0, self.state["water_height"] + dH_dt)  # Prevent negative height

    def get_output(self):
        """Return the current water height (Y)."""
        return self.state["water_height"]
