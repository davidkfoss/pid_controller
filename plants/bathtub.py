from .plant import Plant
import jax.numpy as jnp


class BathtubPlant(Plant):
    def __init__(self, params, disturbance_params):
        """Initialize the bathtub model with parameters."""
        super().__init__(params, disturbance_params)
        self.state["water_height"] = self.params["initial_height_h0"]
        self.gravity = 9.81
        self.target_height = self.params["target_height"]
        self.area_a = self.params["area_a"]
        self.drain_area_c = self.params["drain_area_c"]

    def update(self, control_signal):
        """
        Update the bathtub water height based on:
        - Control input (U)
        - Disturbance (D)
        - Drain flow rate (Q)
        """
        h = self.state["water_height"]  # Current height
        A = self.area_a  # Bathtub cross-sectional area
        C = self.drain_area_c  # Drain cross-sectional area

        # Compute velocity of water exiting through the drain
        V = jnp.sqrt(2 * self.gravity * max(h, 0))  # Ensure non-negative H

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
