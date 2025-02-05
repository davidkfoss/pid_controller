from .plant import Plant
import jax.numpy as jnp


class BathtubPlant(Plant):
    def __init__(self, params):
        """Initialize the bathtub model with parameters."""
        super().__init__(params)
        self.gravity = 9.81
        self.target_height = params["water_height"]
        self.area_a = params["cross_section_a"]
        self.area_c = params["cross_section_c"]

    def update(self, control_signal, plant_state, disturbance):
        """
        Update the bathtub water height based on:
        - Control input (U)
        - Disturbance (D)
        - Drain flow rate (Q)
        """
        h = plant_state["water_height"]  # Current height

        A = self.area_a  # Bathtub cross-sectional area
        C = self.area_c  # Drain cross-sectional area

        # Compute velocity of water exiting through the drain
        V = jnp.sqrt(2 * self.gravity * h)

        # Compute outflow rate Q
        Q = V * C

        # Change in volume
        db_dt = control_signal + disturbance - Q

        # Change in height
        dh_dt = db_dt / A

        # Update water height
        new_height = max(0, h + dh_dt)  # Prevent negative height
        return {"water_height": new_height}

    def get_error(self, state):
        return state["water_height"] - self.target_height
