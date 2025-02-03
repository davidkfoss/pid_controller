from .plant import Plant
import jax.numpy as jnp


class CournotCompetitionPlant(Plant):
    def __init__(self, params):
        """Initialize the Cournot competition model."""
        super().__init__(params)
        self.target_profit = params["target_profit"]
        self.pmax = params["max_price_pmax"]
        self.cm = params["marginal_cost_cm"]

    def update(self, control_signal, plant_state, disturbance):
        """
        Update production levels and calculate new price and profit.
        - `q1` updates based on control signal (U).
        - `q2` updates based on disturbance (D).
        - `p(q) = pmax - q`
        - `profit_p1 = q1 * (p(q) - cm)`
        """

        # Apply control signal (U) to update q1 (firm 1's production)
        # Ensure 0 ≤ q1 ≤ 1
        q1 = jnp.clip(plant_state["q1"] + control_signal, 0, 1)

        q2 = jnp.clip(plant_state["q2"] + disturbance,
                      0, 1)  # Ensure 0 ≤ q2 ≤ 1

        # Compute total production q
        total_q = q1 + q2

        # Compute price
        # Ensure non-negative price
        price = jnp.maximum(self.pmax - total_q, 0)

        # Don't allow unprofitable production
        q1 = q1 * jnp.where(price >= self.cm, 1, 0)  # Smooth transition to 0

        # Compute profit for Firm 1
        profit_p1 = q1 * (price - self.cm)
        return {"q1": q1, "q2": q2, "price": price, "profit_p1": profit_p1}

    def get_error(self, state):
        """Return the difference between target profit and actual profit"""
        return state["profit_p1"] - self.target_profit
