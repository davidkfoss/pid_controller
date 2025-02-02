from .plant import Plant
import numpy as np


class CournotCompetitionPlant(Plant):
    def __init__(self, params, disturbance_params):
        """Initialize the Cournot competition model."""
        super().__init__(params, disturbance_params)
        # Initial production for Firm 1 (bounded between 0 and 1)
        self.state["q1"] = 0.5
        # Initial production for Firm 2 (bounded between 0 and 1)
        self.state["q2"] = 0.5
        self.state["price"] = self.params["max_price_pmax"]
        self.state["profit_p1"] = self.state["q1"] * \
            (self.state["price"] - self.state["marginal_cost_cm"])
        self.target_profit = self.params["target_profit"]

    def update(self, control_signal):
        """
        Update production levels and calculate new price and profit.
        - `q1` updates based on control signal (U).
        - `q2` updates based on disturbance (D).
        - `p(q) = pmax - q`
        - `profit_p1 = q1 * (p(q) - cm)`
        """
        pmax = self.params["max_price_pmax"]
        cm = self.params["marginal_cost_cm"]

        # Apply control signal (U) to update q1 (firm 1's production)
        self.state["q1"] = np.clip(
            self.state["q1"] + control_signal, 0, 1)  # Ensure 0 ≤ q1 ≤ 1

        # Apply disturbance (D) to update q2 (firm 2's production)
        disturbance = self.apply_disturbance()
        self.state["q2"] = np.clip(
            self.state["q2"] + disturbance, 0, 1)  # Ensure 0 ≤ q2 ≤ 1

        # Compute total production q
        total_q = self.state["q1"] + self.state["q2"]

        # Compute price
        # Ensure non-negative price
        self.state["price"] = max(pmax - total_q, 0)

        # Don't allow unprofitable production
        if self.state["price"] < cm:
            self.state["q1"] = 0

        # Compute profit for Firm 1
        self.state["profit_p1"] = self.state["q1"] * (self.state["price"] - cm)

    def get_output(self):
        """Return Firm 1's profit (Y), as the controller regulates this."""
        return self.state["profit_p1"]

    def get_error(self):
        """Return the difference between target profit and actual profit"""
        return self.get_output() - self.target_profit
