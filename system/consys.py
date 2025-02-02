from utils import load_config
from plants import BathtubPlant, CournotCompetitionPlant, MonetaryPolicyPlant
from controllers import PIDController, NeuralNetController


class ControlSystem:
    def __init__(self, plant, controller):
        self.plant = plant
        self.controller = controller

    def run(self, timesteps, target):
        """Runs the simulation for a given number of timesteps."""
        pass


def main():
    # Load configuration
    plant_type, controller_type, disturbance_params, config = load_config()

    # Initialize plant
    if plant_type == "bathtub":
        plant = BathtubPlant(config["BathtubPlant"], disturbance_params)
    elif plant_type == "cournot":
        plant = CournotCompetitionPlant(
            config["CournotCompetitionPlant"], disturbance_params)
    elif plant_type == "centralbank":
        plant = MonetaryPolicyPlant(
            config["MonetaryPolicyPlant"], disturbance_params)
    else:
        raise ValueError(f"Unknown plant type: {plant_type}")

    # Initialize controller
    if controller_type == "classic":
        # Because PID uses learning rate
        controller = PIDController(config["NeuralNetwork"])
    elif controller_type == "nn":
        controller = NeuralNetController(config["NeuralNetwork"])
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    # Create the control system and run it
    consys = ControlSystem(plant, controller)
    # Example with 10 timesteps and target 5
    consys.run(timesteps=10, target=5)


if __name__ == "__main__":
    main()
