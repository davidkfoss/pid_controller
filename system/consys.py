from utils import load_config, plot_mse, plot_pid_mse
from plants import BathtubPlant, CournotCompetitionPlant, MonetaryPolicyPlant
from controllers import PIDController, NeuralNetController
import jax
import jax.numpy as jnp
from datetime import datetime as time
import numpy.random as rnd


class ControlSystem:
    def __init__(self, plant, controller, epochs, timesteps, params, disturbance_params):
        self.plant = plant
        self.controller = controller
        self.epochs = epochs
        self.timesteps = timesteps
        self.grad_fn = jax.value_and_grad(self.run_epoch, argnums=0)
        self.param_history = [params]
        self.mse_history = []
        self.disturbance_params = disturbance_params

    def run(self):
        for _ in range(self.epochs):
            mse, grads = self.grad_fn(self.param_history[-1])
            print(f"Epoch: {len(self.mse_history)+1}, MSE: {mse}")
            self.mse_history.append(mse)
            self.param_history.append(
                self.controller.update_params(self.param_history[-1], grads))
        filename = f"""{self.controller.__class__.__name__}_{self.plant}_{
            self.epochs}_{self.timesteps}_{self.controller.learning_rate}_{time.now()}"""
        if self.controller.__class__.__name__ == "NeuralNetController":
            plot_mse(self.mse_history, filename)
        elif self.controller.__class__.__name__ == "PIDController":
            plot_pid_mse(self.mse_history, self.param_history, filename)

    def run_epoch(self, params):
        error_history = [0, 0]
        plant_state = self.plant.initial_state
        disturbance = self.generate_disturbance()
        for _ in range(self.timesteps):
            plant_state, error = self.run_timestep(
                params, error_history, plant_state, disturbance)
            error_history.append(error)
        return jnp.mean(jnp.square(jnp.array(error_history)))

    def run_timestep(self, params, error_history, plant_state, disturbance):
        control_signal = self.controller.control_signal(
            params, error_history)
        new_plant_state = self.plant.update(
            control_signal, plant_state, disturbance)
        return new_plant_state, self.plant.get_error(new_plant_state)

    def generate_disturbance(self):
        # This is whats in disturbance_params: disturbance_range = -0.01, 0.01
        return rnd.uniform(low=self.disturbance_params["disturbance_range"][0], high=self.disturbance_params["disturbance_range"][1])


def main():
    # Load configuration
    plant_type, controller_type, disturbance_params, config = load_config()

    # Initialize plant
    if plant_type == "bathtub":
        plant = BathtubPlant(config["BathtubPlant"])
    elif plant_type == "cournot":
        plant = CournotCompetitionPlant(
            config["CournotCompetitionPlant"])
    elif plant_type == "centralbank":
        plant = MonetaryPolicyPlant(
            config["MonetaryPolicyPlant"])
    else:
        raise ValueError(f"Unknown plant type: {plant_type}")

    # Initialize controller
    if controller_type == "classic":
        # Because PID uses learning rate
        controller = PIDController(config["ClassicPID"])
        params = jnp.array([config["ClassicPID"]["kp"],
                           config["ClassicPID"]["ki"], config["ClassicPID"]["kd"]])
    elif controller_type == "nn":
        controller = NeuralNetController(config["NeuralNetwork"])
        params = _  # TODO: Initialize neural network parameters
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

    # Create the control system and run it
    consys = ControlSystem(plant, controller, epochs=50,
                           timesteps=100, params=params, disturbance_params=disturbance_params)
    # Example with 10 timesteps and target 5
    consys.run()


if __name__ == "__main__":
    main()
