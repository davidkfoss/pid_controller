from .controller import Controller
import jax.numpy as jnp


class PIDController(Controller):
    def __init__(self, params):
        super().__init__(params)

    def update_params(self, params, grads):
        g_kp = float(grads[0])
        g_ki = float(grads[1])
        g_kd = float(grads[2])
        print(f"g_kp: {g_kp}, g_ki: {g_ki}, g_kd: {g_kd}")

        kp, ki, kd = params
        print(f"kp: {kp}, ki: {ki}, kd: {kd}")

        kp -= self.learning_rate*g_kp
        ki -= self.learning_rate*g_ki
        kd -= self.learning_rate*g_kd

        return [kp, ki, kd]

    def control_signal(self, params, error_history):
        kp, ki, kd = params

        P = kp*error_history[-1]
        I = ki * jnp.sum(jnp.array(error_history))
        D = kd*(error_history[-1]-error_history[-2])

        return P+I+D

    def print_params(self, params):
        kp, ki, kd = params
        print("kp: " + kp + ", ki: " + ki + ", kd: " + kd)
