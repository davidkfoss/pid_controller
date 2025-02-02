from .controller import Controller


class PIDController(Controller):
    def __init__(self, params):
        super().__init__(params)
        self.kp = 0.5
        self.ki = 0.1
        self.kd = 0.05

    def update_params(self, params, grads):
        g_kp = grads[0]
        g_ki = grads[2]
        g_kd = grads[1]

        self.kp -= self.learning_rate*g_kp
        self.kd -= self.learning_rate*g_kd
        self.ki -= self.learning_rate*g_ki

    def control_signal(self, error):
        self.errors.append(error)

        P = self.kp*error
        I = self.ki*sum(self.errors)
        D = self.kd*(errors[-1]-errors[-2]) if len(self.errors) > 1 else 0

        return P+I+D

    def print_params(self):
        print("kp: " + self.kp + ", ki: " + self.ki + ", kd: " + self.kd)
