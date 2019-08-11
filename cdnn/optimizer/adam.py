import numpy as np
from cdnn.model import COptimizer
from cdnn.model import CTensor


# Adaptive Moment Estimation
class Adam(COptimizer):
    def __init__(self, learning_rate: float, beta1: float, beta2: float, epsilon: float):
        COptimizer.__init__(self, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def init(self, var: CTensor):
        if isinstance(var, CTensor):
            var.diff *= 0
            var.opt_var.clear()
            var.opt_var['m'] = np.zeros(var.diff.shape)
            var.opt_var['v'] = np.zeros(var.diff.shape)
            var.opt_var['t'] = 0

    def apply(self, var: CTensor, batch_size: int):
        # if isinstance(var, CTensor):
        #     var.opt_var['t'] += 1
        #     learning_rate_t = self.learning_rate * math.sqrt(1 - pow(self.beta2, var.opt_var['t'])) / (
        #                 1 - pow(self.beta1, var.opt_var['t']))
        #     var.opt_var['m_t'] = self.beta1 * var.opt_var['m_t'] + (1.0 - self.beta1) * var.diff / batch_size
        #     var.opt_var['v_t'] = self.beta2 * var.opt_var['v_t'] + (1.0 - self.beta2) * ((var.diff / batch_size) ** 2)
        #     var.data -= learning_rate_t * var.opt_var['m_t'] / (var.opt_var['v_t'] + self.epsilon) ** 0.5
        #     var.diff *= 0

        if isinstance(var, CTensor):
            var.opt_var['t'] += 1

            var.opt_var['m'] = self.beta1 * var.opt_var['m'] + (1.0 - self.beta1) * (var.diff / batch_size)
            var.opt_var['v'] = self.beta2 * var.opt_var['v'] + (1.0 - self.beta2) * np.square(var.diff / batch_size)

            mt = var.opt_var['m'] / (1 - self.beta1)
            vt = var.opt_var['v'] / (1 - self.beta2)

            var.data -= self.learning_rate * mt / (np.sqrt(vt) + self.epsilon)
            var.diff *= 0
