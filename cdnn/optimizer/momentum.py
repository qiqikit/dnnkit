import numpy as np
from cdnn.model import COptimizer
from cdnn.model import CTensor


# Momentum
class Momentum(COptimizer):
    def __init__(self, learning_rate: float, momentum_rate: float):
        COptimizer.__init__(self, learning_rate=learning_rate)
        self.momentum_rate = momentum_rate

    def init(self, var: CTensor):
        if isinstance(var, CTensor):
            var.diff *= 0
            var.opt_var.clear()
            var.opt_var['mdiff'] = np.zeros(var.diff.shape)

    def apply(self, var: CTensor, batch_size: int):
        if isinstance(var, CTensor):
            var.opt_var['mdiff'] = self.momentum_rate * var.opt_var['mdiff'] + self.learning_rate * var.diff / batch_size
            var.data -= var.opt_var['mdiff']
            var.diff *= 0
