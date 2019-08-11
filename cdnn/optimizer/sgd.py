from cdnn.model import COptimizer
from cdnn.model import CTensor


# Stochastic Gradient Descent
class Sgd(COptimizer):
    def __init__(self, learning_rate: float):
        COptimizer.__init__(self, learning_rate)

    def init(self, var: CTensor):
        if isinstance(var, CTensor):
            var.diff *= 0

    def apply(self, var: CTensor, batch_size: int):
        if isinstance(var, CTensor):
            var.data -= self.learning_rate * var.diff / batch_size
            var.diff *= 0
