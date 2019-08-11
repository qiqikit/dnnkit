from cdnn.optimizer.adam import Adam
from cdnn.optimizer.sgd import Sgd
from cdnn.optimizer.momentum import Momentum
from cdnn.optimizer.nag import Nag


def sgd(learning_rate: float):
    return Sgd(learning_rate)


def adam(learning_rate: float, beta1: float, beta2: float, epsilon: float):
    return Adam(learning_rate, beta1, beta2, epsilon)


def momentum(learning_rate: float, momentum_rate=0.9):
    return Momentum(learning_rate, momentum_rate)


def nag(learning_rate: float, momentum_rate: float):
    return Nag(learning_rate, momentum_rate)
