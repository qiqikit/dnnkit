import numpy as np


class DnnNet(object):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.image_shape = None
        self.image_in = None

    def net_init(self):
        pass

    def set_phase(self, phase: str):
        pass

    def set_learning_rate(self, learning_rate: float):
        pass

    def feed(self, image: np.ndarray, label: np.ndarray):
        pass

    def forward(self):
        pass

    def prediction(self):
        pass

    def backward(self):
        pass

    def net_update(self):
        pass

    def load_weight(self, path: str):
        pass

    def save_weight(self, path: str):
        pass