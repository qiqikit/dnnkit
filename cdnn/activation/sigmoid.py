import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression


# CExpression class
class Sigmoid(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple):
        CExpression.__init__(self, name, input_shape, output_shape)

        if input_shape != output_shape:
            raise Exception("Sigmoid name: %s input_shape != output_shape" % self.name)

        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if input_tensor.shape != output_tensor.shape:
            raise Exception("Sigmoid name: %s input_tensor.shape != output_tensor.shape" % self.name)
        if self.input_shape != input_tensor.shape:
            raise Exception("Sigmoid name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("Sigmoid name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def forward(self):
        CExpression.forward(self)
        # y = 1/(1+exp(-x))
        self.output_tensor.data = 1.0 / (1.0 + np.exp(-self.input_tensor.data))

    def backward(self):
        CExpression.backward(self)
        # dx = y * (1-y)
        self.input_tensor.diff = self.output_tensor.data * (1 - self.output_tensor.data) * self.output_tensor.diff
