import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression


# CExpression class
class Relu(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple):
        CExpression.__init__(self, name, input_shape, output_shape)

        if input_shape != output_shape:
            raise Exception("Relu name: %s input_shape != output_shape" % self.name)

        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if self.input_shape != input_tensor.shape:
            raise Exception("Relu name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("Relu name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def forward(self):
        CExpression.forward(self)
        self.output_tensor.data = np.maximum(self.input_tensor.data, 0)

    def backward(self):
        CExpression.backward(self)
        self.input_tensor.diff = 1.0 * self.output_tensor.diff
        self.input_tensor.diff[self.input_tensor.data < 0] = 0


class LRelu(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, alpha: float):
        CExpression.__init__(self, name, input_shape, output_shape)

        if input_shape != output_shape:
            raise Exception("LRelu name: %s input_shape != output_shape" % self.name)

        self.alpha = alpha
        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if self.input_shape != input_tensor.shape:
            raise Exception("LRelu name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("LRelu name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def forward(self):
        CExpression.forward(self)
        self.output_tensor.data = np.maximum(self.input_tensor.data, 0) + self.alpha * np.minimum(
            self.input_tensor.data, 0)

    def backward(self):
        CExpression.backward(self)
        self.input_tensor.diff = 1.0 * self.output_tensor.diff
        self.input_tensor.diff[self.input_tensor.data < 0] *= self.alpha
