import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression
from cdnn.model import COptimizer


class DropOut(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, phase: str, prob: float):
        CExpression.__init__(self, name, input_shape, output_shape)

        if input_shape != output_shape:
            raise Exception("DropOut name: %s input_shape != output_shape" % self.name)

        self.prob = prob
        self.phase = phase
        self.index = np.random.random(input_shape) < self.prob

        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if input_tensor.shape != output_tensor.shape:
            raise Exception("DropOut name: %s input_tensor.shape != output_tensor.shape" % self.name)
        if self.input_shape != input_tensor.shape:
            raise Exception("DropOut name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("DropOut name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def init_layer(self, opt: COptimizer):
        self.index = np.random.random(self.input_tensor.shape) < self.prob

    def update_layer(self, opt: COptimizer):
        pass

    def forward(self):
        CExpression.forward(self)
        if self.phase == 'train':
            self.output_tensor.data = self.input_tensor.data * self.index
            # self.output_tensor.data /= self.prob
        elif self.phase == 'test':
            self.output_tensor.data = self.input_tensor.data
        else:
            raise Exception('DropOut name: %s phase is not in test or train' % self.name)

    def backward(self):
        CExpression.backward(self)
        if self.phase == 'train':
            # self.input_tensor.diff = self.output_tensor.diff * self.index / self.prob
            self.input_tensor.diff = self.output_tensor.diff * self.index
        elif self.phase == 'test':
            self.input_tensor.diff = self.output_tensor.diff
        else:
            raise Exception('DropOut name: %s phase is not in test or train' % self.name)
