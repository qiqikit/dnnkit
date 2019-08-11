import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression


class Concat(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple):
        CExpression.__init__(self, name, input_shape, output_shape)
        input_shape = np.array(input_shape)

        if len(input_shape.shape) != 2:
            raise Exception("Concat name: %s len(input_shape.shape) != 2" % self.name)
        if input_shape.shape[1] != 4:           # [n * 4]
            raise Exception("Concat name: %s input_shape.shape[1] != 4" % self.name)
        if len(output_shape) != 4:
            raise Exception("Concat name: %s len(output_shape) != 4" % self.name)

        self.input_index = 0
        self.input_count = input_shape.shape[0]
        self.input_tensor = []
        self.output_tensor = None

    def add_input(self, input_tensor: CTensor):
        self.input_tensor.append(input_tensor)

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)
        self.output_tensor = output_tensor

    def forward(self):
        CExpression.forward(self)
        input_data = []
        input_diff = []
        for _tensor in self.input_tensor:
            input_data.append(_tensor.data)
            input_diff.append(_tensor.diff)

        self.output_tensor.data = np.concatenate(input_data, axis=1)
        self.output_tensor.diff = np.concatenate(input_diff, axis=1)

    def backward(self):
        if self.input_index == 0:
            CExpression.backward(self)
            dim_start = 0
            for _tensor in self.input_tensor:
                dim_end = dim_start + _tensor.shape[1]
                _tensor.data = self.output_tensor.data[:, dim_start:dim_end, :, :]
                _tensor.diff = self.output_tensor.diff[:, dim_start:dim_end, :, :]
                dim_start = dim_end

        self.input_index = (self.input_index + 1) % self.input_count
