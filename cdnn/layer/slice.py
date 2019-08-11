import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression


class Slice(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple):
        CExpression.__init__(self, name, input_shape, output_shape)
        output_shape = np.array(output_shape)

        if len(input_shape) != 4:
            raise Exception('Concat name: %s len(input_shape) != 4' % self.name)
        if len(output_shape.shape) != 2:        # output_shape = [n * 4]
            raise Exception('Concat name: %s len(output_shape.shape) != 2' % self.name)
        if output_shape.shape[1] != 4:          # output_shape = [n * 4]
            raise Exception('Concat name: %s output_shape.shape[1] != 4' % self.name)

        self.output_index = 0
        self.output_count = output_shape.shape[0]
        self.input_tensor = None
        self.output_tensor = []

    def add_output(self, output_tensor: CTensor):
        # self.append(self, self.parents, input_tensor)
        self.output_tensor.append(output_tensor)

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)
        self.input_tensor = input_tensor

    def forward(self):
        if self.output_index == 0:
            CExpression.forward(self)
            dim_start = 0

            for _tensor in self.output_tensor:
                dim_end = dim_start + _tensor.shape[1]
                _tensor.data = self.input_tensor.data[:, dim_start:dim_end, :, :]
                _tensor.diff = self.input_tensor.diff[:, dim_start:dim_end, :, :]
                dim_start = dim_end

        self.output_index = (self.output_index + 1) % self.output_count

    def backward(self):
        CExpression.backward(self)
        output_data = []
        output_diff = []
        for _tensor in self.output_tensor:
            output_data.append(_tensor.data)
            output_diff.append(_tensor.diff)

        self.input_tensor.data = np.concatenate(tuple(output_data), axis=1)
        self.input_tensor.diff = np.concatenate(tuple(output_diff), axis=1)
