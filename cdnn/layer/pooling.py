import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression
from cdnn.model import COptimizer
from functools import reduce


class MaxPooling(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, ksize: int, stride: int):
        CExpression.__init__(self, name, input_shape, output_shape)
        assert (stride == ksize)  # only the stride = ksize case is handled

        if len(input_shape) != 4:
            raise Exception("MaxPooling name: %s's input_shape != 4d" % name)

        self.ksize = ksize
        self.stride = stride
        self.batch_size = input_shape[0]

        self.init_argmax = False
        self.argmax = None

        _output_shape = (input_shape[0], input_shape[1], int((input_shape[2] - ksize + stride) / stride),
                         int((input_shape[3] - ksize + stride) / stride))

        if output_shape != _output_shape:
            raise Exception("MaxPooling name: %s output_shape != _output_shape" % self.name)

        self.output_len = reduce(lambda x, y: x * y, _output_shape[0:])
        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if self.input_shape != input_tensor.shape:
            raise Exception("MaxPooling name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("MaxPooling name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def init_layer(self, opt: COptimizer):
        self.init_argmax = False

    def update_layer(self, opt: COptimizer):
        pass

    def forward(self):
        CExpression.forward(self)

        input_col = np.zeros([self.output_len, self.ksize * self.ksize])
        index_col = 0
        for batch in range(self.output_tensor.shape[0]):
            for channel in range(self.output_tensor.shape[1]):
                for row in range(self.output_tensor.shape[2]):
                    for col in range(self.output_tensor.shape[3]):
                        _roi = self.input_tensor.data[batch, channel,
                               row * self.stride:row * self.stride + self.ksize,
                               col * self.stride:col * self.stride + self.ksize]
                        input_col[index_col] = _roi.reshape(-1)
                        index_col += 1

        # init_kernel onece (for gradient check)
        if not self.init_argmax:
            self.init_argmax = True
            self.argmax = np.argmax(input_col, axis=1)

        # max_value = np.zeros(self.output_len)
        # for i in range(max_value.shape[0]):
        #     max_value[i] = input_col[i, self.argmax[i]]
        max_index = self.argmax + np.arange(0, self.output_len) * self.ksize*self.ksize
        max_value = input_col.reshape(-1)[max_index]

        self.output_tensor.data = max_value.reshape(self.output_tensor.shape)

    def backward(self):
        CExpression.backward(self)
        self.input_tensor.diff *= 0

        max_index_shape = (self.input_shape[0], self.input_shape[1],
                           int((self.input_shape[2] - self.ksize + self.stride) / self.stride),
                           int((self.input_shape[3] - self.ksize + self.stride) / self.stride),
                           self.ksize * self.ksize)

        max_index_col = np.zeros([self.output_len, self.ksize * self.ksize])
        for k in range(self.argmax.shape[0]):
            max_index_col[k, self.argmax[k]] = 1
        max_index = max_index_col.reshape(max_index_shape)

        for batch in range(max_index.shape[0]):
            for channel in range(max_index.shape[1]):
                for row in range(max_index.shape[2]):
                    for col in range(max_index.shape[3]):
                        for index in range(max_index.shape[4]):
                            if max_index[batch, channel, row, col, index] == 1:
                                row_index = row * self.stride + int(index / self.ksize)
                                col_index = col * self.stride + int(index % self.ksize)

                                out_diff = self.output_tensor.diff[batch, channel, row, col]
                                self.input_tensor.diff[batch, channel, row_index, col_index] += out_diff


class AvgPooling(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, ksize: int, stride: int):
        CExpression.__init__(self, name, input_shape, output_shape)

        if len(input_shape) != 4:
            raise Exception("MaxPooling name: %s's input_shape != 4d" % name)

        self.ksize = ksize
        self.stride = stride
        self.batch_size = input_shape[0]

        _output_shape = (input_shape[0], input_shape[1], int((input_shape[2] - ksize + stride) / stride),
                         int((input_shape[3] - ksize + stride) / stride))

        if output_shape != _output_shape:
            raise Exception("MaxPooling name: %s output_shape != _output_shape" % self.name)

        self.output_len = reduce(lambda x, y: x * y, _output_shape[0:])
        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if self.input_shape != input_tensor.shape:
            raise Exception("MaxPooling name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("MaxPooling name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def forward(self):
        CExpression.forward(self)

        input_col = np.zeros([self.output_len, self.ksize * self.ksize])
        index_col = 0
        for batch in range(self.output_tensor.shape[0]):
            for channel in range(self.output_tensor.shape[1]):
                for row in range(self.output_tensor.shape[2]):
                    for col in range(self.output_tensor.shape[3]):
                        _roi = self.input_tensor.data[batch, channel,
                               row * self.stride:row * self.stride + self.ksize,
                               col * self.stride:col * self.stride + self.ksize]
                        input_col[index_col] = _roi.reshape(-1)
                        index_col += 1

        mean_value = np.mean(input_col, axis=1)
        self.output_tensor.data = mean_value.reshape(self.output_tensor.shape)

    def backward(self):
        CExpression.backward(self)
        self.input_tensor.diff *= 0

        # self.input_tensor.diff = self.output_tensor.diff / self.ksize
        for batch in range(self.output_tensor.shape[0]):
            for channel in range(self.output_tensor.shape[1]):
                for row in range(self.output_tensor.shape[2]):
                    for col in range(self.output_tensor.shape[3]):
                        for index in range(self.ksize * self.ksize):
                            row_index = row * self.stride + int(index / self.ksize)
                            col_index = col * self.stride + int(index % self.ksize)

                            out_diff = self.output_tensor.diff[batch, channel, row, col]
                            self.input_tensor.diff[batch, channel, row_index, col_index] += out_diff / (
                                        self.ksize * self.ksize)
