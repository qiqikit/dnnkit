import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression
from cdnn.model import COptimizer
from functools import reduce


class FullyConnect(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, output_num=0):
        CExpression.__init__(self, name, input_shape, output_shape)

        self.batch_size = input_shape[0]
        input_len = reduce(lambda x, y: x * y, input_shape[1:])
        self.output_num = output_num
        self.weights = CTensor(name=self.name + '-weights', shape=(input_len, self.output_num), init='msra')
        self.bias = CTensor(name=self.name + '-bias', shape=(self.output_num,), init='msra')

        _output_shape = (self.batch_size, self.output_num)

        if output_shape != _output_shape:
            raise Exception("FullyConnect name: %s output_shape != _output_shape" % self.name)

        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if self.input_shape != input_tensor.shape:
            raise Exception("FullyConnect name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("FullyConnect name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def init_layer(self, opt: COptimizer):
        opt.init(var=self.weights)
        opt.init(var=self.bias)

    def update_layer(self, opt: COptimizer):
        opt.apply(var=self.weights, batch_size=self.batch_size)
        opt.apply(var=self.bias, batch_size=self.batch_size)

    def forward(self):
        CExpression.forward(self)
        input_flatten = self.input_tensor.data.reshape([self.batch_size, -1])
        self.output_tensor.data = np.dot(input_flatten, self.weights.data) + self.bias.data

    def backward(self):
        CExpression.backward(self)
        self.bias.diff *= 0
        self.weights.diff *= 0
        input_flatten = self.input_tensor.data.reshape([self.batch_size, -1])

        for i in range(self.batch_size):
            col_x = input_flatten[i][:, np.newaxis]
            diff_i = self.output_tensor.diff[i][np.newaxis, :]
            self.weights.diff += np.dot(col_x, diff_i)
            self.bias.diff += diff_i.reshape(self.bias.shape)
        next_diff = np.dot(self.output_tensor.diff, self.weights.data.T)
        self.input_tensor.diff = np.reshape(next_diff, self.input_tensor.shape)

    def load(self, file_path: str):
        self.weights.data = np.load(file_path + self.weights.name + '.npy')
        if self.weights.shape != self.weights.data.shape:
            raise Exception("FullyConnect name: %s self.shape != self.weights.data.shape" % self.name)
        self.bias.data = np.load(file_path + self.bias.name + '.npy')
        if self.bias.shape != self.bias.data.shape:
            raise Exception("FullyConnect name: %s self.bias.shape != self.bias.data.shape" % self.name)

    def save(self, file_path: str):
        np.save(file_path + self.weights.name + '.npy', self.weights.data)
        np.save(file_path + self.bias.name + '.npy', self.bias.data)
