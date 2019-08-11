import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression
from functools import reduce


# CExpression class
class Softmax(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple):
        CExpression.__init__(self, name, input_shape, output_shape)
        self.batch_size = input_shape[0]
        self.input_len = reduce(lambda x, y: x * y, input_shape[1:])

        if input_shape[0] != output_shape[0]:
            raise Exception("Softmax name: %s input_shape[0] != output_shape[0]" % self.name)
        if self.input_len != output_shape[1]:
            raise Exception("Softmax name: %s self.input_len != output_shape[1]" % self.name)

        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if self.input_shape != input_tensor.shape:
            raise Exception("Softmax name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("Softmax name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def forward(self):
        CExpression.forward(self)
        input_data = self.input_tensor.data.reshape(self.batch_size, self.input_len)
        input_max = np.max(input_data, axis=1)
        input_max = input_max.reshape(self.batch_size, 1)
        exp_prediction = np.exp(input_data - input_max)

        exp_prediction_sum = np.sum(exp_prediction, axis=1)
        softmax = exp_prediction / exp_prediction_sum.reshape(self.batch_size, 1)

        self.output_tensor.data = softmax

    def backward(self):
        CExpression.backward(self)
        diff = np.zeros([self.batch_size, self.input_len])

        # softmax_diff_ii = np.zeros([self.batch_size, self.input_len, self.input_len])
        # softmax_diff_ii_value = self.output_tensor.data * (1 - self.output_tensor.data)
        # for k in range(self.batch_size):
        #     for i in range(self.input_len):
        #         softmax_diff_ii[k, i, i] = softmax_diff_ii_value[k, i]
        #
        # softmax_diff_ij = np.zeros([self.batch_size, self.input_len, self.input_len])
        # for k in range(self.batch_size):
        #     softmax_diff_ij[k] = -np.dot(self.output_tensor.data[k].reshape(-1, 1),
        #                               self.output_tensor.data[k].reshape(1, -1))
        #
        # eye = np.repeat(np.eye(self.input_len), self.batch_size, axis=1)
        # eye = np.swapaxes(eye.reshape(self.input_len, self.input_len,self.batch_size), 0, 2)
        #
        # softmax_diff = softmax_diff_ij * -1 * (eye - 1) + softmax_diff_ii
        #
        # for k in range(self.batch_size):
        #     diff[k] = softmax_diff[k].dot(self.output_tensor.diff[k])

        for k in range(self.batch_size):
            softmax_diff_ii = np.zeros([self.input_len, self.input_len])
            for i in range(self.input_len):
                softmax_diff_ii[i, i] = self.output_tensor.data[k, i] * (1 - self.output_tensor.data[k, i])

            softmax_diff_ij = -np.dot(self.output_tensor.data[k].reshape(-1, 1),
                                      self.output_tensor.data[k].reshape(1, -1))
            softmax_diff = -1 * (np.eye(self.input_len) - 1) * softmax_diff_ij + softmax_diff_ii

            diff[k] = softmax_diff.dot(self.output_tensor.diff[k])

        self.input_tensor.diff = diff.reshape(self.input_shape)
