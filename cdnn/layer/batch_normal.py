import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression
from cdnn.model import COptimizer
from functools import reduce


class BatchNormal(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple):
        CExpression.__init__(self, name, input_shape, output_shape)

        if input_shape != output_shape:
            raise Exception("BatchNormal name: %s input_shape != output_shape" % self.name)

        self.batch_size = input_shape[0]
        self.epsilon = 0.00001
        self.momentum = 0.9

        self.phase = 'train'
        self.input_len = reduce(lambda x, y: x * y, input_shape[1:])
        self.input_mean = np.zeros([1, self.input_len])
        self.sigma_square = np.zeros([1, self.input_len])
        self.standard_normal = np.zeros([self.batch_size, self.input_len])

        self.gamma = CTensor(name=self.name + '-weights(gama)', shape=(self.input_len, ), init='msra')
        self.beta = CTensor(name=self.name + '-bias(beta)', shape=(self.input_len, ), init='msra')
        self.test_mean = CTensor(name=self.name + '-test_mean', shape=(self.input_len, ), init='zero')
        self.test_sigma_square = CTensor(name=self.name + '-test_sigma_square', shape=(self.input_len, ), init='zero')

        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if input_tensor.shape != output_tensor.shape:
            raise Exception("BatchNormal name: %s input_tensor.shape != output_tensor.shape" % self.name)
        if self.input_shape != input_tensor.shape:
            raise Exception("BatchNormal name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("BatchNormal name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def init_layer(self, opt: COptimizer):
        opt.init(var=self.gamma)
        opt.init(var=self.beta)

    def update_layer(self, opt: COptimizer):
        opt.apply(var=self.gamma, batch_size=self.input_len)
        opt.apply(var=self.beta, batch_size=self.input_len)

    def forward(self):
        CExpression.forward(self)

        input_data = self.input_tensor.data.reshape(self.batch_size, self.input_len)

        if self.phase == 'train':
            self.input_mean = np.mean(input_data, axis=(0,))
            self.sigma_square = np.var(input_data, 0)
            self.test_mean.data = self.momentum * self.test_mean.data + (1 - self.momentum) * self.input_mean
            self.test_sigma_square.data = self.momentum * self.test_sigma_square.data + (
                        1 - self.momentum) * self.sigma_square
        elif self.phase == 'test':
            self.input_mean = self.test_mean.data
            self.sigma_square = self.test_sigma_square.data
        else:
            raise Exception('BatchNormal name: %s phase is not in test or train' % self.name)

        self.standard_normal = (input_data - self.input_mean) / np.sqrt(self.sigma_square + self.epsilon)

        output_data = self.gamma.data * self.standard_normal + self.beta.data
        self.output_tensor.data = output_data.reshape(self.input_shape)

    def backward(self):
        CExpression.backward(self)

        input_data = self.input_tensor.data.reshape(self.batch_size, self.input_len)
        output_diff = self.output_tensor.diff.reshape(self.batch_size, self.input_len)

        self.beta.diff = np.sum(output_diff, axis=0)
        self.gamma.diff = np.sum(output_diff * self.standard_normal, axis=0)

        d_input_hat = output_diff * self.gamma.data

        d_sigma_square = np.sum(d_input_hat * (input_data - self.input_mean), axis=0) * -0.5 * np.power(
            self.sigma_square + self.epsilon, -1.5)

        # d_input_mean = np.sum(d_input_hat * -1 / np.sqrt(self.sigma_square + self.epsilon), axis=0)
        d_input_mean = - np.sum(d_input_hat / np.sqrt(self.sigma_square + self.epsilon),
                                axis=0) - 2 * d_sigma_square * np.sum(input_data - self.input_mean,
                                                                      axis=0) / self.batch_size

        d_input = d_input_hat / np.sqrt(self.sigma_square + self.epsilon) + 2.0 * d_sigma_square * (
                    input_data - self.input_mean) / self.batch_size + d_input_mean / self.batch_size

        self.input_tensor.diff = d_input.reshape(self.input_tensor.shape)

    def load(self, file_path: str):
        self.gamma.data = np.load(file_path + self.gamma.name + '.npy')
        if self.gamma.shape != self.gamma.data.shape:
            raise Exception("BatchNormal name: %s self.gamma.shape != self.gamma.data.shape" % self.name)
        self.beta.data = np.load(file_path + self.beta.name + '.npy')
        if self.beta.shape != self.beta.data.shape:
            raise Exception("BatchNormal name: %s self.beta.shape != self.beta.data.shape" % self.name)

        self.test_mean.data = np.load(file_path + self.test_mean.name + '.npy')
        if self.test_mean.shape != self.test_mean.data.shape:
            raise Exception("BatchNormal name: %s self.test_mean.shape != self.test_mean.data.shape" % self.name)
        self.test_sigma_square.data = np.load(file_path + self.test_sigma_square.name + '.npy')
        if self.test_sigma_square.shape != self.test_sigma_square.data.shape:
            raise Exception(
                "BatchNormal name: %s self.test_sigma_square.shape != self.test_sigma_square.data.shape" % self.name)

    def save(self, file_path: str):
        np.save(file_path + self.gamma.name + '.npy', self.gamma.data)
        np.save(file_path + self.beta.name + '.npy', self.beta.data)
        np.save(file_path + self.test_mean.name + '.npy', self.test_mean.data)
        np.save(file_path + self.test_sigma_square.name + '.npy', self.test_sigma_square.data)
