import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression


# CExpression class
class SoftmaxWithLossTest(CExpression):
    def __init__(self,  name: str, input_shape: tuple):
        CExpression.__init__(self, name, input_shape, (1, ))

        self.batch_size = input_shape[0]

        self.input_tensor = None
        self.label = None
        self.loss = None
        self.predict = None

    def connect(self, input_tensor: CTensor, loss: CTensor):
        self.link_parent(input_tensor)
        self.link_child(loss)

        if self.input_shape != input_tensor.shape:
            raise Exception("SoftmaxWithLoss name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != loss.shape:
            raise Exception("SoftmaxWithLoss name: %s self.output_shape != loss.shape" % self.name)

        self.input_tensor = input_tensor
        self.loss = loss

    def forward(self):
        CExpression.forward(self)
        input_data = self.input_tensor.data.reshape(self.batch_size, -1)
        exp_prediction = np.zeros(input_data.shape)

        softmax = np.zeros(input_data.shape)
        for i in range(self.batch_size):
            input_max = np.max(input_data[i])
            exp_prediction[i] = np.exp(input_data[i] - input_max / 2.0)
            softmax[i] = exp_prediction[i] / np.sum(exp_prediction[i])

        self.predict = softmax
        self.loss.data[0] = 0.0
        for i in range(self.batch_size):
            self.loss.data[0] += -np.sum(self.label.data[i] * np.log(self.predict[i]))

    def backward(self):
        CExpression.backward(self)
        diff = self.predict - self.label.data
        self.input_tensor.diff = diff.reshape(self.input_shape)


class SoftmaxWithLoss(CExpression):
    def __init__(self,  name: str, input_shape: tuple):
        CExpression.__init__(self, name, input_shape, (1, ))

        self.batch_size = input_shape[0]

        self.input_tensor = None
        self.predict = None
        self.label = None
        self.loss = None

    def connect(self, input_tensor: CTensor, loss: CTensor):
        self.link_parent(input_tensor)
        self.link_child(loss)

        if self.input_shape != input_tensor.shape:
            raise Exception("SoftmaxWithLoss name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != loss.shape:
            raise Exception("SoftmaxWithLoss name: %s self.output_shape != loss.shape" % self.name)

        self.input_tensor = input_tensor
        self.loss = loss

    def forward(self):
        CExpression.forward(self)
        input_data = self.input_tensor.data.reshape(self.batch_size, -1)

        input_max = np.max(input_data, axis=1)
        exp_predict = np.exp(input_data - input_max.reshape(self.batch_size, 1) / 2.0)
        exp_predict_sum = np.sum(exp_predict, axis=1)
        softmax = exp_predict / exp_predict_sum.reshape(self.batch_size, 1)

        self.predict = softmax
        log_predict = np.log(self.predict)
        # loss_batch = np.dot(self.label.data, log_predict.transpose())
        # self.loss.data[0] = -np.sum(loss_batch * np.eye(self.batch_size))
        self.loss.data[0] = -np.sum(self.label.data.reshape(-1) * log_predict.reshape(-1))

    def backward(self):
        CExpression.backward(self)
        diff = self.predict - self.label.data
        self.input_tensor.diff = diff.reshape(self.input_shape)
