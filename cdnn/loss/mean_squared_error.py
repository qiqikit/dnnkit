import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression


class MeanSquaredErrorTest(CExpression):
    def __init__(self,  name: str, predict_shape: tuple):
        CExpression.__init__(self, name, predict_shape, (1, ))

        if len(predict_shape) != 2:
            raise Exception("MeanSquaredErrorTest name: %s len(predict_shape) != 2" % self.name)
        self.batch_size = predict_shape[0]

        self.predict = None
        self.label = None
        self.loss = None

    def connect(self, predict: CTensor, loss: CTensor):
        self.link_parent(predict)
        self.link_child(loss)

        if self.input_shape != predict.shape:
            raise Exception("MeanSquaredErrorTest name: %s self.input_shape != predict.shape" % self.name)
        if self.output_shape != loss.shape:
            raise Exception("MeanSquaredErrorTest name: %s self.output_shape != loss.shape" % self.name)

        self.predict = predict
        self.loss = loss

    def forward(self):
        CExpression.forward(self)
        self.loss.data[0] = 0.0
        for k in range(self.batch_size):
            self.loss.data[0] += 0.5 * np.sum(np.square(self.predict.data[k] - self.label.data[k]))

    def backward(self):
        CExpression.backward(self)
        for k in range(self.batch_size):
            self.predict.diff[k] = self.predict.data[k] - self.label.data[k]


class MeanSquaredError(CExpression):
    def __init__(self,  name: str, predict_shape: tuple):
        CExpression.__init__(self, name, predict_shape, (1, ))

        self.batch_size = predict_shape[0]
        self.predict = None
        self.label = None
        self.loss = None

    def connect(self, predict: CTensor, loss: CTensor):
        self.link_parent(predict)
        self.link_child(loss)

        if self.input_shape != predict.shape:
            raise Exception("MeanSquaredError name: %s self.input_shape != predict.shape" % self.name)
        if self.output_shape != loss.shape:
            raise Exception("MeanSquaredError name: %s self.output_shape != loss.shape" % self.name)

        self.predict = predict
        self.loss = loss

    def forward(self):
        CExpression.forward(self)
        self.loss.data[0] = 0.5 * np.sum(np.square(self.predict.data - self.label.data))

    def backward(self):
        CExpression.backward(self)
        self.predict.diff = self.predict.data - self.label.data
