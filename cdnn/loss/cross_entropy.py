import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression


# CrossEntropy for binary classifications
class CrossEntropyBinaryTest(CExpression):
    def __init__(self,  name: str, predict_shape: tuple):
        CExpression.__init__(self, name, predict_shape, (1, ))

        if len(predict_shape) != 2:
            raise Exception("CrossEntropyBinaryTest name: %s len(predict_shape) != 2" % self.name)
        self.batch_size = predict_shape[0]

        self.predict = None
        self.label = None
        self.loss = None

    def connect(self, predict: CTensor, loss: CTensor):
        self.link_parent(predict)
        self.link_child(loss)

        if self.input_shape != predict.shape:
            raise Exception("CrossEntropyBinaryTest name: %s self.input_shape != predict.shape" % self.name)
        if self.output_shape != loss.shape:
            raise Exception("CrossEntropyBinaryTest name: %s self.output_shape != loss.shape" % self.name)

        self.predict = predict
        self.loss = loss

    def forward(self):
        CExpression.forward(self)
        self.loss.data[0] = 0.0
        for k in range(self.batch_size):
            # d = -y[i] * log(p[i]) - (1 - y[i]) * log(1 - p[i])
            self.loss.data[0] += -self.label.data[k] * np.log(self.predict.data[k]) -\
                                      (1.0 - self.label.data[k]) * np.log(1 - self.predict.data[k])

    def backward(self):
        CExpression.backward(self)
        for k in range(self.batch_size):
            # (y[i] - t[i]) / (y[i] * (float_t(1) - y[i]));
            self.predict.diff[k] = (self.predict.data[k] - self.label.data[k]) /\
                                   (self.predict.data[k] * (1 - self.predict.data[k]))


class CrossEntropyBinary(CExpression):
    def __init__(self,  name: str, predict_shape: tuple):
        CExpression.__init__(self, name, predict_shape, (1, ))

        if len(predict_shape) != 2:
            raise Exception("CrossEntropyBinary name: %s len(predict_shape) != 2" % self.name)
        self.batch_size = predict_shape[0]

        self.predict = None
        self.label = None
        self.loss = None

    def connect(self, predict: CTensor, loss: CTensor):
        self.link_parent(predict)
        self.link_child(loss)

        if self.input_shape != predict.shape:
            raise Exception("CrossEntropyBinaryTest name: %s self.input_shape != predict.shape" % self.name)
        if self.output_shape != loss.shape:
            raise Exception("CrossEntropyBinaryTest name: %s self.output_shape != loss.shape" % self.name)

        self.predict = predict
        self.loss = loss

    def forward(self):
        CExpression.forward(self)

        log_predict1 = np.log(self.predict.data)
        # loss_positive = -np.sum(np.dot(self.label.data, log_predict1.transpose()) * np.eye(self.batch_size))
        loss_positive = -np.sum(self.label.data.reshape(-1) * log_predict1.reshape(-1))

        log_predict2 = np.log(1 - self.predict.data)
        # loss_negative = -np.sum(np.dot(1 - self.label.data, 1 - log_predict2.transpose()) * np.eye(self.batch_size))
        loss_negative = -np.sum((1 - self.label.data.reshape(-1)) * (1 - log_predict2.reshape(-1)))

        self.loss.data[0] = loss_positive + loss_negative

    def backward(self):
        CExpression.backward(self)
        # (y[i] - t[i]) / (y[i] * (float_t(1) - y[i]));
        self.predict.diff = (self.predict.data - self.label.data) / (self.predict.data * (1 - self.predict.data))


# CrossEntropy for multi classifications
class CrossEntropyCategoricalTest(CExpression):
    def __init__(self,  name: str, predict_shape: tuple, loss_shape=(1, )):
        CExpression.__init__(self, name, predict_shape, loss_shape)

        if len(predict_shape) != 2:
            raise Exception("CrossEntropyCategoricalTest name: %s len(predict_shape) != 2" % self.name)
        self.batch_size = predict_shape[0]

        self.predict = None
        self.label = None
        self.loss = None

    def connect(self, predict: CTensor, loss: CTensor):
        self.link_parent(predict)
        self.link_child(loss)

        if self.input_shape != predict.shape:
            raise Exception("CrossEntropyBinaryTest name: %s self.input_shape != predict.shape" % self.name)
        if self.output_shape != loss.shape:
            raise Exception("CrossEntropyBinaryTest name: %s self.output_shape != loss.shape" % self.name)

        self.predict = predict
        self.loss = loss

    def forward(self):
        CExpression.forward(self)
        self.loss.data[0] = 0.0

        for k in range(self.batch_size):
            # d = sum(-y[i] * log(p[i]))
            self.loss.data[0] += np.sum(-self.label.data[k] * np.log(self.predict.data[k]))

    def backward(self):
        CExpression.backward(self)
        for k in range(self.batch_size):
            # d[i] = - y[i] / p[i]
            self.predict.diff[k] = -self.label.data[k] / self.predict.data[k]


class CrossEntropyCategorical(CExpression):
    def __init__(self,  name: str, predict_shape: tuple, loss_shape=(1, )):
        CExpression.__init__(self, name, predict_shape, loss_shape)

        if len(predict_shape) != 2:
            raise Exception("CrossEntropyCategoricalTest name: %s len(predict_shape) != 2" % self.name)
        self.batch_size = predict_shape[0]

        self.predict = None
        self.label = None
        self.loss = None

    def connect(self, predict: CTensor, loss: CTensor):
        self.link_parent(predict)
        self.link_child(loss)

        if self.input_shape != predict.shape:
            raise Exception("CrossEntropyBinaryTest name: %s self.input_shape != predict.shape" % self.name)
        if self.output_shape != loss.shape:
            raise Exception("CrossEntropyBinaryTest name: %s self.output_shape != loss.shape" % self.name)

        self.predict = predict
        self.loss = loss

    def forward(self):
        CExpression.forward(self)
        # d = sum(-y[i] * log(p[i]))
        log_predict = np.log(self.predict.data)
        # loss_batch = np.dot(self.label.data, log_predict.transpose())
        # self.loss.data[0] = -np.sum(loss_batch * np.eye(self.batch_size))

        self.loss.data[0] = -np.sum(self.label.data.reshape(-1) * log_predict.reshape(-1))

    def backward(self):
        CExpression.backward(self)
        # d[i] = - y[i] / p[i]
        self.predict.diff = -self.label.data / self.predict.data
