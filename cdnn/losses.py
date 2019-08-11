import cdnn.model as cmodel
from cdnn.loss.softmaxloss import SoftmaxWithLoss
from cdnn.loss.cross_entropy import CrossEntropyBinary
from cdnn.loss.cross_entropy import CrossEntropyCategorical
from cdnn.loss.mean_squared_error import MeanSquaredError


def softmax_withloss(name: str, input_tensor: cmodel.CTensor, label: cmodel.CTensor):
    loss_layer = SoftmaxWithLoss(name=name, input_shape=input_tensor.shape)
    loss_out = cmodel.CTensor(name=name + '-loss', shape=(1,), init='zero')

    loss_layer.connect(input_tensor, loss_out)

    loss_layer.link_parent(label)
    loss_layer.label = label

    return loss_layer, loss_out


def cross_entropy_binary(name: str, predict: cmodel.CTensor, label: cmodel.CTensor):
    loss_layer = CrossEntropyBinary(name=name, predict_shape=predict.shape)
    loss_out = cmodel.CTensor(name=name + '-loss', shape=(1,), init='zero')

    loss_layer.connect(predict, loss_out)

    loss_layer.link_parent(label)
    loss_layer.label = label

    return loss_layer, loss_out


def cross_entropy_categorical(name: str, predict: cmodel.CTensor, label: cmodel.CTensor):
    loss_layer = CrossEntropyCategorical(name=name, predict_shape=predict.shape)
    loss_out = cmodel.CTensor(name=name + '-loss', shape=(1,), init='zero')

    loss_layer.connect(predict, loss_out)

    loss_layer.link_parent(label)
    loss_layer.label = label

    return loss_layer, loss_out


def mean_squared_error(name: str, predict: cmodel.CTensor, label: cmodel.CTensor):
    loss_layer = MeanSquaredError(name=name, predict_shape=predict.shape)
    loss_out = cmodel.CTensor(name=name + '-loss', shape=(1,), init='zero')

    loss_layer.connect(predict, loss_out)

    loss_layer.link_parent(label)
    loss_layer.label = label

    return loss_layer, loss_out
