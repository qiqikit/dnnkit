import cdnn.model as cmodel
from cdnn.activation.relu import Relu
from cdnn.activation.relu import LRelu
from cdnn.activation.sigmoid import Sigmoid
from cdnn.activation.log import Log
from cdnn.activation.tanh import Tanh


# activation
def relu(name: str, input_tensor: cmodel.CTensor):
    input_shape = input_tensor.shape
    _layer = Relu(name=name, input_shape=input_shape, output_shape=input_shape)
    _out = cmodel.CTensor(name=name + '-out', shape=input_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def lrelu(name: str, input_tensor: cmodel.CTensor, alpha: float):
    input_shape = input_tensor.shape
    _layer = LRelu(name=name, input_shape=input_shape, output_shape=input_shape, alpha=alpha)
    _out = cmodel.CTensor(name=name + '-out', shape=input_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def sigmoid(name: str, input_tensor: cmodel.CTensor):
    input_shape = input_tensor.shape
    _layer = Sigmoid(name=name, input_shape=input_shape, output_shape=input_shape)
    _out = cmodel.CTensor(name=name + '-out', shape=input_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def log(name: str, input_tensor: cmodel.CTensor):
    input_shape = input_tensor.shape
    _layer = Log(name=name, input_shape=input_shape, output_shape=input_shape)
    _out = cmodel.CTensor(name=name + '-out', shape=input_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def tanh(name: str, input_tensor: cmodel.CTensor):
    input_shape = input_tensor.shape
    _layer = Tanh(name=name, input_shape=input_shape, output_shape=input_shape)
    _out = cmodel.CTensor(name=name + '-out', shape=input_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out
