import cdnn.model as cmodel
from cdnn.layer.conv2d import Conv2d
from cdnn.layer.pooling import MaxPooling
from cdnn.layer.pooling import AvgPooling
from cdnn.layer.fully_connect import FullyConnect
from cdnn.layer.softmax import Softmax
from cdnn.layer.dropout import DropOut
from cdnn.layer.batch_normal import BatchNormal
from cdnn.layer.concat import Concat
from cdnn.layer.slice import Slice
from functools import reduce


# layers
def conv2d(name: str, input_tensor: cmodel.CTensor, kernel_shape: tuple, stride=1, padding='SAME'):

    batch_size = input_tensor.shape[0]
    ksize = kernel_shape[2]
    output_num = kernel_shape[0]
    input_shape = input_tensor.shape

    if padding == 'SAME':
        output_shape = (batch_size, output_num, int(input_shape[2] / stride), int(input_shape[3] / stride))
    elif padding == 'VALID':
        output_shape = (batch_size, output_num,
                        int((input_shape[2] - ksize + stride) / stride),
                        int((input_shape[3] - ksize + stride) / stride))
    else:
        raise Exception('Conv2dLayer name: %s Unidentified padding type: %s' % (name, padding))
    # output_shape = [int(i) for i in output_shape]
    _layer = Conv2d(name=name, input_shape=input_shape, output_shape=output_shape,
                    kernel_shape=kernel_shape, stride=stride, padding=padding)
    _out = cmodel.CTensor(name=name + '-out', shape=output_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def max_pooling(name: str, input_tensor: cmodel.CTensor, ksize=2, stride=2):

    batch_size = input_tensor.shape[0]
    input_shape = input_tensor.shape

    output_shape = (batch_size, input_shape[1],
                    int((input_shape[2] - ksize + stride) / stride),
                    int((input_shape[3] - ksize + stride) / stride))

    _layer = MaxPooling(name=name, input_shape=input_shape, output_shape=output_shape, ksize=ksize, stride=stride)
    _out = cmodel.CTensor(name=name + '-out', shape=output_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def avg_pooling(name: str, input_tensor: cmodel.CTensor, ksize=2, stride=2):

    batch_size = input_tensor.shape[0]
    input_shape = input_tensor.shape

    output_shape = (batch_size, input_shape[1],
                    int((input_shape[2] - ksize + stride) / stride),
                    int((input_shape[3] - ksize + stride) / stride))

    _layer = AvgPooling(name=name, input_shape=input_shape, output_shape=output_shape, ksize=ksize, stride=stride)
    _out = cmodel.CTensor(name=name + '-out', shape=output_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def full_connect(name: str, input_tensor: cmodel.CTensor, output_num: int):
    batch_size = input_tensor.shape[0]
    input_shape = input_tensor.shape
    output_shape = (batch_size, output_num)

    _layer = FullyConnect(name=name, input_shape=input_shape, output_shape=output_shape, output_num=output_num)
    _out = cmodel.CTensor(name=name + '-out', shape=output_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def dropout(name: str, input_tensor: cmodel.CTensor, phase='train', prob=0.5):
    input_shape = output_shape = input_tensor.shape
    _layer = DropOut(name=name, input_shape=input_shape, output_shape=output_shape, phase=phase, prob=prob)
    _out = cmodel.CTensor(name=name + '-out', shape=output_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def softmax(name: str, input_tensor: cmodel.CTensor):
    input_shape = input_tensor.shape
    input_len = reduce(lambda x, y: x * y, input_shape[1:])
    output_shape = (input_shape[0], input_len)
    _layer = Softmax(name=name, input_shape=input_shape, output_shape=output_shape)
    _out = cmodel.CTensor(name=name + '-out', shape=output_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def batch_normal(name: str, input_tensor: cmodel.CTensor):
    input_shape = output_shape = input_tensor.shape
    _layer = BatchNormal(name=name, input_shape=input_shape, output_shape=output_shape)
    _out = cmodel.CTensor(name=name + '-out', shape=output_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def concat_(name: str, input_tensor: tuple):

    input_shape = []
    batch = 0
    channel = 0
    height = 0
    width = 0

    for _tensor in input_tensor:
        batch = _tensor.shape[0]
        channel += _tensor.shape[1]
        height = _tensor.shape[2]
        width = _tensor.shape[3]
        input_shape.append(_tensor.shape)

    output_shape = (batch, channel, height, width)

    _layer = Concat(name=name, input_shape=tuple(input_shape), output_shape=output_shape)
    _out = cmodel.CTensor(name=name + '-out', shape=output_shape, init='zero')

    for _tensor in input_tensor:
        _layer.add_input(_tensor)
        _layer.connect(_tensor, _out)

    return _layer, _out


def slice_(name: str, input_tensor: cmodel.CTensor, output_shape: tuple):
    _layer = Slice(name=name, input_shape=input_tensor.shape, output_shape=output_shape)

    sum_channel = 0
    out_list = []
    out_index = 0
    for _shape in output_shape:
        sum_channel += _shape[1]
        out_index += 1
        _out = cmodel.CTensor(name=name + '-out_' + str(out_index), shape=_shape, init='zero')
        _layer.add_output(_out)
        _layer.connect(input_tensor=input_tensor, output_tensor=_out)
        out_list.append(_out)

    if sum_channel != input_tensor.shape[1]:
        raise Exception('sum_channel != input_tensor.shape[1]')
    return _layer, tuple(out_list)
