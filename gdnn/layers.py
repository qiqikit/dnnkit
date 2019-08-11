import gdnn.platform as gpm
import cdnn.model as cmodel
import gdnn.model as gmodel

from gdnn.layer.cgtensor import CGTensor
from gdnn.layer.gctensor import GCTensor
from gdnn.layer.pooling import MaxPooling
from gdnn.kernel.pooling import CsPooling


def array_to_buffer(name: str, context: gpm.CsContext, queue: gpm.CsQueue, input_tensor: cmodel.CTensor):

    input_shape = output_shape = input_tensor.shape
    _layer = CGTensor(name=name, context=context, queue=queue,
                      input_shape=input_shape, output_shape=output_shape, data_type='float32')
    _out = gmodel.GTensor(name=name + '-gout', context=context, shape=output_shape, dtype='float32', init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def buffer_to_array(name: str, context: gpm.CsContext, queue: gpm.CsQueue, input_tensor: gmodel.GTensor):

    input_shape = output_shape = input_tensor.shape
    _layer = GCTensor(name=name, context=context, queue=queue,
                      input_shape=input_shape, output_shape=output_shape, data_type='float32')
    _out = cmodel.CTensor(name=name + '-cout', shape=output_shape, init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out


def max_pooling(name: str, context: gpm.CsContext, clc_pooling: CsPooling,
                input_tensor: gmodel.GTensor, ksize=2, stride=2):

    batch_size = input_tensor.shape[0]
    input_shape = input_tensor.shape
    output_shape = (batch_size, input_shape[1],
                    int((input_shape[2] - ksize + stride) / stride),
                    int((input_shape[3] - ksize + stride) / stride))

    _layer = MaxPooling(name=name, clc_pooling=clc_pooling,
                        input_shape=input_shape, output_shape=output_shape, ksize=ksize, stride=stride)
    _out = gmodel.GTensor(name=name + '-out', context=context, shape=output_shape, dtype='float32', init='zero')

    _layer.connect(input_tensor, _out)
    return _layer, _out
