import numpy as np
import pyopencl as cl
import gdnn.platform as gpm

from cdnn.model import CTensor
from gdnn.model import GTensor
from gdnn.model import GExpression


class CGTensor(GExpression):
    def __init__(self, name: str, context: gpm.CsContext, queue: gpm.CsQueue,
                 input_shape: tuple, output_shape: tuple, data_type: str):

        GExpression.__init__(self, name, input_shape, output_shape)

        if input_shape != output_shape:
            raise Exception("CGTensor name: %s input_shape != output_shape" % self.name)
        self.data_type = data_type

        self.context = context
        self.queue = queue
        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: GTensor):

        if self.input_shape != input_tensor.shape:
            raise Exception("CGTensor name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("CGTensor name: %s self.output_shape != output_tensor.shape" % self.name)
        if self.data_type != output_tensor.dtype:
            raise Exception("CGTensor name: %s self.data_type != output_tensor.dtype" % self.name)

        self.link_parent(input_tensor)
        self.link_child(output_tensor)
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def forward(self):
        GExpression.forward(self)
        if self.data_type == 'float32':
            input_data = self.input_tensor.data.astype(np.float32)
        elif self.data_type == 'float64':
            input_data = self.input_tensor.data.astype(np.float64)
        else:
            raise Exception('name: %s ,Unidentified output_type: %s' % (self.name, self.data_type))

        self.output_tensor.data = self.context.get_buffer(input_data, cl.mem_flags.READ_WRITE)

    def backward(self):
        GExpression.backward(self)
        if self.data_type == 'float32':
            input_diff = np.zeros(self.input_shape, np.float32)
        elif self.data_type == 'float64':
            input_diff = np.zeros(self.input_shape, np.float64)
        else:
            raise Exception('name: %s ,Unidentified output_type: %s' % (self.name, self.data_type))

        self.queue.copy(input_diff, self.output_tensor.diff)
        self.input_tensor.diff = input_diff
