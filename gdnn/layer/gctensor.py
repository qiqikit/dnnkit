import numpy as np
import pyopencl as cl
import gdnn.platform as gpm

from cdnn.model import CTensor
from gdnn.model import GTensor
from gdnn.model import GExpression


class GCTensor(GExpression):
    def __init__(self, name: str, context: gpm.CsContext, queue: gpm.CsQueue,
                 input_shape: tuple, output_shape: tuple, data_type: str):

        GExpression.__init__(self, name, input_shape, output_shape)

        if input_shape != output_shape:
            raise Exception("GCTensor name: %s input_shape != output_shape" % self.name)

        self.data_type = data_type
        self.context = context
        self.queue = queue
        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: GTensor, output_tensor: CTensor):

        if self.input_shape != input_tensor.shape:
            raise Exception("GCTensor name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("GCTensor name: %s self.output_shape != output_tensor.shape" % self.name)
        if self.data_type != input_tensor.dtype:
            raise Exception("CGTensor name: %s self.data_type != input_tensor.dtype" % self.name)

        self.link_parent(input_tensor)
        self.link_child(output_tensor)
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def forward(self):
        GExpression.forward(self)
        if self.data_type == 'float32':
            output_data = np.zeros(self.output_shape, np.float32)
        elif self.data_type == 'float64':
            output_data = np.zeros(self.output_shape, np.float64)
        else:
            raise Exception('name: %s ,Unidentified input_type: %s' % (self.name, self.data_type))

        self.queue.copy(output_data, self.input_tensor.data)
        self.output_tensor.data = output_data.astype(np.float64)

    def backward(self):
        GExpression.backward(self)
        if self.data_type == 'float32':
            output_diff = self.output_tensor.diff.astype(np.float32)
        elif self.data_type == 'float64':
            output_diff = self.output_tensor.diff.astype(np.float64)
        else:
            raise Exception('name: %s ,Unidentified input_type: %s' % (self.name, self.data_type))

        self.input_tensor.diff = self.context.get_buffer(output_diff, cl.mem_flags.READ_ONLY)
