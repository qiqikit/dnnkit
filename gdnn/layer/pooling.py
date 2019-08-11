import pyopencl as cl
from gdnn.model import GTensor
from gdnn.model import GExpression
from gdnn.model import GOptimizer
from gdnn.kernel.pooling import CsPooling

from functools import reduce


class MaxPooling(GExpression):
    def __init__(self, name: str, clc_pooling: CsPooling,
                 input_shape: tuple, output_shape: tuple,  ksize: int, stride: int):
        GExpression.__init__(self, name, input_shape, output_shape)

        self.input_type = 'float32'
        self.output_type = 'float32'
        if len(input_shape) != 4:
            raise Exception("MaxPooling name: %s's input_shape != 4d" % name)

        self.context = clc_pooling.program.context
        self.clc_pooling = clc_pooling

        self.ksize = ksize
        self.stride = stride
        self.batch_size = input_shape[0]

        _output_shape = (input_shape[0], input_shape[1], int((input_shape[2] - ksize + stride) / stride),
                         int((input_shape[3] - ksize + stride) / stride))

        if output_shape != _output_shape:
            raise Exception("MaxPooling name: %s output_shape != _output_shape" % self.name)

        self.output_len = reduce(lambda x, y: x * y, _output_shape[0:])

        self.init_argmax = False
        self.argmax = None

        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: GTensor, output_tensor: GTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if self.input_shape != input_tensor.shape:
            raise Exception("MaxPooling name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("MaxPooling name: %s self.output_shape != output_tensor.shape" % self.name)
        if self.input_type != input_tensor.dtype:
            raise Exception("MaxPooling name: %s self.input_type != input_tensor.dtype" % self.name)
        if self.output_type != output_tensor.dtype:
            raise Exception("MaxPooling name: %s self.output_type != output_tensor.dtype" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def init_layer(self, opt: GOptimizer):
        self.init_argmax = False

    def update_layer(self, opt: GOptimizer):
        pass

    def forward(self):
        GExpression.forward(self)

        # init_kernel onece (for gradient check)
        if not self.init_argmax:
            self.init_argmax = True
            self.argmax = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.output_len * 4)

            input_param = (self.ksize, self.ksize, self.stride, self.stride)
            self.clc_pooling.max_index(self.input_tensor.data, self.argmax, self.input_shape, input_param)

        self.clc_pooling.max_value(self.input_tensor.data, self.argmax, self.output_tensor.data, self.output_len)

    def backward(self):
        GExpression.backward(self)

        input_param = (self.ksize, self.ksize, self.stride, self.stride)
        input_len = reduce(lambda x, y: x * y, self.input_shape[0:])

        self.clc_pooling.clear_diff(self.input_tensor.diff, input_len)
        self.clc_pooling.dmax_value(self.output_tensor.diff, self.argmax, self.input_tensor.diff,
                                   self.output_shape, input_param)
        # self.clp_max_pooling.dmax2(self.output_tensor.diff, self.argmax, self.input_tensor.diff,
        # self.input_shape, input_param)
