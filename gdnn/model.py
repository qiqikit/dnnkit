import math
import numpy as np
import pyopencl as cl
from functools import reduce
from cdnn.model import Node
import gdnn.platform as gpm


class GTensor(Node):
    def __init__(self, name: str, context: gpm.CsContext, shape: tuple, dtype: str, init: str):
        Node.__init__(self, name)

        self.context = context
        self.shape = shape
        self.dtype = dtype
        self.data = None
        self.diff = None
        self.opt_var = {}

        if init == 'zero':
            cdata = np.zeros(shape)
            cdiff = np.zeros(shape)
        elif init == 'one':
            cdata = np.ones(shape)
            cdiff = np.zeros(shape)
        elif init == 'msra':
            weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / shape[-1])
            cdata = np.random.standard_normal(shape) / weights_scale
            cdiff = np.zeros(shape)
        elif init == 'normal':
            cdata = np.random.standard_normal(shape)
            cdiff = np.zeros(shape)
        else:
            raise Exception('name: %s ,Unidentified init_kernel type: %s' % (self.name, init))

        if cdata is not None and cdiff is not None:
            if self.dtype == 'float32':
                cdata = cdata.astype(np.float32)
                cdiff = cdiff.astype(np.float32)
            elif self.dtype == 'float64':
                cdata = cdata.astype(np.float64)
                cdiff = cdiff.astype(np.float64)
            else:
                raise Exception('name: %s ,Unidentified dtype: %s' % (self.name, self.dtype))

        if cdata is not None and cdiff is not None:
            cl_mem_flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
            self.data = self.context.get_buffer(cdata, cl_mem_flags)
            self.diff = self.context.get_buffer(cdiff, cl_mem_flags)

    def eval(self):
        if self.recursive:
            for operator in self.parents:
                operator.forward()
        self.feeded = True
        return self.data

    def diff_eval(self):
        if not self.feeded:
            raise Exception('name: %s ,error at self.feeded = False' % (self.name, ))
        if self.recursive:
            for operator in self.childs:
                operator.backward()
        return self.diff


class GOptimizer(object):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def init(self, var: GTensor):
        pass

    def apply(self, var: GTensor, batch_size: int):
        pass


class GExpression(Node):

    def __init__(self, name, input_shape: tuple, output_shape: tuple):
        Node.__init__(self, name)
        self.input_shape = input_shape
        self.output_shape = output_shape

    def link_parent(self, parent: Node):
        for item in self.parents:
            if item.name == parent.name:
                return
        self.parents.append(parent)
        parent.childs.append(self)

    def link_child(self, child: Node):
        for item in self.childs:
            if item.name == child.name:
                return
        self.childs.append(child)
        child.parents.append(self)

    def forward(self):
        if self.recursive:
            for parent in self.parents:
                parent.eval()
        self.feeded = True

    def backward(self):
        if not self.feeded:
            raise Exception('name: %s ,backward() error at self.feeded = False' % self.name)
        if self.recursive:
            for child in self.childs:
                child.diff_eval()

    def load(self, file_path: str):
        pass

    def save(self, file_path: str):
        pass

    def init_layer(self, opt: GOptimizer):
        # opt.init_kernel(var=self)
        pass

    def update_layer(self, opt: GOptimizer):
        # opt.apply(var=self)
        pass
