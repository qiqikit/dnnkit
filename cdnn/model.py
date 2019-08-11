import math
import numpy as np
from functools import reduce


class Node(object):
    def __init__(self, name):
        self.name = name
        self.childs = []
        self.parents = []
        self.feeded = False
        self.recursive = True


class CTensor(Node):
    def __init__(self, name: str, shape: tuple, init: str):
        Node.__init__(self, name)
        self.shape = shape
        self.data = None
        self.diff = None
        self.opt_var = {}

        if init == 'zero':
            self.data = np.zeros(shape)
            self.diff = np.zeros(shape)
        elif init == 'one':
            self.data = np.ones(shape)
            self.diff = np.zeros(shape)
        elif init == 'msra':
            weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / shape[-1])
            self.data = np.random.standard_normal(shape) / weights_scale
            self.diff = np.zeros(shape)
        elif init == 'normal':
            self.data = np.random.standard_normal(shape)
            self.diff = np.zeros(shape)
        elif init == 'none':
            pass
        else:
            raise Exception('name: %s ,Unidentified init_kernel type: %s' % (self.name, init))

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


class COptimizer(object):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def init(self, var: CTensor):
        pass

    def apply(self, var: CTensor, batch_size: int):
        pass


class CExpression(Node):

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

    def init_layer(self, opt: COptimizer):
        # opt.init_kernel(var=self)
        pass

    def update_layer(self, opt: COptimizer):
        # opt.apply(var=self)
        pass



