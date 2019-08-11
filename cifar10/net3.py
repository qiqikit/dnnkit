# import time
import pathlib
import numpy as np
import cdnn.model as cmodel
import cdnn.layers as layers
import cdnn.activations as activation
import cdnn.losses as loss
import cdnn.optimizers as optimizer
from dnn_net import DnnNet


class Cifar10(DnnNet):
    def __init__(self, batch_size: int):
        DnnNet.__init__(self, batch_size=batch_size)

        self.phase = 'train'
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # ################### layers 0 (image)###############################################
        self.image_shape = (self.batch_size, 3,  32, 32)
        self.image_in = cmodel.CTensor('image_in', self.image_shape, init='zero')

        # ################### layers 1 (conv1)###############################################
        self.conv1, self.conv1_out = layers.conv2d('conv1', self.image_in, (6, 3, 5, 5))
        self.relu1, self.relu1_out = activation.relu('relu1', self.conv1_out)
        self.pooling1, self.pooling1_out = layers.max_pooling('pooling1', self.relu1_out, 2, 2)

        # ################### layers 2 (conv2)################################################
        self.conv2, self.conv2_out = layers.conv2d('conv2', self.pooling1_out, (16, 6, 5, 5))
        self.relu2, self.relu2_out = activation.relu('relu2', self.conv2_out)
        self.pooling2, self.pooling2_out = layers.max_pooling('pooling2', self.relu2_out, 2, 2)

        # ################### layers 3 (fc3 )###################################################
        self.fc3, self.fc3_out = layers.full_connect('fc3', self.pooling2_out, 120)
        self.relu3, self.relu3_out = activation.relu('relu3', self.fc3_out)

        # ################### layers 4 (loss)###################################################
        self.fc4, self.fc4_out = layers.full_connect('fc4', self.relu3_out, 84)
        self.relu4, self.relu4_out = activation.relu('relu4', self.fc4_out)

        # ################### layers 5 (loss)###################################################
        self.fc5, self.fc5_out = layers.full_connect('fc5', self.relu4_out, 10)

        # ################### layers 6 (loss)###################################################
        self.label = cmodel.CTensor('label', (self.batch_size, 10), 'zero')
        self.loss, self.loss_out = loss.softmax_withloss('loss', self.fc5_out, self.label)

        # SGD lr=0.001, momentum=0.9
        # self.copt = COptimizer.Momentum(learning_rate=0.001, momentum=0.9)
        self.copt = optimizer.adam(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        # self.copt = COptimizer.GradientDescent(learning_rate=0.000001)

        self.clayer_list = list()
        self.cvar_list = list()

        self.init_layer_list()
        self.net_init()

    def init_layer_list(self):

        self.clayer_list = [self.conv1, self.relu1, self.pooling1,
                            self.conv2, self.relu2, self.pooling2,
                            self.fc3, self.relu3,
                            self.fc4, self.relu4,
                            self.fc5, self.loss]

        self.cvar_list = [self.image_in, self.conv1_out, self.relu1_out, self.pooling1_out,
                          self.conv2_out, self.relu2_out, self.pooling2_out,
                          self.fc3_out, self.relu3_out,
                          self.fc4_out, self.relu4_out,
                          self.fc5_out, self.label, self.loss_out]

    def net_init(self):
        for var in self.cvar_list:
            var.feeded = False
        for layer in self.clayer_list:
            layer.feeded = False
            layer.init_layer(self.copt)

    def set_phase(self, phase: str):
        self.phase = phase

    def set_learning_rate(self, learning_rate: float):
        self.copt.learning_rate = learning_rate

    def feed(self, image: np.ndarray, label: np.ndarray):
        self.image_in.data = image
        self.label.data = label

    def forward(self):
        return self.loss_out.eval()

    def prediction(self):
        return self.loss.predict

    def backward(self):
        return self.image_in.diff_eval()

    def net_update(self):
        for layer in self.clayer_list:
            layer.update_layer(self.copt)

    def load_weight(self, path: str):
        self.conv1.load(path)
        self.conv2.load(path)
        self.fc3.load(path)
        self.fc4.load(path)
        self.fc5.load(path)

    def save_weight(self, path: str):
        dir_path = pathlib.Path(path)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        self.conv1.save(path)
        self.conv2.save(path)
        self.fc3.save(path)
        self.fc4.save(path)
        self.fc5.save(path)
