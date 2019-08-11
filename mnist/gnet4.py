import pathlib
import numpy as np
# import pyopencl as cl

import cdnn.model as cmodel
import cdnn.layers as clayers
import cdnn.losses as losses
import cdnn.activations as cactivations
import cdnn.optimizers as coptimizers

# import gdnn.model as gmodel
import gdnn.platform as gpm
import gdnn.layers as glayers
import gdnn.kernel.pooling as gkp
from dnn_net import DnnNet


class Mnist(DnnNet):
    def __init__(self, batch_size: int):
        DnnNet.__init__(self, batch_size=batch_size)

        self.phase = 'train'
        self.platform = gpm.CsPlatform()
        self.platform.print_platform_info()

        self.context = gpm.CsContext(platform_id=0)
        self.queue = gpm.CsQueue(self.context, device_id=0)

        # cl programs
        self.clp_pooling = gpm.CsProgram(self.context)
        self.clp_pooling.load('gdnn/kernel/pooling.cl', 'source', [])

        # cl classes
        self.clc_max_pooling = gkp.CsPooling(self.clp_pooling, self.queue)

        # ################### layers 0 (image)###############################################
        self.image_shape = (self.batch_size, 1, 28, 28)
        self.image_in = cmodel.CTensor('image_in', self.image_shape, init='zero')

        # ################### layers 1 (conv1)###############################################
        self.conv1, self.conv1_out = clayers.conv2d('conv1', self.image_in, (20, 1, 5, 5), 1, 'SAME')
        self.relu1, self.relu1_out = cactivations.relu('relu1', self.conv1_out)

        self.cg_tensor1, self.cg_tensor1_out = glayers.array_to_buffer('cg_tensor1', self.context, self.queue,
                                                                       self.relu1_out)
        self.gpooling1, self.gpooling1_out = glayers.max_pooling('gpooling1', self.context, self.clc_max_pooling,
                                                                 self.cg_tensor1_out, 2, 2)
        self.gc_tensor1, self.gc_tensor1_out = glayers.buffer_to_array('gpu2cpu1', self.context, self.queue,
                                                                       self.gpooling1_out)
        # ################### layers 2 (conv2)################################################
        self.conv2, self.conv2_out = clayers.conv2d('conv2', self.gc_tensor1_out, (50, 20, 5, 5), 1, 'SAME')
        self.relu2, self.relu2_out = cactivations.relu('relu2', self.conv2_out)

        self.cg_tensor2, self.cg_tensor2_out = glayers.array_to_buffer('cg_tensor2', self.context, self.queue,
                                                                       self.relu2_out)

        self.gpooling2, self.gpooling2_out = glayers.max_pooling('gpooling2', self.context, self.clc_max_pooling,
                                                                 self.cg_tensor2_out, 2, 2)

        self.gc_tensor2, self.gc_tensor2_out = glayers.buffer_to_array('gc_tensor2', self.context, self.queue,
                                                                       self.gpooling2_out)

        # ################### layers 3 (fc3)####################################################
        self.fc3, self.fc3_out = clayers.full_connect('fc3', self.gc_tensor2_out, 500)
        self.relu3, self.relu3_out = cactivations.relu('relu3', self.fc3_out)

        # ################### layers 4 (fc4)#####################################################
        self.dropout4, self.dropout4_out = clayers.dropout('dropout4', self.relu3_out, self.phase, 0.8)
        self.fc4, self.fc4_out = clayers.full_connect('fc4', self.dropout4_out, 10)

        # ################### layers 5 (loss)###################################################
        self.label = cmodel.CTensor('label', (self.batch_size, 10), 'zero')
        self.loss, self.loss_out = losses.softmax_withloss('loss', self.fc4_out, self.label)

        self.copt = coptimizers.adam(0.001, 0.9, 0.999, 1e-8)
        self.gopt = coptimizers.adam(0.001, 0.9, 0.999, 1e-8)

        self.clayer_list = list()
        self.cvar_list = list()
        self.glayer_list = list()
        self.gvar_list = list()

        self.init_layer_list()
        self.net_init()

    def init_layer_list(self):

        self.clayer_list = [self.conv1, self.relu1,
                            self.conv2, self.relu2,
                            self.fc3, self.relu3,
                            self.dropout4, self.fc4, self.loss]

        self.glayer_list = [self.cg_tensor1, self.gpooling1, self.gc_tensor1,
                            self.cg_tensor2, self.gpooling2, self.gc_tensor2]

        self.cvar_list = [self.image_in, self.conv1_out, self.relu1_out, self.gc_tensor1_out,
                          self.conv2_out, self.relu2_out, self.gc_tensor2_out,
                          self.fc3_out, self.relu3_out,
                          self.fc4_out, self.dropout4_out,
                          self.label, self.loss_out]

        self.gvar_list = [self.cg_tensor1_out, self.gpooling1_out,
                          self.cg_tensor2_out, self.gpooling2_out]

    def net_init(self):

        for var in self.cvar_list:
            var.feeded = False
        for layer in self.clayer_list:
            layer.feeded = False
            layer.init_layer(self.copt)

        for var in self.gvar_list:
            var.feeded = False
        for layer in self.glayer_list:
            layer.feeded = False
            layer.init_layer(self.gopt)

    def set_phase(self, phase: str):
        self.phase = phase
        self.dropout4.phase = phase

    def set_learning_rate(self, learning_rate: float):
        self.copt.learning_rate = learning_rate
        self.gopt.learning_rate = learning_rate

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
        for layer in self.glayer_list:
            layer.update_layer(self.gopt)

    def load_weight(self, path: str):
        self.conv1.load(path)
        self.conv2.load(path)
        self.fc3.load(path)
        self.fc4.load(path)

    def save_weight(self, path: str):
        dir_path = pathlib.Path(path)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        self.conv1.save(path)
        self.conv2.save(path)
        self.fc3.save(path)
        self.fc4.save(path)
