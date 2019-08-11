import numpy as np
from cdnn.model import CTensor
from cdnn.model import CExpression
from cdnn.model import COptimizer


# CExpression class
class Conv2d(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, kernel_shape: tuple,
                 stride: int, padding: str):
        CExpression.__init__(self, name, input_shape, output_shape)
        # only the stride = 1 case is handled
        assert(stride == 1)
        # kernel_shape = [output_channels, input_channels, ksize, ksize]
        if len(input_shape) != 4:
            raise Exception("Conv2D name: %s's input_shape != 4d" % name)

        self.batch_size = input_shape[0]

        self.ksize = kernel_shape[2]
        self.stride = stride
        self.padding = padding
        self.output_num = kernel_shape[0]
        self.pad_input = None

        self.weights = CTensor(name=self.name + '-weights', shape=kernel_shape, init='msra')
        self.bias = CTensor(name=self.name + '-bias', shape=(self.output_num, ), init='msra')

        if self.padding == 'SAME':
            _output_shape = (self.batch_size, self.output_num,
                             int(input_shape[2] / stride), int(input_shape[3] / stride))
        elif self.padding == 'VALID':
            _output_shape = (self.batch_size, self.output_num,
                             int((input_shape[2] - self.ksize + stride) / stride),
                             int((input_shape[3] - self.ksize + stride) / stride))
        else:
            raise Exception('Conv2d name: %s Unidentified padding type: %s' % (name, padding))

        if output_shape != _output_shape:
            raise Exception("Conv2d name: %s output_shape != _output_shape" % self.name)

        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if self.input_shape != input_tensor.shape:
            raise Exception("Conv2d name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("Conv2d name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def init_layer(self, opt: COptimizer):
        opt.init(var=self.weights)
        opt.init(var=self.bias)

    def update_layer(self, opt: COptimizer):
        opt.apply(var=self.weights, batch_size=self.batch_size)
        opt.apply(var=self.bias, batch_size=self.batch_size)

    def forward(self):
        CExpression.forward(self)
        for i in range(self.batch_size):
            self.output_tensor.data[i] = self._conv(self.input_tensor.data[i], self.weights, self.bias)

    def _conv(self, input_data: np.ndarray, weights: CTensor, bias: CTensor):
        # padding input_img according to method
        if self.padding == 'SAME':
            pad_size = int(self.ksize / 2)
            pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
            pad_input = np.pad(input_data, pad_width, 'constant', constant_values=0)
        elif self.padding == 'VALID':
            pad_input = input_data
        else:
            raise Exception('Conv2d name: %s Unidentified padding type: %s' % (self.name, self.padding))

        out_shape = (self.output_num,
                     int((pad_input.shape[1] - self.ksize + self.stride) / self.stride),
                     int((pad_input.shape[2] - self.ksize + self.stride) / self.stride))

        image_col = self._im2col(pad_input, self.ksize, self.stride)
        image_col = image_col.swapaxes(0, 1)

        weights_col = weights.data.reshape(self.output_num, -1)
        bias_row = bias.data.reshape(self.output_num, 1)
        conv_out = np.dot(weights_col, image_col) + bias_row

        return conv_out.reshape(out_shape)

    def _im2col(self, image, ksize, stride):
        image_col = []
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[:, i:i + ksize, j:j + ksize].reshape(-1)
                image_col.append(col)

        return np.array(image_col)

    def backward(self):
        CExpression.backward(self)
        self.weights.diff *= 0
        self.bias.diff *= 0
        for i in range(self.batch_size):
            self.input_tensor.diff[i] = self._deconv(self.input_tensor.data[i], self.output_tensor.diff[i],
                                                        self.weights, self.bias)

    def _deconv(self, input_data: np.ndarray, output_diff: np.ndarray, weights: CTensor, bias: CTensor):

        if self.padding == 'SAME':
            pad_size = int(self.ksize / 2)
            pad_input_data = np.pad(input_data, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                                    'constant', constant_values=0)
            pad_output_diff = np.pad(output_diff, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                                     'constant', constant_values=0)
        elif self.padding == 'VALID':
            pad_size = int(self.ksize - 1)
            pad_input_data = input_data
            pad_output_diff = np.pad(output_diff, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                                     'constant', constant_values=0)
        else:
            raise Exception('Conv2d name: %s Unidentified padding type: %s' % (self.name, self.padding))

        output_diff_flatten = output_diff.reshape(self.output_num, -1)
        pad_input_col = self._im2col(pad_input_data, self.ksize, self.stride)

        weights.diff += np.dot(output_diff_flatten, pad_input_col).reshape(weights.diff.shape)
        bias.diff += np.sum(output_diff_flatten, axis=1)

        pad_output_diff = self._im2col(pad_output_diff, self.ksize, self.stride)
        pad_output_diff = pad_output_diff.reshape(pad_output_diff.shape[0], -1, self.ksize*self.ksize)

        pad_output_diff = np.swapaxes(pad_output_diff, 0, 1)
        pad_output_diff = np.swapaxes(pad_output_diff, 1, 2)
        pad_output_diff = pad_output_diff.reshape(-1, pad_output_diff.shape[2])

        flip_weights = np.flip(np.flip(weights.data, 3), 2)
        flip_weights = np.swapaxes(flip_weights, 0, 1)
        flip_weights = flip_weights.reshape(flip_weights.shape[0], -1)

        input_diff = np.dot(flip_weights, pad_output_diff)

        return input_diff.reshape(input_data.shape)

    def load(self, file_path: str):
        self.weights.data = np.load(file_path + self.weights.name + '.npy')
        if self.weights.shape != self.weights.data.shape:
            raise Exception("Conv2d name: %s self.shape != self.weights.data.shape" % self.name)
        self.bias.data = np.load(file_path + self.bias.name + '.npy')
        if self.bias.shape != self.bias.data.shape:
            raise Exception("Conv2d name: %s self.bias.shape != self.bias.data.shape" % self.name)

    def save(self, file_path: str):
        np.save(file_path + self.weights.name + '.npy', self.weights.data)
        np.save(file_path + self.bias.name + '.npy', self.bias.data)


# kernel_shape = [ksize, ksize, input_channels, output_channels]
class Conv2dBak(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, kernel_shape: tuple,
                 stride: int, padding: str):
        CExpression.__init__(self, name, input_shape, output_shape)

        # kernel_shape = [ksize, ksize, input_channels, output_channels]
        if len(input_shape) != 4:
            raise Exception("Conv2D name: %s's input_shape != 4d" % name)

        self.ksize = kernel_shape[0]
        self.stride = stride
        self.output_num = kernel_shape[-1]
        self.padding = padding
        self.col_image = []

        if self.input_shape[3] != kernel_shape[2]:
            raise Exception("Conv2d name: %s self.input_shape[3] != kernel_shape.shape[2]" % self.name)

        self.weights = CTensor(name=self.name + '-weights', shape=kernel_shape, init='normal')
        self.bias = CTensor(name=self.name + '-bias', shape=(self.output_num, ), init='normal')
        self.batch_size = input_shape[0]

        if self.padding == 'SAME':
            _output_shape = [self.batch_size, int(input_shape[1]/stride), int(input_shape[2]/stride), self.output_num]
        elif self.padding == 'VALID':
            _output_shape = [self.batch_size, int((input_shape[1]-self.ksize + 1)/stride),
                             int((input_shape[2]-self.ksize+1)/stride), self.output_num]
        else:
            raise Exception('Conv2d name: %s Unidentified padding type: %s' % (name, padding))

        if output_shape != _output_shape:
            raise Exception("Conv2d name: %s output_shape != _output_shape" % self.name)

        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if self.input_shape != input_tensor.shape:
            raise Exception("Conv2d name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("Conv2d name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def load(self, file_path: str):
        self.weights.data = np.load(file_path + self.weights.name)
        if self.weights.shape != self.weights.data.shape:
            raise Exception("Conv2d name: %s self.shape != self.weights.data.shape" % self.name)

        self.bias.data = np.load(file_path + self.bias.name)
        if self.bias.shape != self.bias.data.shape:
            raise Exception("Conv2d name: %s self.bias.shape != self.bias.data.shape" % self.name)

    def save(self, file_path: str):
        np.save(file_path + self.weights.name, self.weights.data)
        np.save(file_path + self.bias.name, self.bias.data)

    def init_layer(self, opt=COptimizer):
        opt.init(var=self.weights)
        opt.init(var=self.bias)

    def update_layer(self, opt=COptimizer):
        opt.apply(var=self.weights, batch_size=self.batch_size)
        opt.apply(var=self.bias, batch_size=self.batch_size)

    def forward(self):
        CExpression.forward(self)
        self._conv(self.input_tensor, self.output_tensor, self.weights.data, self.bias.data)
        return

    def backward(self):
        CExpression.backward(self)
        self._deconv(self.input_tensor, self.output_tensor, self.weights, self.bias)

    def _conv(self, _input: CTensor, _output: CTensor, weights: np.ndarray, bias: np.ndarray):
        # reshape weights to col
        col_weights = weights.reshape(-1, self.output_num)

        # padding input_img according to method
        if self.padding == 'SAME':
            pad_width = ((0, 0), (int(self.ksize/2), int(self.ksize/2)), (int(self.ksize/2), int(self.ksize/2)), (0, 0))
            batch_img = np.pad(_input.data, pad_width, 'constant', constant_values=0)
        else:
            batch_img = _input.data

        # malloc tmp output_data
        conv_out = np.zeros(_output.data.shape)

        self.col_image = []
        # do dot for every image in batch by im2col dot col_weight
        for i in range(self.batch_size):
            img_i = batch_img[i][np.newaxis, :]
            col_image_i = self._im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(col_image_i, col_weights) + bias, _output.data[0].shape)
            self.col_image.append(col_image_i)
        self.col_image = np.array(self.col_image)

        _output.data = conv_out
        return

    def _deconv(self, _input: CTensor, _output: CTensor, weights: CTensor, bias: CTensor):
        col_eta = np.reshape(_output.diff, [self.batch_size, -1, self.output_num])
        for i in range(self.batch_size):
            weights.diff += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        bias.diff += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.padding == 'VALID':
            pad_width = ((0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0))
            pad_eta = np.pad(_output.diff, pad_width, 'constant', constant_values=0)
        elif self.padding == 'SAME':
            pad_width = ((0, 0), (int(self.ksize/2), int(self.ksize/2)), (int(self.ksize/2), int(self.ksize/2)), (0, 0))
            pad_eta = np.pad(_output.diff, pad_width, 'constant', constant_values=0)
        else:
            raise Exception('Conv2d name: %s Unidentified padding type: %s' % (self.name, self.padding))

        col_pad_eta = np.array([self._im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride)
                                for i in range(self.batch_size)])
        flip_weights = np.flipud(np.fliplr(weights.data))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, weights.shape[2]])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, _input.shape)
        _input.diff = next_eta

    def _im2col(self, image, ksize, stride):
        # image is a 4d CTensor([batchsize, width ,height, channel])
        image_col = []
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
                image_col.append(col)
        image_col = np.array(image_col)

        return image_col


class Conv2dTest(CExpression):
    def __init__(self, name: str, input_shape: tuple, output_shape: tuple, kernel_shape: tuple,
                 stride: int, padding: str):
        CExpression.__init__(self, name, input_shape, output_shape)

        # kernel_shape = [output_channels, input_channels, ksize, ksize]
        if len(input_shape) != 4:
            raise Exception("Conv2D name: %s's input_shape != 4d" % name)

        self.ksize = kernel_shape[2]
        self.stride = stride
        self.output_num = kernel_shape[0]
        self.padding = padding

        self.weights = CTensor(name=self.name + '-weights', shape=kernel_shape, init='normal')
        self.bias = CTensor(name=self.name + '-bias', shape=(self.output_num, ), init='normal')
        self.batch_size = input_shape[0]

        if self.padding == 'SAME':
            _output_shape = [self.batch_size, self.output_num,
                             int(input_shape[2] / stride),
                             int(input_shape[3] / stride)]
        elif self.padding == 'VALID':
            _output_shape = [self.batch_size, self.output_num,
                             int((input_shape[2] - self.ksize + stride) / stride),
                             int((input_shape[3] - self.ksize + stride) / stride)]
        else:
            raise Exception('Conv2d name: %s Unidentified padding type: %s' % (name, padding))

        if output_shape != _output_shape:
            raise Exception("Conv2d name: %s output_shape != _output_shape" % self.name)

        self.input_tensor = None
        self.output_tensor = None

    def connect(self, input_tensor: CTensor, output_tensor: CTensor):
        self.link_parent(input_tensor)
        self.link_child(output_tensor)

        if self.input_shape != input_tensor.shape:
            raise Exception("Conv2d name: %s self.input_shape != input_tensor.shape" % self.name)
        if self.output_shape != output_tensor.shape:
            raise Exception("Conv2d name: %s self.output_shape != output_tensor.shape" % self.name)

        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def init_layer(self, opt: COptimizer):
        opt.init(var=self.weights)
        opt.init(var=self.bias)

    def update_layer(self, opt: COptimizer):
        opt.apply(var=self.weights, batch_size=self.batch_size)
        opt.apply(var=self.bias, batch_size=self.batch_size)

    def forward(self):
        CExpression.forward(self)

        # padding input_img according to method
        if self.padding == 'SAME':
            pad_size = int(self.ksize / 2)
            pad_width = ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size))
            pad_input = np.pad(self.input_tensor.data, pad_width, 'constant', constant_values=0)
        else:
            pad_input = self.input_tensor.data

        for i in range(self.batch_size):
            self.output_tensor.data[i] = self._conv(pad_input[i], self.weights, self.bias, self.stride)

    def _conv(self, input_data: np.ndarray, weights: CTensor, bias: CTensor, stride: int):
        ksize = weights.shape[2]
        output_num = weights.shape[0]

        out_shape = [output_num,
                     int((input_data.shape[1] - ksize + stride) / stride),
                     int((input_data.shape[2] - ksize + stride) / stride)]
        conv_out = np.zeros(out_shape)

        # image_col = self._im2col(input_data, ksize, stride)
        for o in range(weights.shape[0]):
            for i in range(0, input_data.shape[1] - ksize + 1, stride):
                for j in range(0, input_data.shape[2] - ksize + 1, stride):
                    _roi = input_data[:, i:i + ksize, j:j + ksize]
                    conv_out[o, int(i/stride), int(j/stride)] = np.sum(_roi * weights.data[o]) + bias.data[o]

        return conv_out

    def backward(self):
        CExpression.backward(self)

        # padding input_img according to method
        if self.padding == 'SAME':
            pad_size = int(self.ksize / 2)
            pad_width = ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size))
            pad_input = np.pad(self.input_tensor.data, pad_width, 'constant', constant_values=0)
        else:
            pad_input = self.input_tensor.data

        for i in range(self.batch_size):
            input_diff = self._deconv(pad_input[i], self.output_tensor.diff[i], self.weights, self.bias, self.stride)
            if self.padding == 'SAME':
                pad_size = int(self.ksize / 2)
                self.input_tensor.diff[i] = input_diff[:, pad_size:input_diff.shape[1] - pad_size,
                                               pad_size:input_diff.shape[2] - pad_size]
            else:
                self.input_tensor.diff[i] = input_diff

    def _deconv(self, input_data: np.ndarray, output_diff: np.ndarray, weights: CTensor, bias: CTensor, stride: int):

        ksize = weights.shape[2]
        input_diff = np.zeros(input_data.shape)

        for o in range(weights.shape[0]):
            for i in range(0, input_data.shape[1] - ksize + 1, stride):
                for j in range(0, input_data.shape[2] - ksize + 1, stride):
                    diff = output_diff[o, int(i / stride), int(j / stride)]
                    for kc in range(weights.shape[1]):
                        for ki in range(weights.shape[2]):
                            for kj in range(weights.shape[3]):
                                weights.diff[o, kc, ki, kj] += diff * input_data[kc, i+ki, j+kj]
                                input_diff[kc, i+ki, j+kj] += diff * weights.data[o, kc, ki, kj]
                    bias.diff += diff * 1.0

        return input_diff

    def _im2col(self, image, ksize, stride):
        image_col = []
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[:, i:i + ksize, j:j + ksize].reshape([-1])
                image_col.append(col)

        return np.array(image_col)

    def load(self, file_path: str):
        self.weights.data = np.load(file_path + self.weights.name)
        if self.weights.shape != self.weights.data.shape:
            raise Exception("Conv2d name: %s self.shape != self.weights.data.shape" % self.name)
        self.bias.data = np.load(file_path + self.bias.name)
        if self.bias.shape != self.bias.data.shape:
            raise Exception("Conv2d name: %s self.bias.shape != self.bias.data.shape" % self.name)

    def save(self, file_path: str):
        np.save(file_path + self.weights.name, self.weights.data)
        np.save(file_path + self.bias.name, self.bias.data)
