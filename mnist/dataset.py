import struct
import numpy as np
from glob import glob


def load_mnist(path, kind='train'):
    images_path = glob('%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('%s/%s*1-ubyte' % (path, kind))[0]

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        images = images / 255.0
        # input_mean = np.mean(images, axis=(0,))
        # sigma_square = np.var(images, 0)
        # epsilon = 0.00001
        # images = (images - input_mean) / np.sqrt(sigma_square + epsilon)

    label_vector = np.zeros([labels.shape[0], 10])
    for i in range(labels.shape[0]):
        label_vector[i][labels[i]] = 1

    return images, label_vector
