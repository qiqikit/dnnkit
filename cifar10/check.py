import time
import math
import numpy as np
from dnn_net import DnnNet


def check_gradient(_net: DnnNet, image: np.ndarray, label: np.ndarray):

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check...')

    for k in range(_net.image_in.shape[0]):
        for c in range(_net.image_in.shape[1]):
            for i in range(0, _net.image_in.shape[2]):
                for j in range(0, _net.image_in.shape[3]):

                    x0 = image.copy()
                    x1 = image.copy()
                    x2 = image.copy()

                    eps = 0.00001
                    x1[k, c, i, j] -= eps
                    x2[k, c, i, j] += eps

                    _net.net_init()

                    # feed
                    _net.feed(x0, label)
                    _net.forward()              # forward
                    _net.backward()             # backward
                    d0 = _net.image_in.diff[k, c, i, j]

                    _net.feed(x1, label)
                    loss = _net.forward()       # forward
                    _net.backward()             # backward
                    d1 = loss[0]

                    _net.feed(x2, label)
                    loss = _net.forward()       # forward
                    _net.backward()             # backward
                    d2 = loss[0]

                    dx = (d2 - d1) / 2.0 / eps

                    print('[%2d,%2d,%2d,%2d]: (%3.7f, %3.7f)' % (k, c, i, j, d0, dx))
                    if math.fabs(d0 - dx) > 0.000001:
                        raise Exception('math.fabs(d1-dx)')
                        # print('math.fabs(d0x) > 0.000001')


def checker(_net: DnnNet, images: np.ndarray, labels: np.ndarray):

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('start check...')

    image_shape = _net.image_shape
    batch_size = _net.batch_size

    for b in range(1):

        print('check batch %d ...' % b)
        # feed
        image = images[b * batch_size:(b + 1) * batch_size, :].reshape(image_shape)
        label = labels[b * batch_size:(b + 1) * batch_size, :]

        check_gradient(_net, image, label)
        # check_fc_weights(_net, image, label)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('end check.')
