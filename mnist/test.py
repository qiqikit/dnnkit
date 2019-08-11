import time
import numpy as np
from dnn_net import DnnNet


def tester(_net: DnnNet, test_images: np.ndarray, test_labels: np.ndarray):

    _net.set_phase('test')
    batch_size = _net.batch_size
    image_shape = _net.image_shape

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('start test...')

    val_loss = 0
    val_acc = 0

    batch_count = int(test_images.shape[0] / batch_size)
    print(batch_count)

    for i in range(batch_count):

        _net.net_init()

        _image = test_images[i * batch_size:(i + 1) * batch_size].reshape(image_shape)
        _label = test_labels[i * batch_size:(i + 1) * batch_size]

        _net.feed(_image, _label)

        loss = _net.forward()
        batch_loss = loss[0] / batch_size
        _predict = _net.prediction()

        val_loss += batch_loss

        for j in range(batch_size):
            if np.argmax(_predict[j]) == np.argmax(_label[j]):
                val_acc += 1
        print('.')

    print(time.strftime("  %Y-%m-%d %H:%M:%S", time.localtime()))
    print("epoch test val_acc: [%d/%d]  val_loss: %.4f" % (val_acc, batch_count * batch_size, val_loss / batch_count))
    print('end test.')
