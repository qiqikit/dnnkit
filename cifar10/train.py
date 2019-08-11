import time
import numpy as np
from dnn_net import DnnNet


def trainer(_net: DnnNet, train_images: np.ndarray, train_labels: np.ndarray, epoch_count=1):

    _net.set_phase('train')
    batch_size = _net.batch_size
    image_shape = _net.image_shape

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('start train...')

    loss_collect = []
    for epoch in range(epoch_count):

        # train
        train_acc = 0
        train_loss = 0

        # random shuffle
        order = np.arange(train_images.shape[0])
        np.random.shuffle(order)

        images = train_images[order]
        labels = train_labels[order]

        learning_rate = 5e-4 * pow(0.1, float(epoch / 10))
        _net.set_learning_rate(learning_rate)

        batch_count = int(train_images.shape[0] / batch_size)
        for i in range(batch_count):
            _net.net_init()
            # feed
            _image = images[i * batch_size:(i + 1) * batch_size].reshape(image_shape)
            _label = labels[i * batch_size:(i + 1) * batch_size]

            _net.feed(_image, _label)

            # forward
            loss = _net.forward()
            batch_loss = loss[0] / batch_size
            train_loss += batch_loss

            batch_acc = 0
            for j in range(batch_size):
                _predict = _net.prediction()
                if np.argmax(_predict[j]) == np.argmax(_label[j]):
                    batch_acc += 1
                    train_acc += 1

            # backward
            _net.backward()

            # apply gradient
            _net.net_update()

            if i % 5 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("  epoch: %d, batch: %3d, batch_acc: [%d/%d]  batch_loss: %.4f  learning_rate %f" % (
                    epoch, i, batch_acc, batch_size, batch_loss, learning_rate))

        train_loss = train_loss / batch_count
        loss_collect.append(train_loss)

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("  epoch: %d, train_acc: [%d/%d]  train_loss: %.4f" % (
            epoch, train_acc, batch_count * batch_size, train_loss))

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('end train.')
