import pickle
import numpy as np


def load_cifar10(path, kind='train'):
    images = np.zeros([0, 3, 32, 32])
    labels = np.zeros([0, 10])

    if kind == 'train':
        for k in range(1, 6):
            file_name = '/data_batch_%d' % k
            file = open(path + file_name, 'rb')
            data_dict = pickle.load(file, encoding='latin1')
            data_count = len(data_dict['labels'])

            image_data = np.array(data_dict['data']).reshape(data_count, 3, 32, 32)
            label_data = np.array(data_dict['labels'])
            label_vector = np.zeros([data_count, 10])

            for i in range(data_count):
                label_vector[i][label_data[i]] = 1

            images = np.vstack((images, image_data))
            labels = np.vstack((labels, label_vector))

    elif kind == 'test':
        file_name = '/test_batch'
        file = open(path + file_name, 'rb')
        data_dict = pickle.load(file, encoding='latin1')
        data_count = len(data_dict['labels'])

        image_data = np.array(data_dict['data']).reshape(data_count, 3, 32, 32)
        # image_data = np.swapaxes(image_data, 1, 3)

        label_data = np.array(data_dict['labels'])
        label_vector = np.zeros([data_count, 10])

        for i in range(data_count):
            label_vector[i][label_data[i]] = 1

        images = np.vstack((images, image_data))
        # np.concatenate()
        labels = np.vstack((labels, label_vector))

    else:
        raise Exception('load_cifar10(): Unidentified kind type: %s' % (kind, ))

    images = images / 255.0
    return images, labels
