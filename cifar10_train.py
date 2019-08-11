import time
import cifar10.net3 as net
import cifar10.dataset as dataset
import cifar10.train as train

if __name__ == "__main__":
    batch_size = 32

    _net = net.Cifar10(batch_size)
    # _net.load_weight('./cifar10/weights/20190630201659/')
    train_images, train_labels = dataset.load_cifar10('./cifar10/data', 'train')

    train.trainer(_net, train_images, train_labels, 1)
    save_dir = time.strftime('%Y%m%d%H%M%S', time.localtime())
    _net.save_weight('./cifar10/weights/%s/' % (save_dir, ))
