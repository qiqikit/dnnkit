import time
import mnist.net2 as net
import mnist.dataset as dataset
import mnist.train as train

if __name__ == "__main__":
    batch_size = 32

    _net = net.Mnist(batch_size)
    # _net.load_weight('./mnist/weights/20190531003114/')
    train_images, train_labels = dataset.load_mnist('./mnist/data', 'train')

    train.trainer(_net, train_images, train_labels, 1)
    save_dir = time.strftime('%Y%m%d%H%M%S', time.localtime())
    _net.save_weight('./mnist/weights/%s/' % (save_dir, ))
