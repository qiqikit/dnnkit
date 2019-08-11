import mnist.net1 as net
import mnist.dataset as dataset
import mnist.check as check

if __name__ == "__main__":
    batch_size = 3

    _net = net.Mnist(batch_size)
    # _net.load_weight('./mnist/weights/20190531144935/')

    test_images, test_labels = dataset.load_mnist('./mnist/data', 't10k')
    check.checker(_net, test_images, test_labels)
