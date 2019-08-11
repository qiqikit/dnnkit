import mnist.net1 as net
import mnist.dataset as dataset
import mnist.test as test

if __name__ == "__main__":
    batch_size = 16

    _net = net.Mnist(batch_size)
    # _net.load_weight('./mnist/weights/20190531003114/')

    test_images, test_labels = dataset.load_mnist('./mnist/data', 't10k')
    test.tester(_net, test_images, test_labels)
