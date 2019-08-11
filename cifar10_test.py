import cifar10.net3 as net
import cifar10.dataset as dataset
import cifar10.test as test

if __name__ == "__main__":
    batch_size = 64

    _net = net.Cifar10(batch_size)
    # _net.load_weight('./cifar10/weights/20190531144935/')

    test_images, test_labels = dataset.load_cifar10('./cifar10/data', 'test')
    test.tester(_net, test_images, test_labels)
