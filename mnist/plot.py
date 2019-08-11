import matplotlib.pyplot as plt
import mnist.net2


def mnist_plot(net: mnist.net2.Mnist):
    plt.close()

    plt.figure(1)
    i = 1
    node_size = len(net.var_list)

    for var in net.var_list:
        plt.subplot(1, node_size, i)
        print(var.name)
        plt.title(var.name)
        plt.imshow(var.data[0])
        i += 1

    plt.figure(2)
    plt.imshow(net.image_in)
    plt.show()
