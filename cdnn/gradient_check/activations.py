import math
import numpy as np
import cdnn.model as cmodel
import cdnn.activations as activation


def check_relu_x():
    var_in = cmodel.CTensor(name='in', shape=(4, 5, 8, 8), init='zero')
    check_layer, var_out = activation.relu(name='relu', input_tensor=var_in)
    var_out.diff = np.ones(var_out.shape)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):
            for i in range(var_in.shape[2]):
                for j in range(var_in.shape[3]):

                    in_data = np.random.standard_normal(var_in.shape)

                    x0 = in_data.copy()
                    x1 = in_data.copy()
                    x2 = in_data.copy()

                    eps = 0.00001
                    x1[b, c, i, j] -= eps
                    x2[b, c, i, j] += eps

                    var_in.data = x0
                    var_out.eval()          # fp
                    var_in.diff_eval()      # bp
                    d0 = var_in.diff[b, c, i, j]

                    var_in.data = x1
                    var_out.eval()  # fp
                    d1 = np.sum(var_out.data)

                    var_in.data = x2
                    var_out.eval()          # fp
                    d2 = np.sum(var_out.data)
                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_lrelu_x():
    var_in = cmodel.CTensor(name='in', shape=(4, 5, 8, 8), init='zero')
    check_layer, var_out = activation.lrelu(name='lrelu', input_tensor=var_in, alpha=0.7)
    var_out.diff = np.ones(var_out.shape)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):
            for i in range(var_in.shape[2]):
                for j in range(var_in.shape[3]):

                    in_data = np.random.standard_normal(var_in.shape)

                    x0 = in_data.copy()
                    x1 = in_data.copy()
                    x2 = in_data.copy()

                    eps = 0.00001
                    x1[b, c, i, j] -= eps
                    x2[b, c, i, j] += eps

                    var_in.data = x0
                    var_out.eval()          # fp
                    var_in.diff_eval()      # bp
                    d0 = var_in.diff[b, c, i, j]

                    var_in.data = x1
                    var_out.eval()  # fp
                    d1 = np.sum(var_out.data)

                    var_in.data = x2
                    var_out.eval()          # fp
                    d2 = np.sum(var_out.data)
                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_sigmoid_x():
    var_in = cmodel.CTensor(name='in', shape=(4, 5, 8, 8), init='zero')
    check_layer, var_out = activation.sigmoid(name='sigmoid', input_tensor=var_in)
    var_out.diff = np.ones(var_out.shape)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):
            for i in range(var_in.shape[2]):
                for j in range(var_in.shape[3]):

                    in_data = np.random.standard_normal(var_in.shape)

                    x0 = in_data.copy()
                    x1 = in_data.copy()
                    x2 = in_data.copy()

                    eps = 0.00001
                    x1[b, c, i, j] -= eps
                    x2[b, c, i, j] += eps

                    var_in.data = x0
                    var_out.eval()          # fp
                    var_in.diff_eval()      # bp
                    d0 = var_in.diff[b, c, i, j]

                    var_in.data = x1
                    var_out.eval()          # fp
                    d1 = np.sum(var_out.data)

                    var_in.data = x2
                    var_out.eval()          # fp
                    d2 = np.sum(var_out.data)
                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_log_x():
    var_in = cmodel.CTensor(name='in', shape=(4, 5, 8, 8), init='zero')
    check_layer, var_out = activation.log(name='log', input_tensor=var_in)
    var_out.diff = np.ones(var_out.shape)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):
            for i in range(var_in.shape[2]):
                for j in range(var_in.shape[3]):

                    in_data = np.random.random(var_in.shape) + 0.01

                    x0 = in_data.copy()
                    x1 = in_data.copy()
                    x2 = in_data.copy()

                    eps = 0.000001
                    x1[b, c, i, j] -= eps
                    x2[b, c, i, j] += eps

                    var_in.data = x0
                    var_out.eval()          # fp
                    var_in.diff_eval()      # bp
                    d0 = var_in.diff[b, c, i, j]

                    var_in.data = x1
                    var_out.eval()          # fp
                    d1 = np.sum(var_out.data)

                    var_in.data = x2
                    var_out.eval()          # fp
                    d2 = np.sum(var_out.data)
                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_tanh_x():
    var_in = cmodel.CTensor(name='in', shape=(4, 5, 8, 8), init='zero')
    check_layer, var_out = activation.tanh(name='tanh', input_tensor=var_in)
    var_out.diff = np.ones(var_out.shape)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):
            for i in range(var_in.shape[2]):
                for j in range(var_in.shape[3]):

                    in_data = np.random.random(var_in.shape) + 0.01

                    x0 = in_data.copy()
                    x1 = in_data.copy()
                    x2 = in_data.copy()

                    eps = 0.000001
                    x1[b, c, i, j] -= eps
                    x2[b, c, i, j] += eps

                    var_in.data = x0
                    var_out.eval()          # fp
                    var_in.diff_eval()      # bp
                    d0 = var_in.diff[b, c, i, j]

                    var_in.data = x1
                    var_out.eval()          # fp
                    d1 = np.sum(var_out.data)

                    var_in.data = x2
                    var_out.eval()          # fp
                    d2 = np.sum(var_out.data)
                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')


if __name__ == "__main__":
    check_relu_x()
    check_lrelu_x()
    check_sigmoid_x()
    check_log_x()
    check_tanh_x()
