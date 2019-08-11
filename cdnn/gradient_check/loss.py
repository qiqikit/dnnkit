import math
import numpy as np
import cdnn.model as cmodel
import cdnn.losses as losses


def check_softmax_withloss_x():
    var_in = cmodel.CTensor(name='in', shape=(4, 5, 8, 8), init='zero')
    var_label = cmodel.CTensor(name='in', shape=(4, 5*8*8), init='zero')
    var_label.data[0, 0] = 1
    var_label.data[1, 0] = 1
    var_label.data[2, 0] = 1
    var_label.data[3, 0] = 1

    check_layer, var_out = losses.softmax_withloss(name='softmax1', input_tensor=var_in, label=var_label)
    # var_out.diff = np.ones(var_out.shape)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):
            for i in range(var_in.shape[2]):
                for j in range(var_in.shape[3]):

                    in_data = np.random.standard_normal(var_in.shape)

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

                    if math.fabs(d0-dx) > 0.00001:
                        print('d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_cross_entropy_binary_x():
    var_in = cmodel.CTensor(name='in', shape=(10, 1), init='zero')
    var_label = cmodel.CTensor(name='in', shape=(10, 1), init='zero')

    for i in range(var_label.shape[0]):
        var_label.data[i, 0] = 1

    check_layer, var_out = losses.cross_entropy_binary(name='cross_entropy2', predict=var_in, label=var_label)
    # var_out.diff = np.ones(var_out.shape)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):
            in_data = np.random.random(var_in.shape)

            x0 = in_data.copy()
            x1 = in_data.copy()
            x2 = in_data.copy()

            eps = 0.000001
            x1[b, c] -= eps
            x2[b, c] += eps

            var_in.data = x0
            var_out.eval()          # fp
            var_in.diff_eval()      # bp
            d0 = var_in.diff[b, c]

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


def check_cross_entropy_categorical_x():
    var_in = cmodel.CTensor(name='in', shape=(4, 5*8*8), init='zero')
    var_label = cmodel.CTensor(name='in', shape=(4, 5*8*8), init='zero')
    var_label.data[0, 0] = 1
    var_label.data[1, 0] = 1
    var_label.data[2, 0] = 1
    var_label.data[3, 0] = 1

    check_layer, var_out = losses.cross_entropy_categorical(name='cross_entropy1', predict=var_in, label=var_label)
    # var_out.diff = np.ones(var_out.shape)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):

            in_data = np.random.random(var_in.shape)

            x0 = in_data.copy()
            x1 = in_data.copy()
            x2 = in_data.copy()

            eps = 0.000001
            x1[b, c] -= eps
            x2[b, c] += eps

            var_in.data = x0
            var_out.eval()          # fp
            var_in.diff_eval()      # bp
            d0 = var_in.diff[b, c]

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


def check_mean_squared_error_x():
    var_in = cmodel.CTensor(name='in', shape=(4, 5*8*8), init='zero')
    var_label = cmodel.CTensor(name='in', shape=(4, 5*8*8), init='zero')
    var_label.data[0, 0] = 1
    var_label.data[1, 0] = 1
    var_label.data[2, 0] = 1
    var_label.data[3, 0] = 1

    check_layer, var_out = losses.mean_squared_error(name='mean_squared_error1', predict=var_in, label=var_label)
    # var_out.diff = np.ones(var_out.shape)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):

            in_data = np.random.random(var_in.shape)

            x0 = in_data.copy()
            x1 = in_data.copy()
            x2 = in_data.copy()

            eps = 0.000001
            x1[b, c] -= eps
            x2[b, c] += eps

            var_in.data = x0
            var_out.eval()          # fp
            var_in.diff_eval()      # bp
            d0 = var_in.diff[b, c]

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
    check_softmax_withloss_x()
    # check_cross_entropy_binary_x()
    # check_cross_entropy_categorical_x()
    # check_mean_squared_error_x()
