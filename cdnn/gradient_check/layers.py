import math
import numpy as np
import cdnn.model as cmodel
import cdnn.layers as layers


def check_conv2d_x():
    var_in = cmodel.CTensor(name='in', shape=(4, 5, 8, 8), init='zero')
    kernel_shape = (12, 5, 3, 3)
    check_layer, var_out = layers.conv2d(name='conv2d', input_tensor=var_in, kernel_shape=kernel_shape)
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
                        print('d0:%f d2-1:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_conv2d_weights():
    var_in = cmodel.CTensor(name='in', shape=(8, 5, 10, 10), init='zero')
    kernel_shape = (12, 5, 3, 3)
    check_layer, var_out = layers.conv2d(name='conv2d', input_tensor=var_in, kernel_shape=kernel_shape)
    var_out.diff = np.ones(var_out.shape)

    for oc in range(check_layer.weights.shape[0]):
        for ic in range(check_layer.weights.shape[1]):
            for kx in range(check_layer.weights.shape[2]):
                for ky in range(check_layer.weights.shape[3]):
                    in_data = np.random.standard_normal(var_in.shape)

                    x0 = check_layer.weights.data.copy()
                    x1 = check_layer.weights.data.copy()
                    x2 = check_layer.weights.data.copy()

                    eps = 0.00001
                    x1[oc, ic, kx, ky] -= eps
                    x2[oc, ic, kx, ky] += eps

                    var_in.data = in_data

                    check_layer.weights.data = x0
                    var_out.eval()          # fp
                    var_in.diff_eval()      # bp
                    d0 = check_layer.weights.diff[oc, ic, kx, ky]

                    check_layer.weights.data = x1
                    var_out.eval()          # fp
                    d1 = np.sum(var_out.data)

                    check_layer.weights.data = x2
                    var_out.eval()          # fp
                    d2 = np.sum(var_out.data)

                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0 - dx) > 0.000001:
                        print('d0:%f d2-1:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_conv2d_bias():
    var_in = cmodel.CTensor(name='in', shape=(8, 5, 10, 10), init='zero')
    kernel_shape = (12, 5, 3, 3)
    check_layer, var_out = layers.conv2d(name='conv2d', input_tensor=var_in, kernel_shape=kernel_shape)
    var_out.diff = np.ones(var_out.shape)

    for b in range(check_layer.bias.shape[0]):

        in_data = np.random.standard_normal(var_in.shape)

        x0 = check_layer.bias.data.copy()
        x1 = check_layer.bias.data.copy()
        x2 = check_layer.bias.data.copy()

        eps = 0.00001
        x1[b] -= eps
        x2[b] += eps

        var_in.data = in_data

        check_layer.bias.data = x0
        var_out.eval()          # fp
        var_in.diff_eval()      # bp
        d0 = check_layer.bias.diff[b]

        check_layer.bias.data = x1
        var_out.eval()          # fp
        d1 = np.sum(var_out.data)

        check_layer.bias.data = x2
        var_out.eval()          # fp
        d2 = np.sum(var_out.data)

        dx = (d2 - d1) / 2.0 / eps

        if math.fabs(d0 - dx) > 0.000001:
            print('d0:%f d2-1:%f' % (d0, dx))
            raise Exception('gradient check error.')


def check_fc_x():
    var_in = cmodel.CTensor(name='in', shape=(8, 5, 10, 10), init='zero')
    check_layer, var_out = layers.full_connect(name='fc', input_tensor=var_in, output_num=9)
    var_out.diff = np.ones(var_out.shape)

    for k in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):
            for i in range(var_in.shape[2]):
                for j in range(var_in.shape[3]):

                    in_data = np.random.standard_normal(var_in.shape)

                    x0 = in_data.copy()
                    x1 = in_data.copy()
                    x2 = in_data.copy()

                    eps = 0.00001
                    x1[k, c, i, j] -= eps
                    x2[k, c, i, j] += eps

                    var_in.data = x0
                    var_out.eval()          # fp
                    var_in.diff_eval()      # bp
                    d0 = var_in.diff[k, c, i, j]

                    var_in.data = x1
                    var_out.eval()  # fp
                    d1 = np.sum(var_out.data)

                    var_in.data = x2
                    var_out.eval()  # fp
                    d2 = np.sum(var_out.data)

                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f d2-1:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_fc_weights():
    var_in = cmodel.CTensor(name='in', shape=(8, 5, 10, 10), init='zero')
    check_layer, var_out = layers.full_connect(name='fc', input_tensor=var_in, output_num=9)
    var_out.diff = np.ones(var_out.shape)

    for i in range(check_layer.weights.shape[0]):
        for o in range(check_layer.weights.shape[1]):
            in_data = np.random.standard_normal(var_in.shape)

            x0 = check_layer.weights.data.copy()
            x1 = check_layer.weights.data.copy()
            x2 = check_layer.weights.data.copy()

            eps = 0.00001
            x1[i, o] -= eps
            x2[i, o] += eps

            var_in.data = in_data

            check_layer.weights.data = x0
            var_out.eval()  # fp
            var_in.diff_eval()  # bp
            d0 = check_layer.weights.diff[i, o]

            check_layer.weights.data = x1
            var_out.eval()  # fp
            d1 = np.sum(var_out.data)

            check_layer.weights.data = x2
            var_out.eval()  # fp
            d2 = np.sum(var_out.data)

            dx = (d2 - d1) / 2.0 / eps

            if math.fabs(d0 - dx) > 0.000001:
                print('d0:%f d2-1:%f' % (d0, dx))
                raise Exception('gradient check error.')


def check_fc_bias():
    var_in = cmodel.CTensor(name='in', shape=(8, 5, 10, 10), init='zero')
    check_layer, var_out = layers.full_connect(name='fc', input_tensor=var_in, output_num=9)
    var_out.diff = np.ones(var_out.shape)

    for b in range(check_layer.bias.shape[0]):

        in_data = np.random.standard_normal(var_in.shape)

        x0 = check_layer.bias.data.copy()
        x1 = check_layer.bias.data.copy()
        x2 = check_layer.bias.data.copy()

        eps = 0.00001
        x1[b] -= eps
        x2[b] += eps

        var_in.data = in_data

        check_layer.bias.data = x0
        var_out.eval()          # fp
        var_in.diff_eval()      # bp
        d0 = check_layer.bias.diff[b]

        check_layer.bias.data = x1
        var_out.eval()          # fp
        d1 = np.sum(var_out.data)

        check_layer.bias.data = x2
        var_out.eval()          # fp
        d2 = np.sum(var_out.data)

        dx = (d2 - d1) / 2.0 / eps

        if math.fabs(d0 - dx) > 0.000001:
            print('d0:%f d2-1:%f' % (d0, dx))
            raise Exception('gradient check error.')


def check_bn_x():
    var_in = cmodel.CTensor(name='in', shape=(8, 5, 10, 10), init='zero')
    check_layer, var_out = layers.batch_normal(name='bn', input_tensor=var_in)
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
                    var_out.eval()  # fp
                    d2 = np.sum(var_out.data)

                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f d2-1:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_bn_gama():
    var_in = cmodel.CTensor(name='in', shape=(8, 5, 10, 10), init='zero')
    check_layer, var_out = layers.batch_normal(name='bn', input_tensor=var_in)
    var_out.diff = np.ones(var_out.shape)

    for g in range(check_layer.gamma.shape[0]):

        in_data = np.random.standard_normal(var_in.shape)

        x0 = check_layer.gamma.data.copy()
        x1 = check_layer.gamma.data.copy()
        x2 = check_layer.gamma.data.copy()

        eps = 0.00001
        x1[g] -= eps
        x2[g] += eps

        var_in.data = in_data

        check_layer.gamma.data = x0
        var_out.eval()          # fp
        var_in.diff_eval()      # bp
        d0 = check_layer.gamma.diff[g]

        check_layer.gamma.data = x1
        var_out.eval()          # fp
        d1 = np.sum(var_out.data)

        check_layer.gamma.data = x2
        var_out.eval()          # fp
        d2 = np.sum(var_out.data)

        dx = (d2 - d1) / 2.0 / eps

        if math.fabs(d0 - dx) > 0.000001:
            print('d0:%f d2-1:%f' % (d0, dx))
            raise Exception('gradient check error.')


def check_bn_beta():
    var_in = cmodel.CTensor(name='in', shape=(8, 5, 10, 10), init='zero')
    check_layer, var_out = layers.batch_normal(name='bn', input_tensor=var_in)
    var_out.diff = np.ones(var_out.shape)

    for b in range(check_layer.beta.shape[0]):

        in_data = np.random.standard_normal(var_in.shape)

        x0 = check_layer.beta.data.copy()
        x1 = check_layer.beta.data.copy()
        x2 = check_layer.beta.data.copy()

        eps = 0.00001
        x1[b] -= eps
        x2[b] += eps

        var_in.data = in_data

        check_layer.beta.data = x0
        var_out.eval()          # fp
        var_in.diff_eval()      # bp
        d0 = check_layer.beta.diff[b]

        check_layer.beta.data = x1
        var_out.eval()          # fp
        d1 = np.sum(var_out.data)

        check_layer.beta.data = x2
        var_out.eval()          # fp
        d2 = np.sum(var_out.data)

        dx = (d2 - d1) / 2.0 / eps

        if math.fabs(d0 - dx) > 0.000001:
            print('d0:%f d2-1:%f' % (d0, dx))
            raise Exception('gradient check error.')


def check_softmax_x():
    var_in = cmodel.CTensor(name='in', shape=(5, 1, 10, 10), init='zero')
    check_layer, var_out = layers.softmax(name='softmax', input_tensor=var_in)
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
                    var_out.eval()  # fp
                    d2 = np.sum(var_out.data)
                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_dropout_x():
    var_in = cmodel.CTensor(name='in', shape=(8, 5, 10, 10), init='zero')
    check_layer, var_out = layers.dropout(name='dropout', input_tensor=var_in, phase='train', prob=0.9)
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
                    var_out.eval()  # fp
                    d2 = np.sum(var_out.data)
                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_max_pooling_x():
    var_in = cmodel.CTensor(name='in', shape=(1, 1, 16, 16), init='zero')
    check_layer, var_out = layers.max_pooling(name='max_pooling', input_tensor=var_in, ksize=2, stride=2)
    var_out.diff = np.ones(var_out.shape)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):
            for i in range(var_in.shape[2]):
                for j in range(var_in.shape[3]):

                    in_data = np.random.standard_normal(var_in.shape)
                    check_layer.init_argmax = False
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
                    var_out.eval()  # fp
                    d2 = np.sum(var_out.data)
                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_avg_pooling_x():
    var_in = cmodel.CTensor(name='in', shape=(4, 5, 8, 8), init='zero')
    check_layer, var_out = layers.avg_pooling(name='avg_pooling', input_tensor=var_in, ksize=2, stride=2)
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
                    var_out.eval()  # fp
                    d2 = np.sum(var_out.data)
                    dx = (d2 - d1) / 2.0 / eps

                    if math.fabs(d0-dx) > 0.000001:
                        print('d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')


def check_concat_slice_x():
    var_in = cmodel.CTensor(name='in', shape=(8, 10, 6, 6), init='zero')

    slice_shape = ((8, 1, 6, 6), (8, 2, 6, 6), (8, 4, 6, 6), (8, 3, 6, 6))
    slice_layer, slice_out = layers.slice_(name='slice', input_tensor=var_in, output_shape=slice_shape)
    concat_layer, var_out = layers.concat_(name='concat', input_tensor=slice_out)

    var_in.data = np.random.standard_normal(var_in.shape)
    var_out.diff = np.random.standard_normal(var_out.shape)

    var_out.eval()
    var_in.diff_eval()

    d1 = np.sum(np.square(var_in.data) - np.square(var_out.data))
    d2 = np.sum(np.square(var_in.diff) - np.square(var_out.diff))

    if math.fabs(d1) > 0.000001:
        print('d1:%f d2:%f' % (d1, d2))
        raise Exception('gradient check error.')
    if math.fabs(d2) > 0.000001:
        print('d1:%f d2:%f' % (d1, d2))
        raise Exception('gradient check error.')


if __name__ == "__main__":
    # check_conv2d_x()
    # check_conv2d_weights()
    # check_conv2d_bias()

    # check_fc_x()
    # check_fc_weights()
    # check_fc_bias()
    #
    # check_bn_x()
    # check_bn_gama()
    # check_bn_beta()
    #
    # check_dropout_x()
    check_max_pooling_x()
    # check_avg_pooling_x()

    # check_softmax_x()
    # check_concat_slice_x()
