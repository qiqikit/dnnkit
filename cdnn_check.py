import time
import cdnn.gradient_check.layers as layer_check
import cdnn.gradient_check.activations as activation_check
import cdnn.gradient_check.loss as loss_check

if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # layers gradient check
    print('gradient check conv2d...')
    layer_check.check_conv2d_x()
    layer_check.check_conv2d_weights()
    layer_check.check_conv2d_bias()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check fc...')
    layer_check.check_fc_x()
    layer_check.check_fc_weights()
    layer_check.check_fc_bias()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check bn...')
    layer_check.check_bn_x()
    layer_check.check_bn_gama()
    layer_check.check_bn_beta()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check dropout...')
    layer_check.check_dropout_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check max_pooling...')
    layer_check.check_max_pooling_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check avg_pooling...')
    layer_check.check_avg_pooling_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check concat slice...')
    layer_check.check_concat_slice_x()

    # activations gradient check
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check relu...')
    activation_check.check_relu_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check lrelu...')
    activation_check.check_lrelu_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check softmax...')
    layer_check.check_softmax_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check sigmoid...')
    activation_check.check_sigmoid_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check tanh...')
    activation_check.check_tanh_x()

    # losses gradient check
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check softmax_withloss...')
    loss_check.check_softmax_withloss_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check cross_entropy_binary...')
    loss_check.check_cross_entropy_binary_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check cross_entropy_categorical...')
    loss_check.check_cross_entropy_categorical_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check mean_squared_error...')
    loss_check.check_mean_squared_error_x()

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('gradient check finish.')
