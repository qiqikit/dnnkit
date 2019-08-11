import time
import gdnn.gradient_check.layers as layers_check

if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    print('gradient check check_max_pooling_x...')
    layers_check.check_max_pooling_x()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
