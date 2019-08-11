import math
import os
import sys
import numpy as np
import cdnn.model as cmodel
import gdnn.model as gmodel
import gdnn.platform as gpm
import gdnn.layers as glayers
import gdnn.kernel.pooling as gkp


def check_max_pooling_x():
    platform = gpm.CsPlatform()
    platform.print_platform_info()

    context = gpm.CsContext(platform_id=0)
    queue = gpm.CsQueue(context, device_id=0)

    # cl programs
    clp_pooling = gpm.CsProgram(context)
    clp_pooling.load('gdnn/kernel/pooling.cl', 'source', [])

    # cl classes
    clc_max_pooling = gkp.CsPooling(clp_pooling, queue)

    var_in = cmodel.CTensor(name='in', shape=(2, 2, 32, 32), init='zero')

    cpu2gpu1, cpu2gpu1_out = glayers.array_to_buffer(name='cpu2gpu1', context=context, queue=queue, input_tensor=var_in)

    gpooling1, gpooling1_out = glayers.max_pooling(name='gpooling1', context=context, clc_pooling=clc_max_pooling,
                                                   input_tensor=cpu2gpu1_out, ksize=2, stride=2)

    gpu2cpu1, var_out = glayers.buffer_to_array(name='gpu2cpu1', context=context, queue=queue, input_tensor=gpooling1_out)

    var_out.diff = np.ones(var_out.shape, dtype=np.float32)

    for b in range(var_in.shape[0]):
        for c in range(var_in.shape[1]):
            for i in range(var_in.shape[2]):
                for j in range(var_in.shape[3]):

                    in_data = np.random.standard_normal(var_in.shape) * 10
                    gpooling1.init_argmax = False
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

                    # print('test')
                    # print(var_in.data)
                    # print(var_out.data)
                    # print(var_in.diff)

                    if math.fabs(d0-dx) > 0.2:
                        print('xx d0:%f dx:%f' % (d0, dx))
                        raise Exception('gradient check error.')
