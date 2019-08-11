import pyopencl as cl
import pyopencl.cltypes
import gdnn.platform as gp


class CsPooling(object):
    def __init__(self, program: gp.CsProgram, queue: gp.CsQueue):
        self.program = program
        self.queue = queue

        self.max_local_size = 256
        self.max_group_size = 64
        self.max_global_size = self.max_group_size * self.max_local_size

        self.dmax_local_size = 256
        self.dmax_group_size = 64
        self.dmax_global_size = self.dmax_group_size * self.dmax_local_size

        self.argmax_kernel = gp.CsKernel(self.program)
        self.argmax_kernel.set_kernel('pooling_argmax', (self.max_global_size,), (self.max_local_size,))

        self.valmax_kernel = gp.CsKernel(self.program)
        self.valmax_kernel.set_kernel('pooling_valmax', (self.max_global_size,), (self.max_local_size,))

        self.clrdiff_kernel = gp.CsKernel(self.program)
        self.clrdiff_kernel.set_kernel('pooling_clrdiff', (self.dmax_global_size,), (self.dmax_local_size,))

        self.dmax_kernel = gp.CsKernel(self.program)
        self.dmax_kernel.set_kernel('pooling_dmax', (self.dmax_global_size,), (self.dmax_local_size,))

        self.dmax2_kernel = gp.CsKernel(self.program)
        self.dmax2_kernel.set_kernel('pooling_dmax2', (self.dmax_global_size,), (self.dmax_local_size,))

    def max_index(self, input_array_buffer: cl.Buffer, max_index_buffer: cl.Buffer,
                        input_shape: tuple, input_param: tuple):

        int4_shape = cl.cltypes.make_int4(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        int4_param = cl.cltypes.make_int4(input_param[0], input_param[1], input_param[2], input_param[3])

        p1 = input_array_buffer
        p2 = max_index_buffer
        p3 = int4_shape
        p4 = int4_param

        self.argmax_kernel.set_param([p1, p2, p3, p4])
        self.queue.range_kernel(self.argmax_kernel)

    def max_value(self, input_array_buffer: cl.Buffer, max_index_buffer: cl.Buffer, max_value_buffer: cl.Buffer,
                        output_len: int):

        p1 = input_array_buffer
        p2 = max_index_buffer
        p3 = max_value_buffer
        p4 = cl.cltypes.uint(output_len)

        self.valmax_kernel.set_param([p1, p2, p3, p4])
        self.queue.range_kernel(self.valmax_kernel)

    def clear_diff(self, diff: cl.Buffer, diff_len: int):

        self.clrdiff_kernel.set_param([diff, cl.cltypes.uint(diff_len)])
        self.queue.range_kernel(self.clrdiff_kernel)

    def dmax_value(self, output_diff_buffer: cl.Buffer, max_index_buffer: cl.Buffer, input_diff_buffer: cl.Buffer,
                 output_shape: tuple, input_param: tuple):

        int4_shape = cl.cltypes.make_int4(output_shape[0], output_shape[1], output_shape[2], output_shape[3])
        # in dmax_value(): ksize_width = stride_h and ksize_height = stride_v
        # otherwise use dmax_value2
        assert input_param[0] == input_param[2] and input_param[1] == input_param[3]

        p1 = output_diff_buffer
        p2 = max_index_buffer
        p3 = input_diff_buffer
        p4 = int4_shape

        self.dmax_kernel.set_param([p1, p2, p3, p4])
        self.queue.range_kernel(self.dmax_kernel)

    def dmax_value2(self, output_diff_buffer: cl.Buffer, max_index_buffer: cl.Buffer, input_diff_buffer: cl.Buffer,
                 input_shape: tuple, input_param: tuple):

        int4_shape = cl.cltypes.make_int4(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        int4_param = cl.cltypes.make_int4(input_param[0], input_param[1], input_param[2], input_param[3])

        p1 = output_diff_buffer
        p2 = max_index_buffer
        p3 = input_diff_buffer
        p4 = int4_shape
        p5 = int4_param

        self.dmax2_kernel.set_param([p1, p2, p3, p4, p5])
        self.queue.range_kernel(self.dmax2_kernel)