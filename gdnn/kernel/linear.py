import numpy as np
import pyopencl as cl
import pyopencl.cltypes
import gdnn.platform as gp


class CsVector(object):
    def __init__(self, context: gp.CsContext, queue: gp.CsQueue, program: gp.CsProgram):
        self.context = context
        self.queue = queue
        self.program = program

        self.add_local_size = 256
        self.add_group_size = 64
        self.add_global_size = self.add_group_size * self.add_local_size
        self.add_kernel = gp.CsKernel(self.program)
        self.add_kernel.set_kernel('vector_add', (self.add_global_size,), (self.add_local_size,))

        self.mul_local_size = 256
        self.mul_group_size = 64
        self.mul_global_size = self.mul_group_size * self.mul_local_size
        self.mul_kernel = gp.CsKernel(self.program)
        self.mul_kernel.set_kernel('vector_mul', (self.mul_global_size,), (self.mul_local_size,))

        self.sum_local_size = 512
        self.sum_group_size = 16
        self.sum_global_size = self.sum_group_size * self.sum_local_size
        self.sum_kernel = gp.CsKernel(self.program)
        self.sum_kernel.set_kernel('vector_sum', (self.sum_global_size,), (self.sum_local_size,))

    def vadd(self, a: np.ndarray, b: np.ndarray):
        c = np.empty_like(a)
        p1 = self.context.get_buffer(a, cl.mem_flags.READ_ONLY)
        p2 = self.context.get_buffer(b, cl.mem_flags.READ_ONLY)
        p3 = self.context.malloc_buffer(c.nbytes, cl.mem_flags.WRITE_ONLY)

        self.add_kernel.set_param([p1, p2, p3, cl.cltypes.uint(a.shape[0])])
        self.queue.range_kernel(self.add_kernel)
        self.queue.copy(c, p3)
        return c

    def vmul(self, a: np.ndarray, b: np.ndarray):
        c = np.empty_like(a)
        p1 = self.context.get_buffer(a, cl.mem_flags.READ_ONLY)
        p2 = self.context.get_buffer(b, cl.mem_flags.READ_ONLY)
        p3 = self.context.malloc_buffer(c.nbytes, cl.mem_flags.WRITE_ONLY)

        self.mul_kernel.set_param([p1, p2, p3, cl.cltypes.uint(a.shape[0])])
        self.queue.range_kernel(self.mul_kernel)
        self.queue.copy(c, p3)
        return c

    def vsum(self, a: np.ndarray):
        c = np.zeros([self.sum_local_size], dtype=np.float32)

        p1 = self.context.get_buffer(a, cl.mem_flags.READ_ONLY)
        ps = self.context.malloc_buffer(c.nbytes, cl.mem_flags.WRITE_ONLY)

        self.sum_kernel.set_param([p1, ps, cl.LocalMemory(c.nbytes), np.uint32(a.shape[0])])
        self.queue.range_kernel(self.sum_kernel)
        self.queue.copy(c, ps)
        return c.sum()