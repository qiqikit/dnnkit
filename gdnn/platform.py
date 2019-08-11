import pyopencl as cl
import numpy as np
# import os
# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


class CsPlatform(object):
    def __init__(self):
        self.platforms = []
        self.loaded = False

    def load_platform_info(self, show_info: bool):
        self.platforms = cl.get_platforms()

        if show_info:
            if len(self.platforms) == 0:
                print('no opencl platforms was found.')

            for platform in self.platforms:
                # EXTENSIONS NAME PROFILE VENDOR VERSION
                attr_platform_name = getattr(cl.platform_info, 'NAME')
                print(platform.get_info(attr_platform_name))

                devices = platform.get_devices()
                if len(devices) == 0:
                    print('no opencl devices was found.')

                for device in devices:
                    attr_device_name = getattr(cl.device_info, 'NAME')
                    device_name = device.get_info(attr_device_name)
                    print(device_name)

        self.loaded = True
        return len(self.platforms)

    def get_platform_count(self):
        if not self.loaded:
            self.load_platform_info(False)
        return len(self.platforms)

    def print_platform_info(self):
        self.load_platform_info(True)

    def get_platform(self, platform_id: int):
        if not self.loaded:
            self.load_platform_info(False)
        return self.platforms[platform_id]


class CsContext(object):
    def __init__(self, platform_id: int):
        platforms = cl.get_platforms()

        self.platform = platforms[platform_id]
        self.context = None
        self.devices = []
        self.loaded = False

    def get_context(self):
        if not self.loaded:
            self.load_context_info(False)
        return self.context

    def load_context_info(self, show_info: bool):
        self.context = cl.Context(properties=[(cl.context_properties.PLATFORM, self.platform)])
        attr_context_devices = getattr(cl.context_info, 'DEVICES')
        self.devices = self.context.get_info(attr_context_devices)

        if show_info:
            if len(self.devices) == 0:
                print('no opencl devices was found.')

            for device in self.devices:
                attr_device_name = getattr(cl.device_info, 'NAME')
                device_name = device.get_info(attr_device_name)
                print(device_name)

        self.loaded = True

    def get_device_count(self):
        if not self.loaded:
            self.load_context_info(False)
        return len(self.devices)

    def get_devices(self):
        if not self.loaded:
            self.load_context_info(False)
        return self.devices

    def get_device(self, device_id: int):
        if not self.loaded:
            self.load_context_info(False)
        if device_id >= len(self.devices):
            raise Exception("device_id >= len(self.devices)")
        return self.devices[device_id]

    def get_buffer(self, a: np.ndarray, flag):
        if not self.loaded:
            self.load_context_info(False)
        return cl.Buffer(self.context, flag | cl.mem_flags.COPY_HOST_PTR, a.nbytes, hostbuf=a)

    def malloc_buffer(self, nbytes: int, flag):
        if not self.loaded:
            self.load_context_info(False)
        return cl.Buffer(self.context, flag, nbytes)


class CsProgram(object):
    def __init__(self, _context: CsContext):
        self.context = _context.get_context()
        self.devices = _context.get_devices()
        self.program = None
        self.ptype = ''
        self.loaded = False
        self.builded = False

    def get_program(self):
        if not self.loaded:
            raise Exception("self.program is not load")
        if not self.builded:
            raise Exception("self.program is not build")
        return self.program

    def get_context(self):
        return self.context

    def load(self, program: str, ptype: str, param: list):
        self.ptype = ptype

        if ptype == 'source':
            self.load_source(program)
            self.build_program(param)
        elif ptype == 'string':
            self.load_string(program)
            self.build_program(param)
        else:
            raise Exception('unknowned ptype: %s' % (ptype))

    def load_source(self, file_name):
        file = open(file_name, 'r')
        program_source = file.read()

        self.program = cl.Program(self.context, program_source)
        self.loaded = True

    def load_binary(self, file_name, device_id):
        raise Exception('load_binary not implement!')

    def load_string(self, program_source, param: list):
        self.program = cl.Program(self.context, program_source)
        self.loaded = True

    def build_program(self, param: list):
        if not self.loaded:
            raise Exception("self.program is not load")
        self.program.build(param, self.devices)
        self.builded = True


class CsKernel(object):
    def __init__(self, _program: CsProgram):
        self.context = _program.get_context()
        self.program = _program.get_program()
        self.kernel = None
        self.global_size = None
        self.work_size = None
        self.init_kernel = False
        self.init_param = False

    def set_kernel(self, kernel_name: str, global_size: tuple, work_size: tuple):
        self.kernel = cl.Kernel(self.program, kernel_name)
        self.global_size = global_size
        self.work_size = work_size
        self.init_kernel = True

    def set_global_size(self, global_size: tuple):
        self.global_size = global_size

    def set_work_size(self, work_size: tuple):
        self.work_size = work_size

    def set_arg(self, param_id: int, param_value):
        if not self.init_kernel:
            raise Exception('kernel not init_kernel')
        self.kernel.set_arg(param_id, param_value)
        self.init_param = True

    def set_param(self, param_list: list):
        if not self.init_kernel:
            raise Exception('kernel not init_kernel')
        param_id = 0
        for param_value in param_list:
            self.kernel.set_arg(param_id, param_value)
            param_id += 1
        self.init_param = True

    def get_kernel(self):
        if not self.init_kernel:
            raise Exception("kernel not init")
        if not self.init_param:
            raise Exception("param not init")
        return self.kernel

    def get_global_size(self):
        if not self.init_kernel:
            raise Exception("kernel not init")
        return self.global_size

    def get_group_size(self):
        if not self.init_kernel:
            raise Exception("kernel not init")
        return self.global_size / self.work_size

    def get_work_size(self):
        if not self.init_kernel:
            raise Exception("kernel not init")
        return self.work_size


class CsQueue(object):
    def __init__(self, _context: CsContext, device_id: int):
        self.context = _context.get_context()
        self.device_id = device_id
        self.queue = cl.CommandQueue(self.context, _context.get_device(device_id))

    def range_kernel(self, _kernel: CsKernel):
        cl.enqueue_nd_range_kernel(self.queue, _kernel.get_kernel(), _kernel.get_global_size(), _kernel.get_work_size())

    def get_buffer_size(self, buffer: cl.Buffer):
        attr_buffer_size = getattr(cl.mem_info, 'MEM_SIZE')
        return buffer.get_info(attr_buffer_size)

    def copy(self, array: np.ndarray, buffer: cl.Buffer):
        cl.enqueue_copy(self.queue, array, buffer)
