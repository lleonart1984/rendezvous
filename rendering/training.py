import torch
import torch.nn as nn
from rendering.manager import *


class RendererModule(nn.Module):
    r"""
    Module that performs gradient-based operations on graphics or raytracing pipelines.
    :ivar device: Device manager used to render and create resources
    """
    def __init__(self,
                 device: DeviceManager,
                 *args, **kwargs):
        """ Initializes the render module using description for input tensors, parameters and outputs
        """
        super().__init__()
        self.device = device
        if device not in RendererModule.__CACHED_BUFFERS:
            RendererModule.__CACHED_BUFFERS[device] = {}
        self._cached_buffers = RendererModule.__CACHED_BUFFERS[device]
        self._used_buffers = { }
        self.setup()

    __CACHED_BUFFERS = { }

    @staticmethod
    def clear_cache():
        RendererModule.__CACHED_BUFFERS = { }

    def tensor_to_buffer(self, tensor: torch.Tensor, copy_data: bool = True, buffer: Buffer = None) -> Buffer:
        if tensor is None:
            return None
        assert buffer is None or buffer.size == torch.numel(tensor)*4, "Reused buffer has not the required size"
        if buffer is None:
            buffer: Buffer = self._resolve_buffer(torch.numel(tensor)*4)
        self._used_buffers[buffer] = tensor.shape  # save shape for recover later
        if copy_data:
            if tensor.is_cuda:
                buffer.write_gpu_tensor(tensor)
            else:
                buffer.write_direct(tensor)
        return buffer

    def create_buffer(self, shape) -> Buffer:
        num_elements = math.prod(shape)
        buffer = self._resolve_buffer(num_elements * 4)
        self._used_buffers[buffer] = shape
        return buffer

    def buffer_to_tensor(self, buffer: Buffer) -> torch.Tensor:
        assert buffer in self._used_buffers, "Tensor should be turned to a buffer first"
        self._cached_buffers[buffer.size].append(buffer)  # for future reuse
        return buffer.as_gpu_tensor().clone().reshape(self._used_buffers.pop(buffer))

    def _resolve_buffer(self, required_size) -> Buffer:
        if required_size == 0:
            return None
        if required_size not in self._cached_buffers:
            self._cached_buffers[required_size] = []
        if len(self._cached_buffers[required_size]) == 0:
            buffer = self.device.create_buffer(
                required_size,
                BufferUsage.GPU_ADDRESS | BufferUsage.STORAGE | BufferUsage.TRANSFER_SRC | BufferUsage.TRANSFER_DST,
                MemoryProperty.GPU
            )
            self._cached_buffers[required_size].append(buffer)
        return self._cached_buffers[required_size].pop()

    def _free_buffers(self):
        for b in self._used_buffers:
            self._cached_buffers[b.size].append(b)
        self._used_buffers = {}

    def setup(self):
        """
        Creates resources, techniques necessary for rendering process.
        Pipeline should bind buffers provided by the pipeline object and fill them in forward and backward methods.
        """
        pass

    def forward_render(self, input: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the output given the parameters
        """
        pass

    def backward_render(self, input: List[torch.Tensor], output: List[torch.Tensor], output_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the gradient of parameters given the outputs, the gradients of outputs and the original inputs
        """
        pass

    def forward(self, *args):
        outputs = TrainableRendererFunction.apply(*(list(args) + [self]))
        return outputs[0] if len(outputs) == 1 else outputs


# class ParamsRendererModule(RendererModule):
#     def create_params(self) -> List[torch.Tensor]:
#         pass
#
#     def __init__(self,
#                  device: DeviceManager,
#                  input_args: int,
#                  output_args: int,
#                  *args, **kwargs):
#         params = self.create_params()
#         super().__init__(device, input_args + len(params), output_args, *args, **kwargs)
#         self.params_module = nn.ParameterList(nn.Parameter(t) for t in params)
#
#     def get_param_tensor(self, index: int = 0):
#         return self.params_module[index]
#
#     def forward(self, *args):
#         return super().forward(*(list(args) + list(self.params_module)))


class TrainableRendererFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        renderer: RendererModule
        args = list(args)
        renderer = args[-1]
        inputs = args[0:-1]
        ctx.renderer = renderer
        ctx.number_of_inputs = len(inputs)
        clone_inputs = [t.clone() for t in inputs]
        outputs = renderer.forward_render(inputs)
        clone_outputs = [t.clone() for t in outputs]
        ctx.save_for_backward(*(clone_inputs + clone_outputs))
        # ctx.save_for_backward(*(args[0:-1] + [o.detach().clone() for o in outputs]))
        renderer._free_buffers()
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        saved_tensors = list(ctx.saved_tensors)
        renderer = ctx.renderer
        number_of_inputs = ctx.number_of_inputs
        inputs = saved_tensors[0:number_of_inputs] # get saved input tensors
        outputs = saved_tensors[number_of_inputs:] # get saved output tensors
        grad_outputs = list(args) # get output gradients
        grad_inputs = renderer.backward_render(inputs, outputs, grad_outputs)
        renderer._free_buffers()
        return tuple(grad_inputs + [None])  # append None to refer to renderer object passed in forward
