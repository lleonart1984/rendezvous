import torch
import torch.nn as nn
from rendering.manager import *


class RendererModule(nn.Module):
    r"""
    Module that performs gradient-based operations on graphics or raytracing pipelines.
    :ivar device: Device manager used to render
    :ivar input_sizes: list of int indicating the input tensor sizes.
    :ivar input_trainable: list of bool indicating if each input is backprop. If None all input is considered trainable
    :ivar parameter_sizes: list of int indicating the sizes of all internal trainable parameter tensor sizes.
    :ivar output_sizes: list of int indicating the sizes of all output tensor sizes.
    """
    def __init__(self,
                 device: DeviceManager,
                 input_args: int,
                 output_args: int,
                 *args, **kwargs):
        """ Initializes the render module using description for input tensors, parameters and outputs
        """
        super().__init__()
        self.device = device
        self.input_buffers = [None] * input_args
        self.input_grad_buffers = [None] * input_args
        self.output_buffers = [None] * output_args
        self.output_grad_buffers = [None] * output_args
        self.cached_buffers = { }
        self.setup()

    def get_input(self, index = 0):
        return self.input_buffers[index]

    def get_input_gradient(self, index = 0):
        return self.input_grad_buffers[index]

    def get_output(self, index = 0):
        return self.output_buffers[index]

    def get_output_gradient(self, index = 0):
        return self.output_grad_buffers[index]

    def create_output_tensors(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        When implemented creates the tensors for the output
        """
        pass

    def _resolve_buffer(self, required_size):
        if required_size == 0:
            return None
        if required_size not in self.cached_buffers:
            self.cached_buffers[required_size] = []
        if len(self.cached_buffers[required_size]) == 0:
            buffer = self.device.create_buffer(
                required_size,
                BufferUsage.GPU_ADDRESS | BufferUsage.STORAGE | BufferUsage.TRANSFER_SRC | BufferUsage.TRANSFER_DST,
                MemoryProperty.GPU
            )
            self.cached_buffers[required_size].append(buffer)
        return self.cached_buffers[required_size].pop()

    def resolve_inputs(self, sizes):
        assert len(sizes) == len(self.input_buffers), "Incorrect number of input tensors"
        self.input_buffers = [self._resolve_buffer(size) for size in sizes]

    def resolve_input_gradients(self, sizes):
        assert len(sizes) == len(self.input_grad_buffers), "Incorrect number of input tensors"
        self.input_grad_buffers = [self._resolve_buffer(size) for size in sizes]

    def resolve_outputs(self, sizes):
        assert len(sizes) == len(self.output_buffers), "Incorrect number of input tensors"
        self.output_buffers = [self._resolve_buffer(size) for size in sizes]

    def resolve_output_gradients(self, sizes):
        assert len(sizes) == len(self.output_grad_buffers), "Incorrect number of input tensors"
        self.output_grad_buffers = [self._resolve_buffer(size) for size in sizes]

    def free_buffers(self):
        for b in self.input_buffers + self.input_grad_buffers + self.output_buffers + self.output_grad_buffers:
            if b:
                self.cached_buffers[b.size].append(b)

    def setup(self):
        """
        Creates resources, techniques necessary for rendering process.
        Pipeline should bind buffers provided by get_input, get_input_gradient, get_output and get_output_gradient.
        Those buffers will be created/updated with input tensors and backprop gradients
        """
        pass

    def forward_render(self, input_shapes, output_shapes):
        """
        Computes the output given the parameters
        """
        pass

    def backward_render(self, input_shapes, output_shapes):
        """
        Computes the gradient of parameters given the gradients of outputs and the original inputs
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
    def copy_tensor_to_buffer(tensor: torch.Tensor, buffer: Buffer, cpu=True):
        assert torch.numel(tensor)*4 == buffer.size, "Tensor and buffer have different sizes!"
        if cpu:
            buffer.write_direct(tensor)
        else:
            buffer.write_gpu_tensor(tensor)
            # Check
            # check = torch.zeros_like(tensor, device=torch.device('cpu'))
            # buffer.read_direct(check)
            # difference = (check - tensor.to(torch.device('cpu'))).sum()
            # assert difference.item() == 0, "Tensor couldnt copy correclty to buffer"+str(difference.item())

    @staticmethod
    def copy_buffer_to_tensor(tensor, buffer, cpu=True):
        assert torch.numel(tensor)*4 == buffer.size, "Tensor and buffer have different sizes!"
        if cpu:
            buffer.read_direct(tensor)
        else:
            buffer.read_gpu_tensor(tensor)
            # Check
            # check = torch.zeros_like(tensor, device=torch.device('cpu'))
            # buffer.read_direct(check)
            # difference = (check - tensor.to(torch.device('cpu'))).sum()
            # assert difference.item() == 0, "Tensor couldnt copy correclty to buffer"+str(difference.item())


    @staticmethod
    def forward(ctx, *args):
        renderer: RendererModule
        args = list(args)
        renderer = args[-1]
        inputs = args[0:-1]
        outputs = renderer.create_output_tensors(inputs)
        input_shapes = [t.shape for t in inputs]
        output_shapes = [t.shape for t in outputs]
        renderer.resolve_inputs([torch.numel(t)*4 for t in inputs])
        renderer.resolve_outputs([torch.numel(t)*4 for t in outputs])
        ctx.renderer = renderer
        ctx.devices = [t.device == torch.device('cpu') for t in inputs], \
                      [t.device == torch.device('cpu') for t in outputs]
        for i, i_b in zip(inputs, renderer.input_buffers):
            TrainableRendererFunction.copy_tensor_to_buffer(i, i_b, i.device == torch.device('cpu'))
        renderer.forward_render(input_shapes, output_shapes)
        for o, o_b in zip(outputs, renderer.output_buffers):
            TrainableRendererFunction.copy_buffer_to_tensor(o, o_b, o.device == torch.device('cpu'))
        ctx.save_for_backward(*(args[0:-1] + [o.detach().clone() for o in outputs]))
        renderer.free_buffers()
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        saved_tensors = list(ctx.saved_tensors)
        renderer = ctx.renderer
        inputs = saved_tensors[0:len(renderer.input_buffers)] # get saved input tensors
        outputs = saved_tensors[len(renderer.input_buffers):] # get saved output tensors
        grad_outputs = list(args) # get output gradients

        # import matplotlib.pyplot as plt
        # plt.imshow(args[0].detach().reshape(16,16,3).sum(dim=-1).cpu().numpy(), vmin=-10, vmax=10)
        # plt.show()

        grad_inputs = [None if not i.requires_grad else torch.zeros_like(i) for i in inputs] # create input gradients
        input_devices, output_devices = ctx.devices
        input_shapes = [t.shape for t in inputs]
        output_shapes = [t.shape for t in outputs]
        renderer.resolve_inputs([torch.numel(t)*4 for t in inputs])
        renderer.resolve_input_gradients([0 if not t.requires_grad else torch.numel(t)*4 for t in inputs])  # gradients has same dimensions
        renderer.resolve_outputs([torch.numel(t)*4 for t in outputs])
        renderer.resolve_output_gradients([torch.numel(t)*4 for t in grad_outputs])
        for i, i_b, dev in zip(inputs, renderer.input_buffers, input_devices):
            TrainableRendererFunction.copy_tensor_to_buffer(i, i_b, dev)
        for o, o_b, dev in zip(outputs, renderer.output_buffers, output_devices):
            TrainableRendererFunction.copy_tensor_to_buffer(o, o_b, dev)
        for go, go_b, dev in zip(grad_outputs, renderer.output_grad_buffers, output_devices):
            TrainableRendererFunction.copy_tensor_to_buffer(go, go_b, dev)
        renderer.backward_render(input_shapes, output_shapes)
        for i, gi, gi_b, dev in zip(inputs, grad_inputs, renderer.input_grad_buffers, input_devices):
            if i.requires_grad:
                TrainableRendererFunction.copy_buffer_to_tensor(gi, gi_b, dev)
        renderer.free_buffers()
        return tuple(grad_inputs + [None])  # append None to refer to renderer object passed in forward
