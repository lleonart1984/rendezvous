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
                 input_sizes,
                 params_sizes,
                 output_sizes,
                 input_trainable=None,
                 *args, **kwargs):
        """ Initializes the render module using description for input tensors, parameters and outputs
        """
        super().__init__()
        self.device = device
        self.input_sizes = input_sizes
        self.input_trainable = [True]*len(input_sizes) if input_trainable is None or input_trainable is True else \
            [False]*len(input_sizes) if input_trainable is False else input_trainable
        self.params_sizes = params_sizes
        self.output_sizes = output_sizes
        self.input_len = len(input_sizes)
        self.params_len = len(params_sizes)
        self.inputs = []
        self.grad_inputs = []
        for size, trainable in zip(self.input_sizes, self.input_trainable):
            self.inputs.append(device.create_buffer(
                size=size * 4,  # assuming all floats
                usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                memory=MemoryProperty.GPU
            ))
            if trainable:
                self.grad_inputs.append(device.create_buffer(
                    size=size * 4,  # assuming all floats
                    usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                    memory=MemoryProperty.GPU
                ))
            else:
                self.grad_inputs.append(None)
        self.params = []
        self.grad_params = []
        for size in self.params_sizes:
            self.params.append(device.create_buffer(
                size=size * 4,  # assuming all floats
                usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                memory=MemoryProperty.GPU
            ))
            self.grad_params.append(device.create_buffer(
                size=size * 4,  # assuming all floats
                usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                memory=MemoryProperty.GPU
            ))
        self.params_modules = nn.ParameterList(nn.Parameter(torch.zeros(p.size//4)) for p in self.params)
        self.outputs = []
        self.grad_outputs = []
        for size in self.output_sizes:
            self.outputs.append(device.create_buffer(
                size=size * 4,  # assuming all floats
                usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                memory=MemoryProperty.GPU
            ))
            self.grad_outputs.append(
                device.create_buffer(
                    size=size * 4,  # assuming all floats
                    usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                    memory=MemoryProperty.GPU
                ))
        self.setup()

    def get_input(self, index = 0):
        return self.inputs[index]

    def get_input_gradient(self, index = 0):
        return self.grad_inputs[index]

    def get_param(self, index = 0):
        return self.params[index]

    def get_param_tensor(self, index = 0):
        return self.params_modules[index]

    def get_param_gradient(self, index = 0):
        return self.grad_params[index]

    def get_output(self, index = 0):
        return self.outputs[index]

    def get_output_gradient(self, index = 0):
        return self.grad_outputs[index]

    def setup(self):
        """
        Creates resources, techniques necessary for rendering process
        """
        pass

    def forward_render(self):
        """
        Computes the output given the parameters
        """
        pass

    def backward_render(self):
        """
        Computes the gradient of parameters given the gradients of outputs and the original inputs
        """
        pass

    def forward_params(self):
        """
        If overriden, allows to manipulate the list of parameters before forward, for instance, clamping or mapping
        """
        return list(self.params_modules)

    def forward(self, *args):
        outputs = TrainableRendererFunction.apply(*(list(args) + self.forward_params() + [self]))
        return outputs[0] if len(outputs) == 1 else outputs


class TrainableRendererFunction(torch.autograd.Function):

    @staticmethod
    def copy_tensor_to_buffer(tensor, buffer, cpu=True):
        if cpu:
            buffer.write_direct(tensor)
        else:
            buffer.device.copy_gpu_pointer_to_buffer(tensor.data_ptr(), buffer, buffer.size)
        # device: DeviceManager = renderer.device
        # buffer.write_direct(tensor.to(torch.device('cpu')))

    @staticmethod
    def create_tensor_from_buffer(buffer, cpu=True):
        if cpu:
            tensor = torch.zeros(buffer.size // 4)
            buffer.read_direct(tensor)
            return tensor
        else:
            return buffer.create_gpu_tensor()
        # device: DeviceManager = renderer.device
        # tensor = torch.zeros(buffer.size // 4)
        # buffer.read_direct(tensor)
        # tensor = tensor.to(torch_device)
        # return tensor

    @staticmethod
    def resolve_device_for_output(inputs, params):
        if len(inputs) > 0:
            return inputs[0].device
        if len(params) > 0:
            return params[0].device
        return torch.device('cpu')

    @staticmethod
    def forward(ctx, *args):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        renderer: RendererModule
        args = list(args)
        renderer = args[-1]
        inputs = args[0:renderer.input_len]
        params = args[renderer.input_len: renderer.input_len + renderer.params_len]
        torch_device = TrainableRendererFunction.resolve_device_for_output(inputs, params)
        ctx.save_for_backward(*args[0:-1])
        ctx.renderer = renderer
        ctx.devices = [t.device == torch.device('cpu') for t in inputs], \
                      [t.device == torch.device('cpu') for t in params], \
                      torch_device == torch.device('cpu')
        for i, i_b in zip(inputs, renderer.inputs):
            TrainableRendererFunction.copy_tensor_to_buffer(i, i_b, i.device == torch.device('cpu'))
        for p, p_b in zip(params, renderer.params):
            TrainableRendererFunction.copy_tensor_to_buffer(p, p_b, p.device == torch.device('cpu'))
        renderer.forward_render()
        outputs = [
            TrainableRendererFunction.create_tensor_from_buffer(o_b, torch_device == torch.device('cpu'))
            for o_b in renderer.outputs
        ]
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_outputs = list(args)
        saved_tensors = list(ctx.saved_tensors)
        renderer = ctx.renderer
        input_devices, params_devices, output_devices = ctx.devices
        inputs = saved_tensors[0:renderer.input_len]
        params = saved_tensors[renderer.input_len:renderer.input_len + renderer.params_len]
        for i, i_b, dev in zip(inputs, renderer.inputs, input_devices):
            TrainableRendererFunction.copy_tensor_to_buffer(i, i_b, dev)
        for p, p_b, dev in zip(params, renderer.params, params_devices):
            TrainableRendererFunction.copy_tensor_to_buffer(p, p_b, dev)
        for go, go_b in zip(grad_outputs, renderer.grad_outputs):
            TrainableRendererFunction.copy_tensor_to_buffer(go, go_b, output_devices)
        renderer.backward_render()
        gradients = []
        for i, gi, dev in zip(inputs, renderer.grad_inputs, input_devices):
            gradients.append(
                None if gi is None else  # for inputs with no gradients
                TrainableRendererFunction.create_tensor_from_buffer(gi, dev)
            )
        for p, gp, dev in zip(params, renderer.grad_params, params_devices):
            gradients.append(
                TrainableRendererFunction.create_tensor_from_buffer(gp, dev)
            )
        gradients.append(None) # renderer input is not differentiable
        return tuple(gradients)
