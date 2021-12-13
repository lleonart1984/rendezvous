import torch
import torch.nn as nn

from manager import *

class TrainableRenderer:

    def __init__(self, device: DeviceManager, parameter_size: int, output_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.parameter_size = parameter_size
        self.v_parameters = device.create_buffer(
            size=parameter_size * 4,  # assuming all floats
            usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
            memory=MemoryProperty.GPU)
        self.grad_parameters = device.create_buffer(
            size=parameter_size * 4,  # assuming all floats
            usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
            memory=MemoryProperty.GPU)
        self.output_size = output_size
        self.v_output = device.create_buffer(
            size=output_size * 4,  # assuming all floats
            usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
            memory=MemoryProperty.GPU)
        self.grad_output = device.create_buffer(
            size=output_size * 4,  # assuming all floats
            usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
            memory=MemoryProperty.GPU)
        self.setup()

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


class TrainableRendererFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, renderer: TrainableRenderer):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, renderer)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, renderer = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

class RenderingModule(nn.Module):
    pass







