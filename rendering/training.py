import torch
import torch.nn as nn
from rendering.manager import *


class TrainableRenderer:

    def __init__(self, device: DeviceManager, input_size: int, output_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.input_size = input_size
        self.v_input = device.create_buffer(
            size=input_size * 4,  # assuming all floats
            usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
            memory=MemoryProperty.GPU)
        self.grad_input = device.create_buffer(
            size=input_size * 4,  # assuming all floats
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
    def forward(ctx, *args):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        renderer: TrainableRenderer
        input, renderer = args
        ctx.save_for_backward(input)
        ctx.renderer = renderer
        renderer.v_input.write(input)
        with renderer.device.get_copy() as man:
            man.cpu_to_gpu(renderer.v_input)
        renderer.forward_render()
        with renderer.device.get_copy() as man:
            man.gpu_to_cpu(renderer.v_output)
        return renderer.v_output.as_torch()

    @staticmethod
    def backward(ctx, *args):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_output, = args
        input, = ctx.saved_tensors
        renderer = ctx.renderer
        renderer.grad_output.write(grad_output)
        renderer.v_input.write(input)
        with renderer.device.get_copy() as man:
            man.cpu_to_gpu(renderer.grad_output)
            man.cpu_to_gpu(renderer.v_input)
        renderer.backward_render()
        with renderer.device.get_copy() as man:
            man.gpu_to_cpu(renderer.grad_input)
        return renderer.grad_input.as_torch(), None


class RenderingModule(nn.Module):
    def __init__(self, trainable_renderer: TrainableRenderer):
        super().__init__()
        self.trainable_renderer = trainable_renderer

    def forward(self, input: torch.Tensor):
        return TrainableRendererFunction.apply(input, self.trainable_renderer)






