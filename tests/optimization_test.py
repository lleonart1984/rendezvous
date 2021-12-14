from rendering.manager import *
from rendering.training import *

compile_shader_sources('./shaders')

image_width = 512
image_height = 512

presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
                             usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC,
                             debug=True)


class TrainableTest(TrainableRenderer):
    def __init__(self, device: DeviceManager):
        super().__init__(device, 4, 3)

    def setup(self):
        f_pipeline = self.device.create_compute_pipeline()
        f_pipeline.load_compute_shader('./shaders/forward_test.comp.spv')
        f_pipeline.bind_storage_buffer(0, ShaderStage.COMPUTE, lambda: self.v_parameters)
        f_pipeline.bind_storage_buffer(1, ShaderStage.COMPUTE, lambda: self.v_output)
        f_pipeline.close()
        self.f_pipeline = f_pipeline
        b_pipeline = self.device.create_compute_pipeline()
        b_pipeline.load_compute_shader('./shaders/backward_test.comp.spv')
        b_pipeline.bind_storage_buffer(0, ShaderStage.COMPUTE, lambda: self.v_parameters)
        b_pipeline.bind_storage_buffer(1, ShaderStage.COMPUTE, lambda: self.grad_parameters)
        b_pipeline.bind_storage_buffer(2, ShaderStage.COMPUTE, lambda: self.grad_output)
        b_pipeline.close()
        self.b_pipeline = b_pipeline

    def forward_render(self):
        with self.device.get_compute() as man:
            man.set_pipeline(self.f_pipeline)
            man.update_sets(0)
            man.dispatch_threads_1D(3)  # y0, y1, y2

    def backward_render(self):
        with self.device.get_compute() as man:
            man.set_pipeline(self.b_pipeline)
            man.update_sets(0)
            man.dispatch_threads_1D(4)  # x0, x1, x2, x3


model = RenderingModule(TrainableTest(presenter))

input = torch.Tensor([1,2,3,5])
output = model(input)

model = None
presenter = None