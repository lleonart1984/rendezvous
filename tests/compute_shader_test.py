from rendering.manager import *
import time
import numpy as np
import matplotlib.pyplot as plt

compile_shader_sources('./shaders')

image_width = 512
image_height = 512

presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
                             usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC,
                             debug=True)

pipeline = presenter.create_compute_pipeline()
pipeline.load_compute_shader('./shaders/test_compute_shader.comp.spv')
pipeline.bind_storage_image(0, ShaderStage.COMPUTE, lambda: presenter.render_target())
pipeline.close()

with presenter.get_compute() as man:
    man.set_pipeline(pipeline)
    man.dispatch_threads_2D(presenter.width, presenter.height)

with presenter.get_compute() as man:
    man.gpu_to_cpu(presenter.render_target())

plt.imshow(presenter.render_target().as_numpy())
plt.show()

presenter = None
