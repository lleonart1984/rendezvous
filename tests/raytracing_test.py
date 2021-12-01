from rendering.manager import *
import time
import numpy as np
import matplotlib.pyplot as plt
from glm import *

compile_shader_sources('./shaders')

image_width = 512
image_height = 512

presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
                             usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC,
                             debug=True)

vertices = presenter.create_buffer(3 * 3,
                                   BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                                   MemoryProperty.CPU_ACCESSIBLE)
vertices.write([
    vec3(-0.6, 0, 0),
    vec3(0.6, 0, 0),
    vec3(0, 1, 0)
])


geometry_ads = presenter.create_triangle_ads()
geometry_ads.vertex_description(position=vec3)
geometry_ads.append_triangles(vertices)
geometry_ads.build()

scene_ads = presenter.create_scene_ads()
scene_ads.append_geometry(geometry_ads)
scene_ads.build()

pipeline = presenter.create_raytracing_pipeline()
raygen = pipeline.load_rt_generation_shader('./shaders/test_raygen.spv')
rayhit = pipeline.load_rt_closesthit_shader('./shaders/test_rayhit.spv')
raymiss = pipeline.load_rt_miss_shader('./shaders/test_miss.spv')
hitgroup = pipeline.create_hit_group(rayhit)
pipeline.bind_ads(0, ShaderStage.RT_GENERATION, lambda: scene_ads)
pipeline.bind_storage_image(1, ShaderStage.RT_GENERATION, lambda: presenter.render_target())
pipeline.close()

program = presenter.create_rt_program()
program.set_generation(raygen)
program.set_group(0, hitgroup)
program.set_miss(0, raymiss)

with presenter.get_raytracing() as man:
    man.set_pipeline(pipeline)
    man.dispatch_rays(program, presenter.width, presenter.height)

with presenter.get_compute() as man:
    man.gpu_to_cpu(presenter.render_target())

plt.imshow(presenter.render_target().as_numpy())
plt.show()

presenter = None
