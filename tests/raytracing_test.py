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

vertices = presenter.create_structured_buffer(3, BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                                   MemoryProperty.CPU_ACCESSIBLE, position=vec3)
vertices.write([ vec3(-0.6, 0, 0), vec3(0.6, 0, 0), vec3(0, 1, 0) ])

geometry = presenter.create_triangle_collection()
geometry.append(vertices)
geometry_ads = presenter.create_geometry_ads(geometry)

scene_buffer = presenter.create_instance_buffer(1)
scene_buffer[0].transform = mat3x4()
scene_buffer[0].geometry = geometry_ads
scene_ads = presenter.create_scene_ads(scene_buffer)

scratch_buffer = presenter.create_scratch_buffer(geometry_ads, scene_ads)

with presenter.get_raytracing() as man:
    man.build_ads(geometry_ads, scratch_buffer)
    man.build_ads(scene_ads, scratch_buffer)

pipeline = presenter.create_raytracing_pipeline()
raygen = pipeline.load_rt_generation_shader('./shaders/test_raygen.rgen.spv')
rayhit = pipeline.load_rt_closest_hit_shader('./shaders/test_rayhit.rchit.spv')
raymiss = pipeline.load_rt_miss_shader('./shaders/test_raymiss.rmiss.spv')
hit_group = pipeline.create_rt_hit_group(rayhit)
pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION, lambda: scene_ads)
pipeline.bind_storage_image(1, ShaderStage.RT_GENERATION, lambda: presenter.render_target())
pipeline.close()

program = pipeline.create_rt_program()
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
