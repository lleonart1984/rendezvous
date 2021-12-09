from rendering.manager import *
import time
import numpy as np
import matplotlib.pyplot as plt
from glm import *

compile_shader_sources('./shaders')

image_width = 512
image_height = 512

presenter = create_presenter(width=image_width, height=image_height, format=Format.UINT_BGRA_STD, mode=PresenterMode.SDL,
                             usage=ImageUsage.RENDER_TARGET | ImageUsage.TRANSFER_SRC | ImageUsage.TRANSFER_DST,
                             debug=True)

offscreen_image = presenter.create_image(ImageType.TEXTURE_2D, False, Format.UINT_BGRA_UNORM, image_width, image_height,1,1,1,
                                         ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC, MemoryProperty.GPU)

vertices = presenter.create_structured_buffer(3, BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                                   MemoryProperty.CPU_ACCESSIBLE, position=vec3)
vertices.write([ vec3(-0.6, 0, 0), vec3(0.6, 0, 0), vec3(0, 1, 0) ])  # TODO: Write a write_direct, read_direct
with presenter.get_compute() as man:
    man.cpu_to_gpu(vertices)

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
gen_group = pipeline.create_rt_gen_group(raygen)
miss_group = pipeline.create_rt_miss_group(raymiss)
hit_group = pipeline.create_rt_hit_group(rayhit)
pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION, lambda: scene_ads)
pipeline.bind_storage_image(1, ShaderStage.RT_GENERATION, lambda: offscreen_image)
pipeline.close()

program = pipeline.create_rt_program()
program.set_generation(gen_group)
program.set_hit_group(0, hit_group)
program.set_miss(0, miss_group)

with presenter:
    with presenter.get_raytracing() as man:
        man.set_pipeline(pipeline)
        man.dispatch_rays(program, presenter.width, presenter.height)
        man.copy_image(offscreen_image, presenter.render_target())

# with presenter.get_raytracing() as man:
#     man.gpu_to_cpu(presenter.render_target())
#
# plt.imshow(presenter.render_target().as_numpy())
# plt.show()

presenter = None
