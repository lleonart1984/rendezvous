from rendering.manager import *
from glm import *
import matplotlib.pyplot as plt


compile_shader_sources('./shaders')

image_width = 512
image_height = 512

def render():

    presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
                                 usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC | ImageUsage.TRANSFER_DST,
                                 debug=True)

    vertices = presenter.create_structured_buffer(3,
                                       BufferUsage.RAYTRACING_ADS_READ,
                                       MemoryProperty.CPU_DIRECT | MemoryProperty.CPU_ACCESSIBLE, position=vec3)
    vertices.write([ vec3(-0.6, -0.6, 0), vec3(0.6, -0.6, 0), vec3(0, 0.6, 0) ])  # TODO: Write a write_direct, read_direct
    with presenter.get_raytracing() as man:
         man.cpu_to_gpu(vertices)

    geometry = presenter.create_triangle_collection()
    geometry.append(vertices)
    geometry_ads = presenter.create_geometry_ads(geometry)

    scene_buffer = presenter.create_instance_buffer(1)

    scene_buffer[0].transform = mat3x4( 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    scene_buffer[0].mask = 0xff
    scene_buffer[0].id = 0
    scene_buffer[0].offset = 0
    scene_buffer[0].flags = 0
    scene_buffer[0].geometry = geometry_ads

    scene_ads = presenter.create_scene_ads(scene_buffer)

    scratch_buffer = presenter.create_scratch_buffer(geometry_ads, scene_ads)

    with presenter.get_raytracing() as man:
        man.build_ads(geometry_ads, scratch_buffer)
        man.build_ads(scene_ads, scratch_buffer)

    pipeline = presenter.create_raytracing_pipeline()
    raygen = pipeline.load_rt_generation_shader('./shaders/test_basic_raygen.rgen.spv')
    raymiss = pipeline.load_rt_miss_shader('./shaders/test_raymiss.rmiss.spv')
    raymiss2 = pipeline.load_rt_miss_shader('./shaders/test_raymiss2.rmiss.spv')
    rayhit = pipeline.load_rt_closest_hit_shader('./shaders/test_rayhit.rchit.spv')
    gen_group = pipeline.create_rt_gen_group(raygen)
    miss_group = pipeline.create_rt_miss_group(raymiss)
    miss2_group = pipeline.create_rt_miss_group(raymiss2)
    hit_group = pipeline.create_rt_hit_group(rayhit)
    pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION, lambda: scene_ads)
    pipeline.bind_storage_image(1, ShaderStage.RT_GENERATION, lambda: presenter.render_target())
    pipeline.close()

    program = pipeline.create_rt_program(10, 100)
    program.set_generation(gen_group)
    program.set_hit_group(0, hit_group)
    program.set_miss(1, miss_group)
    program.set_miss(0, miss2_group)

    with presenter.get_raytracing() as man:
        man.set_pipeline(pipeline)
        man.dispatch_rays(program, presenter.width, presenter.height)

    with presenter.get_raytracing() as man:
        man.gpu_to_cpu(presenter.render_target())

    plt.imshow(presenter.render_target().as_numpy())
    plt.show()


render()