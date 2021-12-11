from rendering.manager import *
from rendering.scenes import *
from glm import *
import matplotlib.pyplot as plt


compile_shader_sources('./shaders')

image_width = 512
image_height = 512

def render():

    presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
                                 usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC | ImageUsage.TRANSFER_DST,
                                 debug=True)

    scene_builder = SceneBuilder(device=presenter)
    geometry1 = scene_builder.add_geometry_obj("./models/pitagoras.obj")
    scene_builder.add_instance([geometry1], transform=glm.scale(glm.vec3(1.5, 1.5, -1.5)))
    raytracing_scene = scene_builder.build_raytracing_scene()

    pipeline = presenter.create_raytracing_pipeline()
    raygen = pipeline.load_rt_generation_shader('./shaders/test_basic_raygen.rgen.spv')
    raymiss = pipeline.load_rt_miss_shader('./shaders/test_raymiss.rmiss.spv')
    raymiss2 = pipeline.load_rt_miss_shader('./shaders/test_raymiss2.rmiss.spv')
    rayhit = pipeline.load_rt_closest_hit_shader('./shaders/test_rayhit.rchit.spv')
    gen_group = pipeline.create_rt_gen_group(raygen)
    miss_group = pipeline.create_rt_miss_group(raymiss)
    miss2_group = pipeline.create_rt_miss_group(raymiss2)
    hit_group = pipeline.create_rt_hit_group(rayhit)
    pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION | ShaderStage.RT_CLOSEST_HIT, lambda: raytracing_scene.scene_ads)
    pipeline.bind_storage_image(1, ShaderStage.RT_GENERATION, lambda: presenter.render_target())
    pipeline.close()

    program = pipeline.create_rt_program(2, 1)
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