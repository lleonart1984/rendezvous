from rendering.manager import *
from rendering.scenes import *
from glm import *
import matplotlib.pyplot as plt
import time
from techniques.pathtracer import Pathtracer

compile_shader_sources('./shaders/PT')

image_width = 1024
image_height = 1024

def app_loop():

    presenter = create_presenter(image_width, image_height, Format.UINT_BGRA_STD, PresenterMode.SDL,
                                 usage=ImageUsage.RENDER_TARGET | ImageUsage.TRANSFER_DST,
                                 debug=False)

    # presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
    #                              usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC | ImageUsage.TRANSFER_DST,
    #                              debug=True)

    offline_image = presenter.create_image(ImageType.TEXTURE_2D, False, Format.UINT_BGRA_UNORM,
                                           presenter.width, presenter.height, 1, 1, 1,
                                           ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC, MemoryProperty.GPU)

    scene_builder = SceneBuilder(device=presenter)

    wood = scene_builder.add_texture("./models/wood.jpg")
    plate_mat = scene_builder.add_material(
        diffuse=vec3(1, 1, 1),
        diffuse_map=wood
    )
    diffuse_mat = scene_builder.add_material(
    )
    specular_mat = scene_builder.add_material(
        illumination_model_mix=vec4(0,1,0,0)
    )
    mirror_mat = scene_builder.add_material(
        illumination_model_mix=vec4(0,0,1,0)
    )
    fresnel_mat = scene_builder.add_material(
        refraction_index=0.9,
        illumination_model_mix=vec4(0,0,0,1),
    )
    plate = scene_builder.add_geometry_obj("./models/plate.obj")
    bunny = scene_builder.add_geometry_obj("./models/bunnyScene.obj")
    scene_builder.add_instance([plate], material_index=plate_mat, transform=glm.scale(glm.vec3(4.5, 1, 4.5)))
    scene_builder.add_instance([bunny], material_index=diffuse_mat, transform=glm.translate(glm.vec3(-0.5, 0.5, 0.5)))
    scene_builder.add_instance([bunny], material_index=specular_mat, transform=glm.translate(glm.vec3(-0.5, 0.5, -0.5)))
    scene_builder.add_instance([bunny], material_index=mirror_mat, transform=glm.translate(glm.vec3(0.5, 0.5, -0.5)))
    scene_builder.add_instance([bunny], material_index=fresnel_mat, transform=glm.translate(glm.vec3(0.5, 0.5, 0.5)))
    raytracing_scene = scene_builder.build_raytracing_scene()

    camera = Camera()
    camera.PositionAt(vec3(2,1.8,3)).LookAt(vec3(0,0.4,0))

    technique = Pathtracer(raytracing_scene, offline_image, './shaders/PT')
    presenter.load_technique(technique)

    technique.update_camera(camera)

    window = presenter.get_window()

    last_time = time.perf_counter()
    fps = 0
    while True:
        fps += 1
        current_time = time.perf_counter()
        if current_time - last_time >= 1:
            last_time = current_time
            print("FPS: %s" % fps)
            fps = 0

        event, args = window.poll_events()
        if event == Event.CLOSED:
            break

        # camera.PositionAt(glm.rotateY(glm.vec3(1,1,-2), current_time))

        with presenter:
            presenter.dispatch_technique(technique)

    with presenter.get_compute() as man:
        man.gpu_to_cpu(offline_image)
        
    plt.imshow(offline_image.as_numpy()[:,:,(2,1,0)])
    plt.show()


app_loop()

