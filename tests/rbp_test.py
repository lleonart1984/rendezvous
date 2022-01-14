from rendering.training import *
from glm import *
import matplotlib.pyplot as plt
import torch
from rendering.modules import *
import time

image_width = 1024
image_height = 1024

used_device = torch.device('cuda:0')

camera_position = vec3(1.5, 2.5, 1)
camera_target = vec3(0, 0.4, 0)
camera_position = torch.Tensor([*camera_position]).to(used_device)
camera_target = torch.Tensor([*camera_target]).to(used_device)

presenter = create_presenter(image_width, image_height, Format.VEC4, PresenterMode.OFFLINE,
                                 usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_DST | ImageUsage.TRANSFER_SRC,
                                 debug=False)


def create_scene(device: DeviceManager, use_parameters: bool = False):
    """
    Generates the Scene used for the Raytracing processes.
    The scene is generated either using the texture or specifying -2 as material to say
    will use parameters
    """
    scene_builder = SceneBuilder(device=device)

    wood = scene_builder.add_texture("./models/wood.jpg")
    plate_mat = scene_builder.add_material(
        diffuse=vec3(1, 1, 1),
        diffuse_map=wood
    )
    diffuse_mat = scene_builder.add_material(
    )
    specular_mat = scene_builder.add_material(
        illumination_model_mix=vec4(0, 1, 0, 0)
    )
    mirror_mat = scene_builder.add_material(
        illumination_model_mix=vec4(0, 0, 1, 0)
    )
    fresnel_mat = scene_builder.add_material(
        refraction_index=0.9,
        illumination_model_mix=vec4(0, 0, 0, 1),
    )
    plate = scene_builder.add_geometry_obj("./models/plate.obj")
    bunny = scene_builder.add_geometry_obj("./models/bunnyScene.obj")
    scene_builder.add_instance([plate], material_index=-2 if use_parameters else plate_mat, transform=glm.scale(glm.vec3(3.5, 1, 3.5)))
    # scene_builder.add_instance([bunny], material_index=diffuse_mat, transform=glm.translate(glm.vec3(-0.5, 0.5, 0.5)))
    # scene_builder.add_instance([bunny], material_index=specular_mat, transform=glm.translate(glm.vec3(-0.5, 0.6, 0.5)))
    # scene_builder.add_instance([bunny], material_index=mirror_mat, transform=glm.translate(glm.vec3(0.5, 0.5, -0.5)))
    # scene_builder.add_instance([bunny], material_index=fresnel_mat, transform=glm.translate(glm.vec3(0.5, 0.6, -0.5)))
    scene_builder.add_instance([bunny], material_index=fresnel_mat, transform=glm.translate(glm.vec3(0.0, 0.6, 0.0)))
    return scene_builder.build_raytracing_scene()


def generate_pathtraced_target(number_of_samples: int):
    """
    Generates a PT of the scene with the real texture.
    This image will be used as Target
    """
    raytracing_scene = create_scene(presenter)
    path_tracer = RBPRenderer(presenter, (image_height, image_width), number_of_samples)
    path_tracer.set_scene(raytracing_scene)
    t = time.perf_counter()
    image = path_tracer(camera_position, camera_target, None).reshape(image_height, image_width, 3)
    t = time.perf_counter() - t
    print(f"[INFO] Time per sample {t / number_of_samples}s")
    return image


target = generate_pathtraced_target(1000).detach()

plt.imshow(target.cpu().numpy())
plt.show()


class TrainableTexture(nn.Module):
    def __init__(self, device: DeviceManager, texture, output_dim, number_of_samples):
        super().__init__()
        self.texture_dim = texture.shape
        self.Texture = nn.Parameter(texture)
        self.renderer = RBPRenderer(device, output_dim, number_of_samples)

    def show_parameters(self):
        texture = torch.clamp(self.Texture, 0.0, 1.0).reshape(self.texture_dim).detach().cpu()
        plt.imshow(texture.numpy())
        plt.show()

    def forward(self, origins, targets):
        return self.renderer(origins, targets, torch.clamp(self.Texture, 0.0, 1.0))


texture = None

dimensions = [(1<<l, 1<<l, 3) for l in range(3, 10)]

parameter_scene = create_scene(presenter, True)


def visualize_model(model, number_of_samples):
    renderer = RBPRenderer(presenter, (image_height, image_width), number_of_samples)
    renderer.set_scene(model.renderer.scene)
    image = renderer(camera_position, camera_target, torch.clamp(model.Texture, 0.0, 1.0))
    plt.imshow(image.reshape(image_height, image_width, 3).detach().cpu().numpy())
    plt.show()


for d in dimensions:
    if texture is not None:  # previous computed texture
        upscaling = Resample2D (presenter, d, 3)
        texture = upscaling(torch.clamp(texture, 0.0, 1.0))
    else:
        texture = torch.zeros(d, device=used_device)

    plt.imshow(texture.detach().cpu().numpy())
    plt.show()

    model = TrainableTexture(presenter, texture, (image_height, image_width), 10).to(used_device)
    model.renderer.set_scene(parameter_scene)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    for epoch in range(0, 300):
        optimizer.zero_grad()
        output = model(camera_position, camera_target)
        loss = torch.nn.functional.mse_loss(output, target)
        # loss = torch.abs(output - target).sum()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print (loss.item())
            # plt.imshow(torch.abs(output - target).detach().numpy().reshape((image_height, image_width, 3)).sum(axis=-1), vmin=0, vmax=1)
            # plt.show()
            # model.show_grad_output()
            # model.show_grad_input()
            model.show_parameters()
            # visualize_model(model, 100)
        # scheduler.step()

visualize_model(model, 1000)

presenter = None
