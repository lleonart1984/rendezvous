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

presenter = create_presenter(image_width, image_height, Format.PRESENTER, PresenterMode.SDL,
                                 usage=ImageUsage.RENDER_TARGET | ImageUsage.TRANSFER_DST | ImageUsage.TRANSFER_SRC,
                                 debug=True)

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


def generate_pathtraced_target(number_of_samples: int, texture = None):
    """
    Generates a PT of the scene with the real texture.
    This image will be used as Target
    """
    raytracing_scene = create_scene(presenter, texture is not None)
    path_tracer = PTRender(presenter)
    ray_generator = RayGenerator(presenter, (image_height, image_width), 1)
    path_tracer.set_scene(raytracing_scene)
    accumulation = torch.zeros(image_height, image_width, 3, device=used_device)
    t = time.perf_counter()
    for _ in range(number_of_samples):
        rays = ray_generator(camera_position, camera_target)
        accumulation += path_tracer(rays, texture).reshape(image_height, image_width, 3)
    image = accumulation / number_of_samples
    t = time.perf_counter() - t
    print(f"[INFO] Time per sample {t / number_of_samples}s")
    return image


offline_buffer = presenter.create_buffer(image_height*image_width*3*4, BufferUsage.GPU_ADDRESS | BufferUsage.TRANSFER_SRC | BufferUsage.TRANSFER_DST, MemoryProperty.GPU)
offline_image = presenter.create_image(ImageType.TEXTURE_2D, False, Format.VEC3, image_width, image_height, 1, 1, 1, ImageUsage.TRANSFER_SRC | ImageUsage.TRANSFER_DST, MemoryProperty.GPU)


def view_tensor(tensor):
    offline_buffer.write_gpu_tensor(tensor)
    with presenter:
        with presenter.get_graphics() as man:
            man.clear_color(presenter.render_target(), (1,1,0,1))
            man.copy_buffer_to_image(offline_buffer, offline_image)
            man.blit_image(offline_image, presenter.render_target())

def train():

    target = generate_pathtraced_target(1000).detach()
    view_tensor(target)

    plt.imshow(target.cpu().numpy())
    plt.show()

    class TrainableTexture(nn.Module):
        def __init__(self, device: DeviceManager, texture):
            super().__init__()
            self.texture_dim = texture.shape
            self.Texture = nn.Parameter(texture)
            self.renderer = PTRender(device, 400)

        def show_parameters(self):
            texture = torch.clamp(self.Texture, 0.0, 1.0).reshape(self.texture_dim).detach().cpu()
            plt.imshow(texture.numpy())
            plt.show()

        def forward(self, rays):
            # return self.renderer(rays, self.Texture)
            return self.renderer(rays, torch.clamp(self.Texture, 0.0, 1.0))

    parameter_scene = create_scene(presenter, True)

    dimensions = [(1 << l, 1 << l, 3) for l in range(3, 10)]

    ray_generator = RayGenerator(presenter, (image_height, image_width), 1)
    texture = None
    lr = 0.1
    epochs = 2
    for d in dimensions:
        if texture is not None:  # previous computed texture
            upscaling = Resample2D(presenter, d, 3)
            texture = upscaling(torch.clamp(texture, 0.0, 1.0))
        else:
            texture = torch.ones(d, device=used_device) * 0.5
        model = TrainableTexture(presenter, texture).to(used_device)
        model.renderer.set_scene(parameter_scene)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        for epoch in range(0, epochs):
            optimizer.zero_grad()
            samples = 1
            rays = ray_generator(camera_position.repeat(samples, 1), camera_target.repeat(samples, 1))
            output = model(rays).reshape(samples, image_height, image_width, 3).mean(0)
            # plt.imshow(output.detach().cpu().numpy().reshape((image_height, image_width, 3)))
            # plt.imshow(torch.abs(output - target).detach().cpu().numpy().reshape((image_height, image_width, 3)))
            # plt.show()
            loss = torch.nn.functional.mse_loss(output, target)
            # loss = torch.abs(output - target).sum()
            loss.backward()
            optimizer.step()
            view_tensor(output.detach())
            if epoch % 10 == 0:
                print(loss.item())
                # plt.imshow(output.detach().cpu().numpy().reshape((image_height, image_width, 3)))
                # plt.imshow(torch.abs(output - target).detach().cpu().numpy().reshape((image_height, image_width, 3)))
                # plt.show()
                # model.show_grad_output()
                # model.show_grad_input()
                model.show_parameters()
                # visualize_model(model, 100)
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        epochs *= 2

    final = generate_pathtraced_target(1000, model.Texture.detach()).detach()

    plt.imshow(final.cpu().numpy())
    plt.show()

presenter.loop(train)

exit()


texture = None

dimensions = [(1<<l, 1<<l, 3) for l in range(3, 10)]



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

    model = TrainableTexture(presenter, texture, (image_height, image_width), 100).to(used_device)
    model.renderer.set_scene(parameter_scene)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    for epoch in range(0, 30):
        optimizer.zero_grad()
        output = model(camera_position, camera_target)
        loss = torch.nn.functional.mse_loss(output, target)
        # loss = torch.abs(output - target).sum()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
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
