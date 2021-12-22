from rendering.training import *
from glm import *
import matplotlib.pyplot as plt
from techniques.pathtracer import Pathtracer
from techniques.radiativebackprop import *
import os
import torch
import torch.nn as nn
import random

compile_shader_sources('./shaders/PT', force_all=True)
compile_shader_sources('./shaders/RBP', force_all=True)

image_width = 1024
image_height = 1024
camera = Camera()
camera.PositionAt(vec3(1.5, 3.5, 2)).LookAt(vec3(0, 0.4, 0))

presenter = create_presenter(image_width, image_height, Format.VEC4, PresenterMode.OFFLINE,
                                 usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_DST | ImageUsage.TRANSFER_SRC,
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
    scene_builder.add_instance([bunny], material_index=mirror_mat, transform=glm.translate(glm.vec3(0.0, 0.6, 0.0)))
    return scene_builder.build_raytracing_scene()

def generate_pathtraced_target():
    """
    Generates a PT of the scene with the real texture.
    This image will be used as Target
    """
    raytracing_scene = create_scene(presenter)
    technique = Pathtracer(raytracing_scene, presenter.render_target(), './shaders/PT')
    presenter.load_technique(technique)
    technique.update_camera(camera)

    with presenter:
        for _ in range(1600):
            presenter.dispatch_technique(technique)

    with presenter.get_compute() as man:
        man.gpu_to_cpu(presenter.render_target())

    return presenter.render_target().as_numpy()


class TrainableTexture(TrainableRenderer):
    def __init__(self, device: DeviceManager, scene: RaytracingScene):
        super().__init__(
            device,
            512*512*3,  # input parameters for the texture to reconstruct
            presenter.width*presenter.height*3  # output values for the image rendered
        )
        self.device = device
        self.forward_technique = RBPForward(
            scene,
            presenter.width, presenter.height,
            input_parameters=self.v_input,
            output_image=self.v_output,
            shader_folder='./shaders/RBP'
        )
        self.backward_technique = RBPBackward(
            scene,
            presenter.width, presenter.height,
            input_parameters=self.v_input,
            grad_parameters=self.grad_input,
            grad_output=self.grad_output,
            shader_folder='./shaders/RBP'
        )
        self.device.load_technique(self.forward_technique)
        self.device.load_technique(self.backward_technique)
        self.forward_technique.number_of_samples = 16
        self.backward_technique.number_of_samples = 16

    def update_camera(self, camera):
        self.forward_technique.update_camera(camera)
        self.backward_technique.update_camera(camera)

    def forward_render(self):
        self.device.dispatch_technique(self.forward_technique)

    def backward_render(self):
        self.device.dispatch_technique(self.backward_technique)


# Use cached target image if already exists
if False:  # os.path.exists('saved_PT.npy'):
    pt_image = np.load('saved_PT.npy')
else:
    pt_image = generate_pathtraced_target()
    np.save('saved_PT.npy', pt_image)

plt.imshow(pt_image)
plt.show()

target = torch.Tensor(pt_image[:,:,(0,1,2)].flatten())

class MyModule(nn.Module):
    def __init__(self, scene: RaytracingScene, camera: Camera):
        super(MyModule, self).__init__()
        self.renderer = TrainableTexture(presenter, scene)
        self.renderer.update_camera(camera)
        self.module = RenderingModule(self.renderer)
        self.P = torch.nn.Parameter(torch.zeros(512*512*3))
        torch.nn.init.uniform_(self.P, .0, 0.0)

    def forward(self):
        return self.module(torch.clamp(self.P, 0, 1))

    def show_parameters(self):
        plt.imshow(self.P.detach().numpy().reshape((512,512,3)))
        plt.show()

    def show_grad_output(self):
        im = self.renderer.grad_output.as_numpy().reshape((presenter.height, presenter.width, 3))
        # im = np.sum(im, axis=2)
        plt.imshow(im*0.5+0.5)
        plt.show()

    def show_grad_input(self):
        im = self.renderer.grad_input.as_numpy().reshape((512, 512, 3))
        # im = np.sum(im, axis=2)
        plt.imshow(im*0.5+0.5)
        plt.show()


model = MyModule(create_scene(presenter, True), camera)

def visualize_model(model, number_of_passes = 1):
    final_output = torch.zeros(presenter.width*presenter.height*3)
    for _ in range(number_of_passes):
        final_output += model().detach().numpy() / number_of_passes
    plt.imshow(final_output.reshape((presenter.height, presenter.width, 3)))
    plt.show()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(0, 301):
    optimizer.zero_grad()
    output = model()
    loss = torch.nn.functional.mse_loss(output, target)
    # loss = torch.abs(output - target).sum()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print (loss.item())
        plt.imshow(torch.abs(output - target).detach().numpy().reshape((presenter.width, presenter.height, 3)).sum(axis=-1), vmin=0, vmax=1)
        plt.show()
        # model.show_grad_output()
        # model.show_grad_input()
        model.show_parameters()
        visualize_model(model, 1000)

visualize_model(model, 1000)

presenter = None
