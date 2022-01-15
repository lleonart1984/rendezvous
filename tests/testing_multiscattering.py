from rendering.manager import *
import time
import numpy as np
import matplotlib.pyplot as plt
from rendering.tools import *
from rendering.modules import *
import torch
import os


image_width = 512
image_height = 512

presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
                             usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC,
                             debug=False)
presenter.allow_cross_threading()

tools = GridTools(presenter)

used_device = torch.device('cuda:0')
# used_device = torch.device('cpu')

# load grid
if True:  # not os.path.exists('cache/cached_grid.pt'):
    grid = torch.Tensor(tools.load_file_as_numpy('C:/Users/mendez/Desktop/clouds/disney_big.xyz')).to(used_device)
    torch.save(grid, 'cache/cached_grid.pt')
else:
    grid = torch.load('cache/cached_grid.pt').to(used_device)

print("[INFO] Loaded grid")

image_width = 256
image_height = 256

medium_parameters = { 'scattering_albedo': vec3(1.0, 1.0, 1.0), 'density': 100, 'phase_g': 0.3 }

def render_grid(grid, look_position, look_target, image_width, image_height, number_of_samples):
    grid_dim = grid.shape
    box_maxim = vec3(grid_dim[2], grid_dim[1], grid_dim[0]) * 0.5 / max(grid_dim[0], max(grid_dim[1], grid_dim[2]))
    box_minim = -1 * box_maxim
    box_size = box_maxim - box_minim
    ray_generator = RayGenerator(presenter, (image_height, image_width), 1)
    vr_module = VolumeRenderer(presenter)
    vr_module.set_medium(**medium_parameters)
    vr_module.set_box(box_minim, box_size)
    camera_count = torch.numel(look_position)//3
    radiances = torch.zeros(camera_count, image_height, image_width, 3, device=used_device)
    for i in range(number_of_samples):
        rays = ray_generator(look_position, look_target)
        radiances += vr_module(rays, grid).reshape(-1, image_height, image_width, 3)
        if i % 10 == 0:
            print(f"Progress... {i * 100 / number_of_samples}%")
    return radiances/number_of_samples

def view_grid(grid, look_position, look_target, number_of_samples):
    look_position = torch.Tensor([*look_position]).to(used_device)
    look_target = torch.Tensor([*look_target]).to(used_device)
    radiances = render_grid(grid, look_position, look_target, int(image_width*1.4), image_height, number_of_samples)
    plt.figure(figsize=(10, 6))
    plt.imshow(radiances[0].detach().cpu().numpy())
    plt.savefig('result.pdf')
    plt.show()

view_grid(grid, (1.2, -0.2, 0.0), (0,0,0), 1)


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

class TrainableCloud(nn.Module):
    def __init__(self, device, grid_or_dim):
        super().__init__()
        if isinstance(grid_or_dim, torch.Tensor):
            self.Grid = nn.Parameter(grid_or_dim)
        else:
            self.Grid = nn.Parameter(torch.zeros(*grid_or_dim))
        self.vpt_renderer = VolumeRenderer(device)

    def forward(self, *args):
        rays, = args
        return self.vpt_renderer(rays, torch.clamp(self.Grid, 0.0, 1.0))

cameras = [
    glm.rotate(i*360/7, vec3(0,1,0))*vec3(1.2, 0, 0) for i in range(7)
] + [glm.vec3(0.1, 1.2, 0.2), glm.vec3(-0.2,-1.2, 0.1)]

origins = torch.Tensor(np.array([[c.x, c.y, c.z] for c in cameras], dtype=np.float32)).to(used_device)
targets = torch.zeros_like(origins)

if not os.path.exists('cache/cloud_radiances_256.pt'):
    radiances = render_grid(grid, origins, targets, image_width, image_height, 100)
    torch.save(radiances, 'cache/cloud_radiances_256.pt')
else:
    radiances = torch.load('cache/cloud_radiances_256.pt')

# for i in radiances:
#     plt.imshow(i.detach().cpu().numpy())
#     plt.show()

rec_grid = torch.ones(32, 32, 32, device=used_device) * (0.02)

for stage in range(20):
    model = TrainableCloud(presenter, rec_grid).to(used_device)
    model.vpt_renderer.set_medium(**medium_parameters)
    rec_grid_dim = rec_grid.shape
    rec_box_maxim = vec3(rec_grid_dim[2], rec_grid_dim[1], rec_grid_dim[0]) * 0.5 / max(rec_grid_dim[0], max(rec_grid_dim[1], rec_grid_dim[2]))
    rec_box_minim = -1*rec_box_maxim
    rec_box_size = rec_box_maxim - rec_box_minim
    model.vpt_renderer.set_box(rec_box_minim, rec_box_size)
    ray_generator = RayGenerator(presenter, (image_height, image_width), 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.1)

    for e in range(10):
        optimizer.zero_grad()
        rays = ray_generator(origins, targets)
        output = model(rays)
        loss = torch.nn.functional.mse_loss(output.reshape(radiances.shape), radiances)
        loss.backward()
        optimizer.step()
        #if e % 2 == 0:
        #     plt.imshow(output[0].detach().cpu().numpy())
        #     plt.show()
        #    view_grid(torch.clamp(rec_grid, 0.0, 1.0).detach(), (1.2, -0.2, 0.0), (0, 0, 0), 1)
        print(loss.item())
    view_grid(torch.clamp(rec_grid, 0.0, 1.0), (1.2, -0.2, 0.0), (0, 0, 0), 1)
    rec_grid = torch.clamp(rec_grid.detach().clone(), 0.0, 1.0)
