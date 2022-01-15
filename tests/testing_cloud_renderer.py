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
tools = GridTools(presenter)

used_device = torch.device('cuda:0')
# used_device = torch.device('cpu')

# load grid
if not os.path.exists('cache/cached_grid.pt'):
    grid = torch.Tensor(tools.load_file_as_numpy('C:/Users/mendez/Desktop/clouds/disney_big.xyz')).to(used_device)
    torch.save(grid, 'cache/cached_grid.pt')
else:
    grid = torch.load('cache/cached_grid.pt').to(used_device)
grid_dim = grid.shape

box_maxim = vec3(grid_dim[2], grid_dim[1], grid_dim[0]) * 0.5 / max(grid_dim[0], max(grid_dim[1], grid_dim[2]))
box_minim = -1*box_maxim
box_size = box_maxim - box_minim

print("[INFO] Loaded grid")

image_width = 1024
image_height = 1024

medium_parameters = { 'scattering_albedo': vec3(1.0, 1.0, 1.0), 'density': 100, 'phase_g': 0.3 }

def render_grid(grid, look_position, look_target, image_width, image_height, number_of_samples):
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
    plt.figure(figsize=(10, 6), dpi=600)
    plt.imshow(radiances[0].detach().cpu().numpy())
    plt.savefig('result.pdf', dpi=600)
    plt.show()

view_grid(grid, (1.2, -0.2, 0.0), (0,0,0), 20)