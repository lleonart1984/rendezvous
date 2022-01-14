from rendering.manager import *
import time
import numpy as np
import matplotlib.pyplot as plt
from rendering.tools import *
from rendering.modules import *
import torch


image_width = 512
image_height = 512

presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
                             usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC,
                             debug=False)
tools = GridTools(presenter)

used_device = torch.device('cuda:0')
# used_device = torch.device('cpu')

# load grid
grid = torch.Tensor(tools.load_file_as_numpy('C:/Users/mendez/Desktop/clouds/disney_big.xyz')).to(used_device)
grid_dim = grid.shape

box_maxim = vec3(grid_dim[2], grid_dim[1], grid_dim[0]) * 0.5 / max(grid_dim[0], max(grid_dim[1], grid_dim[2]))
box_minim = -1*box_maxim
box_size = box_maxim - box_minim

print("[INFO] Loaded grid")

image_width = 512
image_height = 512

medium_parameters = { 'scattering_albedo': vec3(1, 1, 1), 'density': 10, 'phase_g': 0.875 }

def view_grid(grid, look_position, look_target):
    ray_generator = RayGenerator(presenter, (image_height, int(image_width*1.4)), 0)
    rays = ray_generator(torch.Tensor([*look_position]), torch.Tensor([*look_target]))
    transmittance_module = TransmittanceRenderer(presenter)
    transmittance_module.set_medium(**medium_parameters)
    transmittance_module.set_box(box_minim, box_size)
    transmittances = transmittance_module(rays, grid)
    plt.imshow(transmittances.detach().reshape(image_height, int(image_width*1.4), 3).cpu().numpy())
    plt.show()

view_grid(grid, (0.8, 0.1, 0.5), (0,0,0))


cameras = [
    glm.rotate(i*360/11, vec3(0,1,0))*vec3(1, 0, 0) for i in range(11)
] + [glm.vec3(0.1, 1, 0.2), glm.vec3(-0.2,-1, 0.1)]

origins = torch.Tensor(np.array([[c.x, c.y, c.z] for c in cameras], dtype=np.float32)).to(used_device)
targets = torch.zeros_like(origins)

ray_generator = RayGenerator(presenter, (image_height, image_width), 0)

full_rays = ray_generator(origins, targets)

transmittance_module = TransmittanceRenderer(presenter)
transmittance_module.set_medium(**medium_parameters)
transmittance_module.set_box(box_minim, box_size)

transmittances = transmittance_module(full_rays, grid)

# for i in range(len(cameras)):
#     plt.imshow(transmittances.reshape((len(cameras), image_height, image_width, 3)).cpu().numpy()[i])
#     plt.show()

class TrainableCloud(nn.Module):
    def __init__(self, device, grid_or_dim):
        super().__init__()
        if isinstance(grid_or_dim, torch.Tensor):
            self.Grid = nn.Parameter(grid_or_dim)
        else:
            self.Grid = nn.Parameter(torch.zeros(*grid_or_dim))
        self.transmittance_renderer = TransmittanceRenderer(device)

    def forward(self, *args):
        rays, = args
        return self.transmittance_renderer(rays, torch.clamp(self.Grid, 0.0, 1.0))


rec_sizes = []
rec_size = list(grid.shape)
while rec_size[0] >= 32 and rec_size[1] >= 32 and rec_size[2] >= 32:
    rec_sizes.append(list(rec_size))
    rec_size[0] //= 2
    rec_size[1] //= 2
    rec_size[2] //= 2
list.reverse(rec_sizes)

print("[INFO] Optimization starts...")

_time = time.perf_counter()

rec_grid = None

lr = 0.2

num_epochs = 30

for rec_grid_size in rec_sizes:
    if rec_grid is None:  # first grid
        rec_grid = torch.zeros(*rec_sizes[0], device=used_device)
    else:  # get from upsampling
        upscaling = Resample3D(presenter, rec_grid_size)
        rec_grid = torch.clamp(upscaling(rec_grid), 0.0, 1.0)
        view_grid(rec_grid, (0.8, 0.1, 0.5), (0,0,0))

    model = TrainableCloud(presenter, rec_grid)
    model.transmittance_renderer.set_medium(**medium_parameters)
    model.transmittance_renderer.set_box(box_minim, box_size)

    # num_epochs //= 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
    for epoch in range(0, num_epochs):
        optimizer.zero_grad()
        output = model(full_rays)
        # loss = torch.abs(output - transmittances).sum()
        loss = torch.nn.functional.mse_loss(output, transmittances)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(loss.item())
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']

_time = time.perf_counter() - _time

print(f"[INFO] Optimization took {_time}s")
print(f'[INFO] Final lr={lr}')

rec_grid = torch.clamp(rec_grid, 0.0, 1.0).detach().cpu()

view_grid(rec_grid, (0.8, 0.1, 0.5), (0,0,0))

exit()

output = model(full_rays)

plt.imshow(output.detach().cpu().reshape((image_height * len(cameras), image_width, 3)).numpy())
plt.show()

plt.imshow(transmittances.detach().cpu().reshape((image_height * len(cameras), image_width, 3)).numpy())
plt.show()

rec_grid_size = rec_grid.shape

# show final reconstruction
g = rec_grid.numpy()
plt.imshow(g[:,:,rec_grid_size[2]//2].T, vmin=0, vmax=1)
plt.show()
plt.imshow(g[:,rec_grid_size[1]//2,:].T, vmin=0, vmax=1)
plt.show()
plt.imshow(g[rec_grid_size[0]//2,:,:], vmin=0, vmax=1)
plt.show()

# clean objects
ray_generator = None
transmittance_module = None
presenter = None
exit()











exit()


# Generating dataset
dataset = []
cameras = [
    glm.rotate(i*360/7, vec3(0,1,0))*vec3(1, 0, 0) for i in range(7)
]
for c in cameras:
    generator = TransmittanceGenerator(grid, presenter.render_target())
    presenter.load_technique(generator)
    generator.set_camera(c, vec3(0,0,0))
    generator.set_medium(vec3(1,1,1), 10, 0.875)
    presenter.dispatch_technique(generator)
    with presenter.get_copy() as man:
        man.gpu_to_cpu(generator.rays)
        man.gpu_to_cpu(generator.transmittances)
    dataset.append((generator.rays.as_tensor().to(torch_device), generator.transmittances.as_tensor().to(torch_device)))
    # forward = TransmittanceForward(generator.rays, ivec3(grid.width, grid.height, grid.depth), flatten_grid, transmittances)
    # presenter.load_technique(forward)
    # forward.set_medium(vec3(1,1,1), 10, 0.875)
    # presenter.dispatch_technique(forward)
    # with presenter.get_graphics() as man:
    #    man.gpu_to_cpu(transmittances)
    # plt.imshow(generator.transmittances.as_numpy().reshape((presenter.height, presenter.width, 3)))
    # plt.show()

print("[INFO] Loaded datasets")


# create differentiable transmittance
class DifferentiableTransmittance(RendererModule):
    def __init__(self, device: DeviceManager, grid_dim, output_dim):
        super().__init__(
            device,
            2,  # two inputs, rays and grid densities
            1   # one output, the transmittances
        )
        self.grid_dim = grid_dim   # dimension of the grid.
        self.output_dim = output_dim  # dimension used to create the output tensor.
        self.Grid = nn.Parameter(torch.zeros(grid_dim), requires_grad=True)

    def setup(self):
        self.forward_technique = TransmittanceForward(
            self.get_input(0),
            self.grid_dim,
            self.get_param(),
            self.get_output()
        )
        self.backward_technique = TransmittanceBackward(
            self.get_input(),
            self.grid_dim,
            self.get_param_gradient(),
            self.get_output(),
            self.get_output_gradient()
        )
        self.upsampling = UpSampleGrid()
        self.device.load_technique(self.forward_technique)
        self.device.load_technique(self.backward_technique)
        self.device.load_technique(self.upsampling)

    def forward_render(self):
        self.device.dispatch_technique(self.forward_technique)

    def backward_render(self):
        self.device.dispatch_technique(self.backward_technique)

    # def forward_params(self):
    #     return [torch.clamp(self.P, 0.0, 1.0)]

    def set_medium(self, scattering_albedo: vec3, density: float, phase_g: float):
        self.forward_technique.set_medium(scattering_albedo, density, phase_g)
        self.backward_technique.set_medium(scattering_albedo, density, phase_g)

    def initialize_param(self, grid_dim, grid):
        w, h, d = grid_dim
        grid_buffer = self.device.create_buffer(w*h*d*4, BufferUsage.STORAGE | BufferUsage.TRANSFER_DST | BufferUsage.TRANSFER_SRC,
                                                MemoryProperty.GPU)
        grid_buffer.write_direct(grid.detach().cpu())
        self.upsampling.set_src_grid(grid_dim, grid_buffer)
        self.upsampling.set_dst_grid(self.grid_dim, self.get_param())
        self.device.dispatch_technique(self.upsampling)
        with torch.no_grad():
            self.P[:] = self.get_param().as_tensor()


rec_sizes = []
rec_width, rec_height, rec_depth = grid.width, grid.height, grid.depth

while rec_width >= 32 and rec_height >= 32 and rec_depth >= 32:
    rec_sizes.append((rec_width, rec_height, rec_depth))
    rec_width //= 2
    rec_height //= 2
    rec_depth //= 2

list.reverse(rec_sizes)

last_parameters = None

lr = 0.05

for w, h, d in rec_sizes:
    model = TrainableGrid(presenter, ivec3(w,h,d), ivec2(presenter.width, presenter.height)).to(torch_device)
    model.set_medium(vec3(1,1,1), 10, 0.875)
    if last_parameters is not None:  # something previous to initialize
        prev_dim, prev_grid = last_parameters
        model.initialize_param(prev_dim, prev_grid)
        grid = np.clip(model.get_param_tensor().detach().cpu().numpy().reshape((d,h,w)), 0, 1)
        # plt.imshow(grid[d//2, :, :], vmin=0, vmax=1)
        # plt.show()
        # plt.imshow(grid[:, h//2, :], vmin=0, vmax=1)
        # plt.show()
        plt.imshow(grid[:, :, w//2].T, vmin=0, vmax=1)
        plt.show()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    for epoch in range(0, 11):
        optimizer.zero_grad()
        for i, (rays, transmittances) in enumerate(dataset):
            output = model(rays)
            if i == 0:
                loss = torch.abs(output - transmittances).sum()
                # loss = torch.nn.functional.mse_loss(output, transmittances)
            else:
                loss += torch.abs(output - transmittances).sum()
                # loss += torch.nn.functional.mse_loss(output, transmittances)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print (loss.item())
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']

    grid = np.clip(model.get_param_tensor().detach().cpu().numpy().reshape((d,h,w)), 0, 1)
    # plt.imshow(grid[d//2, :, :], vmin=0, vmax=1)
    # plt.show()
    # plt.imshow(grid[:, h//2, :], vmin=0, vmax=1)
    # plt.show()
    plt.imshow(grid[:, :, w//2].T, vmin=0, vmax=1)
    plt.show()
    last_parameters = ivec3(w, h, d), model.get_param_tensor()


presenter = None
