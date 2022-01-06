from rendering.manager import *
import time
import numpy as np
import matplotlib.pyplot as plt
from rendering.tools import *
from rendering.training import *
from techniques.volumerec import *


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

print("[INFO] Loaded grid")

image_width = 512
image_height = 512

ray_generator = RayGenerator(presenter, (image_width, image_height), 0).to(used_device)
transmittance_module = TransmittanceRenderer(presenter)
transmittance_module.set_medium(vec3(1,1,1), 10, 0.875)

dataset = []
cameras = [
    glm.rotate(i*360/7, vec3(0,1,0))*vec3(1, 0, 0) for i in range(7)
] + [glm.vec3(0.1, 1, 0.2), glm.vec3(-0.2,-1, 0.1)]

for c in cameras:
    ray_generator.origin[:] = torch.Tensor( [c.x, c.y, c.z] )
    ray_generator.target[:] = torch.Tensor( [0, 0, 0] )
    rays = ray_generator()  # generate rays batch
    transmittances = transmittance_module(rays, grid)
    dataset.append((rays.detach().clone(), transmittances.detach().clone()))

# Check transmittances
for r, t in dataset:
    plt.imshow(t.reshape((image_width, image_height, 3)).cpu().numpy())
    plt.show()

class TrainableCloud(nn.Module):
    def __init__(self, device, grid_dim):
        super().__init__()
        self.Grid = nn.Parameter(torch.zeros(*grid_dim))
        self.transmittance_renderer = TransmittanceRenderer(device)

    def set_medium(self, scattering_albedo: glm.vec3, density: float, phase_g: float):
        self.transmittance_renderer.set_medium(scattering_albedo, density, phase_g)

    def forward(self, *args):
        rays, = args
        return self.transmittance_renderer(rays, torch.clamp(self.Grid, 0.0, 1.0))


rec_grid_size = 128

model = TrainableCloud(presenter, (rec_grid_size, rec_grid_size, rec_grid_size)).to(used_device)
model.set_medium(vec3(1,1,1), 10, 0.875)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
num_epochs = 300
for epoch in range(0, num_epochs):
    optimizer.zero_grad()
    for i, (rays, transmittances) in enumerate(dataset):
        output = model(rays)
        if i == 0:
            # loss = torch.abs(output - transmittances).sum()
            loss = torch.nn.functional.mse_loss(output, transmittances)
        else:
            # loss = loss + torch.abs(output - transmittances).sum()
            loss += torch.nn.functional.mse_loss(output, transmittances)
        if epoch == num_epochs-1:
            plt.imshow(output.detach().cpu().reshape((image_width, image_height, 3)).numpy())
            plt.show()
    loss.backward()
    optimizer.step()
    if epoch == 0 or (epoch + 1) % 50 == 0:
        print(loss.item())
    scheduler.step()

# show final reconstruction
g = torch.clamp(model.Grid, 0.0, 1.0).detach().cpu().numpy()
plt.imshow(g[:,:,rec_grid_size//2].T, vmin=0, vmax=1)
plt.show()
plt.imshow(g[:,rec_grid_size//2,:].T, vmin=0, vmax=1)
plt.show()
plt.imshow(g[rec_grid_size//2,:,:], vmin=0, vmax=1)
plt.show()

# clean objects
ray_generator = None
transmittance_module = None
presenter = None

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
