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
                             debug=True)
tools = GridTools(presenter)

# load grid
grid = tools.load_file('C:/Users/mendez/Desktop/clouds/disney_big.xyz', usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC | ImageUsage.TRANSFER_DST)
# flatten_grid, _ = tools.load_file_fatten('C:/Users/mendez/Desktop/clouds/disney_big.xyz', usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC | ImageUsage.TRANSFER_DST)

print("[INFO] Loaded grid")


torch_device = torch.device('cuda:0')

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


# create Trainable rendering
class TrainableGrid(RendererModule):
    def __init__(self, device: DeviceManager, grid_dim, output_dim):
        self.grid_dim = grid_dim
        self.output_dim = output_dim
        super().__init__(
            device,
            [output_dim.x * output_dim.y * 6], # input rays, one for each output transmittance
            [grid_dim.x*grid_dim.y*grid_dim.z],  # parameters for the grid to reconstruct
            [output_dim.x*output_dim.y*3],  # output values for the image rendered
            input_trainable=False  # input is not backprop
        )
        self.P = self.get_param_tensor()

    def setup(self):
        self.forward_technique = TransmittanceForward(
            self.get_input(),
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
