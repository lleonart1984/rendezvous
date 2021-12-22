from rendering.manager import *
import time
import numpy as np
import matplotlib.pyplot as plt
from rendering.tools import *

from techniques.volumerec import TransmittanceGenerator, TransmittanceForward

image_width = 512
image_height = 512

presenter = create_presenter(width=image_width, height=image_height, format=Format.VEC4, mode=PresenterMode.OFFLINE,
                             usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC,
                             debug=True)
tools = GridTools(presenter)

# load grid
grid = tools.load_file('C:/Users/mendez/Desktop/clouds/disney_big.xyz', usage=ImageUsage.STORAGE | ImageUsage.TRANSFER_DST)
flatten_grid, _ = tools.load_file_fatten('C:/Users/mendez/Desktop/clouds/disney_big.xyz')

transmittances = presenter.create_buffer(3*presenter.width*presenter.height*4, usage=BufferUsage.STORAGE | BufferUsage.TRANSFER_SRC, memory=MemoryProperty.GPU)

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
    dataset.append((generator.rays, generator.transmittances))
    forward = TransmittanceForward(ivec3(grid.width, grid.height, grid.depth), flatten_grid, generator.rays, transmittances)
    presenter.load_technique(forward)
    forward.set_medium(vec3(1,1,1), 10, 0.875)
    presenter.dispatch_technique(forward)
    with presenter.get_graphics() as man:
        man.gpu_to_cpu(transmittances)
    plt.imshow(transmittances.as_numpy().reshape((presenter.height, presenter.width, 3)))
    plt.show()




presenter = None
