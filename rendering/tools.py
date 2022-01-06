from rendering.manager import *
from enum import IntEnum
from glm import *
import numpy as np
import os


__SHADERS_TOOLS__ = os.path.dirname(__file__) + "/shaders/Tools"

compile_shader_sources(__SHADERS_TOOLS__)


class SuperVoxelOperation(IntEnum):
    AVERAGE = 0
    MINIMUM = 1
    MAXIMUM = 2


class GridTools:
    def __init__(self, device: DeviceManager):
        self.device = device
        self.compute_super_voxel_pipeline = None
        self.deepness_initialization_pipeline = None
        self.deepness_compute_pipeline = None
        self.chebyshev_distance_pipeline = None
        self.pooling_pipeline = None

    def load_file(self, path: str, mips=1, usage=ImageUsage.TRANSFER_DST | ImageUsage.SAMPLED):
        device: DeviceManager = self.device
        with open(path, 'rb') as f:
            width, height, depth = struct.unpack('iii', f.read(4*3))
            if mips is None:
                mips = int(log2(max(width, max(height, depth))))+1
            resx, resy, resz = struct.unpack('ddd', f.read(8*3))
            data = np.zeros(shape=(depth, height, width), dtype=np.float32)
            for x in range(width):
                for y in range(height):
                    data[:, y, x] = struct.unpack('f'*depth, f.read(4*depth))
        texture = device.create_texure_3D(Format.FLOAT, width, height, depth, mips, usage=usage)
        texture.subresource().write(data)  # only write first subresource if multiple
        with device.get_compute() as man:
            man.cpu_to_gpu(texture.subresource())

        if mips > 1:  # automatically build mipmaps
            for mip in range(1, mips):
                self.pool(texture.subresource(mip - 1), 0, dst_volume=texture.subresource(mip))

        return texture

    def load_file_as_numpy(self, path: str):
        with open(path, 'rb') as f:
            width, height, depth = struct.unpack('iii', f.read(4*3))
            resx, resy, resz = struct.unpack('ddd', f.read(8*3))
            data = np.zeros(shape=(depth, height, width), dtype=np.float32)
            for x in range(width):
                for y in range(height):
                    data[:, y, x] = struct.unpack('f'*depth, f.read(4*depth))
        return data

    def load_file_fatten(self, path: str, mips=1, usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE):
        device: DeviceManager = self.device
        with open(path, 'rb') as f:
            width, height, depth = struct.unpack('iii', f.read(4 * 3))
            if mips is None:
                mips = int(log2(max(width, max(height, depth)))) + 1
            resx, resy, resz = struct.unpack('ddd', f.read(8 * 3))
            data = np.zeros(shape=(depth, height, width), dtype=np.float32)
            for x in range(width):
                for y in range(height):
                    data[:, y, x] = struct.unpack('f' * depth, f.read(4 * depth))
        grid = device.create_buffer(4 * width * height * depth, usage=usage, memory=MemoryProperty.GPU)
        grid.write(data)
        with device.get_compute() as man:
            man.cpu_to_gpu(grid)
        return grid, (width, height, depth)

    def compute_super_voxel(self,
                            src_volume: Image,
                            sv_size: int,
                            sv_operation: SuperVoxelOperation,
                            dst_volume: Image = None):
        device: DeviceManager = self.device
        if dst_volume is None:
            sv_cloud_dim = int(math.ceil(src_volume.width / sv_size)), \
                           int(math.ceil(src_volume.height / sv_size)), \
                           int(math.ceil(src_volume.depth / sv_size))
            dst_volume = device.create_image(
                ImageType.TEXTURE_3D, False, Format.FLOAT,
                sv_cloud_dim[0], sv_cloud_dim[1], sv_cloud_dim[2],
                1, 1, ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC, MemoryProperty.GPU
            )
        if self.compute_super_voxel_pipeline is None:
            pipeline = device.create_compute_pipeline()
            pipeline.parameters = device.create_uniform_buffer(
                sv_size=int,
                sv_operation=int
            )
            pipeline.load_compute_shader(__SHADERS_TOOLS__+"/super_voxel_construction.comp.spv")
            pipeline.bind_storage_image(0, ShaderStage.COMPUTE, lambda: pipeline.src_volume)
            pipeline.bind_storage_image(1, ShaderStage.COMPUTE, lambda: pipeline.dst_volume)
            pipeline.bind_uniform(2, ShaderStage.COMPUTE, lambda: pipeline.parameters)
            pipeline.close()
            self.compute_super_voxel_pipeline = pipeline

        pipeline = self.compute_super_voxel_pipeline
        pipeline.src_volume = src_volume
        pipeline.dst_volume = dst_volume
        pipeline.parameters.sv_size = sv_size
        pipeline.parameters.sv_operation = sv_operation
        with device.get_compute() as man:
            man.set_pipeline(pipeline)
            man.dispatch_threads_1D(dst_volume.width * dst_volume.height * dst_volume.depth)

        return dst_volume

    def compute_deepness_field(self,
                             src_volume: Image,
                             tau_reference: float,
                             direction_samples: int = 1024,
                             dst_volume: Image = None):
        device: DeviceManager = self.device
        if dst_volume is None:
            dst_volume = device.create_image(
                ImageType.TEXTURE_3D, False, Format.FLOAT,
                src_volume.width, src_volume.height, src_volume.depth,
                1, 1, ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC, MemoryProperty.GPU
            )
        if self.deepness_initialization_pipeline is None:
            pipeline = device.create_compute_pipeline()
            pipeline.parameters = device.create_uniform_buffer(
                component_0=int, component_1=int, component_2=int, direction=int,
                tau_reference=float, pad=vec3
            )
            pipeline.load_compute_shader(__SHADERS_TOOLS__+"/orthographic_initialization.comp.spv")
            pipeline.bind_storage_image(0, ShaderStage.COMPUTE, lambda: self.deepness_initialization_pipeline.src_volume)
            pipeline.bind_storage_image(1, ShaderStage.COMPUTE, lambda: self.deepness_initialization_pipeline.dst_volume)
            pipeline.bind_uniform(2, ShaderStage.COMPUTE, lambda: self.deepness_initialization_pipeline.parameters)
            pipeline.close()
            self.deepness_initialization_pipeline = pipeline

        if self.deepness_compute_pipeline is None:
            pipeline = device.create_compute_pipeline()
            pipeline.parameters = device.create_uniform_buffer(
                direction=vec3,
                tau_reference=float
            )
            pipeline.load_compute_shader(__SHADERS_TOOLS__+"/distance_field_construction.comp.spv")
            pipeline.bind_storage_image(0, ShaderStage.COMPUTE, lambda: self.deepness_compute_pipeline.src_volume)
            pipeline.bind_storage_image(1, ShaderStage.COMPUTE, lambda: self.deepness_compute_pipeline.dst_volume)
            pipeline.bind_uniform(2, ShaderStage.COMPUTE, lambda: self.deepness_compute_pipeline.parameters)
            pipeline.close()
            self.deepness_compute_pipeline = pipeline

        pipeline = self.deepness_initialization_pipeline
        pipeline.src_volume = src_volume
        pipeline.dst_volume = dst_volume
        pipeline.parameters.tau_reference = tau_reference
        # initialization to 4 (a distance granted to be larger that any ray inside a -1,1 box
        pipeline.parameters.component_0 = 0  # X
        pipeline.parameters.component_1 = 1  # Y
        pipeline.parameters.component_2 = 2  # Z
        pipeline.parameters.direction = 0
        with device.get_compute() as man:
            man.set_pipeline(pipeline)
            man.dispatch_threads_2D(dst_volume.width, dst_volume.height)

        # passes
        pass_settings = [
            (0, 1, 2, 1),   # x,y, -> z, inc
            (0, 1, 2, -1),  # x,y, -> z, dec
            (0, 2, 1, 1),  # x,z, -> y, inc
            (0, 2, 1, -1),  # x,z, -> y, dec
            (1, 2, 0, 1),  # y,z, -> x, inc
            (1, 2, 0, -1),  # y,z, -> x, dec
        ]
        for comp_0, comp_1, comp_2, dir in pass_settings:
            dim = (dst_volume.width, dst_volume.height, dst_volume.depth)
            pipeline.parameters.component_0 = comp_0
            pipeline.parameters.component_1 = comp_1
            pipeline.parameters.component_2 = comp_2
            pipeline.parameters.direction = dir
            with device.get_compute() as man:
                man.set_pipeline(pipeline)
                man.dispatch_threads_2D(dim[comp_0], dim[comp_1])

        print(f"[INFO] Updated axis aligned directions")

        pipeline = self.deepness_compute_pipeline
        pipeline.src_volume = src_volume
        pipeline.dst_volume = dst_volume

        for sample_index in range(direction_samples):
            # get a random direction
            while True:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
                z = np.random.uniform(-1, 1)
                l_sqr = x*x + y*y + z*z
                if 0.01 < l_sqr < 1:
                    d = vec3(x,y,z) / np.sqrt(l_sqr)
                    break

            # Update min radius in that direction
            pipeline.parameters.direction = d
            pipeline.parameters.tau_reference = tau_reference
            with device.get_compute() as man:
                man.set_pipeline(pipeline)
                man.dispatch_threads_1D(dst_volume.width*dst_volume.height*dst_volume.depth)
            print(f"[INFO] Updated direction {sample_index}={str(d)} from {direction_samples}")

        return dst_volume

    def compute_emptyness_field(self,
                            src_volume: Image,
                            levels: int = None,
                            dst_volume: Image = None):
        device: DeviceManager = self.device
        if dst_volume is None:
            dst_volume = device.create_image(
                ImageType.TEXTURE_3D, False, Format.FLOAT,
                src_volume.width, src_volume.height, src_volume.depth,
                1, 1, ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC, MemoryProperty.GPU
            )
        if self.chebyshev_distance_pipeline is None:
            pipeline = device.create_compute_pipeline()
            pipeline.parameters = device.create_uniform_buffer(level=int)
            pipeline.load_compute_shader(__SHADERS_TOOLS__+"/compute_chebyshev_distance.comp.spv")
            pipeline.bind_storage_image(0, ShaderStage.COMPUTE, lambda: pipeline.src_volume)
            pipeline.bind_storage_image(1, ShaderStage.COMPUTE, lambda: pipeline.dst_volume)
            pipeline.bind_uniform(2, ShaderStage.COMPUTE, lambda: pipeline.parameters)
            pipeline.close()
            self.chebyshev_distance_pipeline = pipeline
        if levels is None:
            max_dim = max(dst_volume.width, max(dst_volume.height, dst_volume.depth))
            levels = int(ceil(log2(max_dim)))
        pipeline = self.chebyshev_distance_pipeline
        pipeline.src_volume = src_volume
        pipeline.dst_volume = dst_volume
        for level in range(levels):
            pipeline.parameters.level = level
            with device.get_compute() as man:
                man.set_pipeline(pipeline)
                man.dispatch_threads_1D(dst_volume.width*dst_volume.height*dst_volume.depth)
        return dst_volume

    def pool(self,
             src_volume: Image,
             operation: int,
             dst_volume: Image = None):
        device: DeviceManager = self.device
        if dst_volume is None:
            dst_volume = device.create_image(
                ImageType.TEXTURE_3D, False, src_volume.format,
                src_volume.width // 2, src_volume.height // 2, src_volume.depth // 2,
                1, 1, ImageUsage.STORAGE | ImageUsage.TRANSFER_SRC, MemoryProperty.GPU
            )
        if self.pooling_pipeline is None:
            pipeline = device.create_compute_pipeline()
            pipeline.parameters = device.create_uniform_buffer(
                src_offset=ivec3, pad0=float,
                src_size=ivec3, pad1=float,
                dst_offset=ivec3, pad2=float,
                dst_size=ivec3,
                operation=int
            )
            pipeline.load_compute_shader(__SHADERS_TOOLS__+"/compute_pooling_3d.comp.spv")
            pipeline.bind_storage_image(0, ShaderStage.COMPUTE, lambda: pipeline.src_volume)
            pipeline.bind_storage_image(1, ShaderStage.COMPUTE, lambda: pipeline.dst_volume)
            pipeline.bind_uniform(2, ShaderStage.COMPUTE, lambda: pipeline.parameters)
            pipeline.close()
            self.pooling_pipeline = pipeline
        pipeline = self.pooling_pipeline
        pipeline.src_volume = src_volume
        pipeline.dst_volume = dst_volume
        pipeline.parameters.src_offset = ivec3(0)
        pipeline.parameters.src_size = ivec3(src_volume.width, src_volume.height, src_volume.depth)
        pipeline.parameters.dst_offset = ivec3(0)
        pipeline.parameters.dst_size = ivec3(dst_volume.width, dst_volume.height, dst_volume.depth)
        pipeline.parameters.operation = operation
        with device.get_compute() as man:
            man.set_pipeline(pipeline)
            man.dispatch_threads_1D(dst_volume.width * dst_volume.height * dst_volume.depth)
        return dst_volume
