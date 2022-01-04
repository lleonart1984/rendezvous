from rendering.manager import *
from rendering.scenes import *
import random
import glm
import os


__VOLUME_RECONSTRUCTION_SHADERS__ = os.path.dirname(__file__)+"/shaders/VR"

compile_shader_sources(__VOLUME_RECONSTRUCTION_SHADERS__)

class TransmittanceGenerator(Technique):
    def __init__(self, grid, output_image):
        super().__init__()
        self.grid = grid
        self.output_image = output_image
        self.width, self.height = output_image.width, output_image.height

    def __setup__(self):
        # rays
        self.rays = self.create_buffer(6 * 4 * self.width * self.height,
                                       BufferUsage.STORAGE | BufferUsage.TRANSFER_SRC | BufferUsage.TRANSFER_DST,
                                       MemoryProperty.GPU)
        # Transmittance
        self.transmittances = self.create_buffer(3 * 4 * self.width * self.height,
                                                 BufferUsage.STORAGE | BufferUsage.TRANSFER_SRC | BufferUsage.TRANSFER_DST,
                                                 MemoryProperty.GPU)
        # camera buffer
        self.camera_buffer = self.create_uniform_buffer(
            ProjToWorld=glm.mat4
        )
        # medium properties
        self.medium_buffer = self.create_uniform_buffer(
            scatteringAlbedo=glm.vec3,
            density=float,
            phase_g=float
        )
        pipeline = self.create_compute_pipeline()
        pipeline.load_compute_shader(__VOLUME_RECONSTRUCTION_SHADERS__+'/generator.comp.spv')
        pipeline.bind_storage_image(0, ShaderStage.COMPUTE, lambda: self.output_image)
        pipeline.bind_storage_image(1, ShaderStage.COMPUTE, lambda: self.grid)
        pipeline.bind_storage_buffer(2, ShaderStage.COMPUTE, lambda: self.rays)
        pipeline.bind_storage_buffer(3, ShaderStage.COMPUTE, lambda: self.transmittances)
        pipeline.bind_uniform(4, ShaderStage.COMPUTE, lambda: self.camera_buffer)
        pipeline.bind_uniform(5, ShaderStage.COMPUTE, lambda: self.medium_buffer)
        pipeline.close()
        self.pipeline = pipeline
        self.set_camera(glm.vec3(0,0,-3), glm.vec3(0,0,0))
        self.set_medium(glm.vec3(1,1,1), 10, 0.875)

    def set_camera(self, look_from: glm.vec3, look_to: glm.vec3):
        # Setup camera
        proj = glm.perspective(45, self.width / self.height, 0.01, 1000)
        view = glm.lookAt(look_from, look_to, glm.vec3(0, 1, 0))
        proj_to_model = glm.inverse(proj * view)
        self.camera_buffer.ProjToWorld = proj_to_model

    def set_medium(self, scattering_albedo: glm.vec3, density: float, phase_g: float):
        self.medium_buffer.scatteringAlbedo = scattering_albedo
        self.medium_buffer.density = density
        self.medium_buffer.phase_g = phase_g

    def __dispatch__(self):
        with self.get_compute() as man:
            man.set_pipeline(self.pipeline)
            man.update_sets(0)
            man.dispatch_threads_2D(self.width, self.height)


class TransmittanceForward(Technique):
    def __init__(self, rays, grid_dim, grid, transmittances):
        super().__init__()
        self.rays = rays  # input
        self.grid = grid  # params
        self.transmittances = transmittances  # output
        self.grid_dim = grid_dim

    def set_medium(self, scattering_albedo: glm.vec3, density: float, phase_g: float):
        self.medium_buffer.scatteringAlbedo = scattering_albedo
        self.medium_buffer.density = density
        self.medium_buffer.phase_g = phase_g

    def __setup__(self):
        # medium properties
        self.medium_buffer = self.create_uniform_buffer(
            scatteringAlbedo=glm.vec3,
            density=float,
            phase_g=float
        )
        pipeline = self.create_compute_pipeline()
        pipeline.load_compute_shader(__VOLUME_RECONSTRUCTION_SHADERS__ + '/forward.comp.spv')
        pipeline.bind_storage_buffer(0, ShaderStage.COMPUTE, lambda: self.grid)
        pipeline.bind_storage_buffer(1, ShaderStage.COMPUTE, lambda: self.rays)
        pipeline.bind_storage_buffer(2, ShaderStage.COMPUTE, lambda: self.transmittances)
        pipeline.bind_uniform(3, ShaderStage.COMPUTE, lambda: self.medium_buffer)
        pipeline.bind_constants(0, ShaderStage.COMPUTE,
            grid_dim = glm.ivec3,
            number_of_rays = int
        )
        pipeline.close()
        self.pipeline = pipeline
        self.set_medium(glm.vec3(1, 1, 1), 10, 0.875)

    def __dispatch__(self):
        with self.get_compute() as man:
            man.set_pipeline(self.pipeline)
            man.update_sets(0)
            ray_count = self.rays.size // (4*3*2)
            man.update_constants(ShaderStage.COMPUTE,
                grid_dim=self.grid_dim,
                number_of_rays=ray_count
            )
            man.dispatch_threads_1D(ray_count)


class TransmittanceBackward(Technique):
    def __init__(self, rays, grid_dim, gradient_densities, transmittances, gradient_transmittances):
        super().__init__()
        self.grid_dim = grid_dim
        self.rays = rays  # buffer with rays configurations (origin, direction)
        self.gradient_densities = gradient_densities  # Flatten grid 512x512x512 used as parameters
        self.transmittances = transmittances  # Float with transmittance for each ray
        self.gradient_transmittances = gradient_transmittances
        self.pipeline = None

    def set_medium(self, scattering_albedo: glm.vec3, density: float, phase_g: float):
        self.medium_buffer.scatteringAlbedo = scattering_albedo
        self.medium_buffer.density = density
        self.medium_buffer.phase_g = phase_g

    def __setup__(self):
        # medium properties
        self.medium_buffer = self.create_uniform_buffer(
            scatteringAlbedo=glm.vec3,
            density=float,
            phase_g=float
        )
        pipeline = self.create_compute_pipeline()
        pipeline.load_compute_shader(__VOLUME_RECONSTRUCTION_SHADERS__ + '/backward.comp.spv')
        pipeline.bind_storage_buffer(0, ShaderStage.COMPUTE, lambda: self.gradient_densities)
        pipeline.bind_storage_buffer(1, ShaderStage.COMPUTE, lambda: self.rays)
        pipeline.bind_storage_buffer(2, ShaderStage.COMPUTE, lambda: self.transmittances)
        pipeline.bind_storage_buffer(3, ShaderStage.COMPUTE, lambda: self.gradient_transmittances)
        pipeline.bind_uniform(4, ShaderStage.COMPUTE, lambda: self.medium_buffer)
        pipeline.bind_constants(0, ShaderStage.COMPUTE,
                                grid_dim=glm.ivec3,
                                number_of_rays=int
                                )
        pipeline.close()
        self.pipeline = pipeline
        self.set_medium(glm.vec3(1, 1, 1), 10, 0.875)

    def __dispatch__(self):
        with self.get_compute() as man:
            man.clear_buffer(self.gradient_densities)  # Zero grad
            man.set_pipeline(self.pipeline)
            man.update_sets(0)
            ray_count = self.rays.size // (4 * 3 * 2)
            man.update_constants(ShaderStage.COMPUTE,
                                 grid_dim=self.grid_dim,
                                 number_of_rays=ray_count
                                 )
            man.dispatch_threads_1D(ray_count)


class UpSampleGrid(Technique):
    def __init__(self):
        self.src_grid = None
        self.dst_grid = None
        self.src_grid_dim = glm.ivec3(0,0,0)
        self.dst_grid_dim = glm.ivec3(0,0,0)

    def set_src_grid(self, grid_dim, grid):
        self.src_grid = grid
        self.src_grid_dim = grid_dim

    def set_dst_grid(self, grid_dim, grid):
        self.dst_grid = grid
        self.dst_grid_dim = grid_dim

    def __setup__(self):
        pipeline = self.create_compute_pipeline()
        pipeline.load_compute_shader(__VOLUME_RECONSTRUCTION_SHADERS__+"/initialize.comp.spv")
        pipeline.bind_storage_buffer(0, ShaderStage.COMPUTE, lambda: self.dst_grid)
        pipeline.bind_storage_buffer(1, ShaderStage.COMPUTE, lambda: self.src_grid)
        pipeline.bind_constants(0, ShaderStage.COMPUTE,
            dst_grid_dim=glm.ivec3, rem0=float,
            src_grid_dim=glm.ivec3, rem1=float
        )
        pipeline.close()
        self.pipeline = pipeline

    def __dispatch__(self):
        with self.get_compute() as man:
            man.set_pipeline(self.pipeline)
            man.update_sets(0)
            man.update_constants(ShaderStage.COMPUTE,
                dst_grid_dim=self.dst_grid_dim,
                src_grid_dim=self.src_grid_dim
            )
            man.dispatch_threads_1D(self.dst_grid_dim.x * self.dst_grid_dim.y * self.dst_grid_dim.z)
            man.gpu_to_cpu(self.dst_grid)