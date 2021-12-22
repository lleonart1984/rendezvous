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
                                       BufferUsage.STORAGE,
                                       MemoryProperty.GPU)
        # Transmittance
        self.transmittances = self.create_buffer(3 * 4 * self.width * self.height,
                                                 BufferUsage.STORAGE,
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
    def __init__(self, grid_dim, grid, rays, transmittances):
        super().__init__()
        self.transmittances = transmittances
        self.rays = rays
        self.grid = grid
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


class VolRecForward(Technique):
    def __init__(self, rays, input_parameters, output_transmittance, shader_folder):
        super().__init__()
        self.rays = rays  # buffer with rays configurations (origin, direction)
        self.input_parameters = input_parameters  # Flatten grid 512x512x512 used as parameters
        self.output_transmittance = output_transmittance  # Float with transmittance for each ray
        self.shader_folder = shader_folder
        self.pipeline = None

    def __setup__(self):
        # create pipeline
        pipeline = self.create_compute_pipeline()
        pipeline.load_compute_shader(self.shader_folder+'/forward.comp.spv')
        pipeline.descriptor_set(0)
        pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION, lambda: self.scene.scene_ads)
        pipeline.bind_storage_buffer(1, ShaderStage.RT_GENERATION, lambda: self.output_image)
        pipeline.bind_uniform(2, ShaderStage.RT_GENERATION, lambda: self.camera_uniform)
        pipeline.bind_constants(0, ShaderStage.RT_GENERATION,
            # Fields
            frame_seed=int,
            number_of_samples=int
        )
        pipeline.descriptor_set(1)
        pipeline.sampler = self.create_sampler_linear()
        pipeline.bind_storage_buffer(0, ShaderStage.RT_CLOSEST_HIT, lambda: self.scene.vertices)
        pipeline.bind_storage_buffer(1, ShaderStage.RT_CLOSEST_HIT, lambda: self.scene.indices)
        pipeline.bind_storage_buffer(2, ShaderStage.RT_CLOSEST_HIT, lambda: self.scene.transforms)
        pipeline.bind_storage_buffer(3, ShaderStage.RT_CLOSEST_HIT, lambda: self.scene.material_buffer)
        pipeline.bind_storage_buffer(4, ShaderStage.RT_CLOSEST_HIT, lambda: self.scene.geometry_descriptions)
        pipeline.bind_storage_buffer(5, ShaderStage.RT_CLOSEST_HIT, lambda: self.scene.instance_descriptions)
        pipeline.bind_texture_combined_array(6, ShaderStage.RT_CLOSEST_HIT, -1, lambda: [
            (texture, pipeline.sampler) for texture in self.scene.textures
        ])
        pipeline.descriptor_set(2)
        pipeline.bind_storage_buffer(0, ShaderStage.RT_CLOSEST_HIT, lambda: self.input_parameters)
        pipeline.close()
        # create program
        program = pipeline.create_rt_program(1, 1)
        program.set_generation(gen_group)
        program.set_hit_group(0, hit_group)
        program.set_miss(0, miss_group)
        # create uniforms
        self.camera_uniform = self.create_uniform_buffer(
            BufferUsage.UNIFORM | BufferUsage.TRANSFER_DST,
            MemoryProperty.GPU,
            # fields
            proj2world=glm.mat4x4
        )
        self.update_camera(Camera())
        self.pipeline = pipeline
        self.program = program

    def update_camera(self, camera):
        self.camera = camera
        self.camera_is_dirty = True
        view, proj = self.camera.build_matrices(self.image_width, self.image_height)
        world_2_proj = proj * view
        proj_2_world = glm.inverse(world_2_proj)
        self.camera_uniform.proj2world = proj_2_world

    def _update_uniforms(self, man: RaytracingManager):
        # Update necessary buffers
        if self.camera_is_dirty:
            man.cpu_to_gpu(self.camera_uniform)

    def __dispatch__(self):
        with self.get_raytracing() as man:
            man.clear_buffer(self.output_image, 0)  # Clear accumulator image
            self._update_uniforms(man)
            # Set pipeline
            man.set_pipeline(self.pipeline)
            man.update_sets(0, 1, 2)
            for _ in range(self.number_of_samples):
                man.update_constants(
                    ShaderStage.RT_GENERATION,
                    frame_seed=self.rnd.randint(0, 10000000),
                    number_of_samples=self.number_of_samples
                )
                # dispatch rays
                man.dispatch_rays(self.program, self.image_width, self.image_height)
            # clear all dirty flags
            self.camera_is_dirty = False