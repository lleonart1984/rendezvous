from rendering.manager import *
from rendering.training import *
from rendering.scenes import *
import random
import os


__RBG_SHADERS__ = os.path.dirname(__file__)+"/shaders/RBP"

compile_shader_sources(__RBG_SHADERS__)


class RBPForward(Technique):
    def __init__(self, scene: RaytracingScene, image_width, image_height, input_parameters, output_image):
        super().__init__()
        self.scene = scene
        self.image_width = image_width
        self.image_height = image_height
        self.input_parameters = input_parameters  # Flatten texture 512x512 used as parameters
        self.output_image = output_image  # Flatten output image of the pathtracer
        self.shader_folder = __RBG_SHADERS__
        self.pipeline = None
        self.program = None
        self.camera_uniform = None
        self.scene_is_dirty = True
        self.number_of_samples = 1
        self.rnd = random.Random()

    def __setup__(self):
        # create pipeline
        pipeline = self.create_raytracing_pipeline()
        ray_gen = pipeline.load_rt_generation_shader(self.shader_folder+'/forward.rgen.spv')
        ray_miss = pipeline.load_rt_miss_shader(self.shader_folder+'/forward.rmiss.spv')
        ray_closest_hit = pipeline.load_rt_closest_hit_shader(self.shader_folder+'/forward.rchit.spv')
        gen_group = pipeline.create_rt_gen_group(ray_gen)
        miss_group = pipeline.create_rt_miss_group(ray_miss)
        hit_group = pipeline.create_rt_hit_group(ray_closest_hit)
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


class RBPBackward(Technique):
    def __init__(self, scene: RaytracingScene, image_width, image_height, input_parameters, grad_parameters, grad_output):
        super().__init__()
        self.scene = scene
        self.image_width = image_width
        self.image_height = image_height
        self.input_parameters = input_parameters  # Flatten texture 512x512 used as parameters
        self.grad_parameters = grad_parameters  # parameters of the scene
        self.grad_output = grad_output  # gradients for the parameters of the scene
        self.shader_folder = __RBG_SHADERS__
        self.pipeline = None
        self.program = None
        self.camera_uniform = None
        self.rnd = random.Random()
        self.number_of_samples = 1

    def __setup__(self):
        # create pipeline
        pipeline = self.create_raytracing_pipeline()
        ray_gen = pipeline.load_rt_generation_shader(self.shader_folder+'/backward.rgen.spv')
        ray_miss = pipeline.load_rt_miss_shader(self.shader_folder+'/backward.rmiss.spv')
        ray_closest_hit = pipeline.load_rt_closest_hit_shader(self.shader_folder+'/backward.rchit.spv')
        gen_group = pipeline.create_rt_gen_group(ray_gen)
        miss_group = pipeline.create_rt_miss_group(ray_miss)
        hit_group = pipeline.create_rt_hit_group(ray_closest_hit)
        pipeline.descriptor_set(0)
        pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION, lambda: self.scene.scene_ads)
        pipeline.bind_storage_buffer(1, ShaderStage.RT_GENERATION, lambda: self.grad_output)
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
        pipeline.bind_storage_buffer(1, ShaderStage.RT_CLOSEST_HIT, lambda: self.grad_parameters)
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
            man.clear_buffer(self.grad_parameters)  # Zero grad
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
