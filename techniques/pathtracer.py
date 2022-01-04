from rendering.manager import *
from rendering.scenes import *
import os

__PT_SHADERS__ = os.path.dirname(__file__)+"/shaders/PT"

compile_shader_sources(__PT_SHADERS__)


class Pathtracer(Technique):
    def __init__(self, scene: RaytracingScene, image):
        super().__init__()
        self.scene = scene
        self.image = image
        self.shader_folder = __PT_SHADERS__
        self.pipeline = None
        self.program = None
        self.camera_uniform = None
        self.scene_is_dirty = True

    def __setup__(self):
        # create accumulation image
        self.accumulation = self.create_image(ImageType.TEXTURE_2D, False,
                                              Format.VEC4, self.image.width, self.image.height,
                                              1, 1, 1, ImageUsage.STORAGE | ImageUsage.TRANSFER_DST, MemoryProperty.GPU)
        # create pipeline
        pipeline = self.create_raytracing_pipeline()
        ray_gen = pipeline.load_rt_generation_shader(self.shader_folder+'/raygen.rgen.spv')
        ray_miss = pipeline.load_rt_miss_shader(self.shader_folder+'/raymiss.rmiss.spv')
        ray_closest_hit = pipeline.load_rt_closest_hit_shader(self.shader_folder+'/rayhit.rchit.spv')
        gen_group = pipeline.create_rt_gen_group(ray_gen)
        miss_group = pipeline.create_rt_miss_group(ray_miss)
        hit_group = pipeline.create_rt_hit_group(ray_closest_hit)
        pipeline.descriptor_set(0)
        pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION, lambda: self.scene.scene_ads)
        pipeline.bind_storage_image(1, ShaderStage.RT_GENERATION, lambda: self.image)
        pipeline.bind_storage_image(2, ShaderStage.RT_GENERATION, lambda: self.accumulation)
        pipeline.bind_uniform(3, ShaderStage.RT_GENERATION, lambda: self.camera_uniform)
        pipeline.bind_constants(0, ShaderStage.RT_GENERATION,
            # Fields
            frame_index=int
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
        self.frame_index = 0
        with self.get_graphics() as man:
            man.clear_color(image=self.accumulation, color=(0,0,0,0))

    def update_camera(self, camera):
        self.camera = camera
        self.camera_is_dirty = True
        view, proj = self.camera.build_matrices(self.image.width, self.image.height)
        world_2_proj = proj * view
        proj_2_world = glm.inverse(world_2_proj)
        self.camera_uniform.proj2world = proj_2_world

    def _update_uniforms(self, man: RaytracingManager):
        # Update necessary buffers
        if self.camera_is_dirty:
            man.cpu_to_gpu(self.camera_uniform)

    def __dispatch__(self):
        with self.get_raytracing() as man:
            self._update_uniforms(man)
            # Set pipeline
            man.set_pipeline(self.pipeline)
            man.update_sets(0)
            man.update_sets(1)
            man.update_constants(
                ShaderStage.RT_GENERATION,
                frame_index=self.frame_index
            )
            # dispatch rays
            man.dispatch_rays(self.program, self.image.width, self.image.height)
            # clear all dirty flags
            self.camera_is_dirty = False
        self.frame_index += 1
