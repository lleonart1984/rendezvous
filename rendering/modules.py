from rendering.training import *
import random
import glm
import os
import numpy as np
from rendering.scenes import *

__MODULES_SHADERS__ = os.path.dirname(__file__)+"/shaders/Modules"
compile_shader_sources(__MODULES_SHADERS__)


class RayGenerator(RendererModule):
    def __init__(self, device, output_dim: (int, int), mode: int, *args, **kwargs):
        self.output_dim = output_dim
        self.mode = mode
        self.camera_buffer = None
        super().__init__(device, *args, **kwargs)

    def setup(self):
        self.camera_buffer = self.device.create_uniform_buffer(
            ProjToWorld=glm.mat4
        )
        self.rays = self.device.create_gpu_wrapped_ptr()
        pipeline = self.device.create_compute_pipeline()
        pipeline.load_compute_shader(__MODULES_SHADERS__+"/raygen.forward.comp.spv")
        pipeline.bind_wrap_gpu(0, ShaderStage.COMPUTE, lambda: self.rays)
        pipeline.bind_uniform(1, ShaderStage.COMPUTE, lambda: self.camera_buffer)
        pipeline.bind_constants(
            0, ShaderStage.COMPUTE,
            dim=glm.ivec2,
            mode=int,
            seed=int
        )
        pipeline.close()
        self.pipeline = pipeline

    def forward_render(self, inputs):
        origins, targets = inputs
        origins = origins.reshape(-1, 3)
        targets = targets.reshape(-1, 3)
        full_rays = torch.zeros(len(origins), self.output_dim[0], self.output_dim[1], 6, device=torch.device('cuda:0'))
        if len(origins) == 1:
            full_rays = full_rays[0]  # squeeze
        for i, (o, t) in enumerate(zip(origins, targets)):
            self.rays.wrap(full_rays, i * self.output_dim[0] * self.output_dim[1] * 6 * 4)
            # Setup camera
            proj = glm.perspective(45, self.output_dim[1] / self.output_dim[0], 0.01, 1000)
            view = glm.lookAt(glm.vec3(*o), glm.vec3(*t), glm.vec3(0, 1, 0))
            proj_to_model = glm.inverse(proj * view)
            self.camera_buffer.ProjToWorld = proj_to_model
            with self.device.get_compute() as man:
                man.set_pipeline(self.pipeline)
                man.update_sets(0)
                man.update_constants(
                    ShaderStage.COMPUTE,
                    dim=glm.ivec2(self.output_dim[1], self.output_dim[0]),
                    mode=self.mode,
                    seed=np.random.randint(0, 10000000)
                )
                man.dispatch_threads_2D(self.output_dim[1], self.output_dim[0])
        return [full_rays.to(origins.device)]


class TransmittanceRenderer(RendererModule):
    def __init__(self, device, *args, **kwargs):
        super().__init__(device, *args, **kwargs)

    def setup(self):
        self.medium_buffer = self.device.create_uniform_buffer(
            scatteringAlbedo=glm.vec3,
            density=float,
            phase_g=float
        )
        self.grid = self.device.create_gpu_wrapped_ptr()
        self.rays = self.device.create_gpu_wrapped_ptr()
        self.transmittances = self.device.create_gpu_wrapped_ptr()
        self.grid_gradients = self.device.create_gpu_wrapped_ptr()
        self.transmittance_gradients = self.device.create_gpu_wrapped_ptr()
        pipeline = self.device.create_compute_pipeline()
        pipeline.load_compute_shader(__MODULES_SHADERS__ + '/transmittance.forward.comp.spv')
        pipeline.bind_wrap_gpu(0, ShaderStage.COMPUTE, lambda: self.grid)
        pipeline.bind_wrap_gpu(1, ShaderStage.COMPUTE, lambda: self.rays)
        pipeline.bind_wrap_gpu(2, ShaderStage.COMPUTE, lambda: self.transmittances)
        pipeline.bind_uniform(3, ShaderStage.COMPUTE, lambda: self.medium_buffer)
        pipeline.bind_constants(0, ShaderStage.COMPUTE,
                                grid_dim=glm.ivec3,
                                number_of_rays=int,
                                box_minim=glm.vec3, pad0=float,
                                box_size=glm.vec3, pad1=float,
                                )
        pipeline.close()
        self.forward_pipeline = pipeline

        pipeline = self.device.create_compute_pipeline()
        pipeline.load_compute_shader(__MODULES_SHADERS__ + '/transmittance.backward.comp.spv')
        pipeline.bind_wrap_gpu(0, ShaderStage.COMPUTE, lambda: self.grid_gradients)
        pipeline.bind_wrap_gpu(1, ShaderStage.COMPUTE, lambda: self.rays)
        pipeline.bind_wrap_gpu(2, ShaderStage.COMPUTE, lambda: self.transmittances)
        pipeline.bind_wrap_gpu(3, ShaderStage.COMPUTE, lambda: self.transmittance_gradients)
        pipeline.bind_uniform(4, ShaderStage.COMPUTE, lambda: self.medium_buffer)
        pipeline.bind_constants(0, ShaderStage.COMPUTE,
                                grid_dim=glm.ivec3,
                                number_of_rays=int,
                                box_minim=glm.vec3, pad0=float,
                                box_size=glm.vec3, pad1=float,
                                )
        pipeline.close()
        self.backward_pipeline = pipeline

    def set_medium(self, scattering_albedo: glm.vec3, density: float, phase_g: float):
        self.medium_buffer.scatteringAlbedo = scattering_albedo
        self.medium_buffer.density = density
        self.medium_buffer.phase_g = phase_g

    def set_box(self, box_minim: glm.vec3, box_size: glm.vec3):
        self.box_minim = box_minim
        self.box_size = box_size

    def forward_render(self, inputs):
        rays, grid = inputs
        output_device = rays.device
        cuda_device = torch.device('cuda:0')
        rays, grid = rays.to(cuda_device), grid.to(cuda_device)
        grid_dim = grid.shape
        ray_count = torch.numel(rays) // 6
        self.rays.wrap(rays)
        self.grid.wrap(grid)
        transmittances = torch.zeros(ray_count * 3, device=cuda_device)
        self.transmittances.wrap(transmittances)
        with self.device.get_compute() as man:
            man.set_pipeline(self.forward_pipeline)
            man.update_sets(0)
            man.update_constants(ShaderStage.COMPUTE,
                grid_dim=glm.ivec3(grid_dim[2], grid_dim[1], grid_dim[0]),
                number_of_rays=ray_count,
                box_minim=self.box_minim,
                box_size=self.box_size
            )
            man.dispatch_threads_1D(ray_count)
        shape = list (rays.shape)
        shape[-1] //= 2  # for each 6 values in the last dimension there is 3 values in transmittances
        return [transmittances.reshape(shape).to(output_device)]

    def backward_render(self, inputs, outputs, output_gradients):
        rays, grid = inputs
        transmittances, = outputs
        transmittance_gradients, = output_gradients
        grid_device = grid.device
        cuda_device = torch.device('cuda:0')
        rays, grid, transmittances, transmittance_gradients = \
            rays.to(cuda_device), \
            grid.to(cuda_device), \
            transmittances.to(cuda_device), \
            transmittance_gradients.to(cuda_device)
        grid_dim = grid.shape
        ray_count = torch.numel(rays) // 6
        self.rays.wrap(rays)
        self.transmittances.wrap(transmittances)
        self.transmittance_gradients.wrap(transmittance_gradients)
        gradients = torch.zeros_like(grid, device=cuda_device)
        self.grid_gradients.wrap(gradients)
        with self.device.get_compute() as man:
            man.set_pipeline(self.backward_pipeline)
            man.update_sets(0)
            man.update_constants(ShaderStage.COMPUTE,
                                 grid_dim=glm.ivec3(grid_dim[2], grid_dim[1], grid_dim[0]),
                                 number_of_rays=ray_count,
                                 box_minim=self.box_minim,
                                 box_size=self.box_size
                                 )
            man.dispatch_threads_1D(ray_count)
        return [None, gradients.to(grid_device)]


class Resample2D(RendererModule):
    def __init__(self, device: DeviceManager, output_dim: (int, int), components, *args, **kwargs):
        self.output_dim = output_dim
        self.components = components
        super().__init__(device, *args, **kwargs)

    def setup(self):
        self.dst_grid = self.device.create_gpu_wrapped_ptr()
        self.src_grid = self.device.create_gpu_wrapped_ptr()
        pipeline = self.device.create_compute_pipeline()
        if self.components == 3:
            pipeline.load_compute_shader(__MODULES_SHADERS__ + "/resampling2d3f.forward.comp.spv")
        else:
            raise Exception("Unsupported component number")
        pipeline.bind_wrap_gpu(0, ShaderStage.COMPUTE, lambda: self.dst_grid)
        pipeline.bind_wrap_gpu(1, ShaderStage.COMPUTE, lambda: self.src_grid)
        pipeline.bind_constants(0, ShaderStage.COMPUTE,
                                dst_grid_dim=glm.ivec2,
                                src_grid_dim=glm.ivec2
                                )
        pipeline.close()
        self.pipeline = pipeline

    def forward_render(self, inputs: List[torch.Tensor]):
        src_grid, = inputs
        src_device = src_grid.device
        cuda_device = torch.device('cuda:0')
        src_grid = src_grid.to(cuda_device)
        self.src_grid.wrap(src_grid)
        dst_tensor = torch.zeros(self.output_dim, device=cuda_device)
        self.dst_grid.wrap(dst_tensor)
        src_grid_dim = src_grid.shape
        dst_grid_dim = self.output_dim
        with self.device.get_compute() as man:
            man.set_pipeline(self.pipeline)
            man.update_sets(0)
            man.update_constants(ShaderStage.COMPUTE,
                dst_grid_dim=glm.ivec2(dst_grid_dim[1], dst_grid_dim[0]),
                src_grid_dim=glm.ivec2(src_grid_dim[1], src_grid_dim[0])
            )
            man.dispatch_threads_1D(dst_grid_dim[0] * dst_grid_dim[1])
        return [dst_tensor.to(src_device)]


class Resample3D(RendererModule):
    def __init__(self, device: DeviceManager, output_dim: (int, int, int), *args, **kwargs):
        self.output_dim = output_dim
        super().__init__(device, *args, **kwargs)

    def setup(self):
        self.dst_grid = self.device.create_gpu_wrapped_ptr()
        self.src_grid = self.device.create_gpu_wrapped_ptr()
        pipeline = self.device.create_compute_pipeline()
        pipeline.load_compute_shader(__MODULES_SHADERS__ + "/resampling3d.forward.comp.spv")
        pipeline.bind_wrap_gpu(0, ShaderStage.COMPUTE, lambda: self.dst_grid)
        pipeline.bind_wrap_gpu(1, ShaderStage.COMPUTE, lambda: self.src_grid)
        pipeline.bind_constants(0, ShaderStage.COMPUTE,
                                dst_grid_dim=glm.ivec3, rem0=float,
                                src_grid_dim=glm.ivec3, rem1=float
                                )
        pipeline.close()
        self.pipeline = pipeline

    def forward_render(self, inputs: List[torch.Tensor]):
        src_grid, = inputs
        src_device = src_grid.device
        cuda_device = torch.device('cuda:0')
        src_grid = src_grid.to(cuda_device)
        self.src_grid.wrap(src_grid)
        dst_tensor = torch.zeros(self.output_dim, device=cuda_device)
        self.dst_grid.wrap(dst_tensor)
        src_grid_dim = src_grid.shape
        dst_grid_dim = self.output_dim
        with self.device.get_compute() as man:
            man.set_pipeline(self.pipeline)
            man.update_sets(0)
            man.update_constants(ShaderStage.COMPUTE,
                dst_grid_dim=glm.ivec3(dst_grid_dim[2], dst_grid_dim[1], dst_grid_dim[0]),
                src_grid_dim=glm.ivec3(src_grid_dim[2], src_grid_dim[1], src_grid_dim[0])
            )
            man.dispatch_threads_1D(dst_grid_dim[0] * dst_grid_dim[1] * dst_grid_dim[2])
        return [dst_tensor.to(src_device)]


class PTRender(RendererModule):
    def __init__(self, device: DeviceManager, forward_samples: int = 20):
        super().__init__(device)
        self.forward_samples = forward_samples

    def set_scene(self, scene: RaytracingScene):
        self.scene = scene

    def setup(self):
        # input
        self.rays = self.device.create_gpu_wrapped_ptr()
        self.seeds = self.device.create_gpu_wrapped_ptr()
        self.texture_param = self.device.create_gpu_wrapped_ptr()
        self.param_dim_buffer = self.device.create_uniform_buffer(param_dim=glm.ivec2)
        self.texture_gradients = self.device.create_gpu_wrapped_ptr()
        self.consts = self.device.create_uniform_buffer(number_of_samples=int)
        # output
        self.radiances = self.device.create_gpu_wrapped_ptr()
        self.radiance_gradients = self.device.create_gpu_wrapped_ptr()
        # create forward pipeline
        if True:
            pipeline = self.device.create_raytracing_pipeline()
            ray_gen = pipeline.load_rt_generation_shader(__MODULES_SHADERS__ + '/prb.forward.rgen.spv')
            ray_miss = pipeline.load_rt_miss_shader(__MODULES_SHADERS__ + '/prb.forward.rmiss.spv')
            ray_closest_hit = pipeline.load_rt_closest_hit_shader(__MODULES_SHADERS__ + '/prb.forward.rchit.spv')
            gen_group = pipeline.create_rt_gen_group(ray_gen)
            miss_group = pipeline.create_rt_miss_group(ray_miss)
            hit_group = pipeline.create_rt_hit_group(ray_closest_hit)

            pipeline.descriptor_set(0)
            pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION, lambda: self.scene.scene_ads)
            pipeline.bind_wrap_gpu(1, ShaderStage.RT_GENERATION, lambda: self.radiances)
            pipeline.bind_wrap_gpu(2, ShaderStage.RT_GENERATION, lambda: self.rays)
            pipeline.bind_wrap_gpu(3, ShaderStage.RT_GENERATION, lambda: self.seeds)
            pipeline.bind_uniform(4, ShaderStage.RT_GENERATION, lambda: self.consts)

            pipeline.descriptor_set(1)
            pipeline.sampler = self.device.create_sampler_linear()
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
            pipeline.bind_wrap_gpu(0, ShaderStage.RT_CLOSEST_HIT, lambda: self.texture_param)
            pipeline.bind_uniform(1, ShaderStage.RT_CLOSEST_HIT, lambda: self.param_dim_buffer)

            pipeline.close()
            # create program
            program = pipeline.create_rt_program(1, 1)
            program.set_generation(gen_group)
            program.set_hit_group(0, hit_group)
            program.set_miss(0, miss_group)
            self.forward_pipeline = pipeline
            self.forward_program = program
        # create backward pipeline
        if True:
            pipeline = self.device.create_raytracing_pipeline()
            ray_gen = pipeline.load_rt_generation_shader(__MODULES_SHADERS__ + '/prb.backward.rgen.spv')
            ray_miss = pipeline.load_rt_miss_shader(__MODULES_SHADERS__ + '/prb.backward.rmiss.spv')
            ray_closest_hit = pipeline.load_rt_closest_hit_shader(__MODULES_SHADERS__ + '/prb.backward.rchit.spv')
            gen_group = pipeline.create_rt_gen_group(ray_gen)
            miss_group = pipeline.create_rt_miss_group(ray_miss)
            hit_group = pipeline.create_rt_hit_group(ray_closest_hit)
            pipeline.descriptor_set(0)
            pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION, lambda: self.scene.scene_ads)
            pipeline.bind_wrap_gpu(1, ShaderStage.RT_GENERATION, lambda: self.radiances)
            pipeline.bind_wrap_gpu(2, ShaderStage.RT_GENERATION, lambda: self.radiance_gradients)
            pipeline.bind_wrap_gpu(3, ShaderStage.RT_GENERATION, lambda: self.rays)
            pipeline.bind_wrap_gpu(4, ShaderStage.RT_GENERATION, lambda: self.seeds)
            pipeline.descriptor_set(1)
            pipeline.sampler = self.device.create_sampler_linear()
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
            pipeline.bind_wrap_gpu(0, ShaderStage.RT_CLOSEST_HIT, lambda: self.texture_param)
            pipeline.bind_wrap_gpu(1, ShaderStage.RT_CLOSEST_HIT, lambda: self.texture_gradients)
            pipeline.bind_uniform(2, ShaderStage.RT_CLOSEST_HIT, lambda: self.param_dim_buffer)
            pipeline.close()
            # create program
            program = pipeline.create_rt_program(1, 1)
            program.set_generation(gen_group)
            program.set_hit_group(0, hit_group)
            program.set_miss(0, miss_group)
            self.backward_pipeline = pipeline
            self.backward_program = program

    def forward_render(self, input: List[torch.Tensor]) -> List[torch.Tensor]:
        rays, param_texture = input
        radiances_device = rays.device
        cuda_device = torch.device('cuda:0')
        rays, param_texture = rays.to(cuda_device), None if param_texture is None else param_texture.to(cuda_device)
        ray_count = torch.numel(rays) // 6
        seeds = torch.randint(0, 100000000, size=(ray_count, 4), device=cuda_device)
        param_dim = glm.ivec2(0) if param_texture is None else glm.ivec2(param_texture.shape[1], param_texture.shape[0])
        self.texture_param.wrap(param_texture)
        radiances = torch.zeros(ray_count, 3, device=cuda_device)
        self.radiances.wrap(radiances)
        self.seeds.wrap(seeds)
        self.rays.wrap(rays)
        self.param_dim_buffer.param_dim = param_dim
        self.consts.number_of_samples = self.forward_samples
        with self.device.get_raytracing() as man:
            man.set_pipeline(self.forward_pipeline)
            man.update_sets(0, 1, 2)
            man.dispatch_rays(self.forward_program, ray_count, 1)  # Linearized rays
        return [radiances.to(radiances_device)]

    def backward_render(self, input: List[torch.Tensor], output: List[torch.Tensor], output_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        rays, param_texture = input
        param_dim = glm.ivec2(0) if param_texture is None else glm.ivec2(param_texture.shape[1], param_texture.shape[0])
        grad_radiances, = output_gradients
        radiances, = output
        output_device = None if param_texture is None else param_texture.device
        cuda_device = torch.device('cuda:0')
        # Output
        grad_parameters = None if param_texture is None else torch.zeros_like(param_texture, device=cuda_device)
        # Move to gpu if necessary
        rays, params = \
            rays.to(cuda_device), \
            None if param_texture is None else param_texture.to(cuda_device)
        ray_count = torch.numel(rays) // 6
        # New radiances
        # radiances = torch.zeros(ray_count, 3, device=cuda_device)
        seeds = torch.randint(0, 100000000, size=(ray_count, 4), device=cuda_device)
        grad_radiances = grad_radiances.to(cuda_device)
        self.rays.wrap(rays)
        self.seeds.wrap(seeds)
        self.radiances.wrap(radiances)
        self.radiance_gradients.wrap(grad_radiances)
        self.texture_param.wrap(params)
        self.texture_gradients.wrap(grad_parameters)
        self.param_dim_buffer.param_dim = param_dim
        # render an extra forward pass to get new radiances
        self.consts.number_of_samples = 1
        with self.device.get_raytracing() as man:
            man.set_pipeline(self.forward_pipeline)
            man.update_sets(0, 1, 2)
            man.dispatch_rays(self.forward_program, ray_count, 1)  # Linearized rays
        # render the backward pass with radiances decopled from the one used for the gradients.
        with self.device.get_raytracing() as man:
            man.set_pipeline(self.backward_pipeline)
            man.update_sets(0, 1, 2)
            man.dispatch_rays(self.backward_program, ray_count, 1)  # Linearized rays
        return [None, None if grad_parameters is None else grad_parameters.to(output_device)]


class RBPRenderer(RendererModule):
    def __init__(self, device: DeviceManager, output_dim: (int, int), number_of_samples: int, *args, **kwargs):
        self.number_of_samples = number_of_samples
        self.output_dim = output_dim
        super().__init__(device, *args, **kwargs)
        self.rnd = random.Random()
        self.ray_generator = RayGenerator(device, output_dim, 1)

    def set_scene(self, scene: RaytracingScene):
        self.scene = scene

    def setup(self):
        self.radiances = self.device.create_gpu_wrapped_ptr()
        self.rays = self.device.create_gpu_wrapped_ptr()
        self.texture_param = self.device.create_gpu_wrapped_ptr()
        self.radiance_gradients = self.device.create_gpu_wrapped_ptr()
        self.texture_gradients = self.device.create_gpu_wrapped_ptr()
        self.random_seed_buffer = self.device.create_uniform_buffer(seed=int)
        self.param_dim_buffer = self.device.create_uniform_buffer(param_dim=glm.ivec2)
        # create forward pipeline
        if True:
            pipeline = self.device.create_raytracing_pipeline()
            ray_gen = pipeline.load_rt_generation_shader(__MODULES_SHADERS__ + '/rbp.forward.rgen.spv')
            ray_miss = pipeline.load_rt_miss_shader(__MODULES_SHADERS__ + '/rbp.forward.rmiss.spv')
            ray_closest_hit = pipeline.load_rt_closest_hit_shader(__MODULES_SHADERS__ + '/rbp.forward.rchit.spv')
            gen_group = pipeline.create_rt_gen_group(ray_gen)
            miss_group = pipeline.create_rt_miss_group(ray_miss)
            hit_group = pipeline.create_rt_hit_group(ray_closest_hit)

            pipeline.descriptor_set(0)
            pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION, lambda: self.scene.scene_ads)
            pipeline.bind_wrap_gpu(1, ShaderStage.RT_GENERATION, lambda: self.radiances)
            pipeline.bind_wrap_gpu(2, ShaderStage.RT_GENERATION, lambda: self.rays)
            pipeline.bind_uniform(3, ShaderStage.RT_GENERATION, lambda: self.random_seed_buffer)

            pipeline.descriptor_set(1)
            pipeline.sampler = self.device.create_sampler_linear()
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
            pipeline.bind_wrap_gpu(0, ShaderStage.RT_CLOSEST_HIT, lambda: self.texture_param)
            pipeline.bind_uniform(1, ShaderStage.RT_CLOSEST_HIT, lambda: self.param_dim_buffer)

            pipeline.close()
            # create program
            program = pipeline.create_rt_program(1, 1)
            program.set_generation(gen_group)
            program.set_hit_group(0, hit_group)
            program.set_miss(0, miss_group)
            self.forward_pipeline = pipeline
            self.forward_program = program
        # create backward pipeline
        if True:
            pipeline = self.device.create_raytracing_pipeline()
            ray_gen = pipeline.load_rt_generation_shader(__MODULES_SHADERS__ + '/rbp.backward.rgen.spv')
            ray_miss = pipeline.load_rt_miss_shader(__MODULES_SHADERS__ + '/rbp.backward.rmiss.spv')
            ray_closest_hit = pipeline.load_rt_closest_hit_shader(__MODULES_SHADERS__ + '/rbp.backward.rchit.spv')
            gen_group = pipeline.create_rt_gen_group(ray_gen)
            miss_group = pipeline.create_rt_miss_group(ray_miss)
            hit_group = pipeline.create_rt_hit_group(ray_closest_hit)
            pipeline.descriptor_set(0)
            pipeline.bind_scene_ads(0, ShaderStage.RT_GENERATION, lambda: self.scene.scene_ads)
            pipeline.bind_wrap_gpu(1, ShaderStage.RT_GENERATION, lambda: self.radiance_gradients)
            pipeline.bind_wrap_gpu(2, ShaderStage.RT_GENERATION, lambda: self.rays)
            pipeline.bind_uniform(3, ShaderStage.RT_GENERATION, lambda: self.random_seed_buffer)
            pipeline.descriptor_set(1)
            pipeline.sampler = self.device.create_sampler_linear()
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
            pipeline.bind_wrap_gpu(0, ShaderStage.RT_CLOSEST_HIT, lambda: self.texture_param)
            pipeline.bind_wrap_gpu(1, ShaderStage.RT_CLOSEST_HIT, lambda: self.texture_gradients)
            pipeline.bind_uniform(2, ShaderStage.RT_CLOSEST_HIT, lambda: self.param_dim_buffer)
            pipeline.close()
            # create program
            program = pipeline.create_rt_program(1, 1)
            program.set_generation(gen_group)
            program.set_hit_group(0, hit_group)
            program.set_miss(0, miss_group)
            self.backward_pipeline = pipeline
            self.backward_program = program

    def forward_render(self, input: List[torch.Tensor]) -> List[torch.Tensor]:
        origins, targets, param_texture = input
        radiances_device = origins.device
        cuda_device = torch.device('cuda:0')
        origins, targets, param_texture = origins.to(cuda_device), targets.to(cuda_device), None if param_texture is None else param_texture.to(cuda_device)
        param_dim = glm.ivec2(0) if param_texture is None else glm.ivec2(param_texture.shape[1], param_texture.shape[0])
        self.texture_param.wrap(param_texture)
        radiances = torch.zeros(self.output_dim[0], self.output_dim[1], 3, device=cuda_device)
        self.radiances.wrap(radiances)
        prepare_resources = self.device.get_raytracing()
        prepare_resources.set_pipeline(self.forward_pipeline)
        prepare_resources.update_sets(0, 1, 2)
        prepare_resources.freeze()
        man = self.device.get_raytracing()
        man.set_pipeline(self.forward_pipeline)
        man.update_sets(0, 1, 2)
        man.dispatch_rays(self.forward_program, self.output_dim[0] * self.output_dim[1], 1)  # Linearized rays
        man.freeze()
        self.param_dim_buffer.param_dim = param_dim
        self.device.submit(prepare_resources)  # Do nothing, just barrier resources to proper states
        for _ in range(self.number_of_samples):
            rays = self.ray_generator(origins, targets)
            self.rays.wrap(rays)
            self.random_seed_buffer.seed = self.rnd.randint(0, 10000000)
            self.device.submit(man)
        return [(radiances / self.number_of_samples).to(radiances_device)]

    def backward_render(self, input: List[torch.Tensor], output: List[torch.Tensor], output_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        origins, targets, param_texture = input
        param_dim = glm.ivec2(0) if param_texture is None else glm.ivec2(param_texture.shape[1], param_texture.shape[0])
        grad_radiances, = output_gradients
        output_device = None if param_texture is None else param_texture.device
        cuda_device = torch.device('cuda:0')
        # Output
        grad_parameters = None if param_texture is None else torch.zeros_like(param_texture, device=cuda_device)
        # Move to gpu if necessary
        origins, targets, params = \
            origins.to(cuda_device), \
            targets.to(cuda_device), \
            None if param_texture is None else param_texture.to(cuda_device)
        grad_radiances = grad_radiances.to(cuda_device)
        self.radiance_gradients.wrap(grad_radiances)
        self.texture_param.wrap(params)
        self.texture_gradients.wrap(grad_parameters)
        # create command buffers
        prepare_resources = self.device.get_raytracing()
        prepare_resources.set_pipeline(self.backward_pipeline)
        prepare_resources.update_sets(0, 1, 2)
        prepare_resources.freeze()
        man = self.device.get_raytracing()
        man.set_pipeline(self.backward_pipeline)
        man.update_sets(0, 1, 2)
        man.dispatch_rays(self.backward_program, self.output_dim[0] * self.output_dim[1], 1)  # Linearized rays
        man.freeze()
        self.param_dim_buffer.param_dim = param_dim
        self.device.submit(prepare_resources)
        for _ in range(1):
            rays = self.ray_generator(origins, targets)
            self.rays.wrap(rays)
            self.random_seed_buffer.seed = self.rnd.randint(0, 10000000)
            self.device.submit(man)
        return [None, None, None if grad_parameters is None else (grad_parameters / self.number_of_samples).to(params.device)]


class VolumeRenderer(RendererModule):

    def __init__(self, device: DeviceManager, forward_samples: int = 20, backward_samples: int = 20, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.forward_samples = forward_samples
        self.backward_samples = backward_samples

    def setup(self):
        self.medium_buffer = self.device.create_uniform_buffer(
            scatteringAlbedo=glm.vec3,
            density=float,
            phase_g=float
        )
        self.grid = self.device.create_gpu_wrapped_ptr()
        self.rays = self.device.create_gpu_wrapped_ptr()
        self.seeds = self.device.create_gpu_wrapped_ptr()
        self.radiances = self.device.create_gpu_wrapped_ptr()
        self.grad_radiances = self.device.create_gpu_wrapped_ptr()
        self.grad_grid = self.device.create_gpu_wrapped_ptr()

        pipeline = self.device.create_compute_pipeline()
        pipeline.load_compute_shader(__MODULES_SHADERS__ + '/vpt.forward.comp.spv')
        pipeline.bind_wrap_gpu(0, ShaderStage.COMPUTE, lambda: self.grid)
        pipeline.bind_wrap_gpu(1, ShaderStage.COMPUTE, lambda: self.rays)
        pipeline.bind_wrap_gpu(2, ShaderStage.COMPUTE, lambda: self.seeds)
        pipeline.bind_wrap_gpu(3, ShaderStage.COMPUTE, lambda: self.radiances)
        pipeline.bind_uniform(4, ShaderStage.COMPUTE, lambda: self.medium_buffer)
        pipeline.bind_constants(0, ShaderStage.COMPUTE,
                                grid_dim=glm.ivec3, number_of_rays=int,
                                box_minim=glm.vec3, pad0=float,
                                box_size=glm.vec3, pad1=float,
                                )
        pipeline.close()
        self.forward_pipeline = pipeline

        pipeline = self.device.create_compute_pipeline()
        pipeline.load_compute_shader(__MODULES_SHADERS__ + '/vpt.backward.comp.spv')
        pipeline.bind_wrap_gpu(0, ShaderStage.COMPUTE, lambda: self.grid)
        pipeline.bind_wrap_gpu(1, ShaderStage.COMPUTE, lambda: self.rays)
        pipeline.bind_wrap_gpu(2, ShaderStage.COMPUTE, lambda: self.seeds)
        pipeline.bind_wrap_gpu(3, ShaderStage.COMPUTE, lambda: self.radiances)
        pipeline.bind_wrap_gpu(4, ShaderStage.COMPUTE, lambda: self.grad_radiances)
        pipeline.bind_wrap_gpu(5, ShaderStage.COMPUTE, lambda: self.grad_grid)
        pipeline.bind_uniform(6, ShaderStage.COMPUTE, lambda: self.medium_buffer)
        pipeline.bind_constants(0, ShaderStage.COMPUTE,
                                grid_dim=glm.ivec3, number_of_rays=int,
                                box_minim=glm.vec3, pad0=float,
                                box_size=glm.vec3, pad1=float,
                                )
        pipeline.close()
        self.backward_pipeline = pipeline


    def set_medium(self, scattering_albedo: glm.vec3, density: float, phase_g: float):
        self.medium_buffer.scatteringAlbedo = scattering_albedo
        self.medium_buffer.density = density
        self.medium_buffer.phase_g = phase_g

    def set_box(self, box_minim: glm.vec3, box_size: glm.vec3):
        self.box_minim = box_minim
        self.box_size = box_size

    def forward_render(self, inputs):
        rays, grid = inputs
        output_device = rays.device
        cuda_device = torch.device('cuda:0')
        rays, grid = rays.to(cuda_device), grid.to(cuda_device)
        grid_dim = grid.shape
        ray_count = torch.numel(rays) // 6
        radiances = torch.zeros(ray_count * 3, device=cuda_device)
        self.rays.wrap_input(rays)
        self.grid.wrap_input(grid)
        self.radiances.wrap(radiances)

        for _ in range(self.forward_samples):
            seeds = torch.randint(0, 1000000000, size=(ray_count, 4), device=cuda_device)
            self.seeds.wrap(seeds)
            with self.device.get_compute() as man:
                man.set_pipeline(self.forward_pipeline)
                man.update_sets(0)
                man.update_constants(ShaderStage.COMPUTE,
                                     grid_dim=glm.ivec3(grid_dim[2], grid_dim[1], grid_dim[0]),
                                     number_of_rays=ray_count,
                                     box_minim=self.box_minim,
                                     box_size=self.box_size
                                     )
                man.dispatch_threads_1D(ray_count)
        shape = list(rays.shape)
        shape[-1] //= 2  # for each 6 values in the last dimension there is 3 values in transmittances
        return [(radiances / self.forward_samples).reshape(shape).to(output_device)]

    def backward_render(self, input: List[torch.Tensor], output: List[torch.Tensor], output_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        rays, grid = input
        cuda_device = torch.device('cuda:0')
        rays, grid = rays.to(cuda_device), grid.to(cuda_device)
        grid_dim = grid.shape
        ray_count = torch.numel(rays) // 6
        grad_radiances, = output_gradients
        output_device = grid.device
        grad_grid = torch.zeros_like(grid, device=cuda_device)
        self.rays.wrap_input(rays)
        self.grid.wrap_input(grid)
        self.grad_radiances.wrap_input(grad_radiances)
        self.grad_grid.wrap(grad_grid)
        for _ in range(self.backward_samples):
            radiances = torch.zeros(ray_count * 3, device=cuda_device)
            self.radiances.wrap(radiances)
            seeds = torch.randint(0, 1000000000, size=(ray_count, 4), device=cuda_device)
            self.seeds.wrap(seeds)
            with self.device.get_compute() as man:
                man.set_pipeline(self.forward_pipeline)
                man.update_sets(0)
                man.update_constants(ShaderStage.COMPUTE,
                                     grid_dim=glm.ivec3(grid_dim[2], grid_dim[1], grid_dim[0]),
                                     number_of_rays=ray_count,
                                     box_minim=self.box_minim,
                                     box_size=self.box_size
                                     )
                man.dispatch_threads_1D(ray_count)
            with self.device.get_compute() as man:
                man.set_pipeline(self.backward_pipeline)
                man.update_sets(0)
                man.update_constants(ShaderStage.COMPUTE,
                                     grid_dim=glm.ivec3(grid_dim[2], grid_dim[1], grid_dim[0]),
                                     number_of_rays=ray_count,
                                     box_minim=self.box_minim,
                                     box_size=self.box_size
                                     )
                man.dispatch_threads_1D(ray_count)
        return [None, (grad_grid).to(output_device)]