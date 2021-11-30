from vulkan import *
from enum import IntFlag
from enum import IntEnum
from rendering import vkw
import math
import numpy as np
from typing import List, Dict, Tuple
import glm
import struct

def compile_shader_sources(directory='.'):
    import os
    import subprocess
    def needs_to_update(source, binary):
        return not os.path.exists(binary) or os.path.getmtime(source) > os.path.getmtime(binary)
    for filename in os.listdir(directory):
        filename = directory+"/"+filename
        if filename.endswith(".vert") or \
                filename.endswith(".frag") or \
                filename.endswith(".comp"):
            binary_file = filename + ".spv"
            if needs_to_update(filename, binary_file):
                p = subprocess.Popen(f'Compile.bat {filename} {binary_file}')
                p.wait()
                print(f'[INFO] Compiled... {filename}')


# ------------ ALIASES ---------------


Event = vkw.Event
Window = vkw.WindowWrapper
GPUTask = vkw.GPUTaskWrapper
Footprint = vkw.SubresourceFootprint


# ------------- ENUMS ---------------


class PresenterMode(IntFlag):
    __no_flags_name__ = 'NONE'
    __all_flags_name__ = 'ALL'
    OFFLINE = 1
    SDL = 2


class QueueType(IntFlag):
    __no_flags_name__ = 'NONE'
    __all_flags_name__ = 'ALL'
    COPY = VK_QUEUE_TRANSFER_BIT
    COMPUTE = VK_QUEUE_COMPUTE_BIT
    GRAPHICS = VK_QUEUE_GRAPHICS_BIT
    RAYTRACING = VK_QUEUE_GRAPHICS_BIT


class BufferUsage(IntFlag):
    VERTEX = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    INDEX = VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    UNIFORM = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    STORAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    TRANSFER_SRC = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    TRANSFER_DST = VK_BUFFER_USAGE_TRANSFER_DST_BIT


class ImageUsage(IntFlag):
    TRANSFER_SRC = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    TRANSFER_DST = VK_IMAGE_USAGE_TRANSFER_DST_BIT
    RENDER_TARGET = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    SAMPLED = VK_IMAGE_USAGE_SAMPLED_BIT
    DEPTH_STENCIL = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    STORAGE = VK_IMAGE_USAGE_STORAGE_BIT


class Format(IntEnum):
    UINT_RGBA = VK_FORMAT_R8G8B8A8_UINT
    UINT_BGRA_STD = VK_FORMAT_B8G8R8A8_SRGB
    UINT_RGBA_STD = VK_FORMAT_R8G8B8A8_SRGB
    FLOAT = VK_FORMAT_R32_SFLOAT
    INT = VK_FORMAT_R32_SINT
    UINT = VK_FORMAT_R32_UINT
    VEC2 = VK_FORMAT_R32G32_SFLOAT
    VEC3 = VK_FORMAT_R32G32B32_SFLOAT
    VEC4 = VK_FORMAT_R32G32B32A32_SFLOAT
    IVEC2 = VK_FORMAT_R32G32_SINT
    IVEC3 = VK_FORMAT_R32G32B32_SINT
    IVEC4 = VK_FORMAT_R32G32B32A32_SINT
    UVEC2 = VK_FORMAT_R32G32_UINT
    UVEC3 = VK_FORMAT_R32G32B32_UINT
    UVEC4 = VK_FORMAT_R32G32B32A32_UINT


class ImageType(IntEnum):
    TEXTURE_1D = VK_IMAGE_TYPE_1D
    TEXTURE_2D = VK_IMAGE_TYPE_2D
    TEXTURE_3D = VK_IMAGE_TYPE_3D


class MemoryProperty(IntFlag):
    """
    Memory configurations
    """
    """
    Efficient memory for reading and writing on the GPU.
    """
    GPU = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    """
    Memory can be mapped for reading and writing from the CPU.
    """
    CPU_ACCESSIBLE = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    """
    Memory can be read and write directly from the CPU
    """
    CPU_DIRECT = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    """ 
    Memory efficient for reading on the CPU a cached device memory.
    """
    GPU_WRITE_CPU_READ = VK_MEMORY_PROPERTY_HOST_CACHED_BIT


class ShaderStage(IntEnum):
    VERTEX = VK_SHADER_STAGE_VERTEX_BIT
    FRAGMENT = VK_SHADER_STAGE_FRAGMENT_BIT
    COMPUTE = VK_SHADER_STAGE_COMPUTE_BIT


class UpdateLevel(IntEnum):
    PIPELINE = 0
    RENDER_PASS = 1
    PER_MATERIAL = 2
    PER_OBJECT = 3


class Filter(IntEnum):
    POINT = VK_FILTER_NEAREST
    LINEAR = VK_FILTER_LINEAR


class MipMapMode(IntEnum):
    POINT = VK_SAMPLER_MIPMAP_MODE_NEAREST
    LINEAR = VK_SAMPLER_MIPMAP_MODE_LINEAR


class AddressMode(IntEnum):
    REPEAT = VK_SAMPLER_ADDRESS_MODE_REPEAT
    CLAMP_EDGE = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
    BORDER_COLOR = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER


class CompareOp(IntEnum):
    NEVER = VK_COMPARE_OP_NEVER
    LESS = VK_COMPARE_OP_LESS,
    EQUAL = VK_COMPARE_OP_EQUAL,
    LESS_OR_EQUAL = VK_COMPARE_OP_LESS_OR_EQUAL,
    GREATER = VK_COMPARE_OP_GREATER,
    NOT_EQUAL = VK_COMPARE_OP_NOT_EQUAL,
    GREATER_OR_EQUAL = VK_COMPARE_OP_GREATER_OR_EQUAL,
    ALWAYS = VK_COMPARE_OP_ALWAYS,


class BorderColor(IntEnum):
    TRANSPARENT_BLACK_FLOAT = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    TRANSPARENT_BLACK_INT = VK_BORDER_COLOR_INT_TRANSPARENT_BLACK,
    OPAQUE_BLACK_FLOAT = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
    OPAQUE_BLACK_INT = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
    OPAQUE_WHITE_FLOAT = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    OPAQUE_WHITE_INT = VK_BORDER_COLOR_INT_OPAQUE_WHITE


# ---- HIGH LEVEL DEFINITIONS ----------


class BinaryFormatter:

    __TYPE_SIZES = {
        int: 4,
        float: 4,  # not really but more legible in graphics context

        glm.vec1: 4,
        glm.float32: 4,
        glm.vec2: 4 * 2,
        glm.vec3: 4 * 3,
        glm.vec4: 4 * 4,

        glm.ivec1: 4 * 1,
        glm.int32: 4 * 1,
        glm.ivec2: 4 * 2,
        glm.ivec3: 4 * 3,
        glm.ivec4: 4 * 4,

        glm.uint32: 4 * 1,
        glm.uvec1: 4 * 1,
        glm.uvec2: 4 * 2,
        glm.uvec3: 4 * 3,
        glm.uvec4: 4 * 4,

        glm.mat2x2: 4 * 2 * 2,
        glm.mat2x3: 4 * 2 * 3,
        glm.mat2x4: 4 * 2 * 4,

        glm.mat3x2: 4 * 3 * 2,
        glm.mat3x3: 4 * 3 * 3,
        glm.mat3x4: 4 * 3 * 4,

        glm.mat4x2: 4 * 4 * 2,
        glm.mat4x3: 4 * 4 * 3,
        glm.mat4x4: 4 * 4 * 4,
    }

    @staticmethod
    def size_of(type):
        assert type in BinaryFormatter.__TYPE_SIZES, f"Not supported type {type}, use int, float or glm types"
        return BinaryFormatter.__TYPE_SIZES[type]

    @staticmethod
    def to_bytes(type, value):
        assert type in BinaryFormatter.__TYPE_SIZES, f"Not supported type {type}, use int, float or glm types"
        if type == int:
            return struct.pack('i', value)
        if type == float:
            return struct.pack('f', value)
        return type.to_bytes(value)

    @staticmethod
    def from_bytes(type, buffer):
        assert type in BinaryFormatter.__TYPE_SIZES, f"Not supported type {type}, use int, float or glm types"
        if type == int:
            return struct.unpack('i', buffer)[0]
        return type.from_bytes(bytes(buffer))


class Resource(object):
    def __init__(self, w_resource: vkw.ResourceWrapper):
        self.w_resource = w_resource

    def get_footprints(self) -> List[Footprint]:
        if self.w_resource.resource_data.is_buffer:
            subresources = 0
        else:
            subresources = self.w_resource.current_slice["array_count"] * self.w_resource.current_slice["mips_count"]
        return [self.w_resource.resource_data.get_staging_footprint_and_offset(i)[0] for i in range(subresources)]


class Buffer(Resource):
    def __init__(self, w_buffer: vkw.ResourceWrapper):
        super().__init__(w_buffer)

    def slice(self, offset, size):
        return Buffer(self.w_resource.slice_buffer(offset, size))

    def write(self, data):
        self.w_resource.write(data)

    def read(self, data):
        self.w_resource.read(data)


class Uniform(Buffer):

    @staticmethod
    def process_layout(fields: Dict[str, type]):
        offset=0
        layout = {}
        for field, type in fields.items():
            field_size = BinaryFormatter.size_of(type)
            layout[field] = (offset, field_size, type)
            offset += field_size
        return layout, offset

    def __init__(self, w_buffer: vkw.ResourceWrapper, layout: Dict[str, Tuple[int, int, type]]):
        self.layout = layout
        super().__init__(w_buffer)
        w_buffer.get_permanent_map()

    def __getattr__(self, item):
        if item == "layout" or item == "w_resource":
            return super(Uniform, self).__getattribute__(item)
        if item not in self.layout:
            return super(Uniform, self).__getattribute__(item)
        offset, size, type = self.layout[item]
        buffer = bytearray(size)
        self.slice(offset, size).read(buffer)
        return BinaryFormatter.from_bytes(type, buffer)

    def __setattr__(self, item, value):
        if item == "layout" or item == "w_resource":
            super(Uniform, self).__setattr__(item, value)
            return
        if item in self.layout:
            offset, size, type = self.layout[item]
            buffer = BinaryFormatter.to_bytes(type, value)
            self.slice(offset, size).write(buffer)
            return
        raise AttributeError(f"Can not set attribute {item}")


class Image(Resource):
    @staticmethod
    def compute_dimension(width: int, height: int, depth: int, mip_level: int):
        return max(1, width // (1 << mip_level)), max(1, height // (1 << mip_level)), max(1, depth // (1 << mip_level))

    def __init__(self, w_image: vkw.ResourceWrapper):
        super().__init__(w_image)
        self.width, self.height, self.depth = Image.compute_dimension(
            w_image.resource_data.vk_description.extent.width,
            w_image.resource_data.vk_description.extent.height,
            w_image.resource_data.vk_description.extent.depth,
            w_image.current_slice["mip_start"]
        )
        self.format = w_image.resource_data.vk_description.format
        self.dimension = w_image.resource_data.vk_description.imageType

    def get_mip_count(self) -> int:
        return self.w_resource.current_slice["mip_count"]

    def get_layer_count(self) -> int:
        return self.w_resource.current_slice["array_count"]

    def slice_mips(self, mip_start, mip_count):
        return Image(self.w_resource.slice_mips(mip_start, mip_count))

    def slice_array(self, array_start, array_count):
        return Image(self.w_resource.slice_array(array_start, array_count))

    def subresource(self, mip=0, layer=0):
        return Image(self.w_resource.subresource(mip, layer))

    def write(self, data):
        self.w_resource.write(data)

    def read(self, data):
        self.w_resource.read(data)

    def as_readonly(self):
        return Image(self.w_resource.as_readonly())

    def as_numpy(self):
        return self.w_resource.as_numpy()


class Pipeline:
    def __init__(self, w_pipeline: vkw.PipelineBindingWrapper):
        self.w_pipeline = w_pipeline

    def __setup__(self):
        pass

    def is_closed(self):
        return self.w_pipeline.initialized

    def close(self):
        self.w_pipeline._build_objects()

    def set_update_level(self, level: UpdateLevel):
        self.w_pipeline.descriptor_set(level)

    def _bind_resource(self, slot: int, stage: ShaderStage, count: int, resolver, vk_descriptor_type):
        self.w_pipeline.binding(
            slot=slot,
            vk_stage=stage,
            vk_descriptor_type=vk_descriptor_type,
            count=count,
            resolver=resolver
        )

    def bind_uniform(self, slot: int, stage: ShaderStage, resolver):
        self._bind_resource(slot, stage, 1, lambda: [resolver()], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)

    def bind_uniform_array(self, slot: int, stage: ShaderStage, count: int, resolver):
        self._bind_resource(slot, stage, count, resolver, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)

    def bind_storage_buffer(self, slot: int, stage: ShaderStage, resolver):
        self._bind_resource(slot, stage, 1, lambda: [resolver()], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)

    def bind_storage_buffer_array(self, slot: int, stage: ShaderStage, count: int, resolver):
        self._bind_resource(slot, stage, count, resolver, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)

    def bind_texture_combined(self, slot: int, stage: ShaderStage, resolver):
        self._bind_resource(slot, stage, 1, lambda: [resolver()], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)

    def bind_texture_combined_array(self, slot: int, stage: ShaderStage, count: int, resolver):
        self._bind_resource(slot, stage, count, resolver, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)

    def bind_texture(self, slot: int, stage: ShaderStage, resolver):
        self._bind_resource(slot, stage, 1, lambda: [resolver()], VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)

    def bind_texture_array(self, slot: int, stage: ShaderStage, count: int, resolver):
        self._bind_resource(slot, stage, count, resolver, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)

    def bind_storage_image(self, slot: int, stage: ShaderStage, resolver):
        self._bind_resource(slot, stage, 1, lambda: [resolver()], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)

    def bind_storage_image_array(self, slot: int, stage: ShaderStage, count: int, resolver):
        self._bind_resource(slot, stage, count, resolver, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)

    def load_shader(self, stage: ShaderStage, path, main_function = 'main'):
        self.w_pipeline.load_shader(vkw.ShaderStageWrapper.from_file(
            device=self.w_pipeline.device,
            vk_stage=stage,
            main_function=main_function,
            path=path))

    def load_fragment_shader(self, path: str, main_function='main'):
        self.load_shader(VK_SHADER_STAGE_FRAGMENT_BIT, path, main_function)

    def load_vertex_shader(self, path: str, main_function='main'):
        self.load_shader(VK_SHADER_STAGE_VERTEX_BIT, path, main_function)

    def load_compute_shader(self, path: str, main_function='main'):
        self.load_shader(VK_SHADER_STAGE_COMPUTE_BIT, path, main_function)


class CommandManager:

    def __init__(self, w_cmdList: vkw.CommandBufferWrapper):
        self.w_cmdList = w_cmdList

    @classmethod
    def get_queue_required(cls) -> int:
        pass

    def is_frozen(self):
        return self.w_cmdList.is_frozen()

    def is_closed(self):
        return self.w_cmdList.is_closed()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.w_cmdList.flush_and_wait()


class CopyManager(CommandManager):
    def __init__(self, w_cmdList: vkw.CommandBufferWrapper):
        super().__init__(w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.COPY

    def gpu_to_cpu(self, resource):
        self.w_cmdList.from_gpu(resource.w_resource)

    def cpu_to_gpu(self, resource):
        self.w_cmdList.to_gpu(resource.w_resource)


class ComputeManager(CopyManager):
    def __init__(self, w_cmdList: vkw.CommandBufferWrapper):
        super().__init__(w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.COMPUTE

    def clear_color(self, image: Image, color):
        self.w_cmdList.clear_color(image.w_resource, color)

    def set_pipeline(self, pipeline: Pipeline):
        if not pipeline.is_closed():
            raise Exception("Error, can not set a pipeline has not been closed.")
        self.w_cmdList.set_pipeline(pipeline=pipeline.w_pipeline)
        self.w_cmdList.update_bindings_level(0)  # after set the pipeline

    def set_render_pass(self):
        self.w_cmdList.update_bindings_level(1)

    def set_material(self):
        self.w_cmdList.update_bindings_level(2)

    def set_object(self):
        self.w_cmdList.update_bindings_level(3)

    def dispatch_groups(self, groups_x: int, groups_y: int = 1, groups_z:int = 1):
        self.w_cmdList.dispatch_groups(groups_x, groups_y, groups_z)

    def dispatch_threads_1D(self, dim_x: int, group_size_x: int = 1024):
        self.dispatch_groups(math.ceil(dim_x/group_size_x))

    def dispatch_threads_2D(self, dim_x: int, dim_y: int, group_size_x: int = 32, group_size_y: int = 32):
        self.dispatch_groups(math.ceil(dim_x/group_size_x), math.ceil(dim_y/group_size_y))


class GraphicsManager(ComputeManager):
    def __init__(self, w_cmdList: vkw.CommandBufferWrapper):
        super().__init__(w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.GRAPHICS


class RaytracingManager(GraphicsManager):
    def __init__(self, w_cmdList: vkw.CommandBufferWrapper):
        super().__init__(w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.RAYTRACING


class DeviceManager:

    def __init__(self):
        self.w_state = None
        self.width = 0
        self.height = 0

    def __bind__(self, w_device: vkw.DeviceWrapper):
        self.w_device = w_device
        self.width = w_device.get_render_target(0).resource_data.vk_description.extent.width
        self.height = w_device.get_render_target(0).resource_data.vk_description.extent.height

    def render_target(self):
        return Image(self.w_device.get_render_target(self.w_device.get_render_target_index()))

    def __setup__(self):
        pass

    def load_technique(self, technique):
        technique.__bind__(self.w_device)
        technique.__setup__()
        return technique

    def dispatch_technique(self, technique):
        technique.__dispatch__()

    def create_buffer(self, size: int, usage: int, memory: MemoryProperty):
        return Buffer(self.w_device.create_buffer(size, usage, memory))

    def create_uniform(self, usage: int = BufferUsage.UNIFORM,
                       memory: MemoryProperty = MemoryProperty.CPU_ACCESSIBLE,
                       **fields):
        layout, size = Uniform.process_layout(fields)
        resource = self.w_device.create_buffer(size, usage, memory)
        # mapped_buffer = resource.resource_data.map_buffer_slice(0, size)
        return Uniform(resource, layout)

    def create_image(self, image_type: ImageType, is_cube: bool, image_format: Format,
                     width: int, height: int, depth: int,
                     mips: int, layers: int,
                     usage: int, memory: MemoryProperty):
        linear = False  # bool(usage & (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        layout = VK_IMAGE_LAYOUT_UNDEFINED
        return Image(self.w_device.create_image(
            image_type, image_format, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT if is_cube else 0,
            VkExtent3D(width, height, depth), mips, layers, linear, layout, usage, memory
        ))

    def create_buffer_vertex(self, vertex_size, vertex_count):
        return self.create_buffer(
            size=vertex_size * vertex_count,
            usage=BufferUsage.VERTEX | BufferUsage.TRANSFER_DST,
            memory=MemoryProperty.GPU
        )

    def create_buffer_index(self, index_count):
        return self.create_buffer(
            size=index_count * 4,
            usage=BufferUsage.INDEX | BufferUsage.TRANSFER_DST,
            memory=MemoryProperty.GPU
        )

    def create_buffer_uniform(self, size):
        return self.create_buffer(
            size=size,
            usage=BufferUsage.UNIFORM | BufferUsage.TRANSFER_DST,
            memory=MemoryProperty.DYNAMIC
        )

    def create_buffer_storage(self, size):
        return self.create_buffer(
            size=size,
            usage=BufferUsage.STORAGE,
            memory=MemoryProperty.GPU
        )

    def create_buffer_staging(self, size):
        return self.create_buffer(
            size=size,
            usage=BufferUsage.TRANSFER_SRC | BufferUsage.TRANSFER_DST,
            memory=MemoryProperty.CPU
        )

    def create_render_target(self, image_format: Format, width: int, height: int):
        return self.create_image(ImageType.TEXTURE_2D, False, image_format,
                                 width, height, 1, 1, 1, ImageUsage.RENDER_TARGET, MemoryProperty.GPU)

    def create_depth_stencil(self, image_format: Format, width: int, height: int):
        return self.create_image(ImageType.TEXTURE_2D, False, image_format,
                                 width, height, 1, 1, 1, ImageUsage.DEPTH_STENCIL, MemoryProperty.GPU)

    def create_texure_1D(self, image_format: Format, width: int, mips=None, layers=1,
                         usage: int = ImageUsage.TRANSFER_DST | ImageUsage.SAMPLED):
        if mips is None:
            mips = int(math.log(width, 2)) + 1
        return self.create_image(ImageType.TEXTURE_1D, False, image_format,
                                 width, 1, 1, mips, layers, usage, MemoryProperty.GPU)

    def create_texure_2D(self, image_format: Format, width: int, height: int, mips=None, layers=1,
                         usage: int = ImageUsage.TRANSFER_DST | ImageUsage.SAMPLED):
        if mips is None:
            mips = int(math.log(max(width, height), 2)) + 1
        return self.create_image(ImageType.TEXTURE_2D, False, image_format,
                                 width, height, 1, mips, layers, usage, MemoryProperty.GPU)

    def create_texure_3D(self, image_format: Format, width: int, height: int, depth: int, mips=None, layers=1,
                         usage: int = ImageUsage.TRANSFER_DST | ImageUsage.SAMPLED):
        if mips is None:
            mips = int(math.log(max(width, height, depth), 2)) + 1
        return self.create_image(ImageType.TEXTURE_3D, False, image_format,
                                 width, height, depth, mips, layers, usage, MemoryProperty.GPU)

    def create_sampler(self,
                       mag_filter: Filter = Filter.POINT,
                       min_filter: Filter = Filter.POINT,
                       mipmap_mode: MipMapMode = MipMapMode.POINT,
                       address_U: AddressMode = AddressMode.REPEAT,
                       address_V: AddressMode = AddressMode.REPEAT,
                       address_W: AddressMode = AddressMode.REPEAT,
                       mip_LOD_bias: float = 0.0,
                       enable_anisotropy: bool = False,
                       max_anisotropy: float = 0.0,
                       enable_compare: bool = False,
                       compare_op: CompareOp = CompareOp.NEVER,
                       min_LOD: float = 0.0,
                       max_LOD: float = 0.0,
                       border_color: BorderColor = BorderColor.TRANSPARENT_BLACK_FLOAT,
                       use_unnormalized_coordinates: bool = False
                       ):
        return self.w_device.create_sampler(mag_filter, min_filter, mipmap_mode,
                                            address_U, address_V, address_W,
                                            mip_LOD_bias, 1 if enable_anisotropy else 0,
                                            max_anisotropy, 1 if enable_compare else 0,
                                            compare_op, min_LOD, max_LOD,
                                            border_color,
                                            1 if use_unnormalized_coordinates else 0)

    def create_sampler_linear(self,
                       address_U: AddressMode = AddressMode.REPEAT,
                       address_V: AddressMode = AddressMode.REPEAT,
                       address_W: AddressMode = AddressMode.REPEAT,
                       mip_LOD_bias: float = 0.0,
                       enable_anisotropy: bool = False,
                       max_anisotropy: float = 0.0,
                       enable_compare: bool = False,
                       compare_op: CompareOp = CompareOp.NEVER,
                       min_LOD: float = 0.0,
                       max_LOD: float = 1000.0,
                       border_color: BorderColor = BorderColor.TRANSPARENT_BLACK_FLOAT,
                       use_unnormalized_coordinates: bool = False
                       ):
        return self.w_device.create_sampler(Filter.LINEAR, Filter.LINEAR, MipMapMode.LINEAR,
                                            address_U, address_V, address_W,
                                            mip_LOD_bias, 1 if enable_anisotropy else 0,
                                            max_anisotropy, 1 if enable_compare else 0,
                                            compare_op, min_LOD, max_LOD,
                                            border_color,
                                            1 if use_unnormalized_coordinates else 0)

    def create_compute_pipeline(self):
        return Pipeline(self.w_device.create_pipeline(
            vk_pipeline_type=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO))

    def create_graphics_pipeline(self):
        return Pipeline(self.w_device.create_pipeline(
            vk_pipeline_type=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO))

    def create_raytracing_pipeline(self):
        return Pipeline(self.w_device.create_pipeline(
            vk_pipeline_type=VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV))

    def _get_queue_manager(self, queue_bits: int):
        return self.w_device.create_cmdList(queue_bits)

    def get_graphics(self) -> GraphicsManager:
        return GraphicsManager(self._get_queue_manager(QueueType.GRAPHICS))

    def get_compute(self) -> ComputeManager:
        return ComputeManager(self._get_queue_manager(QueueType.COMPUTE))

    def get_raytracing(self) -> RaytracingManager:
        return RaytracingManager(self._get_queue_manager(QueueType.RAYTRACING))

    def get_copy(self) -> CopyManager:
        return CopyManager(self._get_queue_manager(QueueType.COPY))

    def flush(self):
        return self.w_device.flush_pending_and_wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


class Presenter(DeviceManager):

    def __init__(self):
        super().__init__()

    def begin_frame(self):
        self.w_device.begin_frame()

    def end_frame(self):
        self.w_device.end_frame()

    def screenshot(self):
        pass

    def get_window(self):
        return self.w_device.get_window()

    def __enter__(self):
        self.begin_frame()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_frame()


def create_presenter(width: int, height: int,
                     format: Format,
                     mode: PresenterMode,
                     usage: int = ImageUsage.RENDER_TARGET,
                     debug: bool = False) -> Presenter:
    state = vkw.DeviceWrapper(
        width=width,
        height=height,
        format=format,
        mode=mode,
        render_usage=usage,
        enable_validation_layers=debug
    )
    presenter = Presenter()
    presenter.__bind__(state)
    return presenter


class Technique(DeviceManager):
    def __dispatch__(self):
        pass


def Extends(class_):
    def wrapper(function):
        setattr(class_, function.__name__, function)
    return wrapper
