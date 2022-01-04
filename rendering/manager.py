from vulkan import *
from enum import IntFlag
from enum import IntEnum

from rendering import vkw
import math
import numpy as np
from typing import List, Dict, Tuple
import glm
import struct
import torch
import os
import subprocess


def compile_shader_sources(directory='.', force_all: bool = False):
    def needs_to_update(source, binary):
        return not os.path.exists(binary) or os.path.getmtime(source) > os.path.getmtime(binary)
    for filename in os.listdir(directory):
        filename = directory+"/"+filename
        filename_without_extension, extension = os.path.splitext(filename)
        if extension == '.glsl':
            stage = os.path.splitext(filename_without_extension)[1][1:]  # [1:] for removing the dot .
            binary_file = filename_without_extension + ".spv"
            if needs_to_update(filename, binary_file) or force_all:
                p = subprocess.Popen(
                    os.path.expandvars('%VULKAN_SDK%/Bin/glslangValidator.exe -r -V --target-env vulkan1.2 ').replace("\\","/")
                    + f'-S {stage} {filename} -o {binary_file}'
                )
                p.wait()
                print(f'[INFO] Compiled... {filename}')


__SHADERS_TOOLS__ = os.path.dirname(__file__) + "/shaders/Tools"

compile_shader_sources(__SHADERS_TOOLS__)

# ------------ ALIASES ---------------


Event = vkw.Event
Window = vkw.WindowWrapper
Footprint = vkw.SubresourceFootprint
ShaderHandler = vkw.ShaderHandlerWrapper


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
    RAYTRACING_ADS = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
    RAYTRACING_ADS_READ = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT


class ImageUsage(IntFlag):
    TRANSFER_SRC = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    TRANSFER_DST = VK_IMAGE_USAGE_TRANSFER_DST_BIT
    RENDER_TARGET = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    SAMPLED = VK_IMAGE_USAGE_SAMPLED_BIT
    DEPTH_STENCIL = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    STORAGE = VK_IMAGE_USAGE_STORAGE_BIT


class Format(IntEnum):
    UINT_RGBA = VK_FORMAT_R8G8B8A8_UINT
    UINT_RGB = VK_FORMAT_R8G8B8_UINT
    UINT_BGRA_STD = VK_FORMAT_B8G8R8A8_SRGB
    UINT_RGBA_STD = VK_FORMAT_R8G8B8A8_SRGB
    UINT_RGBA_UNORM = VK_FORMAT_R8G8B8A8_UNORM
    UINT_BGRA_UNORM = VK_FORMAT_B8G8R8A8_UNORM
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
    CPU_DIRECT = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    """ 
    Memory efficient for reading on the CPU a cached device memory.
    """
    GPU_WRITE_CPU_READ = VK_MEMORY_PROPERTY_HOST_CACHED_BIT


class ShaderStage(IntEnum):
    VERTEX = VK_SHADER_STAGE_VERTEX_BIT
    FRAGMENT = VK_SHADER_STAGE_FRAGMENT_BIT
    COMPUTE = VK_SHADER_STAGE_COMPUTE_BIT
    RT_GENERATION = VK_SHADER_STAGE_RAYGEN_BIT_KHR
    RT_CLOSEST_HIT = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
    RT_MISS = VK_SHADER_STAGE_MISS_BIT_KHR
    RT_ANY_HIT = VK_SHADER_STAGE_ANY_HIT_BIT_KHR
    RT_INTERSECTION_HIT = VK_SHADER_STAGE_INTERSECTION_BIT_KHR


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

        glm.uint64: 8,

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
        if type == glm.uint64:
            return struct.pack('<Q', value.value)
        return type.to_bytes(value)

    @staticmethod
    def from_bytes(type, buffer):
        assert type in BinaryFormatter.__TYPE_SIZES, f"Not supported type {type}, use int, float or glm types"
        if type == int:
            return struct.unpack('i', buffer)[0]
        if type == float:
            return struct.unpack('f', buffer)[0]
        if type == glm.uint64:
            return struct.unpack('Q', buffer)[0]
        if isinstance(buffer, bytearray):
            return type.from_bytes(bytes(buffer))
        return type.from_bytes(ffi.buffer(buffer)[0:BinaryFormatter.size_of(type)])


class Resource(object):
    def __init__(self, device, w_resource: vkw.ResourceWrapper):
        self.w_resource = w_resource
        self.device = device

    def get_footprints(self) -> List[Footprint]:
        if self.w_resource.resource_data.is_buffer:
            subresources = 0
        else:
            subresources = self.w_resource.current_slice["array_count"] * self.w_resource.current_slice["mips_count"]
        return [self.w_resource.resource_data.get_staging_footprint_and_offset(i)[0] for i in range(subresources)]

    def __del__(self):
        self.device = None
        self.w_resource = None


class Buffer(Resource):
    def __init__(self, device, w_buffer: vkw.ResourceWrapper):
        super().__init__(device, w_buffer)
        self.size = w_buffer.current_slice["size"]

    def slice(self, offset, size):
        return Buffer(self.device, self.w_resource.slice_buffer(offset, size))

    def write(self, data):
        self.w_resource.write(data)

    def write_direct(self, data):
        self.write(data)
        with self.device.get_compute() as man:
            man.cpu_to_gpu(self)

    def read(self, data):
        self.w_resource.read(data)

    def read_direct(self, data):
        with self.device.get_compute() as man:
            man.gpu_to_cpu(self)
        self.read(data)

    def structured(self, **fields):
        layout, size = Uniform.process_layout(fields)
        return StructuredBuffer(self.device, self.w_resource, layout, size)

    def as_indices(self):
        return IndexBuffer(self.device, self.w_resource)

    def as_numpy(self, dtype: np.dtype = np.float32()):
        return self.w_resource.as_numpy(dtype)

    def as_tensor(self, dtype: np.dtype = np.float32()):
        tensor =  torch.Tensor(self.as_numpy(dtype))
        tensor.inner_vk_resource = self
        return tensor

    def create_gpu_tensor(self):
        tensor = torch.zeros(self.size // 4).to(torch.device('cuda:0'))
        self.device.copy_buffer_to_gpu_pointer(tensor.data_ptr(), self, self.size)
        return tensor

    def __str__(self):
        gpu_tensor = self.create_gpu_tensor()
        return str(gpu_tensor)+'\n'+str(gpu_tensor.min())+'\n'+str(gpu_tensor.max())


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

    def __init__(self, device, w_buffer: vkw.ResourceWrapper, layout: Dict[str, Tuple[int, int, type]]):
        self.layout = layout
        super().__init__(device, w_buffer)
        w_buffer.get_permanent_map()

    def __getattr__(self, item):
        if item == "layout" or item == "w_resource" or item == "size" or item == "device":
            return super(Uniform, self).__getattribute__(item)
        if item not in self.layout:
            return super(Uniform, self).__getattribute__(item)
        offset, size, type = self.layout[item]
        buffer = bytearray(size)
        self.slice(offset, size).read(buffer)
        return BinaryFormatter.from_bytes(type, buffer)

    def __setattr__(self, item, value):
        if item == "layout" or item == "w_resource" or item == "size" or item == "device":
            super(Uniform, self).__setattr__(item, value)
            return
        if item in self.layout:
            offset, size, type = self.layout[item]
            buffer = BinaryFormatter.to_bytes(type, value)
            self.slice(offset, size).write(buffer)
            return
        raise AttributeError(f"Can not set attribute {item}")


class StructuredBuffer(Buffer):
    def __init__(self, device, w_buffer: vkw.ResourceWrapper, layout: Dict[str, Tuple[int, int, type]], stride: int):
        self.layout = layout
        super().__init__(device, w_buffer)
        w_buffer.get_permanent_map()
        self.stride = stride

    def __getitem__(self, item):
        return Uniform(self.w_resource.slice_buffer(item * self.stride, self.stride), self.layout)


class IndexBuffer(Buffer):
    def __init__(self, device, w_buffer: vkw.ResourceWrapper):
        super().__init__(device, w_buffer)
        w_buffer.get_permanent_map()

    def __getitem__(self, item):
        bytes = bytearray(4)
        self.w_resource.slice_buffer(item * 4, 4).read(bytes)
        return struct.unpack('i', bytes)[0]

    def __setitem__(self, key, value):
        assert isinstance(value, int) or isinstance(value, glm.uint32), "Only integers is supported"
        bytes = struct.pack('i', value)
        self.w_resource.slice_buffer(key * 4, 4).write(bytes)


class Image(Resource):
    @staticmethod
    def compute_dimension(width: int, height: int, depth: int, mip_level: int):
        return max(1, width // (1 << mip_level)), max(1, height // (1 << mip_level)), max(1, depth // (1 << mip_level))

    def __init__(self, device, w_image: vkw.ResourceWrapper):
        super().__init__(device, w_image)
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
        return Image(self.device, self.w_resource.slice_mips(mip_start, mip_count))

    def slice_array(self, array_start, array_count):
        return Image(self.device, self.w_resource.slice_array(array_start, array_count))

    def subresource(self, mip=0, layer=0):
        return Image(self.device, self.w_resource.subresource(mip, layer))

    def write(self, data):
        self.w_resource.write(data)

    def read(self, data):
        self.w_resource.read(data)

    def as_readonly(self):
        return Image(self.device, self.w_resource.as_readonly())

    def as_numpy(self):
        return self.w_resource.as_numpy()


class GeometryCollection:

    def __init__(self, device: vkw.DeviceWrapper):
        self.w_device = device
        self.descriptions = []

    def __del__(self):
        self.w_device = None
        self.descriptions = []

    def get_collection_type(self) -> int:
        pass


class TriangleCollection(GeometryCollection):

    def __init__(self, device: vkw.DeviceWrapper):
        super().__init__(device)

    def append(self, vertices: StructuredBuffer,
                     indices: IndexBuffer = None,
                     transform: StructuredBuffer = None):
        assert transform is None or transform.stride == 4 * 12, "Transform buffer can not be cast to 3x4 float"
        self.descriptions.append((vertices, indices, transform))

    def get_collection_type(self) -> int:
        return 0  # Triangles


class Instance:
    def __init__(self, instance_buffer, index, buffer):
        self.instance_buffer = instance_buffer
        self.index = index
        self.buffer = buffer

    def __get_int_value(self, offset, size):
        bytes = bytearray(4)
        bytes[0:size] = ffi.buffer(self.buffer[offset:offset + size])
        return struct.unpack('i', bytes)[0]

    def __set_int_value(self, offset, size, value):
        bytes = struct.pack('i', value)
        self.buffer[offset:offset + size] = bytes[0:size]

    def _get_transform(self):
        return BinaryFormatter.from_bytes(glm.mat3x4, self.buffer[0:48])

    def _set_transform(self, value):
        self.buffer[0:48] = BinaryFormatter.to_bytes(glm.mat3x4, value)

    transform = property(fget=_get_transform, fset=_set_transform)

    # def _get_mask(self):
    #     return self.__get_int_value(48, 1)
    #
    # def _set_mask(self, value):
    #     self.__set_int_value(48, 1, value)
    #
    # mask = property(fget=_get_mask, fset=_set_mask)
    #
    #
    # def _get_id(self):
    #     return self.__get_int_value(49, 3)
    #
    # def _set_id(self, value):
    #     self.__set_int_value(49, 3, value)
    #
    # id = property(fget=_get_id, fset=_set_id)
    #
    #
    # def _get_flags(self):
    #     return self.__get_int_value(52, 1)
    #
    # def _set_flags(self, value):
    #     self.__set_int_value(52, 1, value)
    #
    # flags = property(fget=_get_flags, fset=_set_flags)
    #
    # def _get_offset(self):
    #     return self.__get_int_value(53, 3)
    #
    # def _set_offset(self, value):
    #     self.__set_int_value(53, 3, value)
    #
    # offset = property(fget=_get_offset, fset=_set_offset)


    def _get_id(self):
        return self.__get_int_value(48, 3)

    def _set_id(self, value):
        self.__set_int_value(48, 3, value)

    id = property(fget=_get_id, fset=_set_id)

    def _get_mask(self):
        return self.__get_int_value(51, 1)

    def _set_mask(self, value):
        self.__set_int_value(51, 1, value)

    mask = property(fget=_get_mask, fset=_set_mask)

    def _get_offset(self):
        return self.__get_int_value(52, 3)

    def _set_offset(self, value):
        self.__set_int_value(52, 3, value)

    offset = property(fget=_get_offset, fset=_set_offset)

    def _get_flags(self):
        return self.__get_int_value(55, 1)

    def _set_flags(self, value):
        self.__set_int_value(55, 1, value)

    flags = property(fget=_get_flags, fset=_set_flags)

    def _get_geometry(self):
        return self.instance_buffer.geometries[self.index]

    def _set_geometry(self, geometry):
        self.instance_buffer.geometries[self.index] = geometry
        self.buffer[56:64] = struct.pack('Q', geometry.handle)

    geometry = property(fget=_get_geometry, fset=_set_geometry)


class InstanceBuffer(Resource):
    def __init__(self, device, w_buffer: vkw.ResourceWrapper, instances: int):
        super().__init__(device, w_buffer)
        self.buffer = self.w_resource.get_permanent_map()
        self.number_of_instances = instances
        self.geometries = [None]*instances  # list to keep referenced geometry's wrappers alive

    def __getitem__(self, item):
        return Instance(self, item, self.buffer[item * 64:item * 64 + 64])


class ADS(Resource):
    def __init__(self, device, w_resource: vkw.ResourceWrapper, handle, scratch_size,
                 info: VkAccelerationStructureCreateInfoKHR, ranges, instance_buffer = None):
        super().__init__(device, w_resource)
        self.ads = w_resource.resource_data.ads
        self.ads_info = info
        self.handle = handle
        self.scratch_size = scratch_size
        self.ranges = ranges
        self.instance_buffer = instance_buffer


class RTProgram:

    def __init__(self, pipeline,
                 w_shader_table: vkw.ResourceWrapper,
                 miss_offset,
                 hit_offset):
        self.pipeline = pipeline
        prop = self.pipeline.w_pipeline.w_device.raytracing_properties
        self.shader_handle_stride = prop.shaderGroupHandleSize
        self.table_buffer = w_shader_table.get_permanent_map()
        self.w_table = w_shader_table
        self.raygen_slice = w_shader_table.slice_buffer(0, self.shader_handle_stride)
        self.miss_slice = w_shader_table.slice_buffer(miss_offset, hit_offset - miss_offset)
        self.hitgroup_slice = w_shader_table.slice_buffer(hit_offset, w_shader_table.get_size() - hit_offset)
        self.miss_offset = miss_offset
        self.hit_offset = hit_offset


    def __del__(self):
        self.pipeline = None
        self.w_table = None
        self.raygen_slice = None
        self.miss_slice = None
        self.hitgroup_slice = None

    def set_generation(self, shader_group: ShaderHandler):
        self.table_buffer[0:self.shader_handle_stride] = shader_group.get_handle()

    def set_miss(self, miss_index: int, shader_group: ShaderHandler):
        self.table_buffer[
        self.miss_offset + self.shader_handle_stride*miss_index:
        self.miss_offset + self.shader_handle_stride*(miss_index+1)] = \
            shader_group.get_handle()

    def set_hit_group(self, hit_group_index, shader_group: ShaderHandler):
        self.table_buffer[
        self.hit_offset + self.shader_handle_stride * hit_group_index:
        self.hit_offset + self.shader_handle_stride * (hit_group_index+1)] = shader_group.get_handle()


class Pipeline:
    def __init__(self, w_pipeline: vkw.PipelineBindingWrapper):
        self.w_pipeline = w_pipeline

    def __setup__(self):
        pass

    def is_closed(self):
        return self.w_pipeline.initialized

    def close(self):
        self.w_pipeline._build_objects()

    def descriptor_set(self, set_slot: int):
        self.w_pipeline.descriptor_set(set_slot)

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

    def bind_scene_ads(self, slot: int, stage: ShaderStage, resolver):
        self._bind_resource(slot, stage, 1, lambda: [resolver()], VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)

    def bind_constants(self, offset: int, stage: ShaderStage, **fields):
        layout, size = Uniform.process_layout(fields)
        self.w_pipeline.add_constant_range(stage, offset, size, layout)

    def load_shader(self, stage: ShaderStage, path, main_function = 'main'):
        return self.w_pipeline.load_shader(vkw.ShaderStageWrapper.from_file(
            device=self.w_pipeline.w_device,
            vk_stage=stage,
            main_function=main_function,
            path=path))

    def load_fragment_shader(self, path: str, main_function='main'):
        return self.load_shader(VK_SHADER_STAGE_FRAGMENT_BIT, path, main_function)

    def load_vertex_shader(self, path: str, main_function='main'):
        return self.load_shader(VK_SHADER_STAGE_VERTEX_BIT, path, main_function)

    def load_compute_shader(self, path: str, main_function='main'):
        return self.load_shader(VK_SHADER_STAGE_COMPUTE_BIT, path, main_function)

    def load_rt_generation_shader(self, path: str, main_function='main'):
        return self.load_shader(VK_SHADER_STAGE_RAYGEN_BIT_KHR, path, main_function)

    def load_rt_closest_hit_shader(self, path: str, main_function='main'):
        return self.load_shader(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, path, main_function)

    def load_rt_miss_shader(self, path: str, main_function='main'):
        return self.load_shader(VK_SHADER_STAGE_MISS_BIT_KHR, path, main_function)

    def create_rt_hit_group(self, closest_hit: int = None, any_hit: int = None, intersection: int = None):
        return self.w_pipeline.create_hit_group(closest_hit, any_hit, intersection)

    def create_rt_gen_group(self, generation_shader_index: int):
        return self.w_pipeline.create_general_group(generation_shader_index)

    def create_rt_miss_group(self, miss_shader_index: int):
        return self.w_pipeline.create_general_group(miss_shader_index)

    def _get_aligned_size(self, size, align):
        return (size + align - 1) & (~(align - 1))

    def create_rt_program(self, max_miss_shader = 10, max_hit_groups = 1000):
        shaderHandlerSize = self.w_pipeline.w_device.raytracing_properties.shaderGroupHandleSize
        groupAlignment = self.w_pipeline.w_device.raytracing_properties.shaderGroupBaseAlignment

        raygen_size = self._get_aligned_size(shaderHandlerSize, groupAlignment)
        raymiss_size = self._get_aligned_size(shaderHandlerSize * max_miss_shader, groupAlignment)
        rayhit_size = self._get_aligned_size(shaderHandlerSize * max_hit_groups, groupAlignment)

        w_buffer = self.w_pipeline.w_device.create_buffer(
            raygen_size + raymiss_size + rayhit_size,
            usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            properties=MemoryProperty.CPU_ACCESSIBLE | MemoryProperty.CPU_DIRECT)
        return RTProgram(self, w_buffer, raygen_size, raygen_size + raymiss_size)


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
        if resource:
            self.w_cmdList.from_gpu(resource.w_resource)

    def cpu_to_gpu(self, resource):
        if resource:
            self.w_cmdList.to_gpu(resource.w_resource)

    def copy_image(self, src_image: Image, dst_image: Image):
        self.w_cmdList.copy_image(src_image.w_resource, dst_image.w_resource)


class ComputeManager(CopyManager):
    def __init__(self, w_cmdList: vkw.CommandBufferWrapper):
        super().__init__(w_cmdList)

    @classmethod
    def get_queue_required(cls) -> int:
        return QueueType.COMPUTE

    def clear_color(self, image: Image, color):
        self.w_cmdList.clear_color(image.w_resource, color)
    
    def clear_buffer(self, buffer: Buffer, value: int = 0):
        self.w_cmdList.clear_buffer(buffer.w_resource, value)

    def set_pipeline(self, pipeline: Pipeline):
        if not pipeline.is_closed():
            raise Exception("Error, can not set a pipeline has not been closed.")
        self.w_cmdList.set_pipeline(pipeline=pipeline.w_pipeline)

    def update_sets(self, *sets):
        for s in sets:
            self.w_cmdList.update_bindings_level(s)

    def update_constants(self, stages, **fields):
        self.w_cmdList.update_constants(stages, **{
            f: BinaryFormatter.to_bytes(type=type(v), value=v)
            for f, v in fields.items()
        })

    def dispatch_groups(self, groups_x: int, groups_y: int = 1, groups_z:int = 1):
        self.w_cmdList.dispatch_groups(groups_x, groups_y, groups_z)

    def dispatch_threads_1D(self, dim_x: int, group_size_x: int = 1024):
        self.dispatch_groups(math.ceil(dim_x/group_size_x))

    def dispatch_threads_2D(self, dim_x: int, dim_y: int, group_size_x: int = 32, group_size_y: int = 32):
        self.dispatch_groups(math.ceil(dim_x/group_size_x), math.ceil(dim_y/group_size_y))

    def dispatch_rays(self, program: RTProgram, dim_x: int, dim_y: int, dim_z: int = 1):
        self.w_cmdList.dispatch_rays(
            program.raygen_slice, program.miss_slice, program.hitgroup_slice,
            dim_x, dim_y, dim_z
        )


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

    def build_ads(self, ads: ADS, scratch_buffer: Buffer):
        self.w_cmdList.build_ads(
            ads.w_resource,
            ads.ads_info,
            ads.ranges,
            scratch_buffer.w_resource)


class DeviceManager:

    def __init__(self):
        self.w_state = None
        self.width = 0
        self.height = 0
        self.__copying_from_gpu_pipeline = None
        self.__copying_to_gpu_pipeline = None

    def __del__(self):
        self.__copying_to_gpu_pipeline = None
        self.__copying_from_gpu_pipeline = None
        self.w_device = None

    def __bind__(self, w_device: vkw.DeviceWrapper):
        self.w_device = w_device
        self.width = w_device.get_render_target(0).resource_data.vk_description.extent.width
        self.height = w_device.get_render_target(0).resource_data.vk_description.extent.height

    def render_target(self):
        return Image(self, self.w_device.get_render_target(self.w_device.get_render_target_index()))

    def load_technique(self, technique):
        technique.__bind__(self.w_device)
        technique.__setup__()
        return technique

    def dispatch_technique(self, technique):
        assert technique.w_device, "Technique is not bound to a device, you must load the technique in some point before dispatching"
        technique.__dispatch__()

    def create_buffer(self, size: int, usage: int, memory: MemoryProperty):
        return Buffer(self, self.w_device.create_buffer(size, usage, memory))

    def create_uniform_buffer(self, usage: int = BufferUsage.UNIFORM,
                       memory: MemoryProperty = MemoryProperty.CPU_ACCESSIBLE,
                       **fields):
        layout, size = Uniform.process_layout(fields)
        resource = self.w_device.create_buffer(size, usage, memory)
        return Uniform(self, resource, layout)

    def create_structured_buffer(self, count:int,
                                    usage: int = BufferUsage.VERTEX | BufferUsage.TRANSFER_DST,
                                    memory: MemoryProperty = MemoryProperty.GPU,
                                    **fields):
        layout, size = Uniform.process_layout(fields)
        resource = self.w_device.create_buffer(size * count, usage, memory)
        return StructuredBuffer(self, resource, layout, size)

    def create_indices_buffer(self, count: int,
                              usage: int = BufferUsage.INDEX | BufferUsage.TRANSFER_DST,
                              memory: MemoryProperty = MemoryProperty.GPU):
        return IndexBuffer(self, self.w_device.create_buffer(count*4, usage, memory))

    def create_image(self, image_type: ImageType, is_cube: bool, image_format: Format,
                     width: int, height: int, depth: int,
                     mips: int, layers: int,
                     usage: int, memory: MemoryProperty):
        linear = False  # bool(usage & (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        layout = VK_IMAGE_LAYOUT_UNDEFINED
        return Image(self, self.w_device.create_image(
            image_type, image_format, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT if is_cube else 0,
            VkExtent3D(width, height, depth), mips, layers, linear, layout, usage, memory
        ))

    def create_triangle_collection(self):
        return TriangleCollection(device=self.w_device)

    def create_geometry_ads(self, collection: GeometryCollection):
        ads, info, ranges, handle, scratch_size = self.w_device.create_ads(
            geometry_type=collection.get_collection_type(),
            descriptions=[
                (v.w_resource, v.stride, None if i is None else i.w_resource, None if t is None else t.w_resource)
                for v, i, t in collection.descriptions
            ]
        )
        return ADS(self, ads, handle, scratch_size, info, ranges)

    def create_scene_ads(self, instance_buffer: InstanceBuffer):
        ads, info, ranges, handle, scratch_size = self.w_device.create_ads(
            geometry_type=VK_GEOMETRY_TYPE_INSTANCES_KHR,
            descriptions=[
                instance_buffer.w_resource
            ]
        )
        return ADS(ads, handle, scratch_size, info, ranges, instance_buffer)

    def create_instance_buffer(self, instances: int, memory: MemoryProperty = MemoryProperty.GPU):
        buffer = self.w_device.create_buffer(instances*64,
                                             BufferUsage.RAYTRACING_ADS_READ | BufferUsage.TRANSFER_DST,
                                             memory)
        return InstanceBuffer(self, buffer, instances)

    def create_scratch_buffer(self, *ads_set):
        size = max(a.scratch_size for a in ads_set)
        return self.create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

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
            vk_pipeline_type=VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR))

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

    def copy_gpu_pointer_to_buffer(self, gpu_pointer, buffer, size):
        if not self.__copying_from_gpu_pipeline:
            # Create pipeline to copy gpu pointer
            pipeline = self.create_compute_pipeline()
            pipeline.bind_storage_buffer(0, ShaderStage.COMPUTE, lambda: self.__copying_from_gpu_pipeline.buffer)
            pipeline.bind_constants(
                0, ShaderStage.COMPUTE,
                pointer=glm.uint64,
                size=int
            )
            pipeline.load_compute_shader(__SHADERS_TOOLS__+"/copy_from_gpu_pointer.comp.spv")
            pipeline.close()
            self.__copying_from_gpu_pipeline = pipeline

        self.__copying_from_gpu_pipeline.buffer = buffer
        with self.get_compute() as man:
            man.set_pipeline(self.__copying_from_gpu_pipeline)
            man.update_sets(0)
            man.update_constants(
                ShaderStage.COMPUTE,
                pointer=glm.uint64(gpu_pointer),
                size=size
            )
            man.dispatch_threads_1D(int(math.ceil(size/(4*128))))
        self.flush()

    def copy_buffer_to_gpu_pointer(self, gpu_pointer, buffer, size):
        if not self.__copying_to_gpu_pipeline:
            # Create pipeline to copy gpu pointer
            pipeline = self.create_compute_pipeline()
            pipeline.bind_storage_buffer(0, ShaderStage.COMPUTE, lambda: self.__copying_to_gpu_pipeline.buffer)
            pipeline.bind_constants(
                0, ShaderStage.COMPUTE,
                pointer=glm.uint64,
                size=int
            )
            pipeline.load_compute_shader(__SHADERS_TOOLS__+"/copy_to_gpu_pointer.comp.spv")
            pipeline.close()
            self.__copying_to_gpu_pipeline = pipeline
        self.__copying_to_gpu_pipeline.buffer = buffer
        with self.get_compute() as man:
            man.set_pipeline(self.__copying_to_gpu_pipeline)
            man.update_sets(0)
            man.update_constants(
                ShaderStage.COMPUTE,
                pointer=glm.uint64(gpu_pointer),
                size=size
            )
            man.dispatch_threads_1D(int(math.ceil(size/(4*128))))


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
    def __setup__(self):
        pass

    def __dispatch__(self):
        pass


def Extends(class_):
    def wrapper(function):
        setattr(class_, function.__name__, function)
    return wrapper
