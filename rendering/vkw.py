"""
Vulkan wrappers to represent:
Device (manage all internal objects for instance, device, swapchain, queues, pools, descriptors)
Resource (manage all internal objects for a resource, memory, staging memory)
CommandPool (creates and flushes commands)
CommandBuffer (represents a command list to be populated)
GPUTask (synchronization object to wait for GPU completation)
"""
from typing import Dict

from vulkan import *
import ctypes
from enum import Enum
import numpy as np
import torch

__TRACE__ = False

def trace_destroying(function):
    def wrapper(self, *args):
        if __TRACE__:
            print(f"[INFO] Destroying {type(self)}...", end='')
        function(self, *args)
        if __TRACE__:
            print("done.")
    return wrapper


class ResourceState:
    def __init__(self, vk_access, vk_stage, vk_layout, queue_index):
        self.vk_access = vk_access
        self.vk_stage = vk_stage
        self.vk_layout = vk_layout
        self.queue_index = queue_index

    def __iter__(self):
        yield self.vk_access
        yield self.vk_stage
        yield self.vk_layout
        yield self.queue_index


_FORMAT_DESCRIPTION = {
    VK_FORMAT_R32_SFLOAT: (1, 'f'),
    VK_FORMAT_R32_SINT: (1, 'i'),
    VK_FORMAT_R32_UINT: (1, 'u'),

    VK_FORMAT_R32G32_SFLOAT: (2, 'f'),
    VK_FORMAT_R32G32_SINT: (2, 'i'),
    VK_FORMAT_R32G32_UINT: (2, 'u'),

    VK_FORMAT_R32G32B32_SFLOAT: (3, 'f'),
    VK_FORMAT_R32G32B32_SINT: (3, 'i'),
    VK_FORMAT_R32G32B32_UINT: (3, 'u'),

    VK_FORMAT_R32G32B32A32_SFLOAT: (4, 'f'),
    VK_FORMAT_R32G32B32A32_SINT: (4, 'i'),
    VK_FORMAT_R32G32B32A32_UINT: (4, 'u'),

    VK_FORMAT_R8G8B8A8_UNORM: (4, 'b'),
    VK_FORMAT_R8G8B8A8_SNORM: (4, 'b'),
    VK_FORMAT_R8G8B8A8_USCALED: (4, 'b'),
    VK_FORMAT_R8G8B8A8_SSCALED: (4, 'b'),
    VK_FORMAT_R8G8B8A8_UINT: (4, 'b'),
    VK_FORMAT_R8G8B8A8_SINT: (4, 'b'),
    VK_FORMAT_R8G8B8A8_SRGB: (4, 'b'),

    VK_FORMAT_B8G8R8A8_UNORM: (4, 'b'),
    VK_FORMAT_B8G8R8A8_SNORM: (4, 'b'),
    VK_FORMAT_B8G8R8A8_USCALED: (4, 'b'),
    VK_FORMAT_B8G8R8A8_SSCALED: (4, 'b'),
    VK_FORMAT_B8G8R8A8_UINT: (4, 'b'),
    VK_FORMAT_B8G8R8A8_SINT: (4, 'b'),
    VK_FORMAT_B8G8R8A8_SRGB: (4, 'b'),
}


_FORMAT_SIZES = {

    VK_FORMAT_R32_SFLOAT: 4,
    VK_FORMAT_R32_SINT: 4,
    VK_FORMAT_R32_UINT: 4,

    VK_FORMAT_R32G32_SFLOAT: 8,
    VK_FORMAT_R32G32_SINT: 8,
    VK_FORMAT_R32G32_UINT: 8,

    VK_FORMAT_R32G32B32_SFLOAT: 12,
    VK_FORMAT_R32G32B32_SINT: 12,
    VK_FORMAT_R32G32B32_UINT: 12,

    VK_FORMAT_R32G32B32A32_SFLOAT: 16,
    VK_FORMAT_R32G32B32A32_SINT: 16,
    VK_FORMAT_R32G32B32A32_UINT: 16,

    VK_FORMAT_R8G8B8A8_UNORM: 4,
    VK_FORMAT_R8G8B8A8_SNORM: 4,
    VK_FORMAT_R8G8B8A8_USCALED: 4,
    VK_FORMAT_R8G8B8A8_SSCALED: 4,
    VK_FORMAT_R8G8B8A8_UINT: 4,
    VK_FORMAT_R8G8B8A8_SINT: 4,
    VK_FORMAT_R8G8B8A8_SRGB: 4,

    VK_FORMAT_B8G8R8A8_UNORM: 4,
    VK_FORMAT_B8G8R8A8_SNORM: 4,
    VK_FORMAT_B8G8R8A8_USCALED: 4,
    VK_FORMAT_B8G8R8A8_SSCALED: 4,
    VK_FORMAT_B8G8R8A8_UINT: 4,
    VK_FORMAT_B8G8R8A8_SINT: 4,
    VK_FORMAT_B8G8R8A8_SRGB: 4,
}


class SubresourceFootprint:
    def __init__(self, dim: tuple, element_stride, row_pitch, slice_pitch, size):
        self.dim = dim
        self.element_stride = element_stride
        self.row_pitch = row_pitch
        self.slice_pitch = slice_pitch
        self.size = size


class ResourceData:
    def __init__(self,
                 device,
                 vk_description,
                 vk_properties,
                 vk_resource,
                 vk_resource_memory,
                 is_buffer,
                 initial_state
                 ):
        self.device = device
        self.vk_description = vk_description
        self.vk_properties = vk_properties
        self.vk_resource = vk_resource
        self.vk_resource_memory = vk_resource_memory
        self.current_state = initial_state
        self.is_buffer = is_buffer
        self.is_ads = False
        self.ads = None
        if is_buffer:
            self.full_slice = {"offset": 0, "size": vk_description.size}
        else:
            self.full_slice = {
                "mip_start": 0, "mip_count": vk_description.mipLevels,
                "array_start": 0, "array_count": vk_description.arrayLayers
            }
        cpu_footprints = []
        cpu_offsets = []
        cpu_offset = 0
        if not self.is_buffer:
            self.element_size = _FORMAT_SIZES[vk_description.format]
            dim = self.vk_description.extent.width, self.vk_description.extent.height, self.vk_description.extent.depth
            for m in range(vk_description.mipLevels):
                size = dim[0] * dim[1] * dim[2] * self.element_size
                for a in range(vk_description.arrayLayers):
                    cpu_footprints.append(SubresourceFootprint(
                        dim, self.element_size, dim[0]*self.element_size, dim[0]*dim[1]*self.element_size, size))
                    cpu_offsets.append(cpu_offset)
                    cpu_offset += size
                dim = max(1, dim[0] // 2), max(1, dim[1] // 2), max(1, dim[2] // 2)
        else:
            self.element_size = 1
            cpu_footprints.append(SubresourceFootprint(
                (vk_description.size, 1, 1), 1, vk_description.size, vk_description.size, vk_description.size))
            cpu_offsets.append(0)
            cpu_offset += vk_description.size

        self.cpu_footprints = cpu_footprints
        self.cpu_offsets = cpu_offsets
        self.cpu_size = cpu_offset
        self.__staging = None  # Staging buffer for uploading and downloading data
        if is_buffer or not(vk_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT):
            self.__staging_footprints = [
                SubresourceFootprint((self.cpu_size, 1, 1), 1, self.cpu_size, self.cpu_size, self.cpu_size)
            ]
            self.__staging_offsets = [0]
        else:
            self.__staging_footprints = []
            self.__staging_offsets = []
            for m in range(vk_description.mipLevels):
                for a in range(vk_description.arrayLayers):
                    layout = vkGetImageSubresourceLayout(device.vk_device, vk_resource, VkImageSubresource(
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        m, a
                    ))
                    cpu_layout = cpu_footprints[m*vk_description.arrayLayers + a]
                    self.__staging_footprints.append(SubresourceFootprint(
                        dim=cpu_layout.dim,
                        element_stride=cpu_layout.element_stride,
                        row_pitch=layout.rowPitch,
                        slice_pitch=layout.depthPitch,
                        size=layout.size
                    ))
                    self.__staging_offsets.append(layout.offset)
        self.__staging_size = self.__staging_offsets[-1] + self.__staging_footprints[-1].size

        self.__permanent_map = None  # Permanent mapped buffer

    def bind_ads(self, ads):
        self.is_ads = True
        self.ads = ads

    vkDestroyAccelerationStructure = None

    @trace_destroying
    def __del__(self):
        if self.vk_resource_memory and self.device:
            if self.is_buffer:
                if self.is_ads:
                    ResourceData.vkDestroyAccelerationStructure(self.device.vk_device, self.ads, None)
                vkDestroyBuffer(self.device.vk_device, self.vk_resource, None)
            else:
                vkDestroyImage(self.device.vk_device, self.vk_resource, None)
            vkFreeMemory(self.device.vk_device, self.vk_resource_memory, None)
        self.__staging = None
        self.device = None
        self.vk_resource_memory = None

    def _resolve_staging(self):
        if self.vk_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:
            return self  # no need for staging because can be mapped
        if self.__staging is None:  # create staging buffer for read and write
            self.__staging = self.device.create_buffer_data(
                self.cpu_size,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        return self.__staging

    def is_image_staging(self):
        return not self.is_buffer and self.vk_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT

    def get_staging_footprint_and_offset(self, subresource_index):
        if self.is_image_staging():
            return self.__staging_footprints[subresource_index], \
                   self.__staging_offsets[subresource_index]
        return self.cpu_footprints[subresource_index], self.cpu_offsets[subresource_index]

    def make_permanent_map(self):
        if self.__permanent_map is not None:
            return
        staging = self._resolve_staging()
        self.__permanent_map = ffi.from_buffer(vkMapMemory(
            self.device.vk_device, staging.vk_resource_memory, 0, self.__staging_size, 0
        ))
        return self.__permanent_map

    def map_subresource(self, index):
        footprint, offset = self.get_staging_footprint_and_offset(index)
        if self.__permanent_map is not None:
            return self.__permanent_map + offset
        staging = self._resolve_staging()
        size = footprint.size
        return ffi.from_buffer(vkMapMemory(
            self.device.vk_device, staging.vk_resource_memory, offset, size, 0
        ))

    def map_buffer_slice(self, offset, size):
        if self.__permanent_map is not None:
            return self.__permanent_map+offset
        staging = self._resolve_staging()
        return ffi.from_buffer(vkMapMemory(
            self.device.vk_device, staging.vk_resource_memory, offset, size, 0
        ))

    def flush_mapped(self):
        # vkFlushMappedMemoryRanges(self.device.vk_device, 1, [
        #     VkMappedMemoryRange(
        #         memory=self._resolve_staging().vk_resource_memory,
        #         offset=0,
        #         size=UINT64_MAX
        #     )
        # ])
        pass

    def invalidate_mapped(self):
        # vkInvalidateMappedMemoryRanges(self.device.vk_device, 1, [
        #     VkMappedMemoryRange(
        #         memory=self._resolve_staging().vk_resource_memory,
        #         offset=0,
        #         size=UINT64_MAX
        #     )
        # ])
        pass

    def unmap(self):
        if self.__permanent_map is None:  # only unmap if permanent map is not active
            vkUnmapMemory(self.device.vk_device, self._resolve_staging().vk_resource_memory)

    def transfer_slice_to_gpu(self, slice, vk_cmdList):
        staging = self._resolve_staging()
        if staging == self:  # no need to transfer
            return
        if self.is_buffer:  # copy buffer to buffer
            offset = slice["offset"]
            size = slice["size"]
            region = VkBufferCopy(srcOffset=offset, dstOffset=offset, size=size)
            vkCmdCopyBuffer(vk_cmdList, staging.vk_resource, self.vk_resource, 1, [region])
        else:
            mip_start = slice["mip_start"]
            mip_count = slice["mip_count"]
            array_start = slice["array_start"]
            array_count = slice["array_count"]
            full_array_count = self.full_slice["array_count"]
            regions = []
            for mip in range(mip_start, mip_start + mip_count):
                for arr in range(array_start, array_start + array_count):
                    subresource_index = arr + mip * full_array_count
                    footprint, offset = self.get_staging_footprint_and_offset(subresource_index)
                    width, height, depth = footprint.dim
                    regions.append(VkBufferImageCopy(offset, 0, 0,
                                                     VkImageSubresourceLayers(VK_IMAGE_ASPECT_COLOR_BIT, mip, arr, 1),
                                                     imageOffset=(0, 0, 0),
                                                     imageExtent=VkExtent3D(width, height, depth)))
            vkCmdCopyBufferToImage(vk_cmdList, staging.vk_resource, self.vk_resource,
                                   VK_IMAGE_LAYOUT_GENERAL, len(regions), regions)

    def transfer_slice_from_gpu(self, slice, vk_cmdList):
        staging = self._resolve_staging()
        if staging == self:  # no need to transfer
            return
        if self.is_buffer:  # copy buffer to buffer
            offset = slice["offset"]
            size = slice["size"]
            region = VkBufferCopy(srcOffset=offset, dstOffset=offset, size=size)
            vkCmdCopyBuffer(vk_cmdList, self.vk_resource, staging.vk_resource, 1, [region])
        else:
            mip_start = slice["mip_start"]
            mip_count = slice["mip_count"]
            array_start = slice["array_start"]
            array_count = slice["array_count"]
            full_array_count = self.full_slice["array_count"]
            regions = []
            for mip in range(mip_start, mip_start + mip_count):
                for arr in range(array_start, array_start + array_count):
                    subresource_index = arr + mip * full_array_count
                    footprint, offset = self.get_staging_footprint_and_offset(subresource_index)
                    width, height, depth = footprint.dim
                    regions.append(VkBufferImageCopy(offset, 0, 0,
                                                     VkImageSubresourceLayers(VK_IMAGE_ASPECT_COLOR_BIT, mip, arr, 1),
                                                     imageOffset=(0, 0, 0),
                                                     imageExtent=VkExtent3D(width, height, depth)))

            vkCmdCopyImageToBuffer(vk_cmdList, self.vk_resource, VK_IMAGE_LAYOUT_GENERAL,
                                   staging.vk_resource, len(regions), regions)

    _FORMAT_TYPE_2_NP_TYPE = {
        'b': 'u1',
        'f': '<f4',
        'i': '<i4',
        'u': '<u4'
    }

    def get_numpy_views(self, slice, buffer_description: np.dtype = None):
        self.make_permanent_map()
        views = []
        if self.is_buffer:  # copy buffer to buffer
            offset = slice["offset"]
            size = slice["size"]
            assert size % buffer_description.itemsize == 0, "Error casting buffer elements, size in bytes must be a multiple"
            views.append(np.frombuffer(
                buffer=ffi.buffer(self.__permanent_map),
                offset=offset,
                dtype=buffer_description,
                count=size//buffer_description.itemsize
            ))
        else:
            components, component_type = _FORMAT_DESCRIPTION[self.vk_description.format]
            numpy_type = component_type
            mip_start = slice["mip_start"]
            mip_count = slice["mip_count"]
            array_start = slice["array_start"]
            array_count = slice["array_count"]
            full_array_count = self.full_slice["array_count"]
            for mip in range(mip_start, mip_start + mip_count):
                for arr in range(array_start, array_start + array_count):
                    subresource_index = arr + mip * full_array_count
                    footprint, offset = self.get_staging_footprint_and_offset(subresource_index)
                    width, height, depth = footprint.dim
                    rwidth = footprint.row_pitch // footprint.element_stride
                    rheight = footprint.slice_pitch // footprint.row_pitch
                    rdepth = footprint.size // footprint.slice_pitch

                    a = np.frombuffer(
                            buffer=ffi.buffer(self.__permanent_map),
                            offset=offset,
                            dtype=ResourceData._FORMAT_TYPE_2_NP_TYPE[component_type],
                            count=footprint.size // (1 if component_type == 'b' else 4)
                        )
                    # remove extra elements
                    if components == 1:
                        a = a.reshape(rdepth, rheight, rwidth)[0:depth, 0:height, 0:width]
                    else:
                        a = a.reshape(rdepth, rheight, rwidth, components)[0:depth, 0:height, 0:width, :]
                    if self.vk_description.imageType == VK_IMAGE_TYPE_2D:
                        a = a[0]
                    elif self.vk_description.imageType == VK_IMAGE_TYPE_2D:
                        a = a[0][0]
                    views.append(a)
        return views[0] if len(views) == 1 else views


class ResourceWrapper:
    """
    Wrapper for a vk resource, memory and and initial state.
    """
    def __init__(self,
                 resource_data: ResourceData,
                 resource_slice: Dict[str, int] = None,
                 ):
        self.resource_data = resource_data
        self.vk_view = None
        if resource_slice is None:
            resource_slice = resource_data.full_slice
        self.current_slice = resource_slice
        self.is_readonly = False

    # @trace_destroying
    def __del__(self):
        # Destroy view if any
        if self.vk_view:
            if self.resource_data.is_buffer:
                vkDestroyBufferView(self.resource_data.device.vk_device, self.vk_view, None)
            else:
                vkDestroyImageView(self.resource_data.device.vk_device, self.vk_view, None)
        self.resource_data = None

    def add_barrier(self, vk_cmdList, state: ResourceState):
        srcAccessMask, srcStageMask, oldLayout, srcQueue = self.resource_data.current_state
        dstAccessMask, dstStageMask, newLayout, dstQueue = state

        if srcQueue == VK_QUEUE_FAMILY_IGNORED:
            dstQueue = VK_QUEUE_FAMILY_IGNORED

        if self.resource_data.is_buffer:
            if not self.resource_data.is_ads:
                barrier = VkBufferMemoryBarrier(
                    buffer=self.resource_data.vk_resource,
                    srcAccessMask=srcAccessMask,
                    dstAccessMask=dstAccessMask,
                    srcQueueFamilyIndex=srcQueue,
                    dstQueueFamilyIndex=dstQueue,
                    offset=0,
                    size=UINT64_MAX
                )
                vkCmdPipelineBarrier(vk_cmdList, srcStageMask, dstStageMask, 0,
                                     0, None, 1, barrier, 0, None)
        else:
            barrier = VkImageMemoryBarrier(
                image=self.resource_data.vk_resource,
                srcAccessMask=srcAccessMask,
                dstAccessMask=dstAccessMask,
                oldLayout=oldLayout,
                newLayout=newLayout,
                srcQueueFamilyIndex=srcQueue,
                dstQueueFamilyIndex=dstQueue,
                subresourceRange=ResourceWrapper.get_subresources(self.current_slice)
            )
            vkCmdPipelineBarrier(vk_cmdList, srcStageMask, dstStageMask, 0,
                                 0, None, 0, None, 1, barrier)
        self.current_state = state

    def as_readonly(self):
        new_slice = dict(self.current_slice)
        rw = ResourceWrapper(self.resource_data, new_slice)
        rw.is_readonly = True
        return rw

    def slice_mips(self, mip_start, mip_count):
        new_slice = dict(self.current_slice)
        new_slice["mip_start"] = self.current_slice["mip_start"] + mip_start
        new_slice["mip_count"] = mip_count
        return ResourceWrapper(self.resource_data, new_slice)

    def slice_array(self, array_start, array_count):
        new_slice = dict(self.current_slice)
        new_slice["array_start"] = self.current_slice["array_start"] + array_start
        new_slice["array_count"] = array_count
        return ResourceWrapper(self.resource_data, new_slice)

    def subresource(self, mip, layer):
        new_slice = dict(self.current_slice)
        new_slice["mip_start"] = self.current_slice["mip_start"] + mip
        new_slice["mip_count"] = 1
        new_slice["array_start"] = self.current_slice["array_start"] + layer
        new_slice["array_count"] = 1
        return ResourceWrapper(self.resource_data, new_slice)

    def slice_buffer(self, offset, size):
        new_slice = dict(self.current_slice)
        new_slice["offset"] = self.current_slice["offset"] + offset
        new_slice["size"] = size
        return ResourceWrapper(self.resource_data, new_slice)

    def write(self, bytes):
        if isinstance(bytes, list):
            bytes = np.array(bytes)
        if isinstance(bytes, np.ndarray):  # get buffer from numpy array
            bytes = bytes.data.cast('b')
        if isinstance(bytes, torch.Tensor):
            bytes = np.array(bytes).data.cast('b')
        if self.resource_data.is_buffer:
            offset = self.current_slice["offset"]
            size = self.current_slice["size"]
            mapped_memory = self.resource_data.map_buffer_slice(offset, size)
            # data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
            ffi.memmove(mapped_memory, bytes, size)
            self.resource_data.flush_mapped()
            self.resource_data.unmap()
        else:
            mip_start = self.current_slice["mip_start"]
            mip_count = self.current_slice["mip_count"]
            array_start = self.current_slice["array_start"]
            array_count = self.current_slice["array_count"]
            full_array_count = self.resource_data.full_slice["array_count"]
            data_offset = 0
            for mip in range(mip_start, mip_start + mip_count):
                for arr in range(array_start, array_start + array_count):
                    subresource_index = arr + mip * full_array_count
                    mapped_memory = self.resource_data.map_subresource(subresource_index)
                    cpu_footprint = self.resource_data.cpu_footprints[subresource_index]
                    stg_footprint,_ = self.resource_data.get_staging_footprint_and_offset(subresource_index)
                    if cpu_footprint.size == stg_footprint.size:
                        ffi.memmove(mapped_memory, bytes[data_offset:], cpu_footprint.size)
                    else:
                        gpu_offset = 0
                        cpu_offset = data_offset
                        for z in range(cpu_footprint.dim[2]):
                            slice_offset = gpu_offset
                            for y in range(cpu_footprint.dim[1]):
                                ffi.memmove(
                                    mapped_memory + slice_offset,
                                    bytes[cpu_offset:],
                                    cpu_footprint.row_pitch)
                                cpu_offset += cpu_footprint.row_pitch
                                slice_offset += stg_footprint.row_pitch
                            gpu_offset += stg_footprint.slice_pitch
                    data_offset += cpu_footprint.size
                    self.resource_data.flush_mapped()
                    self.resource_data.unmap()

    def read(self, bytes):
        if isinstance(bytes, np.ndarray):  # get buffer from numpy array
            bytes = bytes.data.cast('b')
        if self.resource_data.is_buffer:
            offset = self.current_slice["offset"]
            size = self.current_slice["size"]
            mapped_memory = self.resource_data.map_buffer_slice(offset, size)
            self.resource_data.invalidate_mapped()
            ffi.memmove(bytes, mapped_memory, size)
            self.resource_data.unmap()
        else:
            mip_start = self.current_slice["mip_start"]
            mip_count = self.current_slice["mip_count"]
            array_start = self.current_slice["array_start"]
            array_count = self.current_slice["array_count"]
            full_array_count = self.resource_data.full_slice["array_count"]
            data_offset = 0
            for mip in range(mip_start, mip_start + mip_count):
                for arr in range(array_start, array_start + array_count):
                    subresource_index = arr + mip * full_array_count
                    mapped_memory = self.resource_data.map_subresource(subresource_index)
                    self.resource_data.invalidate_mapped()
                    cpu_footprint = self.resource_data.cpu_footprints[subresource_index]
                    stg_footprint, _ = self.resource_data.get_staging_footprint_and_offset(subresource_index)
                    if cpu_footprint.size == stg_footprint.size:
                        ffi.memmove(bytes[data_offset:], mapped_memory, cpu_footprint.size)
                    else:
                        gpu_offset = 0
                        cpu_offset = data_offset
                        for z in range(cpu_footprint.dim[2]):
                            slice_offset = gpu_offset
                            for y in range(cpu_footprint.dim[1]):
                                ffi.memmove(
                                    bytes[cpu_offset:],
                                    mapped_memory + slice_offset,
                                    cpu_footprint.row_pitch)
                                cpu_offset += cpu_footprint.row_pitch
                                slice_offset += stg_footprint.row_pitch
                            gpu_offset += stg_footprint.slice_pitch
                    data_offset += cpu_footprint.size
                    self.resource_data.unmap()

    def get_permanent_map(self):
        return self.resource_data.make_permanent_map()

    def get_element_decription(self):
        if self.resource_data.is_buffer:
            return (1, 'b')
        return _FORMAT_DESCRIPTION[self.resource_data.vk_description.format]

    @staticmethod
    def get_subresources(slice):
        return VkImageSubresourceRange(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=slice["mip_start"],
            levelCount=slice["mip_count"],
            baseArrayLayer=slice["array_start"],
            layerCount=slice["array_count"]
        )

    @staticmethod
    def get_subresource_layers(slice):
        return VkImageSubresourceLayers(
            aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
            mipLevel=slice["mip_start"],
            baseArrayLayer=slice["array_start"],
            layerCount=slice["array_count"]
        )

    def get_view(self):
        if self.vk_view is None:
            if self.resource_data.is_buffer:
                self.vk_view = vkCreateBufferView(self.resource_data.device.vk_device, VkBufferViewCreateInfo(
                    buffer=self.resource_data.vk_resource,
                    offset=self.current_slice["offset"],
                    range=self.current_slice["size"]
                ), None)
            else:
                self.vk_view = vkCreateImageView(self.resource_data.device.vk_device, VkImageViewCreateInfo(
                    image=self.resource_data.vk_resource,
                    viewType=self.resource_data.vk_description.imageType,
                    flags=0,
                    format=self.resource_data.vk_description.format,
                    components=VkComponentMapping(
                        r=VK_COMPONENT_SWIZZLE_IDENTITY,
                        g=VK_COMPONENT_SWIZZLE_IDENTITY,
                        b=VK_COMPONENT_SWIZZLE_IDENTITY,
                        a=VK_COMPONENT_SWIZZLE_IDENTITY,
                    ),
                    subresourceRange=self.get_subresources(self.current_slice)
                ), None)
        return self.vk_view

    def get_size(self):
        """
        Gets the size in bytes of this buffer
        """
        assert self.resource_data.is_buffer, "Can not access to the size of an image"
        return self.current_slice["size"]

    def add_transfer_from_gpu(self, vk_cmdList):
        self.add_barrier(vk_cmdList, state=ResourceState(
            vk_access=VK_ACCESS_TRANSFER_WRITE_BIT,
            vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk_layout=VK_IMAGE_LAYOUT_GENERAL,
            queue_index=VK_QUEUE_FAMILY_IGNORED
        ))
        self.resource_data.transfer_slice_from_gpu(self.current_slice, vk_cmdList)

    def add_transfer_to_gpu(self, vk_cmdList):
        self.add_barrier(vk_cmdList, state=ResourceState(
            vk_access=VK_ACCESS_TRANSFER_READ_BIT,
            vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk_layout=VK_IMAGE_LAYOUT_GENERAL,
            queue_index=VK_QUEUE_FAMILY_IGNORED
        ))
        self.resource_data.transfer_slice_to_gpu(self.current_slice, vk_cmdList)

    def as_numpy(self, buffer_description: np.dtype = None):
        return self.resource_data.get_numpy_views(self.current_slice, buffer_description)


class SamplerWrapper:
    def __init__(self, w_device, vk_sampler):
        self.w_device = w_device
        self.vk_sampler = vk_sampler

    def __del__(self):
        vkDestroySampler(self.w_device.vk_device, self.vk_sampler, None)


class ShaderHandlerWrapper:
    def __init__(self):
        self.handle = None

    def get_handle(self):
        return self.handle

    def _set_handle(self, handle):
        self.handle = handle


class PipelineBindingWrapper:
    def __init__(self, w_device, pipeline_type):
        self.w_device = w_device
        self.pipeline_type = pipeline_type
        self.descriptor_sets_description = [[], [], [], []]
        self.active_set = 0
        self.active_descriptor_set = self.descriptor_sets_description[0]  # by default activate global set
        self.shaders = {}
        self.push_constants = { }
        self.rt_shaders = []  # shader stages
        self.rt_groups = []  # groups for hits
        self.rt_group_handles = []  # handles computed for each group
        self.max_recursion_depth = 1
        self.initialized = False
        self.descriptor_set_layouts = []
        self.descriptor_sets = None
        self.descriptor_pool = None
        self.pipeline_object = None
        self.pipeline_layout = None
        self.current_cmdList = None
        if self.pipeline_type == VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO:
            self.point = VK_PIPELINE_BIND_POINT_COMPUTE
        elif self.pipeline_type == VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO:
            self.point = VK_PIPELINE_BIND_POINT_GRAPHICS
        else:
            self.point = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR

    @trace_destroying
    def __del__(self):
        # Destroy desciptor sets
        vkDestroyDescriptorPool(self.w_device.vk_device, self.descriptor_pool, None)
        # Destroy layouts
        [vkDestroyDescriptorSetLayout(self.w_device.vk_device, dl, None) for dl in self.descriptor_set_layouts]
        # Destroy pipeline layout
        if self.pipeline_layout:
            vkDestroyPipelineLayout(self.w_device.vk_device, self.pipeline_layout, None)
        # Destroy pipeline object
        if self.pipeline_object:
            vkDestroyPipeline(self.w_device.vk_device, self.pipeline_object, None)
        self.descriptor_sets_description = [[], [], [], []]
        self.active_set = 0
        self.active_descriptor_set = None
        self.shaders = {}
        self.descriptor_set_layouts = []
        self.descriptor_sets = None
        self.descriptor_pool = None
        self.pipeline_object = None
        self.pipeline_layout = None
        self.current_cmdList = None
        self.w_device = None

    def descriptor_set(self, set_slot):
        assert not self.initialized, "Error, can not continue pipeline setup after initialized"
        self.active_set = set_slot
        self.active_descriptor_set = self.descriptor_sets_description[set_slot]

    def binding(self, slot, vk_stage, vk_descriptor_type, count, resolver):
        assert not self.initialized, "Error, can not continue pipeline setup after initialized"
        self.active_descriptor_set.append(
            (slot, vk_stage, vk_descriptor_type, count, resolver)
        )

    def add_constant_range(self, stage, field_name, offset, field_size, field_type):
        self.push_constants[field_name] = (offset, stage, field_size, field_type, bytearray(field_size))

    def load_shader(self, w_shader) -> int:
        assert not self.initialized, "Error, can not continue pipeline setup after initialized"
        if self.pipeline_type == VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR:
            self.rt_shaders.append(w_shader)
            return len(self.rt_shaders)-1
        else:
            self.shaders[w_shader.vk_stage_info.stage] = w_shader
            return 0

    def create_hit_group(self,
                         closest_hit_index: int = None,
                         any_hit_index: int = None,
                         intersection_index: int = None):
        type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR \
            if intersection_index is None \
            else VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
        self.rt_groups.append(VkRayTracingShaderGroupCreateInfoKHR(
            type=type,
            generalShader=VK_SHADER_UNUSED_KHR,
            closestHitShader=VK_SHADER_UNUSED_KHR if closest_hit_index is None else closest_hit_index,
            anyHitShader=VK_SHADER_UNUSED_KHR if any_hit_index is None else any_hit_index,
            intersectionShader=VK_SHADER_UNUSED_KHR if intersection_index is None else intersection_index
        ))
        s = ShaderHandlerWrapper()
        self.rt_group_handles.append(s)
        return s

    def create_general_group(self, shader_index: int):
        type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR
        self.rt_groups.append(VkRayTracingShaderGroupCreateInfoKHR(
            type=type,
            generalShader=shader_index,
            closestHitShader=VK_SHADER_UNUSED_KHR,
            anyHitShader=VK_SHADER_UNUSED_KHR,
            intersectionShader=VK_SHADER_UNUSED_KHR
        ))
        s = ShaderHandlerWrapper()
        self.rt_group_handles.append(s)
        return s

    def set_max_recursion(self, depth):
        self.max_recursion_depth = depth

    def _build_objects(self):
        assert not self.initialized, "Error, can not continue pipeline setup after initialized"
        # Builds the descriptor sets layouts
        descriptor_set_layout_bindings = [[], [], [], []]
        descriptor_set_layout_bindings_bound = [[],[],[],[]]
        counting_by_type = {
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: 1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: 1,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: 1,
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: 1,
        }
        var_count_info = []
        for level in range(4):
            has_variable_descriptor = False
            for slot, vk_stage, vk_descriptor_type, count, resolver in self.descriptor_sets_description[level]:
                effect_count = 100 if count == -1 else count
                lb = VkDescriptorSetLayoutBinding(
                    slot,
                    vk_descriptor_type,
                    effect_count, # for unbound descriptor set array
                    vk_stage
                )
                # bound = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT \
                bound = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT \
                    if count == -1 else \
                    0
                if count == -1:
                    has_variable_descriptor = True
                descriptor_set_layout_bindings[level].append(lb)
                descriptor_set_layout_bindings_bound[level].append(bound)
                counting_by_type[vk_descriptor_type] += effect_count * self.w_device.get_number_of_frames()
            var_count_info.append(0 if not has_variable_descriptor else 100)
        self.descriptor_pool = vkCreateDescriptorPool(self.w_device.vk_device, VkDescriptorPoolCreateInfo(
            maxSets=4 * self.w_device.get_number_of_frames(),
            poolSizeCount=6,
            pPoolSizes=[VkDescriptorPoolSize(t, c) for t, c in counting_by_type.items()]
        ), pAllocator=None)

        self.descriptor_set_layouts = []
        for level, lb in enumerate(descriptor_set_layout_bindings):
            bound_info = VkDescriptorSetLayoutBindingFlagsCreateInfo(
                pBindingFlags=descriptor_set_layout_bindings_bound[level],
                bindingCount=len(descriptor_set_layout_bindings_bound[level])
            )
            dslci = VkDescriptorSetLayoutCreateInfo(
                pNext = bound_info,
                bindingCount=len(lb),
                pBindings=lb
            )
            self.descriptor_set_layouts.append(
                vkCreateDescriptorSetLayout(self.w_device.vk_device,
                                        dslci, None))

        # Builds the descriptor sets (one group for each frame)
        descriptor_set_layouts_per_frame = self.descriptor_set_layouts * self.w_device.get_number_of_frames()
        var_count_info = var_count_info * self.w_device.get_number_of_frames()
        var_count = VkDescriptorSetVariableDescriptorCountAllocateInfo(
            descriptorSetCount=0, # len(var_count_info),
            pDescriptorCounts=var_count_info,
        )
        allocate_info = VkDescriptorSetAllocateInfo(
            pNext = var_count,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=len(descriptor_set_layouts_per_frame),
            pSetLayouts=descriptor_set_layouts_per_frame)
        self.descriptor_sets = vkAllocateDescriptorSets(self.w_device.vk_device, allocate_info)
        # Builds pipeline object
        push_constant_ranges = [VkPushConstantRange(
            size=size,
            offset=offset,
            stageFlags=stage
        ) for k, (offset, stage, size, type, data) in self.push_constants.items()]
        pipeline_layout_info = VkPipelineLayoutCreateInfo(
            setLayoutCount=4,
            pSetLayouts=self.descriptor_set_layouts,
            pushConstantRangeCount=len(push_constant_ranges),
            pPushConstantRanges=push_constant_ranges
        )
        self.pipeline_layout = vkCreatePipelineLayout(self.w_device.vk_device, pipeline_layout_info, None)
        if self.pipeline_type == VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO:
            assert VK_SHADER_STAGE_COMPUTE_BIT in self.shaders, "Error, no compute shader bound!"
            self.pipeline_object = vkCreateComputePipelines(self.w_device.vk_device, VK_NULL_HANDLE, 1,
                                                            [
                                                                VkComputePipelineCreateInfo(
                                                                    layout=self.pipeline_layout,
                                                                    stage=self.shaders[
                                                                        VK_SHADER_STAGE_COMPUTE_BIT
                                                                    ].vk_stage_info
                                                                )], None)[0]
        elif self.pipeline_type == VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO:
            pass
        elif self.pipeline_type == VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR:
            self.pipeline_object = self.w_device.vkCreateRayTracingPipelines(
                self.w_device.vk_device, VK_NULL_HANDLE,
                VK_NULL_HANDLE, 1, VkRayTracingPipelineCreateInfoKHR(
                                        layout=self.pipeline_layout,
                                        stageCount=len(self.rt_shaders),
                                        pStages=[s.vk_stage_info for s in self.rt_shaders],
                                        groupCount=len(self.rt_groups),
                                        pGroups=self.rt_groups,
                                        maxPipelineRayRecursionDepth=self.max_recursion_depth
                                    ), None)[0]
            shader_handle_size = self.w_device.raytracing_properties.shaderGroupHandleSize
            all_handles = bytearray(shader_handle_size * len(self.rt_groups))
            self.w_device.vkGetRayTracingShaderGroupHandles(
                self.w_device.vk_device,
                self.pipeline_object,
                0,
                len(self.rt_groups),
                len(self.rt_groups)*shader_handle_size,
                ffi.from_buffer(all_handles)
            )
            for i, s in enumerate(self.rt_group_handles):
                s._set_handle(all_handles[i*shader_handle_size:(i+1)*shader_handle_size])
        self.initialized = True

    def _set_at_cmdList(self, vk_cmdList, queue_index):
        frame = self.w_device.get_frame_index()
        self.current_cmdList = vk_cmdList
        self.current_queue_index = queue_index
        vkCmdBindPipeline(vk_cmdList, self.point, self.pipeline_object)
        vkCmdBindDescriptorSets(
            commandBuffer=self.current_cmdList,
            pipelineBindPoint=self.point,
            layout=self.pipeline_layout,
            firstSet=0,
            descriptorSetCount=4,
            pDescriptorSets=self.descriptor_sets[frame * 4: frame * 4 + 4],
            dynamicOffsetCount=0,
            pDynamicOffsets=None
        )

    def _solve_resolver_as_buffers(self, buffer: ResourceWrapper, vk_descriptor_type):
        if buffer is None:
            # NULL DESCRIPTOR
            return VkDescriptorBufferInfo(
                buffer=None,
                offset=0,
                range=UINT64_MAX
            )
        if vk_descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            return None
        else:
            return VkDescriptorBufferInfo(
                buffer=buffer.resource_data.vk_resource,
                offset=buffer.current_slice["offset"],
                range=buffer.current_slice["size"]
            )

    __SHADER_STAGE_2_PIPELINE_STAGE = {
        VK_SHADER_STAGE_VERTEX_BIT: VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
        VK_SHADER_STAGE_FRAGMENT_BIT: VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_SHADER_STAGE_COMPUTE_BIT: VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_SHADER_STAGE_MISS_BIT_KHR: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_SHADER_STAGE_INTERSECTION_BIT_KHR: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR: VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    }

    def _solve_resolver_as_image(self, t, vk_descriptor_type, vk_shader_stage):

        vk_stage = PipelineBindingWrapper.__SHADER_STAGE_2_PIPELINE_STAGE[vk_shader_stage]

        if vk_descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            image, sampler = t
        else:
            image = t
            sampler = None

        if vk_descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE or \
           vk_descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            vk_access = VK_ACCESS_SHADER_READ_BIT
            vk_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        else:
            if image.w_resource.is_readonly:
                vk_access = VK_ACCESS_SHADER_READ_BIT
            else:
                vk_access = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
            vk_layout = VK_IMAGE_LAYOUT_GENERAL

        image.w_resource.add_barrier(self.current_cmdList, ResourceState(
            vk_access=vk_access,
            vk_layout=vk_layout,
            vk_stage=vk_stage,
            queue_index=self.current_queue_index
        ))
        return VkDescriptorImageInfo(
            imageView=image.w_resource.get_view(),
            imageLayout=vk_layout,
            sampler=None if sampler is None else sampler.vk_sampler
        )

    def _update_level(self, level):
        frame = self.w_device.get_frame_index()
        descriptorWrites = []
        for slot, vk_stage, vk_descriptor_type, count, resolver in self.descriptor_sets_description[level]:
            is_buffer = vk_descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER or \
                        vk_descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER or\
                        vk_descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
            next = None
            if vk_descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:  # set next pointer
                next = VkWriteDescriptorSetAccelerationStructureKHR(
                    accelerationStructureCount=1,
                    pAccelerationStructures=[resolver()[0].w_resource.resource_data.ads]
                )
            if is_buffer:
                descriptors = [self._solve_resolver_as_buffers(b.w_resource if b else None, vk_descriptor_type) for b in resolver()]
            else:
                descriptors = [self._solve_resolver_as_image(t, vk_descriptor_type, vk_stage) for t in resolver()]
            dw = VkWriteDescriptorSet(
                pNext=next,
                dstSet=self.descriptor_sets[frame*4 + level],
                dstBinding=slot,
                descriptorType=vk_descriptor_type,
                descriptorCount=len(descriptors),
                pBufferInfo= descriptors if is_buffer and not next else None,
                pImageInfo= descriptors if not is_buffer else None
            )
            descriptorWrites.append(dw)
        vkUpdateDescriptorSets(
            device=self.w_device.vk_device,
            descriptorWriteCount=len(descriptorWrites),
            pDescriptorWrites=descriptorWrites,
            descriptorCopyCount=0,
            pDescriptorCopies=None
        )



class CommandListState(Enum):
    NONE = 0
    INITIAL = 1
    RECORDING = 2
    EXECUTABLE = 3
    SUBMITTED = 4
    FINISHED = 5


class CommandBufferWrapper:
    """
    Wrapper for a command list.
    """

    def __init__(self, vk_cmdList, pool):
        self.vk_cmdList = vk_cmdList
        self.pool : CommandPoolWrapper = pool
        self.__is_frozen = False
        self.state = CommandListState.INITIAL
        self.current_pipeline = None
        self.shader_groups_size = self.pool.device.raytracing_properties.shaderGroupHandleSize

    @trace_destroying
    def __del__(self):
        self.current_pipeline = None
        self.current_program = None
        self.pool = None

    vkCmdBuildAccelerationStructures=None
    vkBuildAccelerationStructures = None
    vkCmdTraceRays=None

    def begin(self):
        assert self.state == CommandListState.INITIAL, "Error, to begin a cmdList should be in initial state"
        vkBeginCommandBuffer(self.vk_cmdList, VkCommandBufferBeginInfo())
        self.state = CommandListState.RECORDING

    def end(self):
        assert self.state == CommandListState.RECORDING, "Error, to end a cmdList should be in recording"
        vkEndCommandBuffer(self.vk_cmdList)
        self.state = CommandListState.EXECUTABLE
        # self.current_pipeline = None

    def reset(self):
        assert not self.__is_frozen, "Can not reset a frozen cmdList"
        assert self.state == CommandListState.EXECUTABLE, "Error, to reset a cmdList should be in executable state"
        vkResetCommandBuffer(self.vk_cmdList, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)
        self.state = CommandListState.INITIAL

    def flush_and_wait(self):
        self.pool.flush([self])

    def freeze(self):
        if self.is_frozen():
            return
        self.end()
        self.__is_frozen = True

    def close(self):
        self.end()
        self.state = CommandListState.FINISHED

    def is_closed(self):
        return self.state == CommandListState.FINISHED

    def is_frozen(self):
        return self.__is_frozen

    def clear_color(self, w_image: ResourceWrapper, color):
        if not isinstance(color, list) and not isinstance(color, tuple):
            color = list(color)

        image = w_image.resource_data.vk_resource

        new_access, new_stage, new_layout, new_queue = w_image.resource_data.current_state
        if new_layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL or \
                new_layout != VK_IMAGE_LAYOUT_GENERAL or \
                new_layout != VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR:
            new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL

        w_image.add_barrier(self.vk_cmdList, ResourceState(
            vk_access=new_access,
            vk_stage=new_stage,
            vk_layout=new_layout,
            queue_index=new_queue
        ))
        vkCmdClearColorImage(self.vk_cmdList, image, new_layout, VkClearColorValue(color), 1,
                             [ResourceWrapper.get_subresources(w_image.current_slice)])

    def from_gpu(self, w_resource: ResourceWrapper):
        w_resource.add_transfer_from_gpu(self.vk_cmdList)

    def to_gpu(self, w_resource: ResourceWrapper):
        w_resource.add_transfer_to_gpu(self.vk_cmdList)

    def copy_image(self, w_src: ResourceWrapper, w_dst: ResourceWrapper):
        w_src.add_barrier(self.vk_cmdList, ResourceState(
            vk_access=VK_ACCESS_TRANSFER_READ_BIT,
            vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk_layout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            queue_index=VK_QUEUE_FAMILY_IGNORED))
        w_dst.add_barrier(self.vk_cmdList, ResourceState(
            vk_access=VK_ACCESS_TRANSFER_WRITE_BIT,
            vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk_layout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            queue_index=VK_QUEUE_FAMILY_IGNORED))
        subresource_index = w_src.current_slice["array_start"] + w_src.current_slice["mip_start"] * w_src.resource_data.full_slice["array_count"]
        footprint, offset = w_src.resource_data.get_staging_footprint_and_offset(subresource_index)
        w, h, d = footprint.dim
        vkCmdCopyImage(self.vk_cmdList, w_src.resource_data.vk_resource, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       w_dst.resource_data.vk_resource, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, [
                VkImageCopy(
                    ResourceWrapper.get_subresource_layers(w_src.current_slice),
                    VkOffset3D(0,0,0),
                    ResourceWrapper.get_subresource_layers(w_dst.current_slice),
                    VkOffset3D(0,0,0), VkExtent3D(w, h, d)
                )
                       ])

    def set_pipeline(self, pipeline: PipelineBindingWrapper):
        self.current_pipeline = pipeline
        pipeline._set_at_cmdList(self.vk_cmdList, self.pool.queue_index)

    def update_constant(self, field_name, field_data):
        offset, stage, size, type, data = self.current_pipeline.push_constants[field_name]
        data[0:size] = field_data
        vkCmdPushConstants(
            self.vk_cmdList,
            self.current_pipeline.pipeline_layout,
            stage, offset, size, ffi.from_buffer(data)
        )

    def update_bindings_level(self, level):
        self.current_pipeline._update_level(level)

    def dispatch_groups(self, dimx, dimy, dimz):
        vkCmdDispatch(self.vk_cmdList, dimx, dimy, dimz)

    def _get_strided_device_address(self, w_resource: ResourceWrapper, stride):
        if w_resource is None:
            return w_resource
        address = self.pool.device._get_device_address(w_resource)
        return VkStridedDeviceAddressRegionKHR(
            address.deviceAddress, stride, w_resource.current_slice["size"]
        )

    def dispatch_rays(self,
                      w_raygen_table: ResourceWrapper,
                      w_raymiss_table: ResourceWrapper,
                      w_rayhit_table: ResourceWrapper,
                      dimx: int, dimy: int, dimz: int):
        CommandBufferWrapper.vkCmdTraceRays(
            self.vk_cmdList,
            self._get_strided_device_address(w_raygen_table, self.shader_groups_size),
            self._get_strided_device_address(w_raymiss_table, self.shader_groups_size),
            self._get_strided_device_address(w_rayhit_table, self.shader_groups_size),
            VkStridedDeviceAddressRegionKHR(),
            dimx, dimy, dimz
        )
        vkCmdPipelineBarrier(self.vk_cmdList, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, VkMemoryBarrier(
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT
            ), 0, 0, 0, 0)

    def build_ads(self,
                  w_ads: ResourceWrapper,
                  ads_info: VkAccelerationStructureBuildGeometryInfoKHR,
                  ads_ranges,
                  w_scratch: ResourceWrapper):
        vkCmdPipelineBarrier(self.vk_cmdList,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1,
                             VkMemoryBarrier(
                                 srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                                 dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                             ), 0, 0, 0, 0)

        build_info = VkAccelerationStructureBuildGeometryInfoKHR(
            type = ads_info.type,
            geometryCount=ads_info.geometryCount,
            pGeometries=ads_info.pGeometries,
            scratchData=self.pool.device._get_device_address(w_scratch),
            dstAccelerationStructure=w_ads.resource_data.ads,
            mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
        )
        CommandBufferWrapper.vkCmdBuildAccelerationStructures(
            self.vk_cmdList,
            # self.pool.vk_device,
            # None,
            1,
            build_info,
            [ads_ranges]
        )
        # vkCmdPipelineBarrier(self.vk_cmdList,
        #                      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        #                      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        #                      0, 1,
        #                          VkMemoryBarrier(
        #                         srcAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        #                         dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
        #                         ), 0, 0, 0, 0)


class ShaderStageWrapper:

    def __init__(self, vk_stage, main_function, module):
        self.vk_stage_info = VkPipelineShaderStageCreateInfo(
            stage=vk_stage,
            module=module,
            pName=main_function
        )

    @staticmethod
    def from_file(device, vk_stage, path, main_function):
        vk_device = device.vk_device
        if (vk_stage, path, main_function) not in device.loaded_modules:
            with open(path, mode='rb') as f:
                bytecode = f.read(-1)
                info = VkShaderModuleCreateInfo(
                    codeSize=len(bytecode),
                    pCode=bytecode
                )
                device.loaded_modules[(vk_stage, path, main_function)] = vkCreateShaderModule(
                    device=vk_device,
                    pCreateInfo=info,
                    pAllocator=None
                )
        return ShaderStageWrapper(
            vk_stage,
            main_function,
            device.loaded_modules[(vk_stage, path, main_function)])


class CommandPoolWrapper:
    def __init__(self, device, vk_queue, vk_pool, queue_index):
        self.device = device
        self.vk_device = device.vk_device
        self.vk_pool = vk_pool
        self.vk_queue = vk_queue
        self.attached = []  # attached CommandBufferWrapper can be automatically flushed
        self.reusable = []  # commandlists have been submitted and finished that can be reused
        self.queue_index = queue_index

    @trace_destroying
    def __del__(self):
        if len(self.reusable) > 0:
            vkFreeCommandBuffers(self.vk_device, self.vk_pool, len(self.reusable), self.reusable)
        if self.vk_pool:
            vkDestroyCommandPool(self.vk_device, self.vk_pool, None)
        self.attached = []  # attached CommandBufferWrapper can be automatically flushed
        self.reusable = []  # commandlists have been submitted and finished that can be reused
        self.device = None


    def get_cmdList(self):
        """"
        Gets a new command buffer wrapper
        """
        if len(self.reusable) != 0:
            cmdList = self.reusable.pop()
        else:
            # allocate a new one
            cmdList = vkAllocateCommandBuffers(self.vk_device, VkCommandBufferAllocateInfo(
                commandPool=self.vk_pool,
                level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1))[0]
        cmdList_wrapper = CommandBufferWrapper(cmdList, self)
        self.attached.append(cmdList_wrapper)
        cmdList_wrapper.begin()
        return cmdList_wrapper

    def freeze(self, manager):
        manager.freeze()
        self.attached.remove(manager)

    def flush(self, managers=None):
        if managers is None:  # flush all attached (pending) buffers
            managers = self.attached
            self.attached = []
        else:
            for m in managers:
                if not m.is_frozen():
                    self.attached.remove(m)  # remove from attached

        if len(managers) == 0:
            return  # finished task

        cmdLists = []
        for m in managers:
            if m.state == CommandListState.SUBMITTED:
                raise Exception(
                    "Error! submitting a frozen command list already on gpu. Use wait for sync."
                )
            if m.state == CommandListState.FINISHED:
                raise Exception(
                    "Error! submitting a command list already executed. Use a frozen cmdList to repeat submissions."
                )
            if m.state == CommandListState.RECORDING:
                m.end()
            assert m.state == CommandListState.EXECUTABLE, "Error in command list state"
            cmdLists.append(m.vk_cmdList)
            m.state = CommandListState.SUBMITTED

        vkQueueSubmit(self.vk_queue, 1,

                          VkSubmitInfo(
                              commandBufferCount=len(cmdLists),
                              pCommandBuffers=cmdLists
                          )
                      , None)

        vkQueueWaitIdle(self.vk_queue)
        for c in managers:
            if c.is_frozen():
                c.state = CommandListState.EXECUTABLE
            else:
                vkResetCommandBuffer(c.vk_cmdList, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)
                self.reusable.append(c.vk_cmdList)
                c.state = CommandListState.FINISHED


class Event(Enum):
    NONE = 0
    CLOSED = 1


class WindowWrapper:
    def poll_events(self) -> (Event, object):
        pass


class DeviceWrapper:
    def __init__(self, width, height, format, mode, render_usage, enable_validation_layers):
        self.width = width
        self.height = height
        self.__render_target_format = format
        self.mode = mode
        self.enable_validation_layers = enable_validation_layers
        self.render_target_usage = render_usage

        self.vk_device = None
        self.__window = None
        self.__instance = None
        self.__callback = None
        self.__surface = None
        self.__physical_device = None
        self.__swapchain = None
        self.__queues = []  # list of queue objects for each queue family (only one queue used for family)
        self.__queue_index = {}  # map from QueueType to queue index covering the functionalities
        self.__main_manager = None  # Main rendering queue used for presenting
        self.__main_queue_index = -1
        self.__managers = []  # rendering wrappers for each queue type
        self.__render_target_ready = []  # semaphores
        self.__render_target_rendered = [] # semaphores
        self.__render_targets = []
        self.loaded_modules = {} # Cache of all loaded shader modules
        self.__frame_index = 0  # Current Frame
        self.__render_target_index = 0  # Current RT index in swapchain (not necessary the same of frame_index).
        self.__createInstance()
        self.__load_vk_calls()
        self.__createDebugInstance()
        self.__createSurface()
        self.__createPhysicalDevice()
        self.__createQueues()
        self.__createSwapchain()

    @trace_destroying
    def __del__(self):
        vkDeviceWaitIdle(self.vk_device)

        self.__queues = []  # list of queue objects for each queue family (only one queue used for family)
        self.__main_manager = None  # Main rendering queue used for presenting

        # release queue managers
        self.__managers = []  # rendering wrappers for each queue type

        # release render target image views
        self.__render_targets = []

        # semaphores
        [vkDestroySemaphore(self.vk_device, sem, None) for sem in self.__render_target_ready]
        [vkDestroySemaphore(self.vk_device, sem, None) for sem in self.__render_target_rendered]
        self.__render_target_ready = []
        self.__render_target_rendered = []

        # destroy shader module cache
        for k, v in self.loaded_modules.items():
            vkDestroyShaderModule(self.vk_device, v, None)
        self.loaded_modules = {}

        if self.__swapchain:
            self.vkDestroySwapchainKHR(self.vk_device, self.__swapchain, None)
        if self.vk_device:
            vkDestroyDevice(self.vk_device, None)
        if self.__callback:
            self.vkDestroyDebugReportCallbackEXT(self.__instance, self.__callback, None)
        if self.__surface:
            self.vkDestroySurfaceKHR(self.__instance, self.__surface, None)
        if self.__instance:
            vkDestroyInstance(self.__instance, None)
        print('[INFO] Destroyed vulkan instance')

    def __load_vk_calls(self):
        self.vkCreateSwapchainKHR = vkGetInstanceProcAddr(self.__instance, 'vkCreateSwapchainKHR')
        self.vkGetSwapchainImagesKHR = vkGetInstanceProcAddr(self.__instance, 'vkGetSwapchainImagesKHR')
        self.vkGetPhysicalDeviceSurfaceSupportKHR = vkGetInstanceProcAddr(
            self.__instance, 'vkGetPhysicalDeviceSurfaceSupportKHR')
        self.vkCreateDebugReportCallbackEXT = vkGetInstanceProcAddr(
            self.__instance, "vkCreateDebugReportCallbackEXT")
        self.vkQueuePresentKHR = vkGetInstanceProcAddr(self.__instance, "vkQueuePresentKHR")
        self.vkAcquireNextImageKHR = vkGetInstanceProcAddr(self.__instance, "vkAcquireNextImageKHR")
        self.vkDestroyDebugReportCallbackEXT = vkGetInstanceProcAddr(self.__instance, "vkDestroyDebugReportCallbackEXT")
        self.vkDestroySwapchainKHR = vkGetInstanceProcAddr(self.__instance, "vkDestroySwapchainKHR")
        self.vkDestroySurfaceKHR = vkGetInstanceProcAddr(self.__instance, "vkDestroySurfaceKHR")

        self.vkCreateRayTracingPipelines = vkGetInstanceProcAddr(self.__instance, "vkCreateRayTracingPipelinesKHR")
        self.vkGetRayTracingShaderGroupHandles = \
            vkGetInstanceProcAddr(self.__instance, "vkGetRayTracingShaderGroupHandlesKHR")
        CommandBufferWrapper.vkCmdTraceRays = vkGetInstanceProcAddr(self.__instance, "vkCmdTraceRaysKHR")
        self.vkGetAccelerationStructureDeviceAddress = \
            vkGetInstanceProcAddr(self.__instance, "vkGetAccelerationStructureDeviceAddressKHR")
        self.vkCreateAccelerationStructureKHR = vkGetInstanceProcAddr(self.__instance, "vkCreateAccelerationStructureKHR")
        self.vkGetAccelerationStructureBuildSizesKHR = vkGetInstanceProcAddr(
            self.__instance, "vkGetAccelerationStructureBuildSizesKHR")
        CommandBufferWrapper.vkCmdBuildAccelerationStructures = \
            vkGetInstanceProcAddr(self.__instance, "vkCmdBuildAccelerationStructuresKHR")
        CommandBufferWrapper.vkBuildAccelerationStructures = \
            vkGetInstanceProcAddr(self.__instance, "vkBuildAccelerationStructuresKHR")
        self.vkGetPhysicalDeviceProperties2=vkGetInstanceProcAddr(self.__instance, "vkGetPhysicalDeviceProperties2KHR")
        ResourceData.vkDestroyAccelerationStructure = vkGetInstanceProcAddr(self.__instance,
                                                                                 "vkDestroyAccelerationStructureKHR")

    def __createSwapchain(self):
        self.__render_target_extent = VkExtent2D(self.width, self.height)

        if self.mode == 1:  # Offline rendering
            rt = self.create_image(
                VK_IMAGE_TYPE_2D,
                self.__render_target_format,
                0,
                VkExtent3D(self.width, self.height, 1),
                1, 1, False, VK_IMAGE_LAYOUT_UNDEFINED,
                self.render_target_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            )
            self.__render_targets = [rt]
        else:
            swapchain_create = VkSwapchainCreateInfoKHR(
                sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                flags=0,
                surface=self.__surface,
                minImageCount=2,
                imageFormat=self.__render_target_format,
                imageColorSpace=VK_COLORSPACE_SRGB_NONLINEAR_KHR,
                imageExtent=self.__render_target_extent,
                imageArrayLayers=1,
                imageUsage=self.render_target_usage,
                imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount=0,
                pQueueFamilyIndices=None,
                compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=VK_PRESENT_MODE_MAILBOX_KHR,
                clipped=VK_TRUE,
                oldSwapchain=None,
                preTransform=VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
            self.__swapchain = self.vkCreateSwapchainKHR(self.vk_device, swapchain_create, None)
            images = self.vkGetSwapchainImagesKHR(self.vk_device, self.__swapchain)
            self.__render_targets = []
            self.__render_target_ready = []
            self.__render_target_rendered = []
            for img in images:
                # wrap swapchain image
                rt_data = ResourceData(self, vk_description=VkImageCreateInfo(
                        imageType=VK_IMAGE_TYPE_2D,
                        mipLevels=1,
                        arrayLayers=1,
                        extent=VkExtent3D(self.width, self.height, 1),
                        format=self.__render_target_format
                    ),
                    vk_properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    vk_resource=img,
                    vk_resource_memory=None,
                    is_buffer=False,
                    initial_state=ResourceState(
                        vk_access=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        vk_stage=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        vk_layout=VK_IMAGE_LAYOUT_UNDEFINED,
                        queue_index=VK_QUEUE_FAMILY_IGNORED
                    ))
                rt = ResourceWrapper(rt_data)
                self.__render_targets.append(rt)
                # create semaphore for active render target
                self.__render_target_ready.append(vkCreateSemaphore(self.vk_device, VkSemaphoreCreateInfo(), None))
                self.__render_target_rendered.append(vkCreateSemaphore(self.vk_device, VkSemaphoreCreateInfo(), None))

            # Build command buffers for transitioning render target
            self.__presenting = vkAllocateCommandBuffers(
                self.vk_device,
                VkCommandBufferAllocateInfo(
                    commandPool=self.__main_manager.vk_pool,
                    level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=len(self.__render_targets)
                ))
            print(f"[INFO] Swapchain created with {len(images)} images...")

    def __createQueues(self):
        queue_families = vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice=self.__physical_device)
        print("[INFO] %s available queue family" % len(queue_families))

        def get_queue_index(vk_bits):
            min_index = -1
            min_bits = 10000
            for i, queue_family in enumerate(queue_families):
                if queue_family.queueCount > 0 and (queue_family.queueFlags & vk_bits == vk_bits):
                    if min_bits > queue_family.queueFlags:
                        min_bits = queue_family.queueFlags
                        min_index = i
            return min_index

        # Preload queue families
        for bits in range(1, (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT) + 1):
            self.__queue_index[bits] = get_queue_index(bits)
        # Create a single queue for each family
        queues_create = [VkDeviceQueueCreateInfo(sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                                 queueFamilyIndex=i,
                                                 queueCount=min(1, qf.queueCount),
                                                 pQueuePriorities=[1],
                                                 flags=0)
                         for i, qf in enumerate(queue_families)]
        # Add here required extensions
        extensions = [
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            "VK_KHR_acceleration_structure",
            "VK_KHR_ray_tracing_pipeline",
            "VK_KHR_ray_query",
            "VK_KHR_pipeline_library",
            "VK_KHR_deferred_host_operations",
            "VK_KHR_buffer_device_address"
            # "GLSL_EXT_ray_tracing",
            # "GLSL_EXT_ray_query",
            # "GLSL_EXT_ray_flags_primitive_culling",
            # "SPV_KHR_ray_tracing",
            # "SPV_KHR_ray_query"
        ]

        dev_features = self.__physical_devices_features[self.__physical_device]

        rob_features = VkPhysicalDeviceRobustness2FeaturesEXT(
            nullDescriptor=True,
        )

        ads_features = VkPhysicalDeviceAccelerationStructureFeaturesKHR(
            pNext = rob_features,
            accelerationStructure=True,
            # accelerationStructureHostCommands=True,
            # descriptorBindingAccelerationStructureUpdateAfterBind=True,
        )

        rt_features = VkPhysicalDeviceRayTracingPipelineFeaturesKHR(
            pNext = ads_features,
            rayTracingPipeline=True,
        )

        features = VkPhysicalDeviceVulkan12Features(
            pNext=rt_features,
            bufferDeviceAddress=True,
            scalarBlockLayout=True,
            shaderSampledImageArrayNonUniformIndexing=True,
            runtimeDescriptorArray=True,
            descriptorBindingVariableDescriptorCount=True,
            descriptorBindingPartiallyBound=True,
        )

        device_create = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=VkPhysicalDeviceFeatures2(pNext=features, features=dev_features),
            pQueueCreateInfos=queues_create,
            queueCreateInfoCount=len(queues_create),
            # pEnabledFeatures=self.__physical_devices_features[self.__physical_device],
            flags=0,
            enabledLayerCount=len(self.__layers),
            ppEnabledLayerNames=self.__layers,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions
        )
        self.vk_device = vkCreateDevice(self.__physical_device, device_create, None)

        # load calls
        self.vkGetBufferDeviceAddress = vkGetDeviceProcAddr(self.vk_device, "vkGetBufferDeviceAddressKHR")


        self.__queues = [None if qf.queueCount == 0 else vkGetDeviceQueue(
            device=self.vk_device,
            queueFamilyIndex=i,
            queueIndex=0) for i, qf in enumerate(queue_families)]

        # resolve command list manager types for each queue
        self.__managers = []
        for i, qf in enumerate(queue_families):
            pool = vkCreateCommandPool(
                device=self.vk_device,
                pCreateInfo=VkCommandPoolCreateInfo(
                    queueFamilyIndex=i,
                    flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
                ),
                pAllocator=None)
            self.__managers.append(CommandPoolWrapper(self, self.__queues[i], pool, i))

        for i, qf in enumerate(queue_families):
            support_present = self.mode == 1 or self.vkGetPhysicalDeviceSurfaceSupportKHR(
                physicalDevice=self.__physical_device,
                queueFamilyIndex=i,
                surface=self.__surface)
            if qf.queueCount > 0 and (qf.queueFlags & VK_QUEUE_GRAPHICS_BIT) and support_present:
                self.__main_manager = self.__managers[i]
                self.__main_queue_index = i
        print("[INFO] Device and queues created...")

    def __createSurface(self):
        if self.mode == 2:  # SDL Window
            """
            SDL2 Initialization for windows support
            """
            import os
            os.environ["PYSDL2_DLL_PATH"] = "./third-party/SDL2"
            import sdl2
            import sdl2.ext

            if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
                raise Exception(sdl2.SDL_GetError())

            window = sdl2.SDL_CreateWindow(
                'Vulkan Window'.encode('ascii'),
                sdl2.SDL_WINDOWPOS_UNDEFINED,
                sdl2.SDL_WINDOWPOS_UNDEFINED, self.width, self.height, 0)

            wm_info = sdl2.SDL_SysWMinfo()
            sdl2.SDL_VERSION(wm_info.version)
            sdl2.SDL_GetWindowWMInfo(window, ctypes.byref(wm_info))

            if wm_info.subsystem == sdl2.SDL_SYSWM_WINDOWS:
                self.__extensions.append('VK_KHR_win32_surface')
            elif wm_info.subsystem == sdl2.SDL_SYSWM_X11:
                self.__extensions.append('VK_KHR_xlib_surface')
            elif wm_info.subsystem == sdl2.SDL_SYSWM_WAYLAND:
                self.__extensions.append('VK_KHR_wayland_surface')
            else:
                raise Exception("Platform not supported")

            class SDLWindow(WindowWrapper):
                def __init__(self):
                    self.event = sdl2.SDL_Event()

                def poll_events(self) -> (Event, object):
                    if sdl2.SDL_PollEvent(ctypes.byref(self.event)) != 0:
                        if self.event.type == sdl2.SDL_QUIT:
                            return Event.CLOSED, None
                    return Event.NONE, None

            self.__window = SDLWindow()

            def surface_xlib():
                print("Create Xlib surface")
                vkCreateXlibSurfaceKHR = vkGetInstanceProcAddr(self.__instance, "vkCreateXlibSurfaceKHR")
                surface_create = VkXlibSurfaceCreateInfoKHR(
                    sType=VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                    dpy=wm_info.info.x11.display,
                    window=wm_info.info.x11.window,
                    flags=0)
                return vkCreateXlibSurfaceKHR(self.__instance, surface_create, None)

            def surface_wayland():
                print("Create wayland surface")
                vkCreateWaylandSurfaceKHR = vkGetInstanceProcAddr(self.__instance, "vkCreateWaylandSurfaceKHR")
                surface_create = VkWaylandSurfaceCreateInfoKHR(
                    sType=VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
                    display=wm_info.info.wl.display,
                    surface=wm_info.info.wl.surface,
                    flags=0)
                return vkCreateWaylandSurfaceKHR(self.__instance, surface_create, None)

            def surface_win32():
                def get_instance(hWnd):
                    """Hack needed before SDL 2.0.6"""
                    from cffi import FFI
                    _ffi = FFI()
                    _ffi.cdef('long __stdcall GetWindowLongA(void* hWnd, int nIndex);')
                    _lib = _ffi.dlopen('User32.dll')
                    return _lib.GetWindowLongA(_ffi.cast('void*', hWnd), -6)

                print("[INFO] Windows surface created...")
                vkCreateWin32SurfaceKHR = vkGetInstanceProcAddr(self.__instance, "vkCreateWin32SurfaceKHR")
                surface_create = VkWin32SurfaceCreateInfoKHR(
                    sType=VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                    hinstance=get_instance(wm_info.info.win.window),
                    hwnd=wm_info.info.win.window,
                    flags=0)
                return vkCreateWin32SurfaceKHR(self.__instance, surface_create, None)

            surface_mapping = {
                sdl2.SDL_SYSWM_X11: surface_xlib,
                sdl2.SDL_SYSWM_WAYLAND: surface_wayland,
                sdl2.SDL_SYSWM_WINDOWS: surface_win32
            }

            self.__surface = surface_mapping[wm_info.subsystem]()

    def __createDebugInstance(self):
        def debug_callback(*args):
            print('DEBUG: ' + args[5] + ' ' + args[6])
            return 0

        debug_create = VkDebugReportCallbackCreateInfoEXT(
            sType=VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            flags=VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=debug_callback)
        self.__callback = self.vkCreateDebugReportCallbackEXT(self.__instance, debug_create, None)
        print('[INFO] Debug instance created...')

    def __createInstance(self):
        self.__layers = [l.layerName for l in vkEnumerateInstanceLayerProperties()]
        if 'VK_LAYER_KHRONOS_validation' in self.__layers:
            self.__layers = ['VK_LAYER_KHRONOS_validation']
        if 'VK_LAYER_LUNARG_standard_validation' in self.__layers:
            self.__layers = ['VK_LAYER_LUNARG_standard_validation']
        if self.enable_validation_layers and len(self.__layers) == 0:
            raise Exception("validation layers requested, but not layer available!")
        self.__extensions = [e.extensionName for e in vkEnumerateInstanceExtensionProperties(None)]
        if self.enable_validation_layers:
            self.__extensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)
        appInfo = VkApplicationInfo(
            # sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName='Rendering with Python VK',
            applicationVersion=VK_MAKE_VERSION(1, 2, 0),
            pEngineName='pyvulkan',
            engineVersion=VK_MAKE_VERSION(1, 2, 0),
            apiVersion=VK_MAKE_VERSION(1, 2, 0)
        )
        if self.enable_validation_layers:
            instanceInfo = VkInstanceCreateInfo(
                pApplicationInfo=appInfo,
                enabledLayerCount=len(self.__layers),
                ppEnabledLayerNames=self.__layers,
                enabledExtensionCount=len(self.__extensions),
                ppEnabledExtensionNames=self.__extensions
            )
        else:
            instanceInfo = VkInstanceCreateInfo(
                pApplicationInfo=appInfo,
                enabledLayerCount=0,
                enabledExtensionCount=len(self.__extensions),
                ppEnabledExtensionNames=self.__extensions
            )

        self.__instance = vkCreateInstance(instanceInfo, None)
        print("[INFO] Vulkan Instance created...")

    def __createPhysicalDevice(self):
        physical_devices = vkEnumeratePhysicalDevices(self.__instance)
        self.__physical_devices_features = {physical_device: vkGetPhysicalDeviceFeatures(physical_device)
                                            for physical_device in physical_devices}
        physical_devices_properties = {physical_device: vkGetPhysicalDeviceProperties(physical_device)
                                       for physical_device in physical_devices}
        self.__physical_device = physical_devices[0]
        ads_prop=VkPhysicalDeviceAccelerationStructurePropertiesKHR()
        rt_prop = VkPhysicalDeviceRayTracingPipelinePropertiesKHR(pNext=ads_prop)
        vk12_prop = VkPhysicalDeviceVulkan12Properties(pNext=rt_prop)
        prop = VkPhysicalDeviceProperties2(pNext=vk12_prop)
        self.vkGetPhysicalDeviceProperties2(self.__physical_device, prop)
        self.raytracing_properties = rt_prop
        major, minor = VK_VERSION_MAJOR(prop.properties.apiVersion), VK_VERSION_MINOR(prop.properties.apiVersion)
        print("[INFO] Available devices: %s" % [p.deviceName
                                                for p in physical_devices_properties.values()])
        print("[INFO] Selected device: %s\n" % physical_devices_properties[self.__physical_device].deviceName)

    def get_window(self):
        return self.__window

    def begin_frame(self):
        """
        Peek a render target from swap chain and set as current
        """
        if self.mode == 1:  # Offline rendering
            return

        self.__render_target_index = self.vkAcquireNextImageKHR(
            self.vk_device,
            self.__swapchain,
            1000000000,
            self.__render_target_ready[self.__frame_index],
            VK_NULL_HANDLE)

    def get_render_target(self, index):
        return self.__render_targets[index]

    def get_render_target_index(self):
        return self.__render_target_index

    def get_number_of_frames(self):
        return len(self.__render_targets)

    def get_frame_index(self):
        return self.__frame_index

    def flush_pending_and_wait(self):
        for m in self.__managers:
            m.flush()

    def end_frame(self):
        self.flush_pending_and_wait()
        if self.mode == 1:  # offline
            return

        # Finish transitioning the render target to present
        rt: ResourceWrapper = self.get_render_target(self.get_render_target_index())
        cmdList = self.__presenting[self.__render_target_index]  # cmdList for transitioning rt
        vkResetCommandBuffer(commandBuffer=cmdList, flags=VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)
        vkBeginCommandBuffer(cmdList, VkCommandBufferBeginInfo())
        rt.add_barrier(cmdList, ResourceState(
            vk_access=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
            vk_stage=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            vk_layout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            queue_index=VK_QUEUE_FAMILY_IGNORED
        ))
        vkEndCommandBuffer(cmdList)
        # Wait for completation
        submit_create = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmdList],
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.__render_target_ready[self.__frame_index]],
            pWaitDstStageMask=[VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self.__render_target_rendered[self.__frame_index]])
        vkQueueSubmit(self.__main_manager.vk_queue, 1, [submit_create], None)
        # Present render target
        present_create = VkPresentInfoKHR(
            sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.__render_target_rendered[self.__frame_index]],
            swapchainCount=1,
            pSwapchains=[self.__swapchain],
            pImageIndices=[self.__render_target_index],
            pResults=None)
        self.vkQueuePresentKHR(self.__main_manager.vk_queue, present_create)
        if self.enable_validation_layers:
            vkQueueWaitIdle(self.__main_manager.vk_queue)
        # update frame index
        self.__frame_index = (self.__frame_index + 1) % len(self.__render_targets)

    def create_cmdList(self, queue_bits):
        return self.__managers[self.__queue_index[queue_bits]].get_cmdList()

    def __findMemoryType(self, filter, properties):
        mem_properties = vkGetPhysicalDeviceMemoryProperties(self.__physical_device)

        for i, prop in enumerate(mem_properties.memoryTypes):
            if (filter & (1 << i)) and ((prop.propertyFlags & properties) == properties):
                return i

        raise Exception("failed to find suitable memory type!")

    def __resolve_initial_state(self, is_buffer, usage, properties):
        if usage == VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:
            return ResourceState(
                vk_access=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                vk_stage=VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                vk_layout=VK_IMAGE_LAYOUT_UNDEFINED,
                queue_index=VK_QUEUE_FAMILY_IGNORED
            )
        return ResourceState(
            vk_access=VK_ACCESS_TRANSFER_WRITE_BIT,
            vk_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk_layout=VK_IMAGE_LAYOUT_UNDEFINED,
            queue_index=VK_QUEUE_FAMILY_IGNORED
        )

    def create_buffer_data(self, size, usage, properties):
        info = VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )
        buffer = vkCreateBuffer(self.vk_device, info, None)
        mem_reqs = vkGetBufferMemoryRequirements(self.vk_device, buffer)
        allocInfo = VkMemoryAllocateInfo(allocationSize=mem_reqs.size,
                                         pNext=None if not usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT else VkMemoryAllocateFlagsInfo(flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT),
                                         memoryTypeIndex=self.__findMemoryType(mem_reqs.memoryTypeBits, properties))
        buffer_memory = vkAllocateMemory(self.vk_device, allocInfo, None)
        vkBindBufferMemory(self.vk_device, buffer, buffer_memory, 0)
        return ResourceData(self, info, properties, buffer, buffer_memory, True,
                               self.__resolve_initial_state(True, usage, properties))

    def create_buffer(self, size, usage, properties):
        return ResourceWrapper(resource_data=self.create_buffer_data(size, usage, properties))

    def _get_device_address(self, buffer):
        if buffer is None:
            return None
        device_address = self.vkGetBufferDeviceAddress(self.vk_device, VkBufferDeviceAddressInfo(
            buffer=buffer.resource_data.vk_resource
        ))
        add = VkDeviceOrHostAddressKHR()
        add.deviceAddress=device_address + buffer.current_slice["offset"]
        return add

    def _get_device_address_const(self, buffer):
        if buffer is None:
            return None
        device_address = self.vkGetBufferDeviceAddress(self.vk_device, VkBufferDeviceAddressInfo(
            buffer=buffer.resource_data.vk_resource
        ))
        add = VkDeviceOrHostAddressConstKHR()
        add.deviceAddress=device_address + buffer.current_slice["offset"]
        return add

    def _resolve_description(self, geometry_type, element_description):
        if geometry_type == VK_GEOMETRY_TYPE_TRIANGLES_KHR:
            v, v_stride, i, t = element_description
            max_vertex = v.get_size() // v_stride
            data = VkAccelerationStructureGeometryKHR(
                geometryType=geometry_type,
                geometry=VkAccelerationStructureGeometryDataKHR(
                triangles= VkAccelerationStructureGeometryTrianglesDataKHR(
                vertexFormat=VK_FORMAT_R32G32B32_SFLOAT,
                vertexData=self._get_device_address_const(v),
                vertexStride=v_stride,
                maxVertex=max_vertex,
                indexType=VK_INDEX_TYPE_UINT32,
                indexData=self._get_device_address_const(i),
                transformData=self._get_device_address_const(t)
            )))
            if i:
                primitives = i.get_size() // 4 // 3
            else:
                primitives = v.get_size() // v_stride // 3
        elif geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR:
            aabbs = element_description
            data = VkAccelerationStructureGeometryKHR(
                geometryType=geometry_type,
                geometry=VkAccelerationStructureGeometryDataKHR(
                aabbs=VkAccelerationStructureGeometryAabbsDataKHR(
                stride=24,
                data=self._get_device_address_const(aabbs)
            )))
            primitives = aabbs.get_size() // 24
        else:
            instances = element_description
            data = VkAccelerationStructureGeometryKHR(
                geometryType=geometry_type,
                geometry=VkAccelerationStructureGeometryDataKHR(
                instances=VkAccelerationStructureGeometryInstancesDataKHR(
                data=self._get_device_address_const(instances),
                arrayOfPointers=False
            )))
            primitives = instances.get_size() // 64

        range = VkAccelerationStructureBuildRangeInfoKHR(
            primitiveCount=primitives,
            primitiveOffset=0,
            transformOffset=0,
            firstVertex=0
        )
        return data, range

    def create_ads(self, geometry_type=0, descriptions=[]):
        """
        for triangles a description is a tuple in the form (v, v_stride, i, t). From there a rangeinfo can be extracted
        for aabbs a decsription is directly the aabb buffer
        for instances a description is directly the instance buffer
        """
        # Compute the required size of the buffer and the scratch buffer
        structure_type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR if geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR else VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
        datas, ranges = zip(*[list(self._resolve_description(geometry_type, d)) for d in descriptions])
        info = VkAccelerationStructureBuildGeometryInfoKHR(
            type=structure_type,
            mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            geometryCount=len(descriptions),
            pGeometries=datas
        )
        sizes = VkAccelerationStructureBuildSizesInfoKHR()
        self.vkGetAccelerationStructureBuildSizesKHR(self.vk_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                     info, [range.primitiveCount for range in ranges], sizes)
        # Create a buffer to store the ads
        ads_buffer = self.create_buffer(sizes.accelerationStructureSize * 2,
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        # Create object
        create_info = VkAccelerationStructureCreateInfoKHR(
            buffer=ads_buffer.resource_data.vk_resource,
            offset=0, size=sizes.accelerationStructureSize,
            type=structure_type
        )
        ads = self.vkCreateAccelerationStructureKHR(self.vk_device, create_info, None)
        ads_buffer.resource_data.bind_ads(ads)
        query_device_address_info = VkAccelerationStructureDeviceAddressInfoKHR(accelerationStructure=ads)
        device_address = self.vkGetAccelerationStructureDeviceAddress(self.vk_device, query_device_address_info)
        return ads_buffer, info, ranges, device_address, max(sizes.buildScratchSize, sizes.updateScratchSize)*2

    def create_image(self, image_type, image_format, flags, extent, mips, layers, linear,
                     initial_layout, usage, properties):
        info = VkImageCreateInfo(
            imageType=image_type,
            format=image_format,
            flags=flags,
            extent=extent,
            arrayLayers=layers,
            mipLevels=mips,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_LINEAR if linear else VK_IMAGE_TILING_OPTIMAL,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=initial_layout)
        image = vkCreateImage(self.vk_device, info, None)
        mem_reqs = vkGetImageMemoryRequirements(self.vk_device, image)
        allocInfo = VkMemoryAllocateInfo(allocationSize=mem_reqs.size,
                                         memoryTypeIndex=self.__findMemoryType(mem_reqs.memoryTypeBits, properties))
        image_memory = vkAllocateMemory(self.vk_device, allocInfo, None)
        vkBindImageMemory(self.vk_device, image, image_memory, 0)

        resource_data = ResourceData(self, info, properties, image, image_memory, False,
                     self.__resolve_initial_state(False, usage, properties))
        return ResourceWrapper(resource_data)

    def create_sampler(self, mag_filter, min_filter, mipmap_mode, address_U, address_V, address_W,
                       mip_LOD_bias, enable_anisotropy, max_anisotropy, enable_compare,
                       compare_op, min_LOD, max_LOD, border_color, use_unnormalized_coordinates
                       ):
        info = VkSamplerCreateInfo(
            magFilter=mag_filter,
            minFilter=min_filter,
            mipmapMode=mipmap_mode,
            addressModeU=address_U,
            addressModeV=address_V,
            addressModeW=address_W,
            mipLodBias=mip_LOD_bias,
            anisotropyEnable=enable_anisotropy,
            maxAnisotropy=max_anisotropy,
            compareEnable=enable_compare,
            compareOp=compare_op,
            minLod=min_LOD,
            maxLod=max_LOD,
            borderColor=border_color,
            unnormalizedCoordinates=use_unnormalized_coordinates
        )
        return SamplerWrapper(self, vkCreateSampler(self.vk_device, info, None))

    def create_pipeline(self, vk_pipeline_type):
        return PipelineBindingWrapper(w_device=self, pipeline_type=vk_pipeline_type)
