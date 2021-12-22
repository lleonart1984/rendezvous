from rendering.manager import *
import glm
from typing import *
from pywavefront import Wavefront
from PIL import Image


class RaytracingScene:

    __validation_token = object()
    @staticmethod
    def _get_validation_toke():
        return RaytracingScene.__validation_token

    def __init__(self, token):
        assert token == RaytracingScene.__validation_token
        self.scene_ads = None
        self.vertices = None
        self.indices = None
        self.transforms = None
        self.material_buffer = None
        self.textures = None
        self.geometry_descriptions = None
        self.instance_descriptions = None

    def get_scene_ads(self):
        return self.scene_ads

    def get_vertices(self):
        return self.vertices

    def get_indices(self):
        return self.indices

    def get_transforms(self):
        """
        Gives the transforms used by geometries
        """
        return self.transforms

    def get_material_buffer(self):
        """
        For each material, gives the data of such material
        """
        return self.material_buffer

    def get_textures(self):
        """
        List of all textures used in this scene
        """
        return self.textures

    def get_geometry_descriptions(self):
        """
        For each geometry in a geometry group,
        gives the start offset of the first vertex, the first index and the transform index
        """
        return self.geometry_descriptions

    def get_instance_descriptions(self):
        """
        For each instance,
        gives the start offset of the first geometry in geometry descriptions and the material index
        """
        return self.instance_descriptions


class SceneBuilder:

    def __init__(self, device: DeviceManager):
        self.device = device
        self.positions = []
        self.normals = []
        self.texcoords = []
        self.indices = []
        self.transforms = []
        self.geometries = []
        self.textures = []
        self.textures_index = {}
        self.materials = []
        self.instances = []
        self.geometries_in_instances = 0

    def get_vertex_offset(self):
        return len(self.positions)

    def add_vertex(self, pos: glm.vec3, nor: glm.vec3=glm.vec3(0.0,0.0,0.0), tex: glm.vec2=glm.vec2(0.0,0.0)):
        self.positions.append(pos)
        self.normals.append(nor)
        self.texcoords.append(tex)

    def add_indices(self, indices, offset:int=0):
        start = len(self.indices)
        self.indices.extend(map(lambda i: i + offset, indices))
        return start

    def add_transform(self, transform: glm.mat4x4):
        transform = SceneBuilder._cast_to_3x4(transform)
        self.transforms.append(transform)
        return len(self.transforms)-1

    def add_geometry(self, start, count, transform_index = -1):
        self.geometries.append((start, count, transform_index))
        return len(self.geometries)-1

    def add_geometry_obj(self, path: str, transform_id = -1):
        obj = Wavefront(path, strict=True, collect_faces=True, parse=True)
        vertex_offset = self.get_vertex_offset()
        _, val = next(iter(obj.materials.items()))
        v = val.vertices
        min_p = glm.vec3(100000, 100000, 100000)
        max_p = -glm.vec3(100000, 100000, 100000)
        if val.vertex_format == "N3F_V3F":
            vertex_size = 6
            for i in range(len(v) // 6):
                nx, ny, nz, px, py, pz = v[i * 6 + 0], v[i * 6 + 1], v[i * 6 + 2], v[i * 6 + 3], v[i * 6 + 4], \
                                                 v[i * 6 + 5]
                p = glm.vec3(px, py, pz)
                self.add_vertex(p, glm.vec3(nx, ny, nz))
                min_p = glm.min(min_p, p)
                max_p = glm.max(max_p, p)
        elif val.vertex_format == "T2F_N3F_V3F":
            vertex_size = 8
            for i in range(len(v) // 8):
                tu, tv, nx, ny, nz, px, py, pz = v[i * 8 + 0], v[i * 8 + 1], v[i * 8 + 2], v[i * 8 + 3], v[i * 8 + 4], \
                                                 v[i * 8 + 5], v[i * 8 + 6], v[i * 8 + 7]
                p = glm.vec3(px, py, pz)
                self.add_vertex(p, glm.vec3(nx, ny, nz), glm.vec2(tu, tv))
                min_p = glm.min(min_p, p)
                max_p = glm.max(max_p, p)

        max_comp = max(0.000001, max_p.x - min_p.x, max_p.y - min_p.y, max_p.z - min_p.z)
        scale = 1.0 / max_comp
        for i in range(len(v)//vertex_size):
            self.positions[i+vertex_offset] = (self.positions[i+vertex_offset] - (min_p + max_p)*0.5)*scale
        indices = list(range(len(v)//vertex_size))
        index_offset = self.add_indices(indices, vertex_offset)
        return self.add_geometry(index_offset, len(indices))

    def add_texture(self, name: str, data=None):
        index = len(self.textures)
        self.textures.append((name, data))
        self.textures_index[name] = index
        return index

    def get_texture_index(self, name):
        return self.textures_index[name]

    def add_material(self,
                     diffuse: glm.vec3 = glm.vec3(1,1,1),
                     opacity: float = 1,
                     specular: glm.vec3 = glm.vec3(1,1,1),
                     specular_power: float = 40,
                     emissive: glm.vec3 = glm.vec3(0,0,0),
                     refraction_index: float = 1,
                     diffuse_map: int = -1,
                     specular_map: int = -1,
                     bump_map: int = -1,
                     mask_map: int = -1,
                     illumination_model_mix: glm.vec4 = glm.vec4(1, 0, 0, 0)
                     ):
        self.materials.append(
            (
                diffuse,
                opacity,
                specular,
                specular_power,
                emissive,
                refraction_index,
                diffuse_map,
                specular_map,
                bump_map,
                mask_map,
                illumination_model_mix
            )
        )
        return len(self.materials)-1

    @staticmethod
    def _cast_to_3x4(transform: glm.mat4x4):
        return glm.mat3x4(
            transform[0][0], transform[1][0], transform[2][0], transform[3][0],
            transform[0][1], transform[1][1], transform[2][1], transform[3][1],
            transform[0][2], transform[1][2], transform[2][2], transform[3][2],
        )

    def add_instance(self,
                     geometries,
                     material_index: int = -1,
                     transform: glm.mat4x4 = glm.mat4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1),
                     mask:int=0xff):
        transform = SceneBuilder._cast_to_3x4(transform)
        self.instances.append(
            (self.geometries_in_instances, material_index, geometries, transform, mask)
        )
        self.geometries_in_instances += len(geometries)
        return len(self.instances)-1

    def _load_texture(self, path: str, data = None):
        image = Image.open(path)
        width, height = image.size
        image = image.convert("RGBA")
        data = image.getdata()
        texture = self.device.create_texure_2D(Format.UINT_RGBA_UNORM, width, height, 1, 1)
        bytes = bytearray()
        for r,g,b,a in data:
            bytes.append(r)
            bytes.append(g)
            bytes.append(b)
            bytes.append(a)
        texture.write(bytes)
        with self.device.get_copy() as man:
            man.cpu_to_gpu(texture)
        return texture

    def build_raytracing_scene(self):
        s: RaytracingScene = RaytracingScene(RaytracingScene._get_validation_toke())
        device: DeviceManager = self.device

        # create vertices
        if True:
            s.vertices = device.create_structured_buffer(
                count=len(self.positions),
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE | BufferUsage.VERTEX | BufferUsage.RAYTRACING_ADS_READ,
                memory=MemoryProperty.GPU,
                position=glm.vec3,
                normal=glm.vec3,
                texcoord=glm.vec2,
                tangent=glm.vec3,
                binormal=glm.vec3
            )
            vertex_stride = s.vertices.stride
            vertex_cpu_data = bytearray(len(self.positions) * vertex_stride)
            vertex_struct_format = "f"*(vertex_stride//4)
            offset = 0
            for i, (v, n, t) in enumerate(zip(self.positions, self.normals, self.texcoords)):
                vertex_cpu_data[offset:offset+vertex_stride] = struct.pack(vertex_struct_format, v.x, v.y, v.z, n.x, n.y, n.z, t.x, t.y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                offset += vertex_stride
            s.vertices.write(vertex_cpu_data)

        # create indices
        if True:
            s.indices = device.create_indices_buffer(
                count=len(self.indices),
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE | BufferUsage.INDEX | BufferUsage.RAYTRACING_ADS_READ,
                memory=MemoryProperty.GPU
            )
            index_cpu_data = struct.pack('i'*len(self.indices), *self.indices)
            s.indices.write(index_cpu_data)

        # create transforms
        if len(self.transforms) > 0:
            s.transforms = device.create_structured_buffer(
                count=len(self.transforms),
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE | BufferUsage.RAYTRACING_ADS_READ,
                memory=MemoryProperty.GPU,
                # Field
                matrix=glm.mat3x4
            )
            for i, t in enumerate(self.transforms):
                s.transforms[i].matrix = t

        # create materials
        if len(self.materials) > 0:
            s.material_buffer = device.create_structured_buffer(
                count=len(self.materials),
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE,
                memory=MemoryProperty.GPU,
                # Fields
                diffuse=glm.vec3,
                opacity=float,
                specular=glm.vec3,
                specular_power=float,
                emissive=glm.vec3,
                refraction_index=float,
                diff_map=int,
                spec_map=int,
                bump_map=int,
                mask_map=int,
                model=glm.vec4
            )
            for i in range(len(self.materials)):
                mat = s.material_buffer[i]
                mat.diffuse, mat.opacity, mat.specular, mat.specular_power\
                    , mat.emissive, mat.refraction_index\
                    , mat.diff_map, mat.spec_map, mat.bump_map, mat.mask_map\
                    , mat.model = self.materials[i]

        s.textures = [self._load_texture(p, data) for p, data in self.textures]

        to_build = []
        cached_geometry_groups = dict()
        # create geometry_descriptions and instance_decriptions
        # create bottom level ads, instance buffer and top level ads (scene_ads)

        instance_buffer = device.create_instance_buffer(len(self.instances),
                                                        memory=MemoryProperty.CPU_DIRECT)

        if True:
            s.geometry_descriptions = device.create_structured_buffer(
                count=self.geometries_in_instances,
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE,
                memory=MemoryProperty.GPU,
                # Fields
                start_vertex=int,
                start_index=int,
                transform_index=int
            )
            s.instance_descriptions = device.create_structured_buffer(
                count=len(self.instances),
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE,
                memory=MemoryProperty.GPU,
                # Fields
                start_geometry=int,
                material_index=int
            )
            index = 0
            for i, (start, material_index, geometries, instance_transform, mask) in enumerate(self.instances):
                if tuple(geometries) not in cached_geometry_groups:
                    geometry_collection = device.create_triangle_collection()
                    for geom_index in geometries:
                        index_start, index_count, transform_index = self.geometries[geom_index]
                        geometry_collection.append(
                            s.vertices,
                            s.indices.slice(index_start * 4, index_count * 4),
                            None if transform_index == -1 else s.transforms.slice(transform_index*48, 48)
                        )
                    geometry_ads = device.create_geometry_ads(geometry_collection)
                    cached_geometry_groups[tuple(geometries)] = geometry_ads
                    to_build.append(geometry_ads)

                for geom_index in geometries:
                    index_start, index_count, transform_index = self.geometries[geom_index]
                    gd = s.geometry_descriptions[index]
                    gd.start_vertex = 0  # TODO: check if can be used
                    gd.start_index = index_start
                    gd.transform_index = transform_index
                    index += 1

                instance = instance_buffer[i]
                instance.transform = instance_transform
                instance.mask = mask
                instance.offset = 0
                instance.flags = 0x00000001
                instance.geometry = cached_geometry_groups[tuple(geometries)]
                id = s.instance_descriptions[i]
                id.start_geometry = start
                id.material_index = material_index
            s.scene_ads = device.create_scene_ads(instance_buffer)
            to_build.append(s.scene_ads)

        scratch_buffers = [device.create_scratch_buffer(b) for b in to_build]

        with device.get_raytracing() as man:
            # Transfer all buffers to gpu
            man.cpu_to_gpu(instance_buffer)
            man.cpu_to_gpu(s.vertices)
            man.cpu_to_gpu(s.indices)
            man.cpu_to_gpu(s.transforms)
            man.cpu_to_gpu(s.material_buffer)
            man.cpu_to_gpu(s.geometry_descriptions)
            man.cpu_to_gpu(s.instance_descriptions)

        # BUILD all ADSs
        for i, ads in enumerate(to_build):
            with device.get_raytracing() as man:
                man.build_ads(ads, scratch_buffers[i])

        return s


class Camera:
    def __init__(self):
        self.position = glm.vec3(0,0,-2)
        self.target = glm.vec3(0,0,0)
        self.up = glm.vec3(0,1,0)
        self.fov = glm.pi()/4
        self.near_plane = 0.001
        self.far_plane = 1000

    def build_matrices(self, width, height):
        return glm.lookAt(self.position, self.target, self.up), glm.perspective(self.fov, height/width, self.near_plane, self.far_plane)

    def LookAt(self, target: glm.vec3):
        self.target = target
        return self

    def PositionAt(self, position: glm.vec3):
        self.position = position
        return self
