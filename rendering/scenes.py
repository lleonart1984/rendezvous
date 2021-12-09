import manager
from manager import *
import glm


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

    def __init__(self, device: manager.DeviceManager):
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

    def add_vertex(self, pos: glm.vec3, nor: glm.vec3=glm.vec3(0,0,0), tex: glm.vec2=glm.vec2(0,0,0)):
        self.positions.append(pos)
        self.normals.append(nor)
        self.texcoords.append(tex)

    def add_indices(self, indices, offset:int=0):
        start = len(self.indices)
        self.indices.extend(map(lambda i: i + offset, indices))
        return start

    def add_transform(self, transform: glm.mat3x4):
        self.transforms.append(transform)
        return len(self.transforms)-1

    def add_geometry(self, start, count, transform_index = -1):
        self.geometries.append((start, count, transform_index))
        return len(self.geometries)-1

    def add_texture(self, name: str, data=None):
        index = len(self.textures)
        self.textures.append((name, data))
        self.textures_index[name] = index

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

    def add_instance(self,
                     geometries,
                     material_index: int = -1,
                     transform:glm.mat3x4 = glm.mat3x4(1,0,0,0, 0,1,0,0, 0,0,1,0),
                     mask:int=0xff):
        self.instances.append(
            (self.geometries_in_instances, material_index, geometries, transform, mask)
        )
        self.geometries_in_instances += len(geometries)
        return len(self.instances)-1

    def build_raytracing_scene(self):
        s: RaytracingScene = RaytracingScene(RaytracingScene._get_validation_toke())
        device: manager.DeviceManager = self.device

        # create vertices
        if True:
            s.vertices = device.create_structured_buffer(
                count=len(self.positions),
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE | BufferUsage.RAYTRACING_ADS_READ,
                memory=MemoryProperty.GPU,
                # Fields
                position=glm.vec3,
                normal=glm.vec3,
                texcoord=glm.vec2,
                tangent=glm.vec3,
                binormal=glm.vec3
            )
            for i, (v, n, t) in enumerate(zip(self.positions, self.normals, self.texcoords)):
                vertex = s.vertices[i]
                vertex.position = v
                vertex.normal = n
                vertex.texcoord = t

        # create indices
        if True:
            s.indices = device.create_indices_buffer(
                count=len(self.indices),
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE | BufferUsage.RAYTRACING_ADS_READ,
                memory=MemoryProperty.GPU
            )
            for i, index in enumerate(self.indices):
                s.indices[i] = index

        # create transforms
        if True:
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
        if True:
            s.material_buffer = device.create_structured_buffer(
                count=len(self.materials),
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE,
                memory=MemoryProperty.GPU,
                # Fields
                diffuse=glm.vec3,
                opacity=glm.float32,
                specular=glm.vec3,
                specular_power=glm.float32,
                emissive=glm.vec3,
                refraction_index=glm.float32,
                diff_map=glm.int32,
                spec_map=glm.int32,
                bump_map=glm.int32,
                mask_map=glm.int32,
                model=glm.vec4
            )
            for i in range(len(self.materials)):
                mat = s.material_buffer[i]
                mat.diffuse, mat.opacity, mat.specular, mat.specular_power\
                    , mat.emissive, mat.refraction_index\
                    , mat.diff_map, mat.spec_map, mat.bump_map, mat.mask_map\
                    , mat.model = self.materials[i]

        to_build = []

        # create geometry_descriptions and instance_decriptions
        # create bottom level ads, instance buffer and top level ads (scene_ads)
        if True:
            s.geometry_descriptions = device.create_structured_buffer(
                count=self.geometries_in_instances,
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE,
                memory=MemoryProperty.GPU,
                start_vertex=glm.int32,
                start_index=glm.int32,
                transform_index=glm.int32
            )
            s.instance_descriptions = device.create_structured_buffer(
                count=len(self.instances),
                usage=BufferUsage.TRANSFER_DST | BufferUsage.STORAGE,
                memory=MemoryProperty.GPU,
                start_geometry=glm.int32,
                material_index=glm.int32
            )
            index = 0
            instance_buffer = device.create_instance_buffer(len(self.instances))
            for i, (start, material_index, geometries, instance_transform, mask) in enumerate(self.instances):
                geometry_collection = device.create_triangle_collection()
                for index_start, index_count, transform_index in self.geometries:
                    gd = s.geometry_descriptions[index]
                    gd.start_vertex = 0  # TODO: check if can be used
                    gd.start_index = index_start
                    gd.transform_index = transform_index
                    geometry_collection.append(
                        s.vertices,
                        s.indices.slice(index_start * 4, index_count * 4),
                        None if transform_index is None else s.transforms.slice(transform_index*48, 48)
                    )
                    index += 1
                geometry_ads = device.create_geometry_ads(geometry_collection)
                to_build.append(geometry_ads)  # for further build operation
                instance = instance_buffer[i]
                instance.transform = instance_transform
                instance.mask = mask
                instance.offset = 0
                instance.flags = 0
                instance.geometry = geometry_ads
                id = s.instance_descriptions[i]
                id.start_geometry = start
                id.material_index = material_index
            s.scene_ads = device.create_scene_ads(instance_buffer)
            to_build.append(s.scene_ads)

        scratch_buffer = device.create_scratch_buffer(*to_build)

        with device.get_raytracing() as man:
            # BUILD all ADSs
            for ads in to_build:
                man.build_ads(ads, scratch_buffer)
            # Transfer all buffers to gpu
            man.cpu_to_gpu(s.vertices)
            man.cpu_to_gpu(s.indices)
            man.cpu_to_gpu(s.transforms)
            man.cpu_to_gpu(s.material_buffer)
            man.cpu_to_gpu(s.geometry_descriptions)
            man.cpu_to_gpu(s.instance_descriptions)