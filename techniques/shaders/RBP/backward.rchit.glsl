#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_atomic_float : require


#include "./RBPCommon.h"
#include "../Common/PTCommon.h"
#include "../Common/Randoms.h"
#include "../Common/SurfaceScattering.h"
#include "../Common/PTEnvironment.h"


/*
Logic for scattering of light and surface.
*/

layout(scalar, set=1, binding = 0) readonly buffer Vertices { Vertex data[]; } vertices;
layout(scalar, set=1, binding = 1) readonly buffer Indices { int data[]; } indices;
layout(scalar, set=1, binding = 2) readonly buffer Transforms { mat3x4 data[]; } transforms;
layout(scalar, set=1, binding = 3) readonly buffer Materials { Material data[]; } materials;
layout(scalar, set=1, binding = 4) readonly buffer Geometries { GeometryDesc data[]; } geometries;
layout(scalar, set=1, binding = 5) readonly buffer Instances { InstanceDesc data[]; } instances;
layout(set=1, binding = 6) uniform sampler2D textures[100];

layout(scalar, set=2, binding = 0) readonly buffer Parameters { vec3 data[]; } parameters;
layout(scalar, set=2, binding = 1) buffer GradParameters { vec3 data[]; } grad_parameters;

layout(location = 0) rayPayloadInEXT RBPRayHitPayload Payload;
hitAttributeEXT vec2 HitAttribs;

void main() {
    InstanceDesc instance = instances.data[gl_InstanceID];
    GeometryDesc geometry = geometries.data[instance.StartGeometry + gl_GeometryIndexEXT];
    int start_index = geometry.StartIndex + gl_PrimitiveID*3;
    int start_vertex = geometry.StartVertex;

    mat4x3 instance_to_world = gl_ObjectToWorldEXT;
    mat4x3 model_to_instance;
    if (geometry.TransformIndex >= 0)
    model_to_instance = transpose(transforms.data[geometry.TransformIndex]);
    else
    model_to_instance = mat4x3(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0);

    Vertex v0 = vertices.data[start_vertex + indices.data[start_index + 0]];
    Vertex v1 = vertices.data[start_vertex + indices.data[start_index + 1]];
    Vertex v2 = vertices.data[start_vertex + indices.data[start_index + 2]];

    vec3 coord = vec3(1 - HitAttribs.x - HitAttribs.y, HitAttribs.x, HitAttribs.y);

    // Load vertex attributes in model space
    vec3 P = v0.P * coord.x + v1.P * coord.y + v2.P * coord.z;
    vec3 N = v0.N * coord.x + v1.N * coord.y + v2.N * coord.z;
    vec2 C = v0.C * coord.x + v1.C * coord.y + v2.C * coord.z;
    // Transform attributes to world space
    P = instance_to_world * vec4(model_to_instance * vec4(P, 1), 1);
    N = normalize(instance_to_world * vec4(model_to_instance * vec4(N, 0), 0));

    Material material;
    if (instance.MaterialIndex == -2) { // Load material from parameters
        // ASUMING RECONSTRUCTED TEXTURE TO BE 512x512x3 texture
        ivec2 coord = ivec2(C*512);
        material.Diffuse = parameters.data[coord.x + coord.y*512];
        material.Opacity = 1.0;
        material.Specular = vec3(1, 1, 1);
        material.SpecularPower = 40;
        material.Emissive = vec3(0, 0, 0);
        material.RefractionIndex = 1.0/1.1;
        material.DiffuseMap = -1;
        material.SpecularMap = -1;
        material.BumpMap = -1;
        material.MaskMap = -1;
        material.Model = vec4(1.0, 0.0, 0.0, 0.0);
    }
    else
    if (instance.MaterialIndex == -1) { // Load Default material
        material.Diffuse = vec3(1, 1, 1);
        material.Opacity = 1.0;
        material.Specular = vec3(1, 1, 1);
        material.SpecularPower = 40;
        material.Emissive = vec3(0, 0, 0);
        material.RefractionIndex = 1.0/1.1;
        material.DiffuseMap = -1;
        material.SpecularMap = -1;
        material.BumpMap = -1;
        material.MaskMap = -1;
        material.Model = vec4(1.0, 0.0, 0.0, 0.0);
    }
    else // Load material from buffer
    material = materials.data[instance.MaterialIndex];
    // Enhance material with textures
    if (material.DiffuseMap >= 0)
    {
        vec4 diff_texture = texture(textures[material.DiffuseMap], C);
        material.Diffuse *= diff_texture.xyz;
    }

    vec3 V = -Payload.Direction;

    Payload.Position = P;
    SurfaceScatter(Payload.rng_seed, Payload.Position, V, N, material,
    Payload.Direction, Payload.BRDF_cos, Payload.PDF);

    if (instance.MaterialIndex == -2) { // Update grad_parameters
        // Compute Li(Payload.Position, Payload.Direction)
        vec3 Li = vec3(1,1,1); //SampleSkyboxWithSun(Payload.Direction); // TODO: IMPLEMENT A SHADOW RAY HERE OR A PT
        vec3 dx = Payload.grad_output * Li / Payload.PDF;
        // ASUMING RECONSTRUCTED TEXTURE TO BE 512x512x3 texture
        ivec2 coord = ivec2(C*512);
        atomicAdd(grad_parameters.data[coord.x + coord.y*512].x, dx.x);
        atomicAdd(grad_parameters.data[coord.x + coord.y*512].y, dx.y);
        atomicAdd(grad_parameters.data[coord.x + coord.y*512].z, dx.z);
    }
}