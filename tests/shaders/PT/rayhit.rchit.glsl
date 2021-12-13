#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "PTCommon.h"
#include "Randoms.h"

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

layout(location = 0) rayPayloadInEXT RayHitPayload Payload;
hitAttributeEXT vec2 HitAttribs;

float ComputeFresnel(float NdotL, float ratio)
{
	float oneMinusRatio = 1 - ratio;
	float onePlusRatio = 1 + ratio;
	float divOneMinusByOnePlus = oneMinusRatio / onePlusRatio;
	float f = divOneMinusByOnePlus * divOneMinusByOnePlus;
	return (f + (1.0 - f) * pow((1.0 - NdotL), 5));
}

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
        material.Model = vec4(0.0, 0.0, 0.0, 1);
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
    float VdotN = dot(V, N);
    bool entering = VdotN > 0;
    VdotN = entering ? VdotN : -VdotN;
    vec3 fN = entering ? N : -N;
    Payload.Position = P + fN*0.0001;

    // Compute scattering
    float xi = random(Payload.rng_seed);
    if (xi < material.Model.x)// Diffuse
    {
        float NdotL;
        Payload.Direction = randomHSDirectionCosineWeighted(Payload.rng_seed, N, NdotL);
        Payload.BRDF_cos = material.Diffuse;
        Payload.PDF = 1;
        return;
    }
    xi -= material.Model.x;
    if (xi < material.Model.y)// Glossy
    {
        float NdotL;
        Payload.Direction = randomHSDirectionCosineWeighted(Payload.rng_seed, N, NdotL);
        vec3 H = normalize(V + Payload.Direction);
        Payload.BRDF_cos = material.Specular * pow(max(0, dot(H, N)), material.SpecularPower)
        * (2 + material.SpecularPower) / (2*pi);// normalizing blinn model
        Payload.PDF = NdotL;
        return;
    }
    xi -= material.Model.y;

    if (xi < material.Model.z)// mirror
    {
        Payload.Direction = reflect(-V, N);
        Payload.BRDF_cos = material.Specular;
        Payload.PDF = 1;
        return;
    }
    xi -= material.Model.z;

    // Fresnel TODO

    float eta = entering ? material.RefractionIndex : 1 / material.RefractionIndex;
    float F = ComputeFresnel(VdotN, eta);
    vec3 T = refract (-V, fN, eta);
    if (all(equal(T, vec3(0))))// total internal reflection
    F = 1;
    if (xi < material.Model.w*F) // reflection
    {
        Payload.Direction = reflect(-V, fN);
        Payload.BRDF_cos = material.Specular;
        Payload.PDF = 1;
    }
    else
    {
        Payload.Direction = T;
        Payload.BRDF_cos = material.Specular;
        Payload.PDF = 1;
        Payload.Position = P - fN*0.0001; // traspass surface
    }
}