#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) rayPayloadInEXT vec4 ResultColor;
hitAttributeEXT vec2 HitAttribs;

struct Vertex {
    vec3 P;
    vec3 N;
    vec2 C;
    vec3 T;
    vec3 B;
};

layout(scalar, set=0, binding = 2) readonly buffer Vertices {
    Vertex data[];
} vertices;

void main() {
    int triangleIndex = gl_PrimitiveID;
    Vertex v0 = vertices.data[triangleIndex*3 + 0];
    Vertex v1 = vertices.data[triangleIndex*3 + 1];
    Vertex v2 = vertices.data[triangleIndex*3 + 2];
    vec3 coord = vec3(1 - HitAttribs.x - HitAttribs.y, HitAttribs.x, HitAttribs.y);

    vec3 N = v0.N * coord.x + v1.N * coord.y + v2.N * coord.z;
    vec3 light = vec3(1,1,1)*dot(N, normalize(vec3(-1,1,-3)));
    ResultColor = vec4(light.x, light.y, light.z, 1);
}