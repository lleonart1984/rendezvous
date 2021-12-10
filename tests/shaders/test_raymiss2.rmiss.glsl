#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec4 ResultColor;

void main() {
    ResultColor = vec4(1, 1.0, 0.0f, 1.0);
}