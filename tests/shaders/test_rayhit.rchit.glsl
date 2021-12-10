#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec4 ResultColor;
hitAttributeEXT vec2 HitAttribs;

void main() {
    vec4 color = vec4(1,1,0,1);
    ResultColor = vec4(HitAttribs.x, HitAttribs.y, 1, 1);
}