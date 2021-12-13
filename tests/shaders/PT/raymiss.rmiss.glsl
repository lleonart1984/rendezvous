#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "../Common/PTCommon.h"

layout(location = 0) rayPayloadInEXT RayHitPayload Payload;

void main() {
    Payload.BRDF_cos = vec3(0);
    Payload.PDF = 0;
}