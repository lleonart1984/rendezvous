#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#include "../Common/Tensor.h"

#include "../Common/PRBCommon.h"
#include "../Common/PTCommon.h"
#include "../Common/Randoms.h"
#include "../Common/PTEnvironment.h"

// Acceleration structure with the scene
layout(set = 0, binding = 0) uniform accelerationStructureEXT Scene;

// Image with the gradients of the output
BIND_TENSOR(0, 1, vec3, 12, Radiances);
BIND_TENSOR(0, 2, vec3, 12, dRadiances);
BIND_TENSOR(0, 3, vec3, 12, Rays);
BIND_TENSOR(0, 4, uvec4, 16, Seeds);

layout(location = 0) rayPayloadEXT PRBRayHitPayload Payload;

void main() {
    const uint rayFlags = gl_RayFlagsNoneEXT;
    const uint cullMask = 0xFF;
    const uint sbtRecordOffset = 0;
    const uint sbtRecordStride = 0;
    const uint missIndex = 0;
    const float tmin = 0.0f;
    const float tmax = 20.0f;
    const int payloadLocation = 0;

    vec3 ray_origin = TENSOR_ELEMENT(Rays, gl_LaunchIDEXT.x * 2 + 0);
    vec3 ray_direction = TENSOR_ELEMENT(Rays, gl_LaunchIDEXT.x * 2 + 1);

    Payload.rng_seed = TENSOR_ELEMENT(Seeds, gl_LaunchIDEXT.x);
    Payload.Position = ray_origin;
    Payload.Direction = ray_direction;
    Payload.BRDF_cos = vec3(0);

    vec3 output_at_pixel = TENSOR_ELEMENT(Radiances, gl_LaunchIDEXT.x);
    vec3 grad_output_at_pixel = TENSOR_ELEMENT(dRadiances, gl_LaunchIDEXT.x);
    vec3 importance = vec3(1, 1, 1);

    for (int bounce = 0; bounce < 10; bounce ++){
        Payload.dL = grad_output_at_pixel;
        Payload.L = output_at_pixel;

        traceRayEXT(Scene,
        rayFlags,
        cullMask,
        sbtRecordOffset,
        sbtRecordStride,
        missIndex,
        ray_origin,
        tmin,
        ray_direction,
        tmax,
        payloadLocation);

        if (Payload.PDF == 0)// no scattering -> miss -> stop
            break;

        importance *= Payload.BRDF_cos / Payload.PDF;
        ray_origin = Payload.Position;
        ray_direction = Payload.Direction;
    }
}