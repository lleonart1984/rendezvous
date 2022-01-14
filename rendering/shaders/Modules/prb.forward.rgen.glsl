#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "../Common/Tensor.h"
#include "../Common/PTCommon.h"
#include "../Common/Randoms.h"
#include "../Common/PTEnvironment.h"

layout(set = 0, binding = 0) uniform accelerationStructureEXT Scene;
BIND_TENSOR(0, 1, vec3, 12, Radiances);  // Output
BIND_TENSOR(0, 2, vec3, 12, Rays);   // input (rays)
BIND_TENSOR(0, 3, uvec4, 16, Seeds);   // input (random seeds)

layout(set = 0, binding = 4) uniform Consts{
    int number_of_samples;
} consts;

layout(location = 0) rayPayloadEXT RayHitPayload Payload;

void main() {
    const uint rayFlags = gl_RayFlagsNoneEXT;
    const uint cullMask = 0xFF;
    const uint sbtRecordOffset = 0;
    const uint sbtRecordStride = 0;
    const uint missIndex = 0;
    const float tmin = 0.0f;
    const float tmax = 1000.0f;
    const int payloadLocation = 0;

    Payload.rng_seed = TENSOR_ELEMENT(Seeds, gl_LaunchIDEXT.x);

    vec3 total_color = vec3(0,0,0);

    for (int i=0; i < consts.number_of_samples; i++){

        vec3 ray_origin = TENSOR_ELEMENT(Rays, gl_LaunchIDEXT.x * 2 + 0);
        vec3 ray_direction = TENSOR_ELEMENT(Rays, gl_LaunchIDEXT.x * 2 + 1);

        Payload.Position = ray_origin;
        Payload.Direction = ray_direction;
        Payload.BRDF_cos = vec3(0);

        vec3 importance = vec3(1, 1, 1);
        vec3 color = vec3(0, 0, 0);

        for (int bounce = 0; bounce < 10; bounce ++){

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

            if (Payload.PDF == 0)// no scattering -> miss -> sample skybox
            {
                color = importance * SampleSkyboxWithSun(Payload.Direction);
                break;
            }

            importance *= Payload.BRDF_cos / Payload.PDF;
            ray_origin = Payload.Position;
            ray_direction = Payload.Direction;
        }

        total_color += color;
    }

    TENSOR_ELEMENT(Radiances, gl_LaunchIDEXT.x) += total_color / consts.number_of_samples;
}