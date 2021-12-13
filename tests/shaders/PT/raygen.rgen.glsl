#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "PTCommon.h"
#include "Randoms.h"
#include "PTEnvironment.h"

layout(set = 0, binding = 0) uniform accelerationStructureEXT Scene;
layout(set = 0, binding = 1, rgba8) uniform image2D OutputImage;
layout(set = 0, binding = 2, rgba32f) uniform image2D Accumulation;
layout(set = 0, binding = 3) uniform CameraTransforms {
    mat4 ProjToWorld;
} camera;
layout( push_constant ) uniform constants
{
	int frame_index;
} parameters;

layout(location = 0) rayPayloadEXT RayHitPayload Payload;

void CreateScreenRay(in vec2 screen_coord, out vec3 x, out vec3 w)
{
    vec2 coord = screen_coord * 2 - 1;

    vec4 ndcP = vec4(coord, 0, 1);
	ndcP.y *= -1;
	vec4 ndcT = ndcP + vec4(0, 0, 1, 0);

	vec4 viewP = camera.ProjToWorld * ndcP;
	viewP.xyz /= viewP.w;
	vec4 viewT = camera.ProjToWorld * ndcT;
	viewT.xyz /= viewT.w;

	x = viewP.xyz;
	w = normalize(viewT.xyz - viewP.xyz);
}

void main() {
    const uint rayFlags = gl_RayFlagsNoneEXT;
    const uint cullMask = 0xFF;
    const uint sbtRecordOffset = 0;
    const uint sbtRecordStride = 0;
    const uint missIndex = 0;
    const float tmin = 0.0f;
    const float tmax = 20.0f;
    const int payloadLocation = 0;

    Payload.rng_seed = initializeRandom(
        gl_LaunchIDEXT.y + gl_LaunchIDEXT.x * gl_LaunchSizeEXT.y
        + parameters.frame_index * gl_LaunchSizeEXT.x * gl_LaunchSizeEXT.y
    );

    const vec2 screen = vec2(gl_LaunchIDEXT.xy + vec2(random(Payload.rng_seed), random(Payload.rng_seed))) / vec2(gl_LaunchSizeEXT.xy);
    vec3 ray_origin, ray_direction;
    CreateScreenRay(screen, ray_origin, ray_direction);

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

    vec3 acc = imageLoad(Accumulation, ivec2(gl_LaunchIDEXT.xy)).xyz;
    acc += color;
    imageStore(Accumulation, ivec2(gl_LaunchIDEXT.xy), vec4(acc, 1));
    imageStore(OutputImage, ivec2(gl_LaunchIDEXT.xy), vec4(acc / (parameters.frame_index + 1), 1));
}