#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require


#include "./RBPCommon.h"
#include "../Common/PTCommon.h"
#include "../Common/Randoms.h"
#include "../Common/PTEnvironment.h"

// Acceleration structure with the scene
layout(set = 0, binding = 0) uniform accelerationStructureEXT Scene;

// Image with the gradients of the output
layout(scalar, set = 0, binding = 1) readonly buffer GradOuput{
    vec3 data[];
} grad_output;

// Camera setup
layout(set = 0, binding = 2) uniform CameraTransforms {
    mat4 ProjToWorld;
} camera;

// Push constant with frame_seed
layout( push_constant ) uniform constants
{
	int frame_seed;
    int number_of_samples;
} consts;

layout(location = 0) rayPayloadEXT RBPRayHitPayload Payload;

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
        gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x
        + consts.frame_seed * gl_LaunchSizeEXT.x * gl_LaunchSizeEXT.y
    );

    const vec2 screen = vec2(gl_LaunchIDEXT.xy + vec2(random(Payload.rng_seed), random(Payload.rng_seed))) / vec2(gl_LaunchSizeEXT.xy);
    vec3 ray_origin, ray_direction;
    CreateScreenRay(screen, ray_origin, ray_direction);

    Payload.Position = ray_origin;
    Payload.Direction = ray_direction;
    Payload.BRDF_cos = vec3(0);

    vec3 grad_output_at_pixel = grad_output.data[gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x];
    vec3 importance = vec3(1, 1, 1);

    for (int bounce = 0; bounce < 10; bounce ++){
        Payload.grad_output = importance * grad_output_at_pixel / consts.number_of_samples;

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