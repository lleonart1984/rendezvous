#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#include "../Common/Tensor.h"

#include "../Common/Randoms.h"

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

BIND_TENSOR(0, 0, vec3, 12, Rays);

layout(binding = 1) uniform CameraTransforms {
    mat4 ProjToWorld;
} camera;

layout( push_constant ) uniform Constants {
    ivec2 dim;
    int mode; // 0-fix center, 1-stratified, 2-uniform
    int seed; // seed offset provided by the application
} consts;

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

void main(){
    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, consts.dim)))
    return;

    uvec4 seed = initializeRandom(
        gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * consts.dim.x
        + consts.seed * consts.dim.x * consts.dim.y
    );

    vec3 ray_origin, ray_direction;
    switch(consts.mode){
        case 0:
        CreateScreenRay((gl_GlobalInvocationID.xy + 0.5) / consts.dim, ray_origin, ray_direction);
        break;
        case 1:
        CreateScreenRay((gl_GlobalInvocationID.xy + vec2(random(seed), random(seed))) / consts.dim, ray_origin, ray_direction);
        break;
        case 2:
        CreateScreenRay(vec2(random(seed), random(seed)), ray_origin, ray_direction);
        break;
    }

    int index = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * consts.dim.x);
    TENSOR_ELEMENT(Rays, index*2+0) = ray_origin;
    TENSOR_ELEMENT(Rays, index*2+1) = ray_direction;
}