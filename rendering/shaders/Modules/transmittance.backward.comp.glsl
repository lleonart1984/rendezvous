#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_atomic_float : require
#include "../Common/Tensor.h"

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

BIND_TENSOR(0, 0, float, 4, Gradients);

BIND_TENSOR(0, 1, vec3, 12, Rays);

BIND_TENSOR(0, 2, vec3, 12, Transmittances);

BIND_TENSOR(0, 3, vec3, 12, dTransmittances);

layout(binding = 4) uniform MediumProperties {
    vec3 scatteringAlbedo;
    float density;
    float phase_g;
} medium;

layout( push_constant ) uniform Constants {
    ivec3 grid_dim;
    int number_of_rays;
    vec3 box_minim; float pad0;
    vec3 box_size; float pad1;
} consts;

// --- Common registers between functions
float g_density;
vec3 g_box_minim;
vec3 g_box_maxim;
vec3 g_box_size;
ivec3 g_dim;
vec3 g_cellSize;
vec3 g_scatteringAlbedo;

void initialize_registers(){
    g_density = medium.density;
    g_scatteringAlbedo = medium.scatteringAlbedo;
    g_dim = consts.grid_dim;
    g_box_minim = consts.box_minim;
    g_box_size = consts.box_size;
    g_box_maxim = g_box_minim + g_box_size;
    g_cellSize = g_box_size / g_dim;
}

bool rayBoxIntersect(vec3 bMin, vec3 bMax, vec3 P, vec3 D, out float tMin, out float tMax)
{
    // un-parallelize D
    D.x = abs(D).x <= 0.000001 ? 0.000001 : D.x;
    D.y = abs(D).y <= 0.000001 ? 0.000001 : D.y;
    D.z = abs(D).z <= 0.000001 ? 0.000001 : D.z;
    vec3 C_Min = (bMin - P)/D;
    vec3 C_Max = (bMax - P)/D;
	tMin = max(max(min(C_Min[0], C_Max[0]), min(C_Min[1], C_Max[1])), min(C_Min[2], C_Max[2]));
	tMin = max(0.0, tMin);
	tMax = min(min(max(C_Min[0], C_Max[0]), max(C_Min[1], C_Max[1])), max(C_Min[2], C_Max[2]));
	if (tMax <= tMin || tMax <= 0) {
		return false;
	}
	return true;
}

void RegularGridTransmittanceUpdateGradient(vec3 x, vec3 w, float total_d, vec3 dL_dT){
    ivec3 cell = ivec3((x - g_box_minim) * g_dim / (g_box_maxim - g_box_minim));
    cell = clamp(cell, ivec3(0), g_dim - ivec3(1));
    vec3 alpha_inc = g_cellSize / max(vec3(0.000001), abs(w));
	ivec3 side = ivec3(sign(w));
	vec3 corner = (cell + side * 0.5 + vec3(0.5)) * g_cellSize + g_box_minim;
	vec3 alpha = abs(corner - x) / max(vec3(0.000001), abs(w));
    float current_t = 0;
    while(current_t < total_d - 0.0001){
	    float next_t = min(alpha.x, min(alpha.y, alpha.z));
		ivec3 cell_inc = ivec3(
			alpha.x <= alpha.y && alpha.x <= alpha.z,
			alpha.x > alpha.y && alpha.y <= alpha.z,
			alpha.x > alpha.z && alpha.y > alpha.z);
        int voxel_index = cell.x + cell.y * g_dim.x + cell.z * g_dim.x * g_dim.y;
        float dt = next_t - current_t;
        vec3 dT_dsigma = -vec3(dt);
        vec3 grad_inc = dL_dT * dT_dsigma * g_density * g_scatteringAlbedo;
        atomicAdd(TENSOR_ELEMENT(Gradients, voxel_index), grad_inc.x + grad_inc.y + grad_inc.z);
		current_t = next_t;
		alpha += cell_inc * alpha_inc;
		cell += cell_inc * side;
    }
}

void main()
{
    initialize_registers();

    int index = int(gl_GlobalInvocationID.x);
    if (index >= consts.number_of_rays)
    return;

    vec3 ray_origin = TENSOR_ELEMENT(Rays, index*2+0);
    vec3 ray_direction = TENSOR_ELEMENT(Rays, index*2+1);

    vec3 color;
    float tMin, tMax;
    if (rayBoxIntersect(g_box_minim, g_box_maxim, ray_origin, ray_direction, tMin, tMax))
    {
        vec3 x = ray_origin + tMin * ray_direction;
        vec3 w = ray_direction;
        float total_d = tMax - tMin; // TOTAL DISTANCE INSIDE VOLUME
        RegularGridTransmittanceUpdateGradient(x, w, total_d,
        TENSOR_ELEMENT(dTransmittances, index));
    }
}
