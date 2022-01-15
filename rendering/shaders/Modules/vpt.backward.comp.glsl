#version 460
#extension GL_EXT_shader_atomic_float : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#include "../Common/Tensor.h"
#include "../Common/PTEnvironment.h"
#include "../Common/Randoms.h"
#include "../Common/VolumeScattering.h"

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

BIND_TENSOR(0, 0, float, 4, Grid);
BIND_TENSOR(0, 1, vec3, 12, Rays);
BIND_TENSOR(0, 2, uvec4, 16, Seeds);
BIND_TENSOR(0, 3, vec3, 12, Radiances);
BIND_TENSOR(0, 4, vec3, 12, dRadiances);
BIND_TENSOR(0, 5, float, 4, dGrid);

layout(binding = 6) uniform MediumProperties {
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
vec3 g_scatteringAlbedo;
vec3 g_box_minim;
vec3 g_box_maxim;
vec3 g_box_size;
ivec3 g_dim;
vec3 g_cellSize;
float g_phase_g;

void initialize_registers(){
    g_density = medium.density;
    g_scatteringAlbedo = medium.scatteringAlbedo;
    g_phase_g = medium.phase_g;
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

float sample_volume (inout uvec4 seed, vec3 p)
{
    vec3 fcell = g_dim * (p - g_box_minim)/g_box_size;
    fcell += vec3(random(seed), random(seed), random(seed)) - vec3(0.5);
    ivec3 cell = ivec3(fcell);
    if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, g_dim)))
    return 0.0f;
    int voxel_index = cell.x + cell.y * g_dim.x + cell.z * g_dim.x * g_dim.y;
    return TENSOR_ELEMENT(Grid, voxel_index);
}

void backprop_sample_volume(inout uvec4 seed, vec3 p, float de_ddensity){
    vec3 fcell = g_dim * (p - g_box_minim)/g_box_size;
    fcell += vec3(random(seed), random(seed), random(seed)) - vec3(0.5);
    ivec3 cell = ivec3(fcell);
    if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, g_dim)))
    return;
    int voxel_index = cell.x + cell.y * g_dim.x + cell.z * g_dim.x * g_dim.y;
    atomicAdd(TENSOR_ELEMENT(dGrid, voxel_index), de_ddensity);
}

void backprop_PRB_Radiance(inout uvec4 seed, vec3 x, vec3 w, float total_d, vec3 sel, vec3 L, vec3 de_dL)
{
    float scatteringAlbedo = dot(sel, g_scatteringAlbedo);
    while (true){
        float t = -log(1 - random(seed)) / g_density;

        if (t >= total_d - 0.000001)
        return;

        x += t * w;

        uvec4 sample_volume_copy_seed = seed;

        float voxel_density = sample_volume(seed, x);

        if (random(seed) < 1 - voxel_density)// null collision
        {
            uvec4 s = sample_volume_copy_seed;
            float Pn = 1 - voxel_density;
            vec3 dL_dPn = vec3(1, 1, 1);// L = Pn(x)*L(w_n) + Ps(x)*L_s(x)
            float dPn_ddensity = -1;
            float de_ddensity = -dot(de_dL * sel * L / max(0.000001, Pn), vec3(1,1,1));//dot (L / max(0.000001, Pn) * de_dL, dL_dPn) * dPn_ddensity;
            backprop_sample_volume(s, x, de_ddensity);
            total_d -= t;
            continue;
        }

        // Backprop event Ps + Pa
        {
            uvec4 s = sample_volume_copy_seed;
            float Pt = voxel_density;
            vec3 dL_dPt = vec3(1, 1, 1);
            float dPt_ddensity = 1;
            float de_ddensity = dot(de_dL * sel * L / max(0.000001, Pt), vec3(1,1,1));//dot(L / max(0.000001, Pt) * de_dL, dL_dPt) * dPt_ddensity;
            backprop_sample_volume(s, x, de_ddensity);
        }

        if (random(seed) < 1 - scatteringAlbedo) // absorption
            break;

        vec3 we = LightDirection;
        vec3 Le = LightIntensity * EvalHG(w, we, g_phase_g);
        // Compute transmittance
        vec3 xe = x;
        float T = 1;
        float tMin, tMax;
        rayBoxIntersect(g_box_minim, g_box_maxim, xe, we, tMin, tMax);
        float de = tMax - tMin;
        float pdf_T = 1;
        uvec4 transmittance_copy_seed = seed;
        while (true) {
            t = -log(1 - random(seed)) / g_density;
            if (t >= de - 0.000001)
            break; // emitter reached
            xe += we * t;
            T *= 1 - sample_volume(seed, xe);
            if (T < 0.001){
                if (random(seed) >= 0.001)
                {
                    T = 0;
                    break;
                };
                pdf_T *= 0.001;
            }
            de -= t;
        }

//        xe = x;
//        float Ts = 1;
//        de = tMax - tMin;
//        while (true) {
//            t = -log(1 - random(transmittance_copy_seed)) / g_density;
//            if (t >= de - 0.000001)
//            break; // emitter reached
//            xe += we * t;
//            uvec4 sample_volume_copy_seed = transmittance_copy_seed;
//            float r = 1 - sample_volume(transmittance_copy_seed, xe);
//            Ts *= r;
//            if (Ts < 0.001){
//                if (random(transmittance_copy_seed) >= 0.001)
//                    break;
//            }
//            backprop_sample_volume(sample_volume_copy_seed, xe, -dot(de_dL * sel * Le * T / pdf_T / max(0.0000001,r), vec3(1,1,1)));// dot(de_dL*(1)/max(0.000001, r), vec3(1,1,1)));
//            de -= t;
//        }

        L -= T * Le / pdf_T;

        w = SampleHG (seed, w, g_phase_g);
        rayBoxIntersect(g_box_minim, g_box_maxim, x, w, tMin, tMax);
        x += w*tMin;
        total_d = tMax - tMin;
    }
}

void main()
{
    initialize_registers();

    int index = int(gl_GlobalInvocationID.x);
    if (index >= consts.number_of_rays)
    return;

    uvec4 seed = TENSOR_ELEMENT(Seeds, index);
    vec3 ray_origin = TENSOR_ELEMENT(Rays, index*2+0);
    vec3 ray_direction = TENSOR_ELEMENT(Rays, index*2+1);

    vec3 L = TENSOR_ELEMENT(Radiances, index);
    vec3 dL = TENSOR_ELEMENT(dRadiances, index);

    for (int i=0; i<3; i++){

        vec3 sel = vec3(0);
        if (g_scatteringAlbedo.x == g_scatteringAlbedo.y && g_scatteringAlbedo.x == g_scatteringAlbedo.z)
        {
            sel = vec3(1)/3.0;
        }
        else
            sel[i] = 1.0;

        float tMin, tMax;
        if (rayBoxIntersect(g_box_minim, g_box_maxim, ray_origin, ray_direction, tMin, tMax))
        {
            vec3 x = ray_origin + tMin * ray_direction;
            vec3 w = ray_direction;
            float total_d = tMax - tMin;// TOTAL DISTANCE INSIDE VOLUME
            backprop_PRB_Radiance(seed, x, w, total_d, sel, L, dL);
            //total_color += SDT_Radiance(seed, x, w, total_d) / 3.0;
        }
    }
}
