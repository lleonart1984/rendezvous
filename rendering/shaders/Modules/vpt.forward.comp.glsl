#version 460
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

vec3 RegularGridTransmittance(inout uvec4 seed, vec3 x, vec3 w, float total_d){
    ivec3 cell = ivec3((x - g_box_minim) * g_dim / g_box_size);
    cell = clamp(cell, ivec3(0), g_dim - ivec3(1));
    vec3 alpha_inc = g_cellSize / max(vec3(0.000001), abs(w));
	ivec3 side = ivec3(sign(w));
	vec3 corner = (cell + side * 0.5 + vec3(0.5)) * g_cellSize + g_box_minim;
	vec3 alpha = abs(corner - x) / max(vec3(0.000001), abs(w));
    float tau = 0;
    float current_t = 0;
    float log_pdf = 0;
    float max_scat = max(g_scatteringAlbedo.x, max(g_scatteringAlbedo.y, g_scatteringAlbedo.z));
    while(current_t < total_d - 0.0001){
	    float next_t = min(alpha.x, min(alpha.y, alpha.z));
		ivec3 cell_inc = ivec3(
			alpha.x <= alpha.y && alpha.x <= alpha.z,
			alpha.x > alpha.y && alpha.y <= alpha.z,
			alpha.x > alpha.z && alpha.y > alpha.z);
        int voxel_index = cell.x + cell.y * g_dim.x + cell.z * g_dim.x * g_dim.y;

        float voxel_density = TENSOR_ELEMENT(Grid, voxel_index);

        tau += (next_t - current_t) * voxel_density * g_density;
		current_t = next_t;
		alpha += cell_inc * alpha_inc;
		cell += cell_inc * side;

        if (tau * max_scat > 10)
        {
            // Apply russian roulette
            if (random(seed) >= 0.1)
            return vec3(0,0,0);
            log_pdf += log(0.1);
        }
    }
    return exp(-tau * g_scatteringAlbedo - log_pdf);
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

vec3 DT_Radiance(inout uvec4 seed, vec3 x, vec3 w, float total_d, vec3 sel){
    float sel_scatteringAlbedo = dot(sel, g_scatteringAlbedo);
    vec3 T = RegularGridTransmittance(seed, x, w, total_d);
    vec3 accum = T * SampleSkyboxWithSun(w);
    bool scatters = false;
    while (true){
        float t = -log(max(0.0000001, 1 - random(seed))) / max(0.0000001, g_density);
        if (t >= total_d)
            return accum + (scatters ? SampleSkybox(w) : vec3(0));
        x += w * t; // potential event position
        if (random(seed) < sample_volume(seed, x)) // event collision
        {
            if (random(seed) >= sel_scatteringAlbedo)
            // absorption event
            return accum;
            // scattering event
            scatters = true;
            // direct light contribution (NEE)
            vec3 env_dir = LightDirection;// SampleHG(seed, w, g_phase_g);
            float tMin, tMax;
            rayBoxIntersect(g_box_minim, g_box_maxim, x, env_dir, tMin, tMax);
            T = RegularGridTransmittance(seed, x, env_dir, tMax - tMin);
            //float d = max(0, tMax - tMin);
            accum += T * LightIntensity * EvalHG(w, env_dir, g_phase_g);

            w = SampleHG(seed, w, g_phase_g);
            rayBoxIntersect(g_box_minim, g_box_maxim, x, w, tMin, tMax);
            x += tMin * w;
            total_d = tMax - tMin;
        }
        else{
            total_d -= t; // null collision
        }
    }
    return accum; // No necessary here but to please the editor compiler.
}


vec3 PRB_Radiance(inout uvec4 seed, vec3 x, vec3 w, float total_d, vec3 sel)
{
    float scatteringAlbedo = dot(sel, g_scatteringAlbedo);
    vec3 L = vec3(0);
    while (true){
        float t = -log(1 - random(seed)) / g_density;

        if (t >= total_d - 0.000001)
        {
            L += SampleSkybox(w);
            break;
        }

        x += t * w;

        float voxel_density = sample_volume(seed, x);

        if (random(seed) < 1 - voxel_density) // null collision
        {
            total_d -= t;
            continue;
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

        L += T * Le / pdf_T;

        w = SampleHG (seed, w, g_phase_g);
        rayBoxIntersect(g_box_minim, g_box_maxim, x, w, tMin, tMax);
        x += w*tMin;
        total_d = tMax - tMin;
    }

    return L;
}


/// TODO: BAD IMPLEMENTATION OF VOLUME RENDERING IN PATH-REPLAY BACKPROPAGATION PAPER
vec3 SDT_Radiance(inout uvec4 seed, vec3 x, vec3 w, float total_d){
    vec3 W = vec3(1);
    while (true){
        float t = -log(max(0.0000001, 1 - random(seed))) / max(0.0000001, g_density);
        if (t >= total_d)
            return W * SampleSkyboxWithSun(w);
        if (all(lessThan(W, vec3(0.000001)))) // threshold termination
        return vec3(0);
        x += w * t; // potential event position
        if (random(seed) < sample_volume(seed, x)) // event
        {
            W *= g_scatteringAlbedo;
            // scattering
            w = SampleHG(seed, w, g_phase_g);
            float tMin, tMax;
            rayBoxIntersect(g_box_minim, g_box_maxim, x, w, tMin, tMax);
            x += tMin * w;
            total_d = tMax - tMin;
        }
        else{
            total_d -= t; // null collision
        }
    }
    return vec3(0); // No necessary here but to please the editor compiler.
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

    vec3 total_color = vec3(0, 0, 0);

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
            total_color += PRB_Radiance(seed, x, w, total_d, sel) * sel;
            //total_color += DT_Radiance(seed, x, w, total_d, sel) * sel;
            //total_color += SDT_Radiance(seed, x, w, total_d) / 3.0;
        }
        else
        total_color += SampleSkybox(ray_direction) * sel;
    }
    TENSOR_ELEMENT(Radiances, index) += total_color;
}
