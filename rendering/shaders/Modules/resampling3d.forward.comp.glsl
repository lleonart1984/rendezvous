#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#include "../Common/Tensor.h"

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

BIND_TENSOR(0, 0, float, 4, Dst);
BIND_TENSOR(0, 1, float, 4, Src);

layout( push_constant ) uniform Constants {
    ivec3 dst_grid_dim; float rem0;
    ivec3 src_grid_dim; float rem1;
} consts;

vec3 dst_cell_size;
vec3 src_cell_size;

vec3 from_dst_to_src(vec3 p){
    return p * dst_cell_size / src_cell_size;
}

vec3 from_src_to_dst(vec3 p){
    return p * src_cell_size / dst_cell_size;
}

float vol_int (vec3 a_min, vec3 a_max, vec3 b_min, vec3 b_max){
    vec3 i_min = max(a_min, b_min);
    vec3 i_max = min(a_max, b_max);
    vec3 sizes = i_max -  i_min;
    return sizes.x * sizes.y * sizes.z;
}

void main(){
    ivec3 dst_grid_dim = consts.dst_grid_dim;
    ivec3 src_grid_dim = consts.src_grid_dim;

    int index = int(gl_GlobalInvocationID.x);
    int dst_x = index % dst_grid_dim.x;
    int dst_y = (index / dst_grid_dim.x) % dst_grid_dim.y;
    int dst_z = index / (dst_grid_dim.x * dst_grid_dim.y);

    if (dst_z >= dst_grid_dim.z)
    return;

    dst_cell_size = vec3(1) / dst_grid_dim;
    src_cell_size = vec3(1) / src_grid_dim;

    vec3 sample_size = max(dst_cell_size, src_cell_size)/dst_cell_size;
    float sample_vol = sample_size.x * sample_size.y * sample_size.z;

    vec3 dst_start = vec3(dst_x + 0.5 - sample_size.x * 0.5, dst_y + 0.5 - sample_size.y*0.5, dst_z + 0.5 - sample_size.z * 0.5);
    vec3 dst_end = vec3(dst_x + 0.5 + sample_size.x * 0.5, dst_y + 0.5 + sample_size.y*0.5, dst_z + 0.5 + sample_size.z * 0.5);

    vec3 src_start = from_dst_to_src(dst_start);
    vec3 src_end = from_dst_to_src(dst_end);

    ivec3 src_start_cell = ivec3(src_start);
    ivec3 src_end_cell = ivec3(src_end);

    src_start_cell = max(src_start_cell, ivec3(0));
    src_end_cell = min(src_end_cell, src_grid_dim - 1);

    float total = 0;
    for (int z = src_start_cell.z; z <= src_end_cell.z; z++)
    for (int y = src_start_cell.y; y <= src_end_cell.y; y++)
    for (int x = src_start_cell.x; x <= src_end_cell.x; x++)
    {
        vec3 c_min = from_src_to_dst(vec3(x,y,z));
        vec3 c_max = from_src_to_dst(vec3(x + 1, y + 1, z + 1));
        total += TENSOR_ELEMENT(Src, z * (src_grid_dim.x * src_grid_dim.y) + y * src_grid_dim.x + x) *
            vol_int (dst_start, dst_end, c_min, c_max) / sample_vol;
    }

    TENSOR_ELEMENT(Dst, index) = total;
}