#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#include "../Common/Tensor.h"

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

BIND_TENSOR(0, 0, vec3, 12, Dst);
BIND_TENSOR(0, 1, vec3, 12, Src);

layout( push_constant ) uniform Constants {
    ivec2 dst_img_dim;
    ivec2 src_img_dim;
} consts;

vec2 dst_cell_size;
vec2 src_cell_size;

vec2 from_dst_to_src(vec2 p){
    return p * dst_cell_size / src_cell_size;
}

vec2 from_src_to_dst(vec2 p){
    return p * src_cell_size / dst_cell_size;
}

float area_int (vec2 a_min, vec2 a_max, vec2 b_min, vec2 b_max){
    vec2 i_min = max(a_min, b_min);
    vec2 i_max = min(a_max, b_max);
    vec2 sizes = i_max -  i_min;
    return sizes.x * sizes.y;
}

void main(){
    ivec2 dst_img_dim = consts.dst_img_dim;
    ivec2 src_img_dim = consts.src_img_dim;

    int index = int(gl_GlobalInvocationID.x);
    int dst_x = index % dst_img_dim.x;
    int dst_y = index / dst_img_dim.x;

    if (dst_y >= dst_img_dim.y)
    return;

    dst_cell_size = vec2(1) / dst_img_dim;
    src_cell_size = vec2(1) / src_img_dim;

    vec2 sample_size = max(dst_cell_size, src_cell_size)/dst_cell_size;
    float sample_area = sample_size.x * sample_size.y;

    vec2 dst_start = vec2(dst_x + 0.5 - sample_size.x*0.5, dst_y + 0.5 - sample_size.y*0.5);
    vec2 dst_end = vec2(dst_x + 0.5 + sample_size.x*0.5, dst_y + 0.5 + sample_size.y*0.5);

    vec2 src_start = from_dst_to_src(dst_start);
    vec2 src_end = from_dst_to_src(dst_end);

    ivec2 src_start_cell = ivec2(src_start);
    ivec2 src_end_cell = ivec2(src_end);

    src_start_cell = max(src_start_cell, ivec2(0));
    src_end_cell = min(src_end_cell, src_img_dim - 1);

    vec3 total = vec3(0);
    for (int y = src_start_cell.y; y <= src_end_cell.y; y++)
    for (int x = src_start_cell.x; x <= src_end_cell.x; x++)
    {
        vec2 c_min = from_src_to_dst(vec2(x, y));
        vec2 c_max = from_src_to_dst(vec2(x + 1, y + 1));
        total += TENSOR_ELEMENT(Src, y * src_img_dim.x + x) *
            area_int (dst_start, dst_end, c_min, c_max) / sample_area;
    }

    TENSOR_ELEMENT(Dst, index) = total;
}