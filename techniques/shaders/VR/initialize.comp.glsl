#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(scalar, binding = 0) buffer DstGrid {
    float data[];
} dst_grid;

layout(scalar, binding = 1) readonly buffer SrcGrid {
    float data[];
} src_grid;

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

    dst_cell_size = vec3(1) / max ( dst_grid_dim.x, max( dst_grid_dim.y, dst_grid_dim.z));
    src_cell_size = vec3(1) / max ( src_grid_dim.x, max( src_grid_dim.y, src_grid_dim.z));

    vec3 dst_start = vec3(dst_x, dst_y, dst_z);
    vec3 dst_end = vec3(dst_x + 1, dst_y + 1, dst_z + 1);

    vec3 src_start = from_dst_to_src(dst_start);
    vec3 src_end = from_dst_to_src(dst_end);

    /*ivec3 src_cell = ivec3(dst_x / 2, dst_y / 2, dst_z / 2);
    float s =src_grid.data[src_cell.z * (src_grid_dim.x * src_grid_dim.y) + src_cell.y * src_grid_dim.x + src_cell.x];
    dst_grid.data[index]= s;
    return;*/

    ivec3 src_start_cell = ivec3(src_start);
    ivec3 src_end_cell = ivec3(src_end);

    float total = 0;
    for (int z = src_start_cell.z; z <= src_end_cell.z; z++)
    for (int y = src_start_cell.y; y <= src_end_cell.y; y++)
    for (int x = src_start_cell.x; x <= src_end_cell.x; x++)
    {
        vec3 c_min = from_src_to_dst(vec3(x,y,z));
        vec3 c_max = from_src_to_dst(vec3(x + 1, y + 1, z + 1));
        total += src_grid.data[z * (src_grid_dim.x * src_grid_dim.y) + y * src_grid_dim.x + x] *
            vol_int (dst_start, dst_end, c_min, c_max);
    }

    dst_grid.data[index] = total;
}