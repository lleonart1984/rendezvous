#version 450
layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0, r32f) uniform image3D SrcGrid;
layout (binding = 1, r32f) uniform image3D DstGrid;

layout (binding = 2) uniform Parameters {
    ivec3 src_offset;
    ivec3 src_size;
    ivec3 dst_offset;
    ivec3 dst_size;
    int operation; // 0 - ave, 1 - min, 2 - max
} parameters;

ivec3 src_offset;
ivec3 src_size;
ivec3 dst_offset;
ivec3 dst_size;
int operation ;

float volume_intersection(vec3 a1, vec3 a2, vec3 b1, vec3 b2){
    float x_min = max(a1.x, b1.x);
    float x_max = min(a2.x, b2.x);
    float y_min = max(a1.y, b1.y);
    float y_max = min(a2.y, b2.y);
    float z_min = max(a1.z, b1.z);
    float z_max = min(a2.z, b2.z);
    return max(0, x_max-x_min) * max(0, y_max-y_min) * max(0, z_max - z_min);
}

void main(){
    src_offset = parameters.src_offset;
    src_size = parameters.src_size;
    dst_offset = parameters.dst_offset;
    dst_size = parameters.dst_size;
    operation = parameters.operation;

    int index = int(gl_GlobalInvocationID.x);
    int vx = index % dst_size.x;
    int vy = (index / dst_size.x) % dst_size.y;
    int vz = index / (dst_size.x * dst_size.y);
    if (vz >= dst_size.z)
    return;// kernel coordinate outside volume

    vec3 uv_min = vec3(vx, vy, vz) / dst_size;
    vec3 uv_max = vec3(vx+1, vy+1, vz+1) / dst_size;
    vec3 src_coord_min = uv_min * src_size + src_offset;
    vec3 src_coord_max = uv_max * src_size + src_offset;

    vec4 acc = vec4(0);
    switch(operation){
        case 0: break; // keep 0.
        case 1: acc = vec4(1.0)/vec4(0.0); // pos inf
        break;
        case 2: acc = vec4(-1.0)/vec4(0.0); // neg inf
        break;
    }

    float total_sample_volume =
    (src_coord_max.x - src_coord_min.x)*
    (src_coord_max.y - src_coord_min.y)*
    (src_coord_max.z - src_coord_min.z);

    for (int z = int(src_coord_min.z); z <= int(src_coord_max.z); z++)
    for (int y = int(src_coord_min.y); y <= int(src_coord_max.y); y++)
    for (int x = int(src_coord_min.x); x <= int(src_coord_max.x); x++)
    {
        switch(operation){
            case 0: // AVERAGE
            vec3 vox_min = vec3(x,y,z);
            vec3 vox_max = vec3(x+1, y+1, z+1);
            float weight = volume_intersection(vox_min, vox_max, src_coord_min, src_coord_max)/total_sample_volume;
            acc += weight * imageLoad(SrcGrid, ivec3(x,y,z));
            break;
            case 1:
            acc = min(acc, imageLoad(SrcGrid, ivec3(x,y,z)));
            break;
            case 2:
            acc = max(acc, imageLoad(SrcGrid, ivec3(x,y,z)));
            break;
        }
    }

    imageStore(DstGrid, dst_offset + ivec3(vx, vy, vz), acc);
}
