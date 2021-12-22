#version 450
layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0, r32f) uniform image3D SrcGrid;
layout (binding = 1, r32f) uniform image3D DstGrid;

layout (binding = 2) uniform Parameters {
    int level;
} parameters;


void main(){
    ivec3 dim = imageSize(DstGrid);
    int index = int(gl_GlobalInvocationID.x);
    int vx = index % dim.x;
    int vy = (index / dim.x) % dim.y;
    int vz = index / (dim.x * dim.y);
    if (vz >= dim.z)
    return;// kernel coordinate outside volume

    ivec3 coord = ivec3(vx,vy,vz);

    float maxDim = max(dim.x, max(dim.y, dim.z));
    float voxel_size = 1.0 / maxDim; // assuming a normalization -0.5, 0.5

    if (parameters.level == 0){
        float value = imageLoad(SrcGrid, coord).x;
        imageStore(DstGrid, coord, vec4(value == 0 ? 0.5*voxel_size:0 ));
        return;
    }

    float ref_value = voxel_size * 0.5 * (1 << parameters.level); // center value
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++)
    {
        ivec3 adjCoord = clamp(
                ivec3(coord + vec3(0.5) + vec3(dx, dy, dz) * (ref_value/voxel_size*0.5 + 0.5)),
                ivec3(0), dim - ivec3(1));
        if (imageLoad(DstGrid, adjCoord).x < ref_value * 0.4999)// check neighboors
        return;
    }

    imageStore(DstGrid, coord, vec4(ref_value));
}
