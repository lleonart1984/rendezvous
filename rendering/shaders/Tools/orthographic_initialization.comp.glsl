#version 450
layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (binding = 0, r32f) uniform image3D SrcGrid;
layout (binding = 1, r32f) uniform image3D DstGrid;

layout (binding = 2) uniform Parameters {
    int component_0;
    int component_1;
    int component_2;
    int direction;
    float tau_reference;
} parameters;

void main(){
    ivec3 grid_dim = imageSize(DstGrid);
    if (parameters.direction == 0) // initialization
    {
        for (int i=0; i<grid_dim[parameters.component_2]; i++)
        {
            ivec3 coordinate = ivec3(0);
            // SWIZZLE
            coordinate[parameters.component_0] = int(gl_GlobalInvocationID.x);
            coordinate[parameters.component_1] = int(gl_GlobalInvocationID.y);
            coordinate[parameters.component_2] = i;
            imageStore(DstGrid, coordinate, vec4(4));
        }
        return;
    }

    int max_dim = max(grid_dim.x, max(grid_dim.y, grid_dim.z));
    float voxel_size = 1.0 / max_dim; // assuming a grid normalization -0.5, 0.5

    int start = parameters.direction > 0 ? 0 : grid_dim[parameters.component_2];
    int end = parameters.direction > 0 ? grid_dim[parameters.component_2]: -1;
    float accum = 0;
    float accum_radius = 0;
    for (int i=start; i != end; i += parameters.direction){
        ivec3 coordinate = ivec3(0);
        coordinate[parameters.component_0] = int(gl_GlobalInvocationID.x);
        coordinate[parameters.component_1] = int(gl_GlobalInvocationID.y);
        coordinate[parameters.component_2] = i;
        float value = imageLoad(SrcGrid, coordinate).x * voxel_size;
        accum += value;
        if (accum > parameters.tau_reference) // deep enough
            accum_radius += voxel_size;
        float prev_radius = imageLoad(DstGrid, coordinate).x;
        imageStore(DstGrid, coordinate, vec4(min(prev_radius, accum_radius)));
    }
}

