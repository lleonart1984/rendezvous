#version 450
layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0, r32f) uniform image3D SrcGrid;
layout (binding = 1, r32f) uniform image3D DstGrid;

layout (binding = 2) uniform Parameters {
    int super_voxel_size;
    int operation;
} parameters;

void main()
{
    ivec3 dim = imageSize(DstGrid);
    int index = int(gl_GlobalInvocationID.x);
    int dst_x = index % dim.x;
    int dst_y = (index / dim.x)%dim.y;
    int dst_z = index / (dim.x*dim.y);
    ivec3 sv_cell = ivec3(dst_x, dst_y, dst_z);

    if (sv_cell.z >= dim.z)
        return;

    int operation = parameters.operation;

    float accum = 0;

    switch(operation){
        case 0: // AVERAGE
            accum = 0.0;
            break;
        case 1: // MIN
            accum = 1.0/0.0;
            break;
        case 2: // MAX
            accum = -1.0/0.0;
            break;
    }

    for (int z = -1; z <= parameters.super_voxel_size; z++)
        for (int y = -1; y <= parameters.super_voxel_size; y++)
            for (int x = -1; x <= parameters.super_voxel_size; x++)
            {
                float value = imageLoad(SrcGrid, sv_cell*parameters.super_voxel_size + ivec3(x,y,z)).x;
                switch(operation)
                {
                    case 0: // AVERAGE
                        accum += value;
                        break;
                    case 1: // MIN
                        accum = min(value, accum);
                        break;
                    case 2:
                        accum = max(value, accum);
                        break;
                }
            }

    if (parameters.operation == 0)
        imageStore(DstGrid, sv_cell, vec4(accum)/pow(parameters.super_voxel_size + 2, 3));
    else
        imageStore(DstGrid, sv_cell, vec4(accum));
}