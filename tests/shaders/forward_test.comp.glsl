#version 450
#extension GL_EXT_scalar_block_layout: require

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout (std430, binding = 0) readonly buffer Parameter {
    float data[];
} parameters;

layout (std430, binding = 1) buffer Outputs {
    float data[];
} output_image;

void main()
{
    int y_index = int(gl_GlobalInvocationID.x);
    if (y_index >= 3)
    return;
    float value = 0;
    for (int x_index = 0; x_index < 4; x_index++)
    value += parameters.data[x_index] * (y_index+1);
    output_image.data[y_index] = value + 1;
}