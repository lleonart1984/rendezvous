#version 450

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer Parameter {
    float data[];
} parameters;

layout (binding = 1) buffer GradParameter {
    float data[];
} grad_parameters;

layout (binding = 1) readonly buffer GradOutput {
    float data[];
} grad_output;

void main()
{
    /*
    y0 = x0 + x1 + x2 + x3
    y1 = x0*2 + x1*2 + x2*2 + x3*2
    y2 = x0*3 + x1*3 + x2*3 + x3*3
    */
    int x_index = int(gl_GlobalInvocationID.x);
    if (x_index >= 4)
    return;
    float dvalue = 0;
    for (int y_index = 0; y_index < 3; y_index++)
    dvalue += grad_output.data[y_index] * (y_index+1);
    grad_parameters.data[x_index] = dvalue;
}