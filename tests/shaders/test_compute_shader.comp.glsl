#version 460

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (binding = 0, rgba32f) uniform image2D output_image;

void main()
{
    ivec2 dim = imageSize(output_image);
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    imageStore( output_image, coord, vec4( coord.x / float(dim.x), coord.y / float(dim.y), 0.5, 1 ) );
}