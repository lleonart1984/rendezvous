#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require

layout (local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout (buffer_reference, std430, buffer_reference_align = 4) buffer Block {
    uint data;
};

layout( push_constant ) uniform Constants {
    uint64_t src_pointer;
    uint64_t dst_pointer;
    int size_in_bytes;
    ivec3 padd0;
} consts;

void main()
{
    int block_index = int(gl_GlobalInvocationID.x);
    int start = block_index * 128;
    int size_in_uints = consts.size_in_bytes / 4;

    if (start >= size_in_uints)
    return;

    int end = min(size_in_uints, start + 128);

    uint64_t src_pointer = consts.src_pointer;
    uint64_t dst_pointer = consts.dst_pointer;

    for (int i = start; i < end; i++)
        Block(dst_pointer + uint64_t(i*4)).data = Block(src_pointer + uint64_t(i*4)).data;
}