#ifndef TENSOR_H
#define TENSOR_H

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require

#define BIND_TENSOR(s, b, element_type, element_size, block_name) \
const int SIZE_OF_##block_name = element_size; \
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Buffer##block_name { element_type data; };\
layout(set=s, binding=b) uniform Uniform##block_name { uint64_t ptr; } block_name;
#define TENSOR_ELEMENT(block_name, index) Buffer##block_name(block_name.ptr + SIZE_OF_##block_name * (index)).data

#endif