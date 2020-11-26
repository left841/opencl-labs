__kernel void matrix_mul(__global const int* a, __global const int* b, __global int* c, size_t n, size_t m, size_t l)
{
    size_t id = get_global_id(0);
}