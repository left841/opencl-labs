__kernel void matrix_mul_simple(__global int* a, __global int* b, __global int* c, ulong n, ulong m, ulong l)
{
    size_t g_id0 = get_global_id(0);
    size_t g_id1 = get_global_id(1);

    __private int res = 0;

    for (size_t i = 0; i < m; ++i)
        res += a[g_id0 * m + i] * b[i * l + g_id1];
    c[l * g_id0 + g_id1] = res;
}

__kernel void matrix_mul(__global int* a, __global int* b, __global int* c, ulong n, ulong m, ulong l)
{
    size_t g_id0 = get_global_id(0);
    size_t g_id1 = get_global_id(1);
    size_t l_id0 = get_local_id(0);
    size_t l_id1 = get_local_id(1);

    __local int sub_arr_a[16][16];
    __local int sub_arr_b[16][16];
    __private int res = 0;

    for (size_t i = 0; i < m / 16; ++i)
    {
        sub_arr_a[l_id0][l_id1] = a[g_id0 * m + i * 16 + l_id1];
        sub_arr_b[l_id0][l_id1] = b[(i * 16 + l_id0) * l + g_id1];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t j = 0; j < 16; ++j)
            res += sub_arr_a[l_id0][j] * sub_arr_b[j][l_id1];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[l * g_id0 + g_id1] = res;
}

__kernel void matrix_mul_img(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, ulong n, ulong m, ulong l)
{
    size_t g_id0 = get_global_id(0);
    size_t g_id1 = get_global_id(1);
    size_t l_id0 = get_local_id(0);
    size_t l_id1 = get_local_id(1);

    __local int sub_arr_a[16][16];
    __local int sub_arr_b[16][16];
    __private int res = 0;

    for (size_t i = 0; i < m / 16; ++i)
    {
        int2 a_coord = (int2)(i * 16 + l_id1, g_id0);
        int2 b_coord = (int2)(g_id1, i * 16 + l_id0);

        sub_arr_a[l_id0][l_id1] = read_imagei(a, a_coord).x;
        sub_arr_b[l_id0][l_id1] = read_imagei(b, b_coord).x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t j = 0; j < 16; ++j)
            res += sub_arr_a[l_id0][j] * sub_arr_b[j][l_id1];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int2 c_cord = (int2)(g_id1, g_id0);
    write_imagei(c, c_cord, (int4)(res, 0, 0, 1));
}
