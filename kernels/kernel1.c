__kernel void nothing(__global char* input, __global char* output)
{
    size_t ind = get_global_id(0);
    output[ind] = input[ind] + '>';
}