__kernel void print_some()
{
    ulong id1 = get_group_id(0);
    ulong id2 = get_local_id(0);
    ulong id3 = get_global_id(0);
    printf("I am from %lu block, %lu thread (global index: %lu)\r\n", id1, id2, id3);
}

__kernel void add_thread_index(__global int* a)
{
    ulong id = get_global_id(0);
    a[id] += id;
}
