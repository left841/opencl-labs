#ifndef __COMMON_DEVICES_H__
#define __COMMON_DEVICES_H__

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <CL/cl.h>

#define CL_CHECK_AND_RET(func) do \
{ \
    cl_int ret = CL_SUCCESS; \
    if (ret = func != CL_SUCCESS) \
    { \
        std::cerr << "Error " << ret << " in " << #func << std::endl; \
        exit(1); \
    } \
} \
while (0)

#define CL_CHECK_AND_RET2(ret, func) do \
{ \
    if (ret != CL_SUCCESS) \
    { \
        std::cerr << "Error " << ret << " in " << #func << std::endl; \
        exit(1); \
    } \
} \
while (0)

cl_device_id get_any_gpu_device();
cl_device_id get_any_cpu_device();

cl_program create_program_from_file(cl_context ctx, const char* file, cl_int* ret);

#endif // __COMMON_DEVICES_H__
