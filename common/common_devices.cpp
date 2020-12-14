#include <vector>
#include "common_devices.h"

cl_device_id get_any_gpu_device()
{
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS)
        return nullptr;

    std::vector<cl_platform_id> plfrm_ids(platform_count);
    if (clGetPlatformIDs(platform_count, plfrm_ids.data(), nullptr) != CL_SUCCESS)
        return nullptr;

    for (cl_platform_id i: plfrm_ids)
    {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(i, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count) != CL_SUCCESS)
            return nullptr;

        std::vector<cl_device_id> device_vec(device_count);

        if (clGetDeviceIDs(i, CL_DEVICE_TYPE_GPU, device_count, device_vec.data(), nullptr) != CL_SUCCESS)
            return nullptr;

        if (device_vec.size() > 0)
        {
            cl_device_id id = device_vec.front();
            return id;
        }
    }
    return nullptr;
}

cl_device_id get_any_cpu_device()
{
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS)
        return nullptr;

    std::vector<cl_platform_id> plfrm_ids(platform_count);
    if (clGetPlatformIDs(platform_count, plfrm_ids.data(), nullptr) != CL_SUCCESS)
        return nullptr;

    for (cl_platform_id i: plfrm_ids)
    {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(i, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count) != CL_SUCCESS)
            return nullptr;

        std::vector<cl_device_id> device_vec(device_count);

        if (clGetDeviceIDs(i, CL_DEVICE_TYPE_ALL, device_count, device_vec.data(), nullptr) != CL_SUCCESS)
            return nullptr;

        for (cl_device_id j: device_vec)
        {
            cl_device_type t;
            clGetDeviceInfo(j, CL_DEVICE_TYPE, sizeof(cl_device_type), &t, nullptr);
            if (t == CL_DEVICE_TYPE_CPU)
                return j;
        }
    }
    return nullptr;
}

cl_program create_program_from_file(cl_context ctx, const char* file, cl_int* ret)
{
    std::fstream kernel_file(file, std::ios::in);
    std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    kernel_file.close();
    const char* kernel_code_p = kernel_code.c_str();
    size_t kernel_code_len = kernel_code.size();

    return clCreateProgramWithSource(ctx, 1, &kernel_code_p, &kernel_code_len, ret);
}
