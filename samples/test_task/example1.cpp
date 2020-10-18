#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <CL/cl.h>

void ctx_callback(const char* err, const void* tmp1, size_t tmp2, void* tmp3)
{
    std::cout << "something is not good" << std::endl;
    std::cout << err << std::endl;
}

int main(int argc, char** argv)
{
    std::fstream kernel_file;
    size_t gpu_memory_use_size = 1000000000;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-f") == 0)
        {
            kernel_file.open(argv[++i], std::ios::in);
        }
        else if (strcmp(argv[i], "-s") == 0)
        {
            gpu_memory_use_size = atoll(argv[++i]);
        }
    }

    cl_int ret = CL_SUCCESS;
    cl_uint platform_count = 0;
    clGetPlatformIDs(0, nullptr, &platform_count);
    
    std::vector<cl_platform_id> plfrm_ids(platform_count);
    clGetPlatformIDs(platform_count, plfrm_ids.data(), nullptr);

    char s[256];

    cl_device_id cur_device;

    std::cout << "Platforms: " << platform_count << std::endl;
    for (cl_uint i = 0; i < platform_count; ++i)
    {
        std::cout << "platform: " << i << std::endl;

        clGetPlatformInfo(plfrm_ids[i], CL_PLATFORM_NAME, 256, s, nullptr);
        std::cout << s << std::endl;

        clGetPlatformInfo(plfrm_ids[i], CL_PLATFORM_VERSION, 256, s, nullptr);
        std::cout << s << std::endl;

        clGetPlatformInfo(plfrm_ids[i], CL_PLATFORM_VENDOR, 256, s, nullptr);
        std::cout << s << std::endl;

        cl_uint device_count = 0;

        clGetDeviceIDs(plfrm_ids[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
        std::cout << "\nDevices: " << device_count << std::endl;

        std::vector<cl_device_id> device_vec(device_count);

        clGetDeviceIDs(plfrm_ids[i], CL_DEVICE_TYPE_ALL, device_count, device_vec.data(), nullptr);

        for (cl_uint j = 0; j < device_count; ++j)
        {
            std::cout << "device: " << j << std::endl;

            cl_device_type t;
            clGetDeviceInfo(device_vec[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &t, nullptr);
            std::cout << t << std::endl;

            clGetDeviceInfo(device_vec[j], CL_DEVICE_NAME, 256, s, nullptr);
            std::cout << s << std::endl;

            clGetDeviceInfo(device_vec[j], CL_DEVICE_VERSION, 256, s, nullptr);
            std::cout << s << std::endl;

            clGetDeviceInfo(device_vec[j], CL_DEVICE_VENDOR, 256, s, nullptr);
            std::cout << s << std::endl;

            cl_ulong u;
            clGetDeviceInfo(device_vec[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &u, nullptr);
            std::cout << "memory: " << u << std::endl;

            cur_device = device_vec[j];
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    cl_context main_ctx = clCreateContext(nullptr, 1, &cur_device, ctx_callback, nullptr, &ret);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clCreateContext: " << ret << std::endl;
        return 1;
    }

    cl_command_queue main_queue = clCreateCommandQueueWithProperties(main_ctx, cur_device, nullptr, &ret);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clCreateCommandQueue: " << ret << std::endl;
        return 1;
    }

    std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    kernel_file.close();
    const char* kernel_code_p = kernel_code.c_str();
    size_t kernel_code_len = kernel_code.size();

    cl_program prog = clCreateProgramWithSource(main_ctx, 1, &kernel_code_p, &kernel_code_len, &ret);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clCreateProgramWithSource: " << ret << std::endl;
        return 1;
    }

    ret = clBuildProgram(prog, 1, &cur_device, nullptr, nullptr, nullptr);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clBuildProgram: " << ret << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(prog, "nothing", &ret);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clCreateKernel: " << ret << std::endl;
        return 1;
    }

    std::vector<char> buf1_data(gpu_memory_use_size);

    cl_mem buffer1 = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf1_data.size(), buf1_data.data(), &ret);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clCreateBuffer: " << ret << std::endl;
        return 1;
    }

    cl_mem buffer2 = clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY, buf1_data.size(), nullptr, &ret);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clCreateBuffer: " << ret << std::endl;
        return 1;
    }

    ret = clEnqueueWriteBuffer(main_queue, buffer1, CL_TRUE, 0, buf1_data.size(), buf1_data.data(), 0, nullptr, nullptr);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clEnqueueWriteBuffer: " << ret << std::endl;
        return 1;
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer1);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clSetKernelArg: " << ret << std::endl;
        return 1;
    }

    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer2);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clSetKernelArg: " << ret << std::endl;
        return 1;
    }

    size_t group_size;
    ret = clGetKernelWorkGroupInfo(kernel, cur_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group_size, nullptr);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clSetKernelArg: " << ret << std::endl;
        return 1;
    }
    std::cout << "workgroup size: " << group_size << std::endl;

    if (gpu_memory_use_size % group_size != 0)
    {
        std::cout << "Wrong size" << std::endl;
        return 1;
    }

    ret = clEnqueueNDRangeKernel(main_queue, kernel, 1, nullptr, &gpu_memory_use_size, &group_size, 0, nullptr, nullptr);
    if (ret != CL_SUCCESS)
    {
        std::cout << "Error in clEnqueueNDRangeKernel: " << ret << std::endl;
        return 1;
    }

    clFinish(main_queue);

    ret = clEnqueueReadBuffer(main_queue, buffer2, CL_TRUE, 0, buf1_data.size(), buf1_data.data(), 0, nullptr, nullptr);

    std::cout << buf1_data[0];

    clReleaseMemObject(buffer1);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(main_queue);
    clReleaseContext(main_ctx);
}