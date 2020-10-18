#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <CL/cl.h>

cl_device_id get_any_gpu_device()
{
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS)
        return nullptr;

    std::vector<cl_platform_id> plfrm_ids(platform_count);
    if (clGetPlatformIDs(platform_count, plfrm_ids.data(), nullptr) != CL_SUCCESS)
        return nullptr;

    for (cl_platform_id i : plfrm_ids)
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

int main(int argc, char** argv)
{
    cl_int ret = CL_SUCCESS;

    cl_device_id cur_device = get_any_gpu_device();
    if (cur_device == nullptr)
        return std::cerr << "Error in get_any_gpu_device" << std::endl, 1;

    cl_context main_ctx = clCreateContext(nullptr, 1, &cur_device, nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateContext: " << ret << std::endl, 1;

    cl_command_queue main_queue = clCreateCommandQueueWithProperties(main_ctx, cur_device, nullptr, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateCommandQueueWithProperties: " << ret << std::endl, 1;

    std::fstream kernel_file("kernels/task1.cl", std::ios::in);
    std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    kernel_file.close();
    const char* kernel_code_p = kernel_code.c_str();
    size_t kernel_code_len = kernel_code.size();

    cl_program prog = clCreateProgramWithSource(main_ctx, 1, &kernel_code_p, &kernel_code_len, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateProgramWithSource: " << ret << std::endl, 1;

    if (ret = clBuildProgram(prog, 1, &cur_device, nullptr, nullptr, nullptr) != CL_SUCCESS)
        return std::cerr << "Error in clBuildProgram: " << ret << std::endl, 1;

    cl_kernel kernel = clCreateKernel(prog, "add_thread_index", &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateKernel: " << ret << std::endl, 1;

    size_t group_size;
    if (ret = clGetKernelWorkGroupInfo(kernel, cur_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group_size, nullptr) != CL_SUCCESS)
        return std::cerr << "Error in clGetKernelWorkGroupInfo: " << ret << std::endl, 1;

    size_t global_work_size = group_size * 16;
    std::vector<int> v(global_work_size);
    cl_mem buff = clCreateBuffer(main_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, v.size() * sizeof(int), v.data(), &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateBuffer: " << ret << std::endl, 1;

    if (ret = clEnqueueWriteBuffer(main_queue, buff, CL_TRUE, 0, v.size() * sizeof(int), v.data(), 0, nullptr, nullptr) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueWriteBuffer: " << ret << std::endl, 1;

    if (ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buff) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;

    if (ret = clEnqueueNDRangeKernel(main_queue, kernel, 1, nullptr, &global_work_size, &group_size, 0, nullptr, nullptr) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueNDRangeKernel: " << ret << std::endl, 1;

    if (ret = clFinish(main_queue) != CL_SUCCESS)
        return std::cerr << "Error in clFinish: " << ret << std::endl, 1;

    if (ret = clEnqueueReadBuffer(main_queue, buff, CL_TRUE, 0, v.size() * sizeof(int), v.data(), 0, nullptr, nullptr) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueReadBuffer: " << ret << std::endl, 1;

    for (int& i: v)
        std::cout << i << ' ';
    std::cout << std::endl;

    if (ret = clReleaseKernel(kernel) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseKernel: " << ret << std::endl, 1;
    if (ret = clReleaseProgram(prog) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseProgram: " << ret << std::endl, 1;
    if (ret = clReleaseCommandQueue(main_queue) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseCommandQueue: " << ret << std::endl, 1;
    if (ret = clReleaseContext(main_ctx) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseContext: " << ret << std::endl, 1;
}
