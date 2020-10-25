#include <ctime>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <random>
#include <CL/cl.h>
#include "omp.h"
#include "common_devices.h"

int saxpy(size_t n, float a, float* x, size_t incx, float* y, size_t incy)
{
    for (size_t i = 0; i < n; ++i)
        y[i * incy] += x[i * incx] * a;
    return 0;
}

int daxpy(size_t n, double a, double* x, size_t incx, double* y, size_t incy)
{
    for (size_t i = 0; i < n; ++i)
        y[i * incy] += x[i * incx] * a;
    return 0;
}

int saxpy_omp(size_t n, float a, float* x, size_t incx, float* y, size_t incy)
{
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
        y[i * incy] += x[i * incx] * a;
    return 0;
}

int daxpy_omp(size_t n, double a, double* x, size_t incx, double* y, size_t incy)
{
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
        y[i * incy] += x[i * incx] * a;
    return 0;
}

int saxpy_device(size_t n, float a, float* x, size_t incx, float* y, size_t incy, cl_device_id device, size_t group_size)
{
    cl_int ret = CL_SUCCESS;

    cl_context main_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateContext: " << ret << std::endl, 1;

    cl_command_queue main_queue = clCreateCommandQueueWithProperties(main_ctx, device, nullptr, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateCommandQueueWithProperties: " << ret << std::endl, 1;

    cl_program prog = create_program_from_file(main_ctx, "kernels/task2.cl", &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateProgramWithSource: " << ret << std::endl, 1;

    if (ret = clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr) != CL_SUCCESS)
        return std::cerr << "Error in clBuildProgram: " << ret << std::endl, 1;

    cl_kernel kernel = clCreateKernel(prog, "saxpy", &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateKernel: " << ret << std::endl, 1;

    cl_event buf_e[2];

    cl_mem buff_y = clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, n * incy * sizeof(float), y, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateBuffer: " << ret << std::endl, 1;

    cl_mem buff_x = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, n * incx * sizeof(float), x, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateBuffer: " << ret << std::endl, 1;

    if (ret = clEnqueueWriteBuffer(main_queue, buff_y, CL_FALSE, 0, n * incy * sizeof(float), y, 0, nullptr, &buf_e[0]) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueWriteBuffer: " << ret << std::endl, 1;

    if (ret = clEnqueueWriteBuffer(main_queue, buff_x, CL_FALSE, 0, n * incx * sizeof(float), x, 0, nullptr, &buf_e[1]) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueWriteBuffer: " << ret << std::endl, 1;

    if (ret = clSetKernelArg(kernel, 0, sizeof(float), &a) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;
    if (ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buff_x) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;
    if (ret = clSetKernelArg(kernel, 2, sizeof(size_t), &incx) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;
    if (ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &buff_y) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;
    if (ret = clSetKernelArg(kernel, 4, sizeof(size_t), &incy) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;

    cl_event main_e;
    if (ret = clEnqueueNDRangeKernel(main_queue, kernel, 1, nullptr, &n, &group_size, 2, buf_e, &main_e) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueNDRangeKernel: " << ret << std::endl, 1;

    if (ret = clEnqueueReadBuffer(main_queue, buff_y, CL_TRUE, 0, n * incy * sizeof(float), y, 1, &main_e, nullptr) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueReadBuffer: " << ret << std::endl, 1;

    if (ret = clFinish(main_queue) != CL_SUCCESS)
        return std::cerr << "Error in clFinish: " << ret << std::endl, 1;

    if (ret = clReleaseMemObject(buff_x) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseMemObject: " << ret << std::endl, 1;
    if (ret = clReleaseMemObject(buff_y) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseMemObject: " << ret << std::endl, 1;
    if (ret = clReleaseKernel(kernel) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseKernel: " << ret << std::endl, 1;
    if (ret = clReleaseProgram(prog) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseProgram: " << ret << std::endl, 1;
    if (ret = clReleaseCommandQueue(main_queue) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseCommandQueue: " << ret << std::endl, 1;
    if (ret = clReleaseContext(main_ctx) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseContext: " << ret << std::endl, 1;
    return 0;
}

int daxpy_device(size_t n, double a, double* x, size_t incx, double* y, size_t incy, cl_device_id device, size_t group_size)
{
    cl_int ret = CL_SUCCESS;

    cl_context main_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateContext: " << ret << std::endl, 1;

    cl_command_queue main_queue = clCreateCommandQueueWithProperties(main_ctx, device, nullptr, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateCommandQueueWithProperties: " << ret << std::endl, 1;

    cl_program prog = create_program_from_file(main_ctx, "kernels/task2.cl", &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateProgramWithSource: " << ret << std::endl, 1;

    if (ret = clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr) != CL_SUCCESS)
        return std::cerr << "Error in clBuildProgram: " << ret << std::endl, 1;

    cl_kernel kernel = clCreateKernel(prog, "daxpy", &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateKernel: " << ret << std::endl, 1;

    cl_event buf_e[2];

    cl_mem buff_y = clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, n * incy * sizeof(double), y, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateBuffer: " << ret << std::endl, 1;

    cl_mem buff_x = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, n * incx * sizeof(double), x, &ret);
    if (ret != CL_SUCCESS)
        return std::cerr << "Error in clCreateBuffer: " << ret << std::endl, 1;

    if (ret = clEnqueueWriteBuffer(main_queue, buff_y, CL_FALSE, 0, n * incy * sizeof(double), y, 0, nullptr, &buf_e[0]) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueWriteBuffer: " << ret << std::endl, 1;

    if (ret = clEnqueueWriteBuffer(main_queue, buff_x, CL_FALSE, 0, n * incx * sizeof(double), x, 0, nullptr, &buf_e[1]) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueWriteBuffer: " << ret << std::endl, 1;

    if (ret = clSetKernelArg(kernel, 0, sizeof(double), &a) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;
    if (ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buff_x) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;
    if (ret = clSetKernelArg(kernel, 2, sizeof(size_t), &incx) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;
    if (ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &buff_y) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;
    if (ret = clSetKernelArg(kernel, 4, sizeof(size_t), &incy) != CL_SUCCESS)
        return std::cout << "Error in clSetKernelArg: " << ret << std::endl, 1;

    cl_event main_e;
    if (ret = clEnqueueNDRangeKernel(main_queue, kernel, 1, nullptr, &n, &group_size, 2, buf_e, &main_e) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueNDRangeKernel: " << ret << std::endl, 1;

    if (ret = clEnqueueReadBuffer(main_queue, buff_y, CL_TRUE, 0, n * incy * sizeof(double), y, 1, &main_e, nullptr) != CL_SUCCESS)
        return std::cerr << "Error in clEnqueueReadBuffer: " << ret << std::endl, 1;

    if (ret = clFinish(main_queue) != CL_SUCCESS)
        return std::cerr << "Error in clFinish: " << ret << std::endl, 1;

    if (ret = clReleaseMemObject(buff_x) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseMemObject: " << ret << std::endl, 1;
    if (ret = clReleaseMemObject(buff_y) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseMemObject: " << ret << std::endl, 1;
    if (ret = clReleaseKernel(kernel) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseKernel: " << ret << std::endl, 1;
    if (ret = clReleaseProgram(prog) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseProgram: " << ret << std::endl, 1;
    if (ret = clReleaseCommandQueue(main_queue) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseCommandQueue: " << ret << std::endl, 1;
    if (ret = clReleaseContext(main_ctx) != CL_SUCCESS)
        return std::cerr << "Error in clReleaseContext: " << ret << std::endl, 1;
    return 0;
}

int main(int argc, char** argv)
{
    cl_int ret = CL_SUCCESS;

    cl_device_id gpu_device = get_any_gpu_device();
    cl_device_id cpu_device = get_any_cpu_device();

    size_t n = 6400000;
    size_t inc_x = 4;
    size_t inc_y = 3;
    size_t group_size = 256;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-n") == 0)
        {
            n = atoll(argv[++i]);
        }
        else if (strcmp(argv[i], "-x") == 0)
        {
            inc_x = atoll(argv[++i]);
        }
        else if (strcmp(argv[i], "-y") == 0)
        {
            inc_y = atoll(argv[++i]);
        }
        else if (strcmp(argv[i], "-g") == 0)
        {
            group_size = atoll(argv[++i]);
        }
    }

    size_t x_size = n * inc_x;
    size_t y_size = n * inc_y;

    std::cout << "float" << std::endl;
    {
        std::mt19937 mt;
        std::uniform_real_distribution<float> urd;
        float a = urd(mt);

        std::vector<float> x1(x_size), x2, x3;
        std::vector<float> y1(y_size), y2, y3;

        for (float& i: x1)
            i = urd(mt);
        for (float& i: y1)
            i = urd(mt);

        x3 = x1;
        y3 = y1;
        double t1 = omp_get_wtime();
        saxpy(n, a, x3.data(), inc_x, y3.data(), inc_y);
        t1 = omp_get_wtime() - t1;
        std::cout << "time simple: " << t1 << std::endl;

        x2 = x1;
        y2 = y1;
        double t3 = omp_get_wtime();
        saxpy_omp(n, a, x2.data(), inc_x, y2.data(), inc_y);
        t3 = omp_get_wtime() - t3;
        std::cout << "time omp: " << t3 << std::endl;
        if (x2 == x3)
            std::cout << "correct" << std::endl;
        else
            std::cout << "wrong" << std::endl;

        if (gpu_device != nullptr)
        {
            x2 = x1;
            y2 = y1;
            double t2 = omp_get_wtime();
            ret = saxpy_device(n, a, x2.data(), inc_x, y2.data(), inc_y, gpu_device, group_size);
            t2 = omp_get_wtime() - t2;
            if (ret != 0)
                return std::cout << "Error in saxpy_device" << std::endl, 1;
            else
                std::cout << "time gpu: " << t2 << std::endl;
            if (x2 == x3)
                std::cout << "correct" << std::endl;
            else
                std::cout << "wrong" << std::endl;
        }
        else
            std::cout << "gpu device not found" << std::endl;

        if (cpu_device != nullptr)
        {
            x2 = x1;
            y2 = y1;
            double t2 = omp_get_wtime();
            ret = saxpy_device(n, a, x2.data(), inc_x, y2.data(), inc_y, cpu_device, group_size);
            t2 = omp_get_wtime() - t2;
            if (ret != 0)
                return std::cout << "Error in saxpy_device" << std::endl, 1;
            else
                std::cout << "time cpu: " << t2 << std::endl;
        }
        else
            std::cout << "cpu device not found" << std::endl;
    }
    std::cout << std::endl;
    std::cout << "double" << std::endl;
    {
        std::mt19937_64 mt;
        std::uniform_real_distribution<double> urd;
        double a = urd(mt);

        std::vector<double> x1(x_size), x2, x3;
        std::vector<double> y1(y_size), y2, y3;

        for (double& i: x1)
            i = urd(mt);
        for (double& i: y1)
            i = urd(mt);

        x3 = x1;
        y3 = y1;
        double t1 = omp_get_wtime();
        daxpy(n, a, x3.data(), inc_x, y3.data(), inc_y);
        t1 = omp_get_wtime() - t1;
        std::cout << "time simple: " << t1 << std::endl;

        x2 = x1;
        y2 = y1;
        double t3 = omp_get_wtime();
        daxpy_omp(n, a, x2.data(), inc_x, y2.data(), inc_y);
        t3 = omp_get_wtime() - t3;
        std::cout << "time omp: " << t3 << std::endl;
        if (x2 == x3)
            std::cout << "correct" << std::endl;
        else
            std::cout << "wrong" << std::endl;

        if (gpu_device != nullptr)
        {
            x2 = x1;
            y2 = y1;
            double t2 = omp_get_wtime();
            ret = daxpy_device(n, a, x2.data(), inc_x, y2.data(), inc_y, gpu_device, group_size);
            t2 = omp_get_wtime() - t2;
            if (ret != 0)
                return std::cout << "Error in saxpy_device" << std::endl, 1;
            else
                std::cout << "time gpu: " << t2 << std::endl;
            if (x2 == x3)
                std::cout << "correct" << std::endl;
            else
                std::cout << "wrong" << std::endl;
        }
        else
            std::cout << "gpu device not found" << std::endl;

        if (cpu_device != nullptr)
        {
            x2 = x1;
            y2 = y1;
            double t2 = omp_get_wtime();
            ret = daxpy_device(n, a, x2.data(), inc_x, y2.data(), inc_y, cpu_device, group_size);
            t2 = omp_get_wtime() - t2;
            if (ret != 0)
                return std::cout << "Error in saxpy_device" << std::endl, 1;
            else
                std::cout << "time cpu: " << t2 << std::endl;
        }
        else
            std::cout << "cpu device not found" << std::endl;
    }
}
