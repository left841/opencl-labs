#include <cstring>
#include <cstdlib>
#include <limits>
#include <random>
#include <iostream>
#include <vector>
#include <array>
#include <CL/cl.h>
#include "omp.h"
#include "common_devices.h"

constexpr const size_t LOCAL_SIZE_PRM = 16;

template<typename Type>
class matrix
{
private:
    Type* arr;
    std::array<size_t, 2> length_v;

public:
    matrix(size_t height, size_t width) : length_v{ height, width }
    { arr = new Type[length_v[0] * length_v[1]]; }

    matrix(const matrix& m) : length_v(m.length_v)
    {
        arr = new Type[length_v[0] * length_v[1]];
        for (size_t i = 0; i < length_v[0] * length_v[1]; ++i)
            arr[i] = m.arr[i];
    }

    matrix& operator=(const matrix& m)
    {
        if (&m != this)
        {
            ~matrix();
            matrix(m);
        }
        return *this;
    }

    ~matrix()
    { delete[] arr; }

    bool operator==(const matrix& m)
    {
        if ((length_v[0] != m.length_v[0]) || (length_v[1] != m.length_v[1]))
            return false;
        for (size_t i = 0; i < length_v[0] * length_v[1]; ++i)
            if (arr[i] != m.arr[i])
                return false;
        return true;
    }

    bool operator!=(const matrix& m)
    { return !operator==(m); }

    Type* operator[](size_t n)
    { return arr + length_v[1] * n; }

    const Type* operator[](size_t n) const
    { return arr + length_v[1] * n; }

    size_t size() const
    { return length_v[0] * length_v[1]; }

    const std::array<size_t, 2>& length() const
    { return length_v; }

    Type* data()
    { return arr; }

    const Type* data() const
    { return arr; }
};

void matrix_mul(const matrix<int>& a, const matrix<int>& b, matrix<int>& c)
{
    for (size_t i = 0; i < a.length()[0]; ++i)
        for (size_t j = 0; j < b.length()[1]; ++j)
        {
            c[i][j] = 0;
            for (size_t k = 0; k < a.length()[1]; ++k)
                c[i][j] += a[i][k] * b[k][j];
        }
}

void matrix_mul_omp(const matrix<int>& a, const matrix<int>& b, matrix<int>& c)
{
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(a.length()[0]); ++i)
        for (size_t j = 0; j < b.length()[1]; ++j)
        {
            c[i][j] = 0;
            for (size_t k = 0; k < a.length()[1]; ++k)
                c[i][j] += a[i][k] * b[k][j];
        }
}

void matrix_mul_cl_simple(const matrix<int>& a, const matrix<int>& b, matrix<int>& c, cl_device_id device)
{
    cl_int ret = CL_SUCCESS;
    size_t n = a.length()[0], m = a.length()[1], l = b.length()[1];
    size_t size[2] = { n, l }, group_size[2] = { LOCAL_SIZE_PRM, LOCAL_SIZE_PRM };

    cl_context main_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
    CL_CHECK_AND_RET2(ret, clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret));

    cl_command_queue main_queue = clCreateCommandQueue(main_ctx, device, 0, &ret);
    CL_CHECK_AND_RET2(ret, clCreateCommandQueue(main_ctx, device, 0, &ret));

    cl_program prog = create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret);
    CL_CHECK_AND_RET2(ret, create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret));

    CL_CHECK_AND_RET(clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(prog, "matrix_mul_simple", &ret);
    CL_CHECK_AND_RET2(ret, clCreateKernel(prog, "matrix_mul_simple", &ret));

    cl_mem a_buf = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(int), const_cast<int*>(a.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(int), const_cast<int*>(a.data()), &ret));
    cl_mem b_buf = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(int), const_cast<int*>(b.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(int), const_cast<int*>(b.data()), &ret));
    cl_mem c_buf = clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c.size() * sizeof(int), c.data(), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c.size() * sizeof(int), c.data(), &ret));

    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, a_buf, CL_TRUE, 0, a.size() * sizeof(int), const_cast<int*>(a.data()), 0, nullptr, nullptr));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, b_buf, CL_TRUE, 0, b.size() * sizeof(int), const_cast<int*>(b.data()), 0, nullptr, nullptr));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, c_buf, CL_TRUE, 0, c.size() * sizeof(int), c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 3, sizeof(size_t), &n));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 4, sizeof(size_t), &m));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 5, sizeof(size_t), &l));

    cl_event main_e;
    double t = omp_get_wtime();
    CL_CHECK_AND_RET(clEnqueueNDRangeKernel(main_queue, kernel, 2, nullptr, size, group_size, 0, nullptr, &main_e));
    CL_CHECK_AND_RET(clWaitForEvents(1, &main_e));
    std::cout << "kernel_time: " << omp_get_wtime() - t << std::endl;

    CL_CHECK_AND_RET(clEnqueueReadBuffer(main_queue, c_buf, CL_TRUE, 0, c.size() * sizeof(int), c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clFinish(main_queue));

    CL_CHECK_AND_RET(clReleaseMemObject(a_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(b_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(c_buf));
    CL_CHECK_AND_RET(clReleaseKernel(kernel));
    CL_CHECK_AND_RET(clReleaseProgram(prog));
    CL_CHECK_AND_RET(clReleaseCommandQueue(main_queue));
    CL_CHECK_AND_RET(clReleaseContext(main_ctx));
}

void matrix_mul_cl_simple_optimized(const matrix<int>& a, const matrix<int>& b, matrix<int>& c, cl_device_id device)
{
    cl_int ret = CL_SUCCESS;
    size_t n = a.length()[0], m = a.length()[1], l = b.length()[1];
    size_t size[2] = { l, n }, group_size[2] = { LOCAL_SIZE_PRM, LOCAL_SIZE_PRM };

    cl_context main_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
    CL_CHECK_AND_RET2(ret, clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret));

    cl_command_queue main_queue = clCreateCommandQueue(main_ctx, device, 0, &ret);
    CL_CHECK_AND_RET2(ret, clCreateCommandQueue(main_ctx, device, 0, &ret));

    cl_program prog = create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret);
    CL_CHECK_AND_RET2(ret, create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret));

    CL_CHECK_AND_RET(clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(prog, "matrix_mul_simple_optimized", &ret);
    CL_CHECK_AND_RET2(ret, clCreateKernel(prog, "matrix_mul_simple_optimized", &ret));

    cl_mem a_buf = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(int), const_cast<int*>(a.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(int), const_cast<int*>(a.data()), &ret));
    cl_mem b_buf = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(int), const_cast<int*>(b.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(int), const_cast<int*>(b.data()), &ret));
    cl_mem c_buf = clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c.size() * sizeof(int), c.data(), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c.size() * sizeof(int), c.data(), &ret));

    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, a_buf, CL_TRUE, 0, a.size() * sizeof(int), const_cast<int*>(a.data()), 0, nullptr, nullptr));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, b_buf, CL_TRUE, 0, b.size() * sizeof(int), const_cast<int*>(b.data()), 0, nullptr, nullptr));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, c_buf, CL_TRUE, 0, c.size() * sizeof(int), c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 3, sizeof(size_t), &n));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 4, sizeof(size_t), &m));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 5, sizeof(size_t), &l));

    cl_event main_e;
    double t = omp_get_wtime();
    CL_CHECK_AND_RET(clEnqueueNDRangeKernel(main_queue, kernel, 2, nullptr, size, group_size, 0, nullptr, &main_e));
    CL_CHECK_AND_RET(clWaitForEvents(1, &main_e));
    std::cout << "kernel_time: " << omp_get_wtime() - t << std::endl;

    CL_CHECK_AND_RET(clEnqueueReadBuffer(main_queue, c_buf, CL_TRUE, 0, c.size() * sizeof(int), c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clFinish(main_queue));

    CL_CHECK_AND_RET(clReleaseMemObject(a_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(b_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(c_buf));
    CL_CHECK_AND_RET(clReleaseKernel(kernel));
    CL_CHECK_AND_RET(clReleaseProgram(prog));
    CL_CHECK_AND_RET(clReleaseCommandQueue(main_queue));
    CL_CHECK_AND_RET(clReleaseContext(main_ctx));
}

void matrix_mul_cl(const matrix<int>& a, const matrix<int>& b, matrix<int>& c, cl_device_id device)
{
    cl_int ret = CL_SUCCESS;
    size_t n = a.length()[0], m = a.length()[1], l = b.length()[1];
    size_t size[2] = { n, l }, group_size[2] = { LOCAL_SIZE_PRM, LOCAL_SIZE_PRM };

    cl_context main_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
    CL_CHECK_AND_RET2(ret, clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret));

    cl_command_queue main_queue = clCreateCommandQueue(main_ctx, device, 0, &ret);
    CL_CHECK_AND_RET2(ret, clCreateCommandQueue(main_ctx, device, 0, &ret));

    cl_program prog = create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret);
    CL_CHECK_AND_RET2(ret, create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret));

    CL_CHECK_AND_RET(clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(prog, "matrix_mul", &ret);
    CL_CHECK_AND_RET2(ret, clCreateKernel(prog, "matrix_mul", &ret));

    cl_mem a_buf = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(int), const_cast<int*>(a.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(int), const_cast<int*>(a.data()), &ret));
    cl_mem b_buf = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(int), const_cast<int*>(b.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(int), const_cast<int*>(b.data()), &ret));
    cl_mem c_buf = clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c.size() * sizeof(int), c.data(), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c.size() * sizeof(int), c.data(), &ret));

    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, a_buf, CL_TRUE, 0, a.size() * sizeof(int), const_cast<int*>(a.data()), 0, nullptr, nullptr));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, b_buf, CL_TRUE, 0, b.size() * sizeof(int), const_cast<int*>(b.data()), 0, nullptr, nullptr));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, c_buf, CL_TRUE, 0, c.size() * sizeof(int), c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 3, sizeof(size_t), &n));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 4, sizeof(size_t), &m));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 5, sizeof(size_t), &l));

    cl_event main_e;
    double t = omp_get_wtime();
    CL_CHECK_AND_RET(clEnqueueNDRangeKernel(main_queue, kernel, 2, nullptr, size, group_size, 0, nullptr, &main_e));
    CL_CHECK_AND_RET(clWaitForEvents(1, &main_e));
    std::cout << "kernel_time: " << omp_get_wtime() - t << std::endl;

    CL_CHECK_AND_RET(clEnqueueReadBuffer(main_queue, c_buf, CL_TRUE, 0, c.size() * sizeof(int), c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clFinish(main_queue));

    CL_CHECK_AND_RET(clReleaseMemObject(a_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(b_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(c_buf));
    CL_CHECK_AND_RET(clReleaseKernel(kernel));
    CL_CHECK_AND_RET(clReleaseProgram(prog));
    CL_CHECK_AND_RET(clReleaseCommandQueue(main_queue));
    CL_CHECK_AND_RET(clReleaseContext(main_ctx));
}

void matrix_mul_cl_optimized(const matrix<int>& a, const matrix<int>& b, matrix<int>& c, cl_device_id device)
{
    cl_int ret = CL_SUCCESS;
    size_t n = a.length()[0], m = a.length()[1], l = b.length()[1];
    size_t size[2] = { l, n }, group_size[2] = { LOCAL_SIZE_PRM, LOCAL_SIZE_PRM };

    cl_context main_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
    CL_CHECK_AND_RET2(ret, clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret));

    cl_command_queue main_queue = clCreateCommandQueue(main_ctx, device, 0, &ret);
    CL_CHECK_AND_RET2(ret, clCreateCommandQueue(main_ctx, device, 0, &ret));

    cl_program prog = create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret);
    CL_CHECK_AND_RET2(ret, create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret));

    CL_CHECK_AND_RET(clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(prog, "matrix_mul_optimized", &ret);
    CL_CHECK_AND_RET2(ret, clCreateKernel(prog, "matrix_mul_optimized", &ret));

    cl_mem a_buf = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(int), const_cast<int*>(a.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(int), const_cast<int*>(a.data()), &ret));
    cl_mem b_buf = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(int), const_cast<int*>(b.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(int), const_cast<int*>(b.data()), &ret));
    cl_mem c_buf = clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c.size() * sizeof(int), c.data(), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c.size() * sizeof(int), c.data(), &ret));

    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, a_buf, CL_TRUE, 0, a.size() * sizeof(int), const_cast<int*>(a.data()), 0, nullptr, nullptr));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, b_buf, CL_TRUE, 0, b.size() * sizeof(int), const_cast<int*>(b.data()), 0, nullptr, nullptr));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, c_buf, CL_TRUE, 0, c.size() * sizeof(int), c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 3, sizeof(size_t), &n));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 4, sizeof(size_t), &m));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 5, sizeof(size_t), &l));

    cl_event main_e;
    double t = omp_get_wtime();
    CL_CHECK_AND_RET(clEnqueueNDRangeKernel(main_queue, kernel, 2, nullptr, size, group_size, 0, nullptr, &main_e));
    CL_CHECK_AND_RET(clWaitForEvents(1, &main_e));
    std::cout << "kernel_time: " << omp_get_wtime() - t << std::endl;

    CL_CHECK_AND_RET(clEnqueueReadBuffer(main_queue, c_buf, CL_TRUE, 0, c.size() * sizeof(int), c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clFinish(main_queue));

    CL_CHECK_AND_RET(clReleaseMemObject(a_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(b_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(c_buf));
    CL_CHECK_AND_RET(clReleaseKernel(kernel));
    CL_CHECK_AND_RET(clReleaseProgram(prog));
    CL_CHECK_AND_RET(clReleaseCommandQueue(main_queue));
    CL_CHECK_AND_RET(clReleaseContext(main_ctx));
}

void matrix_mul_cl_img(const matrix<int>& a, const matrix<int>& b, matrix<int>& c, cl_device_id device)
{
    cl_int ret = CL_SUCCESS;
    size_t n = a.length()[0], m = a.length()[1], l = b.length()[1];
    size_t size[2] = {n, l}, group_size[2] = { LOCAL_SIZE_PRM, LOCAL_SIZE_PRM };

    cl_context main_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
    CL_CHECK_AND_RET2(ret, clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret));

    cl_command_queue main_queue = clCreateCommandQueue(main_ctx, device, 0, &ret);
    CL_CHECK_AND_RET2(ret, clCreateCommandQueue(main_ctx, device, 0, &ret));

    cl_program prog = create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret);
    CL_CHECK_AND_RET2(ret, create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret));

    CL_CHECK_AND_RET(clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(prog, "matrix_mul_img", &ret);
    CL_CHECK_AND_RET2(ret, clCreateKernel(prog, "matrix_mul_img", &ret));

    cl_image_format form;
    form.image_channel_order = CL_R;
    form.image_channel_data_type = CL_SIGNED_INT32;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {0, 0, 1};

    cl_mem a_buf = clCreateImage2D(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &form, m, n, 0, const_cast<int*>(a.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateImage2D(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &form, m, n, 0, const_cast<int*>(a.data()), &ret));
    cl_mem b_buf = clCreateImage2D(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &form, l, m, 0, const_cast<int*>(b.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateImage2D(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &form, l, m, 0, const_cast<int*>(b.data()), &ret));
    cl_mem c_buf = clCreateImage2D(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &form, l, n, 0, c.data(), &ret);

    region[0] = m;
    region[1] = n;
    CL_CHECK_AND_RET(clEnqueueWriteImage(main_queue, a_buf, CL_TRUE, origin, region, 0, 0, const_cast<int*>(a.data()), 0, nullptr, nullptr));
    region[0] = l;
    region[1] = m;
    CL_CHECK_AND_RET(clEnqueueWriteImage(main_queue, b_buf, CL_TRUE, origin, region, 0, 0, const_cast<int*>(b.data()), 0, nullptr, nullptr));
    region[0] = l;
    region[1] = n;
    CL_CHECK_AND_RET(clEnqueueWriteImage(main_queue, c_buf, CL_TRUE, origin, region, 0, 0, c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 3, sizeof(size_t), &n));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 4, sizeof(size_t), &m));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 5, sizeof(size_t), &l));

    cl_event main_e;
    double t = omp_get_wtime();
    CL_CHECK_AND_RET(clEnqueueNDRangeKernel(main_queue, kernel, 2, nullptr, size, group_size, 0, nullptr, &main_e));
    CL_CHECK_AND_RET(clWaitForEvents(1, &main_e));
    std::cout << "kernel_time: " << omp_get_wtime() - t << std::endl;

    CL_CHECK_AND_RET(clEnqueueReadImage(main_queue, c_buf, CL_TRUE, origin, region, 0, 0, c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clFinish(main_queue));

    CL_CHECK_AND_RET(clReleaseMemObject(a_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(b_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(c_buf));
    CL_CHECK_AND_RET(clReleaseKernel(kernel));
    CL_CHECK_AND_RET(clReleaseProgram(prog));
    CL_CHECK_AND_RET(clReleaseCommandQueue(main_queue));
    CL_CHECK_AND_RET(clReleaseContext(main_ctx));
}

void matrix_mul_cl_img_optimized(const matrix<int>& a, const matrix<int>& b, matrix<int>& c, cl_device_id device)
{
    cl_int ret = CL_SUCCESS;
    size_t n = a.length()[0], m = a.length()[1], l = b.length()[1];
    size_t size[2] = {l, n}, group_size[2] = { LOCAL_SIZE_PRM, LOCAL_SIZE_PRM };

    cl_context main_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
    CL_CHECK_AND_RET2(ret, clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret));

    cl_command_queue main_queue = clCreateCommandQueue(main_ctx, device, 0, &ret);
    CL_CHECK_AND_RET2(ret, clCreateCommandQueue(main_ctx, device, 0, &ret));

    cl_program prog = create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret);
    CL_CHECK_AND_RET2(ret, create_program_from_file(main_ctx, "kernels/matrix_mul.cl", &ret));

    CL_CHECK_AND_RET(clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(prog, "matrix_mul_img_optimized", &ret);
    CL_CHECK_AND_RET2(ret, clCreateKernel(prog, "matrix_mul_img_optimized", &ret));

    cl_image_format form;
    form.image_channel_order = CL_R;
    form.image_channel_data_type = CL_SIGNED_INT32;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {0, 0, 1};

    cl_mem a_buf = clCreateImage2D(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &form, m, n, 0, const_cast<int*>(a.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateImage2D(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &form, m, n, 0, const_cast<int*>(a.data()), &ret));
    cl_mem b_buf = clCreateImage2D(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &form, l, m, 0, const_cast<int*>(b.data()), &ret);
    CL_CHECK_AND_RET2(ret, clCreateImage2D(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &form, l, m, 0, const_cast<int*>(b.data()), &ret));
    cl_mem c_buf = clCreateImage2D(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &form, l, n, 0, c.data(), &ret);

    region[0] = m;
    region[1] = n;
    CL_CHECK_AND_RET(clEnqueueWriteImage(main_queue, a_buf, CL_TRUE, origin, region, 0, 0, const_cast<int*>(a.data()), 0, nullptr, nullptr));
    region[0] = l;
    region[1] = m;
    CL_CHECK_AND_RET(clEnqueueWriteImage(main_queue, b_buf, CL_TRUE, origin, region, 0, 0, const_cast<int*>(b.data()), 0, nullptr, nullptr));
    region[0] = l;
    region[1] = n;
    CL_CHECK_AND_RET(clEnqueueWriteImage(main_queue, c_buf, CL_TRUE, origin, region, 0, 0, c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 3, sizeof(size_t), &n));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 4, sizeof(size_t), &m));
    CL_CHECK_AND_RET(clSetKernelArg(kernel, 5, sizeof(size_t), &l));

    cl_event main_e;
    double t = omp_get_wtime();
    CL_CHECK_AND_RET(clEnqueueNDRangeKernel(main_queue, kernel, 2, nullptr, size, group_size, 0, nullptr, &main_e));
    CL_CHECK_AND_RET(clWaitForEvents(1, &main_e));
    std::cout << "kernel_time: " << omp_get_wtime() - t << std::endl;

    CL_CHECK_AND_RET(clEnqueueReadImage(main_queue, c_buf, CL_TRUE, origin, region, 0, 0, c.data(), 0, nullptr, nullptr));

    CL_CHECK_AND_RET(clFinish(main_queue));

    CL_CHECK_AND_RET(clReleaseMemObject(a_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(b_buf));
    CL_CHECK_AND_RET(clReleaseMemObject(c_buf));
    CL_CHECK_AND_RET(clReleaseKernel(kernel));
    CL_CHECK_AND_RET(clReleaseProgram(prog));
    CL_CHECK_AND_RET(clReleaseCommandQueue(main_queue));
    CL_CHECK_AND_RET(clReleaseContext(main_ctx));
}

int main(int argc, char** argv)
{
    size_t size_n = 256, size_m = 256, size_l = 256;
    double t;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-s") == 0)
        {
            size_n = atoll(argv[++i]);
            size_m = atoll(argv[++i]);
            size_l = atoll(argv[++i]);
        }
    }

    cl_device_id gpu_id = get_any_gpu_device();
    cl_device_id cpu_id = get_any_cpu_device();

    matrix<int> a(size_n, size_m);
    matrix<int> b(size_m, size_l);
    std::mt19937 mt(static_cast<unsigned>(time(0)));
    std::uniform_int_distribution<int> uid(0, 3);
    for (size_t i = 0; i < a.length()[0]; ++i)
        for (size_t j = 0; j < a.length()[1]; ++j)
            a[i][j] = uid(mt);
    for (size_t i = 0; i < b.length()[0]; ++i)
        for (size_t j = 0; j < b.length()[1]; ++j)
            b[i][j] = uid(mt);

    matrix<int> c(size_n, size_l);
    t = omp_get_wtime();
    matrix_mul(a, b, c);
    std::cout << "simple time: " << omp_get_wtime() - t << std::endl;

    matrix<int> c_omp(size_n, size_l);
    t = omp_get_wtime();
    matrix_mul_omp(a, b, c_omp);
    t = omp_get_wtime() - t;
    if (c == c_omp)
        std::cout << "omp time: " << t << std::endl;
    else
        std::cout << "omp wrong" << std::endl;
    std::cout << std::endl;

    if (gpu_id != nullptr)
    {
        {
            std::cout << "gpu simple" << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl_simple(a, b, c_dev, gpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
        {
            std::cout << "gpu simple optimized" << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl_simple_optimized(a, b, c_dev, gpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
        {
            std::cout << "gpu" << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl(a, b, c_dev, gpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
        {
            std::cout << "gpu optimized " << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl_optimized(a, b, c_dev, gpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
        {
            std::cout << "gpu img" << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl_img(a, b, c_dev, gpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
        {
            std::cout << "gpu img optimized" << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl_img_optimized(a, b, c_dev, gpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
    }
    else
        std::cout << "gpu device not found" << std::endl;

    if (cpu_id != nullptr)
    {
        {
            std::cout << "cpu simple" << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl_simple(a, b, c_dev, cpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
        {
            std::cout << "cpu simple optimized" << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl_simple_optimized(a, b, c_dev, cpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
        {
            std::cout << "cpu" << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl(a, b, c_dev, cpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
        {
            std::cout << "cpu optimized " << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl_optimized(a, b, c_dev, cpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
        {
            std::cout << "cpu img" << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl_img(a, b, c_dev, cpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
        {
            std::cout << "cpu img optimized" << std::endl;
            matrix<int> c_dev(size_n, size_l);
            t = omp_get_wtime();
            matrix_mul_cl_img_optimized(a, b, c_dev, cpu_id);
            t = omp_get_wtime() - t;
            if (c == c_dev)
                std::cout << "all time: " << t << std::endl;
            else
                std::cout << "wrong" << std::endl;
            std::cout << std::endl;
        }
    }
    else
        std::cout << "cpu device not found" << std::endl;
}