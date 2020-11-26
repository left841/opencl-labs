#include <iostream>
#include <random>
#include <ctime>
#include <omp.h>
#include <CL/cl.h>
#include "common_devices.h"

#define CL_CHECK_AND_RET(func) \
do \
{ \
    cl_int ret; \
    if (ret = func != CL_SUCCESS) \
    { \
        std::cerr << "Error " << ret << " in " << #func << std::endl; \
        exit(1); \
    } \
} \
while(0) \

#define CL_CHECK_AND_RET2(ret, func) \
do \
{ \
    if (ret != CL_SUCCESS) \
    { \
        std::cerr << "Error " << ret << " in " << #func << std::endl; \
        exit(1); \
    } \
} \
while(0) \

template<typename Type>
class matrix
{
private:
    Type* arr;
    size_t size_, length_;

public:
    matrix(const size_t& height, const size_t& width): size_(height), length_(width)
    { arr = new Type[size_ * length_]; }

    matrix(const matrix<Type>& m): size_(m.size_), length_(m.length_), arr(new Type[size_ * length_])
    {
        for (size_t i = 0; i < size_; ++i)
            for (size_t j = 0; j < length_; ++j)
                (*this)[i][j] = m[i][j];
    }

    matrix<Type>& operator=(const matrix<Type>& m)
    {
        if (this == &m)
            return *this;
        delete[] arr;
        size_ = m.size_;
        length_ = m.length_;
        arr = new Type[size_ * length_];
        for (size_t i = 0; i < size_; ++i)
            for (size_t j = 0; j < length_; ++j)
                arr[i][j] = m.arr[i][j];
    }

    ~matrix()
    { delete[] arr; }

    bool operator==(const matrix<Type>& m)
    {
        if ((size_ != m.size_) || (length_ != m.length_))
            return false;
        size_t all_length = length_ * size_;
        for (size_t i = 0; i < all_length; ++i)
            if (arr[i] != m.arr[i])
                return false;
        return true;
    }

    bool operator!=(const matrix<Type>& m)
    { return !this->operator==(m); }

    Type* operator[](size_t n)
    { return arr + length_ * n; }

    const Type* operator[](size_t n) const
    { return arr + length_ * n; }

    size_t height() const
    { return size_; }

    size_t width() const
    { return length_; }

    size_t size() const
    { return length_ * size_; }

    Type* data()
    { return arr; }

    const Type* data() const
    { return arr; }

    void clear()
    {
        size_t all_length = length_ * size_;
        for (size_t i = 0; i < all_length; ++i)
            arr[i] = Type();
    }
};

void matrix_mul(const matrix<int>& a, const matrix<int>& b, matrix<int>& c)
{
    for (size_t i = 0; i < a.height(); ++i)
        for (size_t k = 0; k < b.width(); ++k)
        {
            c[i][k] = 0;
            for (size_t j = 0; j < a.width(); ++j)
                c[i][k] += a[i][j] * b[j][k];
        }
}

void matrix_mul_omp(const matrix<int>& a, const matrix<int>& b, matrix<int>& c)
{
    ptrdiff_t a_height = a.height();
    #pragma omp parallel for
    for (ptrdiff_t i = 0; i < a_height; ++i)
        for (size_t k = 0; k < b.width(); ++k)
        {
            c[i][k] = 0;
            for (size_t j = 0; j < a.width(); ++j)
                c[i][k] += a[i][j] * b[j][k];
        }
}

void matrix_mul_cl(const matrix<int>& a, const matrix<int>& b, matrix<int>& c, cl_device_id device)
{
    cl_int ret = CL_SUCCESS;

    cl_context main_ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
    CL_CHECK_AND_RET2(ret, clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret));

    cl_command_queue main_queue = clCreateCommandQueueWithProperties(main_ctx, device, nullptr, &ret);
    CL_CHECK_AND_RET2(ret, clCreateCommandQueueWithProperties(main_ctx, device, nullptr, &ret));

    cl_program prog = create_program_from_file(main_ctx, "kernels/task3.cl", &ret);
    CL_CHECK_AND_RET2(ret, create_program_from_file(main_ctx, "kernels/task3.cl", &ret));

    CL_CHECK_AND_RET(clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr));

    cl_kernel kernel = clCreateKernel(prog, "matrix_mul", &ret);
    CL_CHECK_AND_RET2(ret, clCreateKernel(prog, "matrix_mul", &ret));

    // buffer creating
    cl_event buf_e[3]{0};

    cl_mem buff_a = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size(), const_cast<matrix<int>&>(a).data(), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size(), const_cast<matrix<int>&>(a).data(), &ret));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, buff_a, CL_FALSE, 0, a.size(), a.data(), 0, nullptr, &buf_e[0]));

    cl_mem buff_b = clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size(), const_cast<matrix<int>&>(b).data(), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size(), const_cast<matrix<int>&>(b).data(), &ret));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, buff_b, CL_FALSE, 0, b.size(), b.data(), 0, nullptr, &buf_e[1]));

    cl_mem buff_c = clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c.size(), c.data(), &ret);
    CL_CHECK_AND_RET2(ret, clCreateBuffer(main_ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c.size(), c.data(), &ret));
    CL_CHECK_AND_RET(clEnqueueWriteBuffer(main_queue, buff_a, CL_FALSE, 0, c.size(), c.data(), 0, nullptr, &buf_e[2]));


    CL_CHECK_AND_RET(clReleaseKernel(kernel));
    CL_CHECK_AND_RET(clReleaseProgram(prog));
    CL_CHECK_AND_RET(clReleaseCommandQueue(main_queue));
    CL_CHECK_AND_RET(clReleaseContext(main_ctx));
}

int main(size_t argc, char** argv)
{
    cl_int ret = CL_SUCCESS;

    double t;
    size_t n = 16, m = 16, l = 16;
    for (int i = 1; i < argc; ++i)
    {
        if ((strcmp(argv[i], "-s") == 0) || (strcmp(argv[i], "-size") == 0))
        {
            n = atoll(argv[++i]);
            m = atoll(argv[++i]);
            l = atoll(argv[++i]);
        }
    }

    matrix<int> a(n, m), b(m, l), c1(n, l), c2(n, l);

    std::mt19937 mt(static_cast<unsigned>(time(0)));
    std::uniform_int_distribution<int> uid(-4, 4);

    for (size_t i = 0; i < a.height(); ++i)
        for (size_t j = 0; j < a.width(); ++j)
            a[i][j] = uid(mt);

    for (size_t i = 0; i < b.height(); ++i)
        for (size_t j = 0; j < b.width(); ++j)
            b[i][j] = uid(mt);

    t = omp_get_wtime();
    matrix_mul(a, b, c1);
    std::cout << "simple: " << omp_get_wtime() - t << std::endl;

    t = omp_get_wtime();
    matrix_mul_omp(a, b, c2);
    std::cout << "omp: " << omp_get_wtime() - t << std::endl;

    if (c1 == c2)
        std::cout << "omp correct" << std::endl;
    else
        std::cout << "omp wrong" << std::endl;
    c2.clear();

    cl_device_id gpu_device = get_any_gpu_device();
    if (gpu_device != nullptr)
    {
        t = omp_get_wtime();
        matrix_mul_cl(a, b, c2, gpu_device);
        std::cout << "gpu: " << omp_get_wtime() - t << std::endl;

        if (c1 == c2)
            std::cout << "gpu correct" << std::endl;
        else
            std::cout << "gpu wrong" << std::endl;
        c2.clear();
    }
    else
        std::cout << "gpu device not found" << std::endl;

    cl_device_id cpu_device = get_any_cpu_device();
    if (cpu_device != nullptr)
    {
        t = omp_get_wtime();
        matrix_mul_cl(a, b, c2, cpu_device);
        std::cout << "cpu: " << omp_get_wtime() - t << std::endl;

        if (c1 == c2)
            std::cout << "cpu correct" << std::endl;
        else
            std::cout << "cpu wrong" << std::endl;
    }
    else
        std::cout << "cpu device not found" << std::endl;
}