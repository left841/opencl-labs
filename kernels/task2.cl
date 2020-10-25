__kernel void saxpy(float a, __global float* x, size_t incx, __global float* y, size_t incy)
{
    size_t id = get_global_id(0);
    y[id * incy] += x[id * incx] * a;
}

__kernel void daxpy(double a, __global double* x, size_t incx, __global double* y, size_t incy)
{
    size_t id = get_global_id(0);
    y[id * incy] += x[id * incx] * a;
}
