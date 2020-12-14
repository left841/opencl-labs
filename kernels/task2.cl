__kernel void saxpy(float a, __global float* x, ulong incx, __global float* y, ulong incy)
{
    size_t id = get_global_id(0);
    y[id * incy] += x[id * incx] * a;
}

__kernel void daxpy(double a, __global double* x, ulong incx, __global double* y, ulong incy)
{
    size_t id = get_global_id(0);
    y[id * incy] += x[id * incx] * a;
}
