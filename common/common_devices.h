#ifndef __COMMON_DEVICES_H__
#define __COMMON_DEVICES_H__

#include <fstream>
#include <CL/cl.h>

cl_device_id get_any_gpu_device();
cl_device_id get_any_cpu_device();

cl_program create_program_from_file(cl_context ctx, const char* file, cl_int* ret);

#endif // __COMMON_DEVICES_H__
