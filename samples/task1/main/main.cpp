#include <CL/cl.h>
#include <iostream>

int main(int argc, char** argv)
{
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);

    cl_platform_id* platform = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platform, nullptr);

    for (cl_uint i = 0; i < platformCount; ++i)
    {
        char platformName[128];
        clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 128, platformName, nullptr);
        std::cout << platformName << std::endl;
    }
    return 0;
}
