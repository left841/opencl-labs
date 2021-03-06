cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project("opencl_labs")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib/Release)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib/Debug)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib/Release)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB kernels RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "kernels/*")

foreach(kernel ${kernels})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${kernel}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}/${kernel}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${kernel}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/${kernel}")
endforeach(kernel ${kernels})

set(common_headers "${CMAKE_CURRENT_SOURCE_DIR}/common")

add_subdirectory(common)
add_subdirectory(samples)
