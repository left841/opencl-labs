get_filename_component(local_directory_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(sample_name "${global_directory_name}_${local_directory_name}")

project(${sample_name})

file(GLOB src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
file(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")

add_executable(${sample_name} ${src} ${headers})

target_include_directories(${sample_name} PUBLIC ${headers} ${common_headers} ${OpenCL_INCLUDE_DIRS})

target_compile_options(${sample_name} PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${sample_name} ${common_libs} ${OpenCL_LIBRARY})
