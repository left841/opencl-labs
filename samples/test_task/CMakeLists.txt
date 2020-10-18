get_filename_component(sample_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

project(${sample_name})

file(GLOB src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
file(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")

add_executable(${sample_name} ${src} ${headers})

target_include_directories(${sample_name} PUBLIC ${headers} ${OpenCL_INCLUDE_DIRS})

target_link_libraries(${sample_name} ${OpenCL_LIBRARY})
