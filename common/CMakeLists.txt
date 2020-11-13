project("common_lib")

file(GLOB src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
file(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")

add_library(common_libs ${src} ${headers})

target_include_directories(common_libs PUBLIC ${OpenCL_INCLUDE_DIRS})
