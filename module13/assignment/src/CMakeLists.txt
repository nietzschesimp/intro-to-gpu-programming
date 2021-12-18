# Create executable for vector operations
add_executable(even_odd even_odd.cc opencl_utils.cc)
target_include_directories(even_odd PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(even_odd PUBLIC ${OpenCL_LIBRARIES})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/even_odd.cl ${CMAKE_BINARY_DIR}/src)
