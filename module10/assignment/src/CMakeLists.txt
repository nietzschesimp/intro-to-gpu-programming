# Create executable for vector operations
add_executable(vector_operations vector_operations.cc)
target_include_directories(vector_operations PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(vector_operations PUBLIC ${OpenCL_LIBRARIES})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vector_operations.cl ${CMAKE_BINARY_DIR}/src)

